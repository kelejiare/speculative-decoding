"""Model loading and compatibility helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def patch_transformers_cache() -> None:
    """Make MiniCPM-o remote code compatible with Transformers 5 cache APIs."""
    import transformers

    if int(transformers.__version__.split(".", 1)[0]) < 5:
        return
    from transformers import cache_utils
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, legacy_cache):
            cache = cls()
            for layer_index, layer_past in enumerate(legacy_cache or ()):
                cache.update(layer_past[0], layer_past[1], layer_index)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache

    def legacy_getitem(cache, layer_index):
        layer = cache.layers[layer_index]
        return layer.keys, layer.values

    if not hasattr(DynamicCache, "__getitem__"):
        DynamicCache.__getitem__ = legacy_getitem
    encoder_decoder = getattr(cache_utils, "EncoderDecoderCache", None)
    if encoder_decoder is not None and not hasattr(encoder_decoder, "__getitem__"):
        encoder_decoder.__getitem__ = (
            lambda self, index: legacy_getitem(self.self_attention_cache, index)
        )
    if hasattr(DynamicCache, "key_cache"):
        return

    class CacheTensorList(list):
        def __init__(self, cache, attribute):
            super().__init__(getattr(layer, attribute) for layer in cache.layers)
            self.cache = cache
            self.attribute = attribute

        def __setitem__(self, index, value):
            super().__setitem__(index, value)
            if isinstance(index, slice):
                for layer, tensor in zip(self.cache.layers[index], value):
                    setattr(layer, self.attribute, tensor)
            else:
                setattr(self.cache.layers[index], self.attribute, value)

    def make_view(attribute):
        def getter(cache):
            return CacheTensorList(cache, attribute)

        def setter(cache, tensors):
            for layer, tensor in zip(cache.layers, tensors):
                setattr(layer, attribute, tensor)

        return property(getter, setter)

    DynamicCache.key_cache = make_view("keys")
    DynamicCache.value_cache = make_view("values")
    dynamic_layer = getattr(cache_utils, "DynamicLayer", None)
    if dynamic_layer is not None and not hasattr(dynamic_layer, "key_cache"):
        dynamic_layer.key_cache = property(
            lambda self: self.keys, lambda self, value: setattr(self, "keys", value)
        )
        dynamic_layer.value_cache = property(
            lambda self: self.values, lambda self, value: setattr(self, "values", value)
        )


def patch_whisper_attention(model: Any) -> None:
    """Adapt the pre-5.x remote Whisper wrapper to Transformers 5."""
    import transformers

    if int(transformers.__version__.split(".", 1)[0]) < 5:
        return
    try:
        from transformers.models.whisper.modeling_whisper import WhisperAttention
    except ImportError:
        return
    for module in model.modules():
        if not isinstance(module, WhisperAttention) or getattr(
            module, "_minicpmo_dflash_patched", False
        ):
            continue
        original = module.forward

        def make_wrapper(function):
            def wrapped(*args, **kwargs):
                if "past_key_value" in kwargs:
                    kwargs["past_key_values"] = kwargs.pop("past_key_value")
                result = function(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    return result[0], result[1], kwargs.get("past_key_values")
                return result

            return wrapped

        module.forward = make_wrapper(original)
        module._minicpmo_dflash_patched = True


def patch_torchaudio_io() -> None:
    """Avoid torchaudio's optional torchcodec dependency for plain WAV I/O."""
    try:
        import soundfile as sf
        import torchaudio
    except ImportError:
        return
    if getattr(torchaudio, "_minicpmo_dflash_patched", False):
        return

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **_):
        del normalize
        stop = None if num_frames is None or num_frames < 0 else frame_offset + num_frames
        data, sample_rate = sf.read(
            uri, start=frame_offset, stop=stop, dtype="float32", always_2d=True
        )
        array = data.T.copy() if channels_first else data.copy()
        return torch.from_numpy(array), sample_rate

    def save(uri, source, sample_rate, channels_first=True, **_):
        array = source.detach().cpu().float().numpy()
        if array.ndim == 1:
            array = array[:, None]
        elif channels_first:
            array = array.T
        sf.write(uri, array, int(sample_rate), format="WAV")

    torchaudio.load = load
    torchaudio.save = save
    torchaudio._minicpmo_dflash_patched = True


def resolve_model_path(model_id: str | Path) -> str:
    """Resolve a local checkpoint path or a Hugging Face repo id.

    Absolute or existing local paths are returned unchanged. Anything else is
    treated as a repo id and materialised with ``snapshot_download`` into the
    HF cache, so downstream ``assets`` lookups (reference audio, token2wav)
    keep working on local files; repeated runs hit the cache.
    """
    from huggingface_hub import snapshot_download

    path = Path(model_id).expanduser()
    if path.is_absolute() or path.is_dir():
        return str(path.resolve())
    return snapshot_download(str(model_id))


def load_minicpmo(
    model_path: str | Path,
    device: torch.device | str,
    *,
    initialize_vocoder: bool,
    token2wav_path: str | Path | None = None,
) -> Any:
    """Load MiniCPM-o from a local checkpoint or a Hugging Face repo id."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = resolve_model_path(model_path)
    device = torch.device(device)
    patch_transformers_cache()
    patch_torchaudio_io()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tts_config = getattr(config, "tts_config", None)
    if tts_config is not None:
        for name, default in (("top_p", 1.0), ("top_k", 0), ("repetition_penalty", 1.0)):
            if not hasattr(tts_config, name):
                setattr(tts_config, name, default)

    model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_path)
    if not getattr(model_class, "_minicpmo_dflash_patched", False):
        original_init = model_class.__init__

        def init_with_post_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.post_init()

        model_class.__init__ = init_with_post_init
        model_class._minicpmo_dflash_patched = True

    import transformers

    dtype_keyword = (
        {"dtype": torch.bfloat16}
        if int(transformers.__version__.split(".", 1)[0]) >= 5
        else {"torch_dtype": torch.bfloat16}
    )
    model = model_class.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        **dtype_keyword,
    ).to(device)
    model.eval()
    patch_whisper_attention(model)
    if initialize_vocoder:
        assets = token2wav_path or Path(model_path) / "assets" / "token2wav"
        model.init_tts(model_dir=str(assets))
    return model


def set_cuda_device(device: torch.device) -> None:
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
