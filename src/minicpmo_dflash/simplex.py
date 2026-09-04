"""Single-turn TTS inference with MiniCPM-o 4.5 + DFlash."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before torch initializes CUDA. Video inputs run the Qwen3
# prefill within ~1 GB of a 24 GB card, where allocator fragmentation alone
# causes OOM; expandable segments reclaim most of it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import librosa
import torch
from PIL import Image

from .draft import load_draft_model
from .runtime import load_minicpmo, resolve_model_path, set_cuda_device
from .speculative import (
    DFlashDecoder,
    add_dflash_sampling_arguments,
    dflash_sampling_from_args,
)


DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_DRAFT = "cs-qyzhang/MiniCPM-o-4_5-tts-dflash"


def load_image(path: Path) -> Image.Image:
    """Flatten transparent images onto white; the vision tower expects RGB."""
    image = Image.open(path)
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    flattened = Image.new("RGB", rgba.size, (255, 255, 255))
    flattened.paste(rgba, mask=rgba.split()[-1])
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--no-video-audio", action="store_true")
    parser.add_argument("--max-slice-nums", type=int, default=None)
    parser.add_argument("--max-inp-length", type=int, default=8192)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--output", default="outputs/simplex_dflash.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vocoder-device",
        default=None,
        help="Token2wav device; defaults to --device. A separate GPU frees "
        "~2 GB on --device, which video inputs need for the vision tower and "
        "resampler.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Thinker text budget; 4096 matches the official chat() default.",
    )
    parser.add_argument("--max-tts-tokens", type=int, default=2048)
    parser.add_argument("--min-tts-tokens", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Thinker text-sampling temperature. Omitted by default so the "
        "official chat defaults apply: Thinker 0.7 (prepare_generation_config, "
        "do_sample=True) and Talker 0.8 (TTSSamplingParams).",
    )
    add_dflash_sampling_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dflash_sampling = dflash_sampling_from_args(args)
    device = torch.device(args.device)
    set_cuda_device(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    reference = Path(args.ref_audio or Path(model_path) / "assets" / "HT_ref_audio.wav")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {reference}")

    media_content: list = []
    if args.image is not None:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        media_content.append(load_image(image_path))
    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        media_content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": str(video_path),
                    "stack_frames": 1,
                    "use_ffmpeg": True,
                    "use_audio": not args.no_video_audio,
                },
            }
        )

    if args.text is None:
        if args.image is not None and args.video is not None:
            text = "请用简洁的语言介绍图片和视频的内容。"
        elif args.image is not None:
            text = "请用简洁的语言介绍这张图片的内容。"
        elif args.video is not None:
            text = "请用简洁的语言介绍这段视频的内容。"
        else:
            text = "你好，欢迎使用 MiniCPM-o 四点五 DFlash 语音合成。"
    else:
        text = args.text

    # Default slicing turns each 960x540 frame into 4x64 tokens and blows the
    # 8192-token budget within ~30s of video; one block per frame matches the
    # duplex streaming setup.
    max_slice_nums = args.max_slice_nums
    if max_slice_nums is None and args.video is not None:
        max_slice_nums = 1

    # The official non-streaming TTS path slices hidden states between
    # <|tts_bos|> and <|tts_eos|>; an answer truncated by the token budget
    # never reaches <|tts_eos|> and silently skips audio generation, so keep
    # the budget generous when lowering --max-new-tokens manually.

    print(f"Loading MiniCPM-o from {model_path}")
    vocoder_device = torch.device(args.vocoder_device) if args.vocoder_device else device
    # Token2wav is constructed on the current CUDA device; when it shares the
    # main card, load_minicpmo initializes it in place, otherwise initialize
    # it after switching the current device (same trick as duplex).
    model = load_minicpmo(
        model_path, device, initialize_vocoder=vocoder_device == device
    )
    if vocoder_device != device:
        set_cuda_device(vocoder_device)
        model.init_tts(model_dir=str(Path(model_path) / "assets" / "token2wav"))
        set_cuda_device(device)
    if args.video is not None:
        # The remote NavitSigLip vision tower only implements eager attention;
        # with the default vision_batch_size=16 a batch of 960x540 frames
        # (1032 patches each) spikes multi-GiB of attention weights and OOMs
        # next to the resident model. Duplex avoids this by feeding one frame
        # per call.
        model.config.vision_batch_size = 2
    decoder = None
    if args.disable_dflash:
        print("DFlash disabled; using the original Transformers TTS decoder")
    else:
        draft_path = resolve_model_path(args.draft_model)
        print(f"Loading DFlash from {draft_path}")
        draft = load_draft_model(draft_path, device)
        decoder = DFlashDecoder(
            model.tts,
            draft,
            device,
            seed=args.seed,
            sampling=dflash_sampling,
        )
        # The official non-streaming chat path calls MiniCPMTTS.generate.
        # Replacing only this method keeps the official prompt, dense
        # text-hidden condition, and Token2wav paths intact.
        def speculative_generate(*call_args, **call_kwargs):
            call_kwargs["max_new_token"] = args.max_tts_tokens
            call_kwargs["min_new_token"] = args.min_tts_tokens
            return decoder.generate(*call_args, **call_kwargs)

        model.tts.generate = speculative_generate

    reference_audio, _ = librosa.load(reference, sr=16000, mono=True)
    if media_content:
        system_instruction = "请用这种声音风格回答用户的问题，不要添加任何解释。"
        user_content = [*media_content, text]
    else:
        system_instruction = "请用这种声音风格直接朗读用户给出的内容，不要添加任何解释。"
        user_content = [f"请朗读以下内容： {text}"]
    messages = [
        {
            "role": "system",
            "content": [
                "模仿音频样本的音色并生成新的内容。",
                reference_audio,
                system_instruction,
            ],
        },
        {"role": "user", "content": user_content},
    ]
    # Token2wav allocates its intermediate tensors on the current CUDA
    # device, so keep it pointed at the vocoder card during chat(); Thinker,
    # Talker, and DFlash all use their explicit device arguments and are
    # unaffected (same arrangement as duplex).
    set_cuda_device(vocoder_device)
    try:
        chat_kwargs = dict(
            msgs=messages,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            use_tts_template=True,
            generate_audio=True,
            output_audio_path=str(output),
            max_inp_length=args.max_inp_length,
            max_slice_nums=max_slice_nums,
            # The remote code defaults to num_beams=3; beam search re-runs the
            # full sequence through the LM at finalization, which OOMs a 24 GB
            # card once media inputs push the prompt past a few thousand
            # tokens.
            **({"num_beams": 1} if media_content else {}),
        )
        # Omitting temperature keeps the official defaults: Thinker 0.7 via
        # prepare_generation_config, Talker 0.8 via TTSSamplingParams().
        if args.temperature is not None:
            chat_kwargs["temperature"] = args.temperature
        answer = model.chat(**chat_kwargs)
    finally:
        set_cuda_device(device)
    print(f"Text: {answer}")
    if decoder is not None:
        print(decoder.report())
    if not output.is_file():
        raise RuntimeError(
            "The model returned without writing the output WAV; the answer "
            "likely hit the --max-new-tokens budget before <|tts_eos|>, so "
            "raise the budget for longer answers"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
