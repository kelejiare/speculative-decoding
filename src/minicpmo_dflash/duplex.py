"""Recorded full-duplex inference with MiniCPM-o 4.5 + DFlash TTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf
import torch

from .draft import load_draft_model
from .runtime import load_minicpmo, resolve_model_path, set_cuda_device
from .simplex import DEFAULT_DRAFT, DEFAULT_MODEL
from .speculative import (
    DFlashDecoder,
    add_dflash_sampling_arguments,
    dflash_sampling_from_args,
)
from .timeline import mix_timeline, render_duplex_video, to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="./omni_duplex1.mp4")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--output-video", default="outputs/duplex_dflash.mp4")
    parser.add_argument("--output-wav", default="outputs/duplex_dflash.wav")
    parser.add_argument("--output-log", default="outputs/duplex_dflash.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vocoder-device",
        default=None,
        help="Token2wav device; defaults to --device. A separate GPU avoids "
        "24 GB duplex cache OOM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--batch-vision-feed", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default="Streaming Omni Conversation. Respond naturally and concisely.",
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

    video_path = Path(args.video).expanduser().resolve()
    model_path = resolve_model_path(args.model)
    reference = Path(args.ref_audio or Path(model_path) / "assets" / "HT_ref_audio.wav")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {reference}")
    output_video = Path(args.output_video).expanduser().resolve()
    output_wav = Path(args.output_wav).expanduser().resolve()
    output_log = Path(args.output_log).expanduser().resolve()
    for path in (output_video, output_wav, output_log):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MiniCPM-o from {model_path}")
    # as_duplex() initializes Token2wav itself. Loading it here as well would
    # keep two ~2 GB vocoders alive and exhaust a 24 GB card.
    model = load_minicpmo(model_path, device, initialize_vocoder=False)
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
        model.tts.generate_chunk = decoder.generate_chunk

    from minicpmo.utils import get_video_frame_audio_segments

    print(f"Extracting one-second chunks from {video_path}")
    frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        str(video_path), stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
    )
    print(f"Input chunks: {len(audio_segments)}")
    reference_audio, _ = librosa.load(reference, sr=16000, mono=True)
    # stepaudio2-minicpmo 0.1.1 uses the current CUDA device internally, so
    # Token2wav lands on ``--vocoder-device`` while Thinker/Talker/DFlash stay
    # on ``--device``. Both default to the same card; splitting them leaves
    # headroom for the duplicated stream cache at turn boundaries.
    vocoder_device = (
        torch.device(args.vocoder_device) if args.vocoder_device else device
    )
    set_cuda_device(vocoder_device)
    duplex = model.as_duplex(
        device=str(device), generate_audio=True, enable_float16=False
    )
    duplex.prepare(
        prefix_system_prompt=args.system_prompt,
        ref_audio=reference_audio,
        prompt_wav_path=str(reference),
    )

    results_log: list[dict] = []
    timed_output_audio: list[tuple[int, object]] = []
    for chunk_index, audio_chunk in enumerate(audio_segments):
        frame_list = []
        if chunk_index < len(frames) and frames[chunk_index] is not None:
            frame_list.append(frames[chunk_index])
            if (
                stacked_frames is not None
                and chunk_index < len(stacked_frames)
                and stacked_frames[chunk_index] is not None
            ):
                frame_list.append(stacked_frames[chunk_index])
        duplex.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list,
            max_slice_nums=args.max_slice_nums,
            batch_vision_feed=args.batch_vision_feed,
        )
        result = duplex.streaming_generate(
            prompt_wav_path=str(reference),
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens_per_chunk,
            decode_mode="sampling",
        )
        waveform = result.get("audio_waveform")
        if waveform is not None:
            timed_output_audio.append((chunk_index, waveform))
        chunk_stats = (
            decoder.last_stats
            if decoder is not None and not result["is_listen"]
            else None
        )
        record = {
            "chunk_idx": chunk_index,
            "is_listen": bool(result["is_listen"]),
            "text": result.get("text", ""),
            "end_of_turn": bool(result.get("end_of_turn", False)),
            "current_time": int(result.get("current_time", chunk_index)),
            "audio_samples": len(to_numpy(waveform)) if waveform is not None else 0,
            "tts_codec_tokens": chunk_stats.output_tokens if chunk_stats else 0,
            "dflash_proposed": chunk_stats.proposed if chunk_stats else 0,
            "dflash_accepted": chunk_stats.accepted if chunk_stats else 0,
        }
        results_log.append(record)
        if record["is_listen"]:
            print(f"[{chunk_index:03d}] listen...")
        else:
            print(f"[{chunk_index:03d}] speak> {record['text']}")

    output_log.write_text(
        json.dumps(results_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sf.write(
        output_wav,
        mix_timeline(timed_output_audio, len(audio_segments)),
        24000,
    )
    render_duplex_video(
        video_path=str(video_path),
        output_video_path=str(output_video),
        results_log=results_log,
        timed_output_audio=timed_output_audio,
        ai_audio_path=str(output_wav),
    )
    if decoder is not None:
        print(decoder.report())
    print(f"Saved WAV:   {output_wav}")
    print(f"Saved video: {output_video}")
    print(f"Saved log:   {output_log}")


if __name__ == "__main__":
    main()
