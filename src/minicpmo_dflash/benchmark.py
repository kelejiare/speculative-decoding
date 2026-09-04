"""DFlash simplex benchmark: diverse prompts, per-sample EAL / tokens-per-round.

Loads the Thinker + DFlash drafter once, runs a built-in suite of diverse
simplex samples (Chinese/English, read-aloud, QA, translation, poetry,
numbers, tongue twisters, ...), and reports the speculative decoding
acceptance rate, EAL (accepted draft tokens per round) and emitted tokens
per round for each sample, per language, per task category, and overall.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Same rationale as simplex.py: keep the allocator from fragmenting on the
# 24 GB cards this project targets.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import librosa
import torch

from .draft import load_draft_model
from .runtime import load_minicpmo, resolve_model_path, set_cuda_device
from .speculative import (
    DFlashDecoder,
    DecodeStats,
    add_dflash_sampling_arguments,
    dflash_sampling_from_args,
)
from .simplex import DEFAULT_DRAFT, DEFAULT_MODEL


@dataclass(frozen=True)
class BenchmarkSample:
    """One benchmark prompt.

    ``task`` selects the chat framing: ``read`` wraps the text in the
    read-aloud instruction used by simplex.py, anything else (``answer``)
    uses the question-answering framing so the model speaks its own reply.
    """

    id: str
    language: str  # zh / en
    category: str  # read / qa / translate / style
    task: str  # read / answer
    text: str


BUILTIN_SAMPLES: list[BenchmarkSample] = [
    # --- Chinese, read-aloud ---
    BenchmarkSample(
        "zh-read-short", "zh", "read", "read",
        "你好，欢迎使用 MiniCPM-o 四点五 DFlash 语音合成基准测试。",
    ),
    BenchmarkSample(
        "zh-read-news", "zh", "read", "read",
        "据有关部门消息，今年前三季度全国铁路发送旅客突破三十亿人次，"
        "同比增长百分之八点六，其中动车组列车承担了主要运力。",
    ),
    BenchmarkSample(
        "zh-read-prose", "zh", "read", "read",
        "傍晚的湖面泛起一层金色的光，远处的山峦在云雾中若隐若现，"
        "微风拂过，带来阵阵草木的清香。",
    ),
    BenchmarkSample(
        "zh-read-poem", "zh", "read", "read",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    ),
    BenchmarkSample(
        "zh-read-tongue", "zh", "read", "read",
        "四是四，十是十，十四是十四，四十是四十。",
    ),
    BenchmarkSample(
        "zh-read-numbers", "zh", "read", "read",
        "订单编号 20260901，金额 128.5 元，将于 9 月 15 日前发货，"
        "客服电话 400-820-5638。",
    ),
    BenchmarkSample(
        "zh-read-mixed", "zh", "read", "read",
        "我们团队使用 PyTorch 和 Transformers 框架训练模型，"
        "在 A100 GPU 上大约跑了 72 个小时。",
    ),
    # --- Chinese, QA ---
    BenchmarkSample(
        "zh-qa-science", "zh", "qa", "answer",
        "为什么天空是蓝色的？请用两句话解释。",
    ),
    BenchmarkSample(
        "zh-qa-knowledge", "zh", "qa", "answer",
        "请用三句话介绍一下长城。",
    ),
    BenchmarkSample(
        "zh-qa-advice", "zh", "qa", "answer",
        "最近工作压力很大，请给我三个缓解压力的建议。",
    ),
    # --- Chinese, translation ---
    BenchmarkSample(
        "zh-trans-zh2en", "zh", "translate", "answer",
        "请把下面这句话翻译成英文并朗读出来：今天天气很好，我们去公园散步吧。",
    ),
    BenchmarkSample(
        "zh-trans-en2zh", "zh", "translate", "answer",
        "请把这句话翻译成中文并朗读出来：The early bird catches the worm.",
    ),
    # --- Chinese, style ---
    BenchmarkSample(
        "zh-style-emotion", "zh", "style", "answer",
        "请用开心的语气，给我的朋友说一句生日祝福。",
    ),
    # --- English, read-aloud ---
    BenchmarkSample(
        "en-read-short", "en", "read", "read",
        "Hello, welcome to the MiniCPM-o DFlash speech synthesis benchmark.",
    ),
    BenchmarkSample(
        "en-read-tech", "en", "read", "read",
        "Speculative decoding uses a small draft model to propose several "
        "tokens at once, and a larger verifier accepts or rejects them in "
        "a single forward pass.",
    ),
    BenchmarkSample(
        "en-read-quote", "en", "read", "read",
        "The only way to do great work is to love what you do. "
        "If you haven't found it yet, keep looking.",
    ),
    BenchmarkSample(
        "en-read-tongue", "en", "read", "read",
        "She sells seashells by the seashore, and the shells she sells "
        "are seashells, I'm sure.",
    ),
    # --- English, QA ---
    BenchmarkSample(
        "en-qa-science", "en", "qa", "answer",
        "What is photosynthesis? Answer in two short sentences.",
    ),
    BenchmarkSample(
        "en-qa-fact", "en", "qa", "answer",
        "Tell me one interesting fact about the ocean.",
    ),
    BenchmarkSample(
        "en-qa-math", "en", "qa", "answer",
        "If a train travels 120 kilometers in one and a half hours, "
        "what is its average speed?",
    ),
]


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vocoder-device",
        default=None,
        help="Token2wav device; defaults to --device. A separate GPU frees "
        "~2 GB on --device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Thinker text-sampling temperature. Omitted by default so the "
        "official chat defaults apply: Thinker 0.7 (prepare_generation_config, "
        "do_sample=True) and Talker 0.8 (TTSSamplingParams).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Thinker text budget; 4096 matches the official chat() default.",
    )
    parser.add_argument("--max-inp-length", type=int, default=8192)
    parser.add_argument("--max-tts-tokens", type=int, default=2048)
    parser.add_argument("--min-tts-tokens", type=int, default=50)
    parser.add_argument(
        "--samples",
        default=None,
        help="JSON file overriding the built-in suite; a list of objects "
        "with keys id, language, category, task (read/answer), text.",
    )
    parser.add_argument("--languages", default=None, help="Comma filter, e.g. zh,en")
    parser.add_argument(
        "--categories", default=None, help="Comma filter, e.g. read,qa,translate,style"
    )
    parser.add_argument("--ids", default=None, help="Comma-separated sample ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/benchmark_dflash.json",
        help="Where to write the JSON results.",
    )
    parser.add_argument(
        "--audio-dir",
        default="outputs/benchmark",
        help="Directory for the per-sample WAVs produced by the benchmark.",
    )
    add_dflash_sampling_arguments(parser)
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    if args.samples is not None:
        path = Path(args.samples).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = [BenchmarkSample(**item) for item in raw]
    else:
        samples = list(BUILTIN_SAMPLES)
    if args.languages:
        keep = {token.strip() for token in args.languages.split(",")}
        samples = [sample for sample in samples if sample.language in keep]
    if args.categories:
        keep = {token.strip() for token in args.categories.split(",")}
        samples = [sample for sample in samples if sample.category in keep]
    if args.ids:
        keep = {token.strip() for token in args.ids.split(",")}
        samples = [sample for sample in samples if sample.id in keep]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("The sample filters selected no benchmark samples")
    return samples


def build_messages(sample: BenchmarkSample, reference_audio) -> list[dict]:
    """Mirror simplex.py's chat framing for read-aloud vs. answer tasks."""
    if sample.task == "read":
        system_instruction = "请用这种声音风格直接朗读用户给出的内容，不要添加任何解释。"
        user_content = [f"请朗读以下内容： {sample.text}"]
    else:
        system_instruction = "请用这种声音风格回答用户的问题，不要添加任何解释。"
        user_content = [sample.text]
    return [
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


def snapshot_stats(stats: DecodeStats) -> tuple[int, int, int, int, float]:
    return (
        stats.rounds,
        stats.proposed,
        stats.accepted,
        stats.output_tokens,
        stats.seconds,
    )


def stats_metrics(
    rounds: int, proposed: int, accepted: int, output_tokens: int
) -> dict[str, float]:
    return {
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "eal": accepted / rounds if rounds else 0.0,
        "tokens_per_round": output_tokens / rounds if rounds else 0.0,
    }


def aggregate(records: list[dict]) -> dict:
    """Micro-average over ok records, plus macro EAL across samples."""
    usable = [record for record in records if record["status"] == "ok"]
    if not usable:
        return {"samples": 0}
    rounds = sum(record["rounds"] for record in usable)
    proposed = sum(record["proposed"] for record in usable)
    accepted = sum(record["accepted"] for record in usable)
    output_tokens = sum(record["output_tokens"] for record in usable)
    decode_seconds = sum(record["decode_seconds"] for record in usable)
    summary = {
        "samples": len(usable),
        "rounds": rounds,
        "proposed": proposed,
        "accepted": accepted,
        "output_tokens": output_tokens,
        "decode_seconds": round(decode_seconds, 3),
        "macro_eal": round(
            sum(record["eal"] for record in usable) / len(usable), 4
        ),
    }
    metrics = stats_metrics(rounds, proposed, accepted, output_tokens)
    summary.update({key: round(value, 4) for key, value in metrics.items()})
    return summary


def group_aggregate(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def print_table(title: str, records: list[dict], overall: dict | None = None) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'id':<20} {'lang':<4} {'category':<9} {'rounds':>6} {'acc%':>6} "
        f"{'EAL':>6} {'tok/rnd':>7} {'codec':>6} {'decode_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        if record["status"] != "ok":
            print(f"{record['id']:<20} {record['language']:<4} "
                  f"{record['category']:<9} {'[' + record['status'] + ']':>6}")
            continue
        print(
            f"{record['id']:<20} {record['language']:<4} "
            f"{record['category']:<9} {record['rounds']:>6} "
            f"{100 * record['acceptance_rate']:>6.1f} {record['eal']:>6.2f} "
            f"{record['tokens_per_round']:>7.2f} {record['output_tokens']:>6} "
            f"{record['decode_seconds']:>8.2f}"
        )
    if overall and overall.get("samples"):
        print("-" * len(header))
        print(
            f"{'overall':<20} {'':<4} "
            f"{'':<9} {overall['rounds']:>6} "
            f"{100 * overall['acceptance_rate']:>6.1f} {overall['eal']:>6.2f} "
            f"{overall['tokens_per_round']:>7.2f} {overall['output_tokens']:>6} "
            f"{overall['decode_seconds']:>8.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.disable_dflash:
        raise ValueError(
            "--disable-dflash is incompatible with the benchmark: acceptance "
            "statistics require the DFlash drafter."
        )
    dflash_sampling = dflash_sampling_from_args(args)
    samples = load_samples(args)

    device = torch.device(args.device)
    set_cuda_device(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_json = Path(args.output).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    reference = Path(args.ref_audio or Path(model_path) / "assets" / "HT_ref_audio.wav")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {reference}")

    print(f"Loading MiniCPM-o from {model_path}")
    vocoder_device = torch.device(args.vocoder_device) if args.vocoder_device else device
    model = load_minicpmo(
        model_path, device, initialize_vocoder=vocoder_device == device
    )
    if vocoder_device != device:
        set_cuda_device(vocoder_device)
        model.init_tts(model_dir=str(Path(model_path) / "assets" / "token2wav"))
        set_cuda_device(device)

    draft_path = resolve_model_path(args.draft_model)
    print(f"Loading DFlash from {draft_path}")
    draft = load_draft_model(draft_path, device)
    decoder = DFlashDecoder(
        model.tts,
        draft,
        device,
        seed=args.seed,
        sampling=dflash_sampling,
        verbose=False,
    )

    def speculative_generate(*call_args, **call_kwargs):
        call_kwargs["max_new_token"] = args.max_tts_tokens
        call_kwargs["min_new_token"] = args.min_tts_tokens
        return decoder.generate(*call_args, **call_kwargs)

    model.tts.generate = speculative_generate

    reference_audio, _ = librosa.load(reference, sr=16000, mono=True)

    records: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        audio_path = audio_dir / f"{sample.id}.wav"
        print(f"\n[{index}/{len(samples)}] {sample.id} "
              f"({sample.language}/{sample.category}): {sample.text[:40]}...")
        # Re-seed both random streams per sample so every sample is
        # individually reproducible regardless of what ran before it.
        if decoder.runner.generator is not None:
            decoder.runner.generator.manual_seed(args.seed)
        decoder.draft_generator.manual_seed(5489)

        record = asdict(sample)
        record |= {
            "status": "ok",
            "answer": None,
            "rounds": 0,
            "proposed": 0,
            "accepted": 0,
            "output_tokens": 0,
            "truncated": False,
            "acceptance_rate": 0.0,
            "eal": 0.0,
            "tokens_per_round": 0.0,
            "decode_seconds": 0.0,
            "wall_seconds": 0.0,
            "audio": str(audio_path),
        }
        before = snapshot_stats(decoder.total)
        start = time.perf_counter()
        try:
            messages = build_messages(sample, reference_audio)
            # Token2wav allocates on the current CUDA device (same as simplex).
            set_cuda_device(vocoder_device)
            try:
                chat_kwargs = dict(
                    msgs=messages,
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    use_tts_template=True,
                    generate_audio=True,
                    output_audio_path=str(audio_path),
                    max_inp_length=args.max_inp_length,
                )
                # Omitting temperature keeps the official defaults: Thinker
                # 0.7 via prepare_generation_config, Talker 0.8 via
                # TTSSamplingParams().
                if args.temperature is not None:
                    chat_kwargs["temperature"] = args.temperature
                answer = model.chat(**chat_kwargs)
            finally:
                set_cuda_device(device)
            record["answer"] = answer
            rounds, proposed, accepted, output_tokens, decode_seconds = (
                after - pre
                for after, pre in zip(snapshot_stats(decoder.total), before)
            )
            record.update(
                rounds=rounds,
                proposed=proposed,
                accepted=accepted,
                output_tokens=output_tokens,
                truncated=bool(decoder.last_stats.truncated),
                decode_seconds=round(decode_seconds, 3),
            )
            record.update(
                {
                    key: round(value, 4)
                    for key, value in stats_metrics(
                        rounds, proposed, accepted, output_tokens
                    ).items()
                }
            )
            if not audio_path.is_file():
                # Text answer hit --max-new-tokens before <|tts_eos|>; the
                # official path silently skips audio generation.
                record["status"] = "no_audio"
        except Exception as error:  # Keep running the rest of the suite.
            record["status"] = "error"
            record["error"] = repr(error)
        record["wall_seconds"] = round(time.perf_counter() - start, 3)
        if record["status"] == "ok":
            print(
                f"  accepted {record['accepted']}/{record['proposed']} "
                f"({record['acceptance_rate']:.3f}), EAL={record['eal']:.2f}, "
                f"tokens/round={record['tokens_per_round']:.2f}, "
                f"{record['output_tokens']} codec tokens, "
                f"decode {record['decode_seconds']:.2f}s"
            )
        else:
            print(f"  [{record['status']}] {record.get('error', '')}")
        records.append(record)

    summary = {
        "overall": aggregate(records),
        "by_language": group_aggregate(records, "language"),
        "by_category": group_aggregate(records, "category"),
    }
    print_table("Per-sample results", records, overall=summary["overall"])

    result = {
        "config": {
            "model": args.model,
            "draft_model": args.draft_model,
            "device": str(device),
            "seed": args.seed,
            "thinker_temperature": (
                args.temperature if args.temperature is not None else 0.7
            ),
            "tts_temperature": 0.8,  # official TTSSamplingParams default
            "tts_sampling_note": (
                "Talker top_k/top_p truncation is intentionally disabled in "
                "this implementation (full-vocab multinomial, see README)"
            ),
            "draft_temperature": dflash_sampling.draft_temperature,
            "accept_top_k": dflash_sampling.accept_top_k,
            "accept_min_prob": dflash_sampling.accept_min_prob,
            "block_tokens": draft.block_size - 1,
            "max_new_tokens": args.max_new_tokens,
            "max_tts_tokens": args.max_tts_tokens,
        },
        "samples": records,
        "summary": summary,
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved results: {output_json}")
    print(f"Saved audio:   {audio_dir}")


if __name__ == "__main__":
    main()
