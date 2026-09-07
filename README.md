# MiniCPM-o 4.5 TTS DFlash

This project attaches a DFlash drafter (<https://huggingface.co/cs-qyzhang/MiniCPM-o-4_5-tts-dflash>) trained with `speculators` to the
Talker/TTS decoder of MiniCPM-o 4.5, and exposes three command-line entries:

- Simplex: text, image or video input is turned into speech through the official
  `model.chat()` path;
- Duplex: video and audio are streamed in one-second chunks while the model
  keeps listening and speaking, producing a time-aligned WAV, MP4 and JSON log;
- Benchmark: DFlash acceptance rate and accepted length per round are measured
  over a built-in suite of simplex samples.

DFlash accelerates only the TTS codec-token decoding performed by the 20-layer,
hidden-size-768 Talker. Text decoding by the main Qwen3 Thinker, as well as the
vision and audio encoders, still runs the original Hugging Face Transformers
remote code.

## Default paths

```text
base model: openbmb/MiniCPM-o-4_5
draft:      cs-qyzhang/MiniCPM-o-4_5-tts-dflash
duplex in:  ./omni_duplex1.mp4
voice ref:  <base model>/assets/HT_ref_audio.wav
```

`--model` and `--draft-model` accept either a Hugging Face repo id or a local
directory; when a repo id is given, the first run downloads it into the HF
cache. The `model.safetensors` of the draft checkpoint already contains
`t2d` / `d2t`, so neither a training-data directory nor an installation of
`speculators` is required at runtime.

## Environment

```bash
uv sync
```

An NVIDIA GPU and a system FFmpeg are required. Dependencies are managed by
`pyproject.toml` / `uv.lock`.

## Device placement

By default all three commands put every component on `cuda:0`: `--device`
selects the main device hosting the Thinker/Talker and DFlash, and Token2wav is
placed on the same card unless `--vocoder-device` is given. The duplex streaming
cache and the vision encoding of a video input both consume a substantial amount
of memory, so Token2wav can be split onto a second card when one card is not
enough:

```bash
uv run minicpmo-dflash-duplex --device cuda:0 --vocoder-device cuda:1
```

## Simplex TTS

```bash
uv run minicpmo-dflash-simplex \
  --text "你好，欢迎使用 MiniCPM-o 四点五 DFlash 语音合成。" \
  --tts-draft-temperature 0.8 \
  --output outputs/simplex_dflash.wav
```

`--ref-audio` changes the cloned voice, and `--device` / `--vocoder-device`
select the GPUs.

### Image/video plus text

`--image` / `--video` mix media with text input, and the two may be used
together. A video is expanded at one frame per second and carries its audio
track by default; `--no-video-audio` keeps only the picture:

```bash
uv run minicpmo-dflash-simplex \
  --image ./transformer.png \
  --output outputs/simplex_image_dflash.wav

uv run minicpmo-dflash-simplex \
  --video ./omni_duplex1.mp4 \
  --output outputs/simplex_video_dflash.wav
```

When `--text` is omitted, the default instruction for a media input is
“请用简洁的语言介绍这张图片/这段视频的内容。”, and `--max-new-tokens` defaults
to 4096, matching the official `chat()`. For a long video, raise
`--max-inp-length` accordingly; when a single card runs out of memory, move
Token2wav to a second card as described in “Device placement”.

## Duplex video

```bash
uv run minicpmo-dflash-duplex \
  --video ./omni_duplex1.mp4 \
  --tts-draft-temperature 0.8 \
  --output-video outputs/duplex_dflash.mp4 \
  --output-wav outputs/duplex_dflash.wav \
  --output-log outputs/duplex_dflash.json
```

The CLI prints the DFlash proposed/accepted counts for every speak chunk, and
summarizes the acceptance rate and EAL (draft tokens accepted per round on
average) at the end.

## DFlash sampling parameters

Both the simplex and the duplex commands accept the following parameters, and
the short aliases are available as well:

| Parameter | Default | Semantics |
|---|---:|---|
| `--tts-draft-temperature` / `--draft-temperature` | `0.8` | draft proposal temperature; `-1` follows the verifier automatically, `0` uses argmax |
| `--tts-draft-accept-top-k` / `--accept-top-k` | `0` | `0` uses distribution-preserving acceptance; a positive value enables lenient top-k acceptance |
| `--tts-draft-accept-min-prob` / `--accept-min-prob` | `0.0` | a positive value enables verifier-probability-threshold acceptance, which takes precedence over top-k |

Speculative decoding uses distribution-preserving speculative sampling: the
draft proposes tokens from `q = softmax(logits/T)`, the verifier accepts with
`min(1, p/q)`, and after a rejection the next token is resampled from the
residual distribution `max(0, p-q)`. Under the default configuration
(`accept-top-k=0` and `accept-min-prob=0`) the output distribution is therefore
identical to that of the original Talker, and `draft-temperature` affects only
the acceptance rate and the speed, never that distribution. A greedy verifier
switches the draft back to argmax automatically. Lenient top-k mode accepts more
proposals but no longer guarantees distribution preservation:

```bash
uv run minicpmo-dflash-duplex --accept-top-k 5
```

`--disable-dflash` skips loading the draft checkpoint and falls back to the
model's native Transformers TTS decoder:

```bash
# simplex
uv run minicpmo-dflash-simplex --disable-dflash --text "基线测试"

# duplex
uv run minicpmo-dflash-duplex --disable-dflash --video ./omni_duplex1.mp4
```

Text and speech generation follow the sampling configuration of the official
`chat()` by default; pass `--temperature` explicitly to override it. The draft
proposal temperature is the only non-official parameter introduced by this
project.

## DFlash acceptance benchmark

`minicpmo-dflash-benchmark` loads the model and the drafter once and then runs a
suite of diverse simplex samples in sequence (Chinese/English crossed with
read-aloud/QA/translation/style, including poetry, numbers, tongue twisters and
mixed-language text, 20 built-in samples in total). It records the DFlash
acceptance rate, the EAL (draft tokens accepted per round on average) and
tokens/round (codec tokens actually emitted per round) for each sample, and
aggregates them by language and by task category; the results are written to
JSON:

```bash
uv run minicpmo-dflash-benchmark \
  --output outputs/benchmark_dflash.json \
  --audio-dir outputs/benchmark
```

Filtering and extension parameters that are used most often:

- `--languages zh,en` / `--categories read,qa,translate,style` / `--ids ...` / `--limit N`:
  select a subset of the built-in samples;
- `--samples my_suite.json`: replace the built-in suite with a custom one, given
  as an array of objects shaped `{"id", "language", "category", "task": "read"|"answer", "text"}`;
- the DFlash sampling parameters (`--tts-draft-temperature`,
  `--tts-draft-accept-top-k`, `--tts-draft-accept-min-prob`) behave exactly as
  in the simplex and duplex commands, so acceptance rates under different
  acceptance strategies can be compared directly.

A failure on a single sample does not abort the whole suite. The JSON result
records the text answer, the codec-token count, the decoding time and the
end-to-end time for each sample; `acceptance_rate/EAL/tokens_per_round` in the
summary are micro-averaged over rounds, while `macro_eal` is the arithmetic mean
of the per-sample EAL.

## Implementation entry points

- `src/minicpmo_dflash/draft.py`: self-contained DFlash Qwen3 draft network and
  checkpoint loading;
- `src/minicpmo_dflash/speculative.py`: Talker KV cache, `p/q` acceptance,
  residual sampling and rollback;
- `src/minicpmo_dflash/simplex.py`: replacement for `MiniCPMTTS.generate`;
- `src/minicpmo_dflash/duplex.py`: replacement for `MiniCPMTTS.generate_chunk`;
- `src/minicpmo_dflash/benchmark.py`: acceptance-rate / EAL benchmark over the
  diverse simplex samples;
- `src/minicpmo_dflash/timeline.py`: shared timeline and subtitle alignment for
  the duplex WAV, subtitles and video.
