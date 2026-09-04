# MiniCPM-o 4.5 TTS DFlash Drafter Training Guide

> The author is not deeply familiar with TTS models or their training. This
> document records faithfully the data construction and training configuration
> used for the DFlash drafter in this repository, as a reference for
> reproduction; the recipe is not optimal, and the corpus mixture, the
> hyperparameters and the draft depth were each ablated only incompletely.

## 1. Background

### 1.1 MiniCPM-o 4.5 and its TTS decoder

MiniCPM-o 4.5 consists of three parts: the Thinker (an 8B Qwen3 LLM) handles
multimodal understanding and text decoding, the Talker decodes the TTS codec
tokens, and Code2Wav turns the codec back into a 24 kHz waveform. The Talker is
an autoregressive decoder with a Llama architecture, 20 layers and a hidden size
of 768; its vocabulary holds 6562 audio codec tokens, and the last id (6561) is
the codec EOS. The Talker takes a conditioned input (the sequence of text tokens
produced by the Thinker) and emits audio tokens one step at a time
autoregressively. Being a standard autoregressive model, it can in principle be
accelerated directly with speculative decoding.

### 1.2 Key design: the draft is fed embeddings rather than token ids

Generic speculative-decoding training frameworks (EAGLE, EAGLE-3, DFlash and the
like) assume by default that the draft model consumes the same input as the
verifier, namely token ids, and reconstruct the input during training with
`embed_tokens(input_ids)`. The input at each Talker position, however, is not a
plain token-embedding lookup:

- the condition part (the text tokens given by the Thinker) is computed per
  position as `emb_text(token) + projector_semantic(thinker_hidden_state)`, with
  the projection L2-normalized according to the checkpoint configuration; that
  is, an element-wise sum of the text embedding and the projected LLM-side
  features;
- the codec feedback part (the audio tokens already generated) is
  `emb_code[0](audio_code)`.

If a token-id-based training framework were reused, the draft model would see
only the lookup embedding at the anchor position and could not learn the input
representation that the verifier actually consumes. This inference has not been
verified rigorously, but the pipeline described here proceeds under that
assumption. Each training sample therefore carries the per-position input
embedding directly: the draft's anchor slot is filled with the recorded verifier
input embedding instead of a token id. This is the main difference between the
training pipeline described here and conventional LLM drafter training.

### 1.3 The DFlash drafter

DFlash is a block-diffusion drafter derived from Z Lab's block-diffusion
research and implemented inside the `speculators` framework. In each speculative
round the draft performs a single forward pass: the input is `block_size` slots
arranged as `[anchor token, <MASK> × (block_size-1)]`, attention is
bidirectional within the block and can see the context preceding the anchor. The
verifier's hidden states from several layers are compressed by `fc + RMSNorm`
and injected into the K/V of every draft layer (KV-injection attention); this is
how the draft perceives the verifier's internal features. One forward pass fills
in `block_size - 1` speculative tokens (`block_size=8` in this work, that is
seven tokens), the verifier checks the whole block in a single forward pass,
accepts the longest correct prefix and adds one bonus token. The embedding,
`lm_head` and `verifier_norm` are copied from the verifier and frozen; the
trainable parameters are only `fc`, the two norms and the N draft decoder
layers.

All of these mechanisms come from upstream `speculators`; the only change this
work makes to the framework is the one described in Section 1.2, replacing token
ids with embeddings as the input.

## 2. Training pipeline overview

```
MiniCPM-o-4_5 checkpoint
  │ Step 1: Talker extraction, stripped out as a standalone LlamaForCausalLM verifier
  ▼
verifier checkpoint
  │ Step 2: teacher-forcing bulk recording (vllm-omni, Code2Wav skipped)
  │         five corpora; input_embeds / hidden_states / codec trajectory per request
  ▼
recorded data
  │ Step 3: merge and finalize (truncated samples filtered out)
  ▼
training dataset (459k samples)
  │ Step 4: DFlash training (speculators, embeddings as input)
  ▼
DFlash checkpoint (default drafter of this repository, 4 layers / FFN 2304 / full attention / block 8)
```

## 3. Corpus selection and request construction

### 3.1 Corpora

The condition input of the Talker contains only the text-token embeddings
produced by the LLM side, without modality features such as image or audio, so
building the dataset requires only plain-text corpora. In addition, TTS is used
mostly for everyday conversation rather than the coding / math / QA traffic
typical of chatbots, and MiniCPM-o 4.5 targets mainly Chinese and English. The
following Chinese and English utterance corpora were selected accordingly:

| Corpus | Language | Request construction | Notes |
| --- | --- | --- | --- |
| DailyDialog | English | mixed (see 3.2) | everyday multi-turn conversation |
| Topical-Chat | English | mixed | topical chit-chat |
| NaturalConv | Chinese | mixed | everyday multi-turn conversation |
| LCCC (base) | Chinese | mixed | utterances are detokenized by removing spaces first |
| open-perfectblend | mostly English | single-turn teacher forcing | only samples whose assistant output contains no newline are kept, 15,000 of them |

perfectblend was added because the TTS output shows a noticeable accuracy
problem when the LLM-side output contains newlines, as in lists or code blocks.
To keep the training distribution close to text that is genuinely suitable for
reading aloud, only samples whose assistant part is a single line were kept from
perfectblend.

### 3.2 The repeat / multi-turn / mixed request constructions

The goal is to train the TTS rather than the LLM, so there is no need to let the
Thinker generate the assistant reply autoregressively; a single teacher-forcing
prefill suffices (see Section 4.1). Every training request therefore carries its
own answer, and there are two basic construction modes:

1. repeat mode: one utterance from the corpus serves as both the input and the
   output, with the user turn being `Please repeat the sentence: <utterance>`
   and the assistant turn being `<utterance>`. The TTS then generates codec
   conditioned on already seeing what it is about to say, which is the most
   direct form of coverage.
2. multi-turn mode: for corpora that are already multi-turn conversations, the
   k-th request carries the first 2k utterances alternating as user / assistant
   turns (cumulatively), and the last assistant turn is the teacher-forcing
   target.

For the four conversational corpora (DailyDialog, LCCC, NaturalConv and
Topical-Chat) the final choice is a mixed combination of the two: dialogue turns
are numbered from 1, odd turns (user turns) produce a repeat request for that
turn's utterance, and even turns (assistant turns) produce a multi-turn request
accumulated up to that turn. For example, given the raw corpus

```json
["你好，最近在忙什么？", "我在学吉他，挺有意思的。"]
```

two requests are generated:

```json
[
  { "role": "system",    "content": "You are MiniCPM-o, a helpful multimodal assistant that can understand images, audio and video, and respond in text and speech." },
  { "role": "user",      "content": "Please repeat the sentence: 你好，最近在忙什么？" },
  { "role": "assistant", "content": "你好，最近在忙什么？" }
]
```

```json
[
  { "role": "system",    "content": "You are MiniCPM-o, a helpful multimodal assistant that can understand images, audio and video, and respond in text and speech." },
  { "role": "user",      "content": "你好，最近在忙什么？" },
  { "role": "assistant", "content": "我在学吉他，挺有意思的。" }
]
```

Under the mixed mode the same utterance appears both as a repeat target and as
multi-turn context, covering the conditional distributions of reading a short
sentence aloud and of continuing a real conversation at the same time.

## 4. Data generation (vllm-omni)

### 4.1 The teacher-forcing recording design

The recording pipeline is implemented on a vllm-omni fork, with the following
design:

1. Two-stage dump pipeline: during recording the pipeline is trimmed from three
   stages to Thinker + Talker, and Code2Wav is not loaded at all. Code2Wav only
   affects waveform synthesis and is irrelevant to drafter training, so removing
   it speeds up data generation considerably.
2. Teacher forcing: the serving layer wraps the assistant part carried by the
   request into `<|tts_bos|>answer<|tts_eos|>`, disables
   `add_generation_prompt` and `continue_final_message`, and fixes the Thinker's
   `max_tokens` to 1. The Thinker then performs a single prefill and yields the
   hidden states for the whole assistant segment, skipping the decode stage
   entirely; the hidden states are handed to the Talker as usual and the Talker
   runs fully autoregressively. `use_tts_template=true` and
   `enable_thinking=false` are forced at the same time so that the
   `<|tts_bos|>` boundary matches the real audio path exactly.
3. Prefix caching disabled: prefix caching skips re-embedding the cached prefix,
   which leaves gaps in the recorded `inputs_embeds`, so it has to be turned off
   in recording mode.
4. Capture layers: the hidden states of Talker layers `2, 6, 10, 14, 17` are
   recorded, five layers spread evenly over the 20-layer Talker, and the last
   layer (layer 20) is appended automatically by the framework as the source of
   the verifier teacher logits. Once this layer configuration is fixed, the
   training side has to match it verbatim.

### 4.2 The recorded data format

Each request is recorded as one self-contained record, which forms the format
contract between this pipeline and any reimplementation:

| Field | Shape | Meaning |
| --- | --- | --- |
| `input_embeds` | `[C+2+T, 768]` | the actual input embedding at each Talker position: the first `C+2` are the condition (`emb_text + projector_semantic(thinker_hidden)` summed per position, plus the two boundaries `<text_eos>` and `<audio_bos>`), followed by the codec feedback `emb_code[0](audio_codes[t-1])` |
| `hidden_states` | `[C+2+T, 6, 768]` | the capture layers `{2,6,10,14,17}` plus the last layer, which is a pre-final-norm state, so the training framework applies `verifier_norm` itself |
| `loss_mask` | `[C+2+T]` | 0 over the condition prefix, 1 at the codec positions |
| codec trajectory | `[T]` | the audio-code sequence actually sampled by the Talker; EOS is not written into the sequence and serves as the last supervision target |

Here `C` is the number of condition tokens and `T` the number of generated codec
tokens. What is recorded is the server's real sampling trajectory rather than a
deterministic replay, and the sampling parameters are saved together with each
sample.

After recording, a merge-and-finalize pass is run: only samples whose codec is
non-empty and ends normally with EOS are kept, and those truncated by the Talker
step limit are discarded. The vocabulary mappings `d2t` / `t2d` are identity
mappings, so draft ids `0..6561` correspond one-to-one with codec ids
`0..6561`, and the DFlash mask token `6562` is not part of the prediction
vocabulary. The output is an Arrow dataset consumed by `speculators`.

The five corpora merge into 460k samples in total; the tensor payload is about
10.5 KB per token in bf16, for a total size of roughly 1 TB, so reproduction
requires enough disk space reserved in advance.

## 5. Training (speculators)

### 5.1 Modifications made to speculators

Relative to upstream speculators, the changes are concentrated on taking
embeddings rather than token ids as input:

1. Data pipeline: a training sample is allowed to carry only `input_embeds`
   without `input_ids`;
2. DFlash core: the anchor slot (slot 0 of every block) no longer receives a
   token id; it is overwritten directly with the verifier's per-position input
   embedding.

### 5.2 Extracting the Talker as a standalone verifier

Before training, the Talker is stripped out of the MiniCPM-o 4.5 checkpoint into
a standalone `LlamaForCausalLM` directory: `tts.emb_code.0` becomes
`model.embed_tokens`, `tts.head_code.0` (with weight_norm recomposed) becomes
`lm_head`, and `tts.model.norm` becomes `model.norm`, keeping the 20-layer
backbone intact. One row is appended to both the embedding and `lm_head` as the
DFlash mask token (id 6562), ensuring that the mask token never collides with a
real codec id. The training framework loads the frozen verifier weights from
here and computes the teacher logits.

### 5.3 Draft architecture

The draft decoder architecture selected after ablation (see Section 5.5), as a
Qwen3Config JSON:

```json
{
  "model_type": "qwen3",
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 768,
  "intermediate_size": 2304,
  "num_hidden_layers": 4,
  "num_attention_heads": 12,
  "num_key_value_heads": 12,
  "head_dim": 64,
  "vocab_size": 6563,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]
}
```

That is, a 4-layer draft decoder with the Qwen3 architecture: hidden size 768
(matching the Talker), FFN 2304, 12-head full attention with no sliding window,
and a vocabulary of 6563 (6562 codec tokens plus one mask token). It holds about
30M parameters, and at inference the cost of one draft forward pass is far below
that of the 20-layer verifier.

### 5.4 Training configuration

The complete training configuration is as follows; the entry point is the DFlash
training script of speculators, the data is the merged dataset from Section 4.2,
and the verifier is the extraction product from Section 5.2:

| Item | Value | Notes |
| --- | --- | --- |
| speculator type | dflash (`sample_from_anchor=false`) | |
| block size | 8 | eight slots per block, that is seven speculative tokens per round; written into the checkpoint |
| target layer ids | `2 6 10 14 17` | the verifier auxiliary layers used for KV injection, identical to the recorded layer configuration |
| loss | KL (draft logits vs teacher logits) | positional decay `fixed-exp-decay` (gamma 4.0, the upstream default) |
| optimizer / lr | Muon / 3e-4 | the lr matches the official DFlash example; the optimizer was not compared rigorously |
| epochs | 8 | `checkpoint_best`, the one with the lowest validation loss, is used |
| sampling alignment (optional variant) | temperature 0.8 / top-k 25 / top-p 0.85 | aligns the training distribution with the truncated sampling of the verifier at deployment (only the temperature applies on the draft side) |

### 5.5 Training results and ablations

The complete training process is recorded publicly in a wandb project, see
[MiniCPM-o 4.5 tts dflash](https://wandb.ai/cs-qyzhang-hust/MiniCPM%20o%204.5%20tts%20dflash).

Final metrics measured end to end with `minicpmo-dflash-benchmark` over the 20
built-in simplex samples (seed 42), with the draft architecture selected above
(4 layers / FFN 2304 / full attention / block size 8), draft temperature 0.8,
verifier temperature 0.8 over the full codec vocabulary, distribution-preserving
acceptance (`accept-top-k=0`, `accept-min-prob=0`) and seven speculative tokens
per round:

| Group | Samples | Rounds | Acceptance rate | EAL | tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 20 | 1724 | 22.24% | 1.5568 | 2.5516 |
| en | 7 | 602 | 25.72% | 1.8007 | 2.7973 |
| zh | 13 | 1122 | 20.37% | 1.4260 | 2.4198 |
| qa | 6 | 812 | 23.82% | 1.6675 | 2.6626 |
| read | 11 | 815 | 21.17% | 1.4822 | 2.4785 |
| style | 1 | 42 | 20.07% | 1.4048 | 2.4048 |
| translate | 2 | 55 | 16.36% | 1.1455 | 2.1091 |

The `overall` row covers the whole suite, `en` / `zh` split it by language and
the last four rows split it by task category. Of the 12068 proposed tokens 2684
were accepted and 4399 codec tokens were emitted, in 46.5 s of decoding. EAL is
out of the seven tokens proposed per round; the acceptance rate, EAL and
tokens/round are micro-averaged over rounds, while the arithmetic mean of the
per-sample EAL is 1.5316.

The ablations are far from exhaustive — one training run per combination,
without multiple seeds: draft depth {2,3,4,5} × FFN {1536,2304,3072} × attention
{full, sliding 1024, sliding 2048}, plus part of the combinations of block size
{4,6,8,12,16}. After weighing the measured acceptance rate / EAL against the
cost of a draft forward pass:

- 4 layers + FFN 2304 + full attention + block size 8 was selected, which is
  this repository's default checkpoint
  `MiniCPM-o-4_5-tts-dflash-checkpoints-l4-bs8-ffn2304-full`;
- larger block sizes (12/16) show a clear decay in the hit rate of the far slots
  and did not improve the end-to-end result;
- the sliding-window variants (1024/2048) showed no advantage on this task, so
  full attention was adopted in the end.

## 6. Artifacts and deployment

Usage in this repository (see [README.md](./README.md)):

```bash
# simplex: load the checkpoint and run distribution-preserving speculative decoding
uv run minicpmo-dflash-simplex \
    --draft-model /path/to/MiniCPM-o-4_5-tts-dflash-checkpoints-l4-bs8-ffn2304-full/checkpoint_best \
    --text "你好，请介绍一下自己。"

# benchmark: per-sample acceptance rate / EAL / tokens-per-round
uv run minicpmo-dflash-benchmark
```

The loading and validation logic lives in
[src/minicpmo_dflash/draft.py](./src/minicpmo_dflash/draft.py): a
self-contained implementation of the Qwen3-style DFlash layers (KV-injection
attention, the anchor block mask and RoPE) that reads the checkpoint's
`config.json` and `model.safetensors` directly and checks that
`speculators_model_type == "dflash"`, that `sample_from_anchor == false`, and
that `t2d`/`d2t` are consistent.

## 7. Sampling settings at inference

Sampling on the inference side has three layers: the acceptance rule, the
verifier (Talker) sampling, and the draft proposal. The default configuration
aims to preserve the output distribution of the original Talker exactly, so the
acceleration does not change the quality distribution of the generated result.

Acceptance rule (the default, distribution-preserving): the standard
speculative-sampling algorithm (see
[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) and
[Chen et al., 2023](https://arxiv.org/abs/2302.01318)). A candidate is accepted
with probability `min(1, p(candidate) / q(candidate))`; on rejection the next
token is resampled from the residual distribution `normalize(max(0, p - q))`.
Under this rule the final output distribution is exactly the same as that of the
unaccelerated Talker. When the verifier is greedy (temperature 0) the rule
switches automatically to an argmax comparison, accepting only when the
verifier's argmax equals the candidate.

The temperatures on the verifier and draft sides are set independently and may
differ. The verifier side follows the official simplex implementation: codec
sampling at temperature 0.8 / top-k 25 / top-p 0.85 (the `TTSSamplingParams`
defaults), which determines the quality distribution of the final output. The
draft side defaults to temperature 0.8 and samples candidates from
`q = softmax(logits / 0.8)`; this affects only the acceptance rate and the
speed, never the output distribution. Matching the verifier temperature gave a
higher measured acceptance rate. The draft temperature may also be set to `-1`
to follow the verifier temperature, or to `0` to propose with argmax, through
the CLI parameter `--tts-draft-temperature`.

Optional lenient acceptance: distribution consistency is traded for a higher
acceptance rate and more speed, following the lenient path of llama.cpp-omni. It
takes effect only when the verifier is not greedy, and once enabled the draft
switches back to argmax proposals automatically, since acceptance no longer
depends on the q distribution:

- `--tts-draft-accept-min-prob P`: the candidate is accepted when the verifier's
  probability for it is at least P, and this rule takes precedence;
- `--tts-draft-accept-top-k K`: the candidate is accepted when it falls inside
  the verifier's top-K probability set;
- on the lenient path every checked position still consumes one verifier sample,
  and that sample is used directly as the output when the candidate is rejected.
  Measurements with top-k=5 show the output accuracy staying at a high level: on
  the Seed TTS dataset the accuracy is almost the same as with the original
  sampling method.

## 8. Other notes

- The recipe is not optimal: the corpus mixture, the number of auxiliary layers
  (five is on the high side, DFlash examples commonly use three) and the
  positional decay `gamma` were never tuned systematically.
- Neither multimodal input nor duplex mode took part in training, yet both work
  in practice: all training data comes from plain-text, single-shot (simplex)
  scenarios and contains neither multimodal input such as image or video nor
  duplex (realtime) sessions, but the resulting drafter performs well under
  multimodal input and in duplex mode as measured. The explanation is that
  multimodal information has already been fused inside the Thinker into the
  representation of its output text tokens, so the condition input of the Talker
  is always just the text tokens of the LLM output and the conditional
  distribution seen by the drafter does not change in essence; duplex mode
  mostly changes the LLM side, namely when and what to listen and speak in a
  streaming fashion, while the codec generation mechanism of the Talker is the
  same as in simplex.
