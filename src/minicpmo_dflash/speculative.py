"""DFlash speculative decoding for the MiniCPM-o Talker."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from .draft import DFlashDraftModel


@dataclass(frozen=True)
class DFlashSamplingConfig:
    """Sampling controls matching llama.cpp-omni's TTS DFlash runtime."""

    draft_temperature: float = 0.8
    accept_top_k: int = 0
    accept_min_prob: float = 0.0

    def __post_init__(self) -> None:
        invalid_negative = (
            self.draft_temperature < 0.0 and self.draft_temperature != -1.0
        )
        if (
            not math.isfinite(self.draft_temperature)
            or invalid_negative
            or self.draft_temperature > 10.0
        ):
            raise ValueError("draft_temperature must be -1 (auto) or in [0, 10]")
        if not 0 <= self.accept_top_k <= 64:
            raise ValueError("accept_top_k must be in [0, 64]")
        if not math.isfinite(self.accept_min_prob) or not (
            0.0 <= self.accept_min_prob <= 1.0
        ):
            raise ValueError("accept_min_prob must be in [0, 1]")


def add_dflash_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add llama.cpp-omni-compatible DFlash sampling flags to a CLI."""

    group = parser.add_argument_group("TTS DFlash sampling")
    group.add_argument(
        "--disable-dflash",
        action="store_true",
        help=(
            "Disable DFlash completely: do not load the draft checkpoint and "
            "use the original Transformers TTS decoder."
        ),
    )
    group.add_argument(
        "--tts-draft-temperature",
        "--draft-temperature",
        dest="tts_draft_temperature",
        type=float,
        default=0.8,
        help=(
            "Draft proposal temperature: 0.8 by default, -1 follows the "
            "verifier temperature, and 0 uses argmax proposals."
        ),
    )
    group.add_argument(
        "--tts-draft-accept-top-k",
        "--accept-top-k",
        dest="tts_draft_accept_top_k",
        type=int,
        default=0,
        help=(
            "0 uses distribution-preserving textbook acceptance; a positive "
            "value enables biased top-k acceptance."
        ),
    )
    group.add_argument(
        "--tts-draft-accept-min-prob",
        "--accept-min-prob",
        dest="tts_draft_accept_min_prob",
        type=float,
        default=0.0,
        help=(
            "Verifier-probability threshold for biased lenient acceptance; "
            "when positive it takes precedence over accept-top-k."
        ),
    )


def dflash_sampling_from_args(args: argparse.Namespace) -> DFlashSamplingConfig:
    return DFlashSamplingConfig(
        draft_temperature=args.tts_draft_temperature,
        accept_top_k=args.tts_draft_accept_top_k,
        accept_min_prob=args.tts_draft_accept_min_prob,
    )


def _sample_row(
    probabilities: torch.Tensor, generator: torch.Generator | None
) -> int:
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def speculative_accept_sample(
    probabilities: torch.Tensor,
    candidate: int,
    generator: torch.Generator | None,
    proposal_probabilities: torch.Tensor | None = None,
) -> tuple[int, bool]:
    """Apply textbook speculative acceptance to one proposed token.

    With a sampled draft distribution ``q``, acceptance is
    ``min(1, p(candidate) / q(candidate))`` and rejection samples from
    ``normalize(max(0, p - q))``.  An argmax draft is the delta-distribution
    special case; sample-and-compare is distributionally equivalent and uses
    one verifier draw, matching the old strict implementation.
    """

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional tensor")
    if candidate < 0 or candidate >= probabilities.numel():
        return _sample_row(probabilities, generator), False
    if proposal_probabilities is None:
        emitted = _sample_row(probabilities, generator)
        return emitted, emitted == candidate
    if proposal_probabilities.shape != probabilities.shape:
        raise ValueError("proposal and verifier distributions must have the same shape")

    p_candidate = probabilities[candidate]
    q_candidate = proposal_probabilities[candidate]
    if q_candidate > 0:
        accept_probability = torch.clamp(p_candidate / q_candidate, max=1.0)
    else:
        accept_probability = torch.ones_like(p_candidate)
    coin = torch.rand(
        (), device=probabilities.device, generator=generator, dtype=torch.float32
    )
    if coin < accept_probability:
        return candidate, True

    residual = torch.clamp(probabilities - proposal_probabilities, min=0.0)
    residual_mass = residual.sum()
    if not torch.isfinite(residual_mass) or residual_mass <= 0:
        return _sample_row(probabilities, generator), False
    return _sample_row(residual / residual_mass, generator), False


@dataclass
class DecodeStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    output_tokens: int = 0
    truncated: bool = False
    seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.output_tokens / self.rounds if self.rounds else 0.0


class TalkerRunner:
    """KV-cached verifier runner over ``MiniCPMTTS.model``."""

    def __init__(
        self,
        tts: Any,
        aux_layer_ids: tuple[int, ...],
        device: torch.device,
        seed: int,
    ) -> None:
        self.backbone = tts.model
        self.head = tts.head_code[0]
        self.embed_code = tts.emb_code[0]
        self.eos_code = int(tts.config.num_audio_tokens) - 1
        self.aux_layer_ids = aux_layer_ids
        self.device = device
        self.past = None
        self.temperature = 0.8
        self.top_k = 0
        self.top_p = 1.0
        self.repetition_penalty = 1.05
        self.repetition_window = 16
        self.repetition_by_frequency = True
        self.min_tokens_to_keep = 1
        self.generator = None
        if seed >= 0:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)

    @torch.inference_mode()
    def forward(
        self, embeddings: torch.Tensor, position_start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_len = embeddings.shape[0]
        position_ids = torch.arange(
            position_start, position_start + query_len, device=self.device
        ).unsqueeze(0)
        outputs = self.backbone(
            inputs_embeds=embeddings.unsqueeze(0),
            position_ids=position_ids,
            cache_position=position_ids[0],
            past_key_values=self.past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.past = outputs.past_key_values
        auxiliary = torch.cat(
            [outputs.hidden_states[index][0] for index in self.aux_layer_ids], dim=-1
        )
        return auxiliary, outputs.last_hidden_state[0]

    def _repetition_counts(
        self, histories: list[list[int]], vocabulary_size: int
    ) -> torch.Tensor:
        rows = torch.zeros(
            (len(histories), self.repetition_window), dtype=torch.long, device=self.device
        )
        valid = torch.zeros_like(rows, dtype=torch.bool)
        for row, history in enumerate(histories):
            recent = history[-self.repetition_window :]
            if recent:
                values = torch.tensor(recent, dtype=torch.long, device=self.device)
                rows[row, -len(recent) :] = values
                valid[row, -len(recent) :] = True
        counts = (F.one_hot(rows, vocabulary_size) * valid.unsqueeze(-1)).sum(1)
        if not self.repetition_by_frequency:
            counts = counts.bool().to(counts.dtype)
        return counts

    def distribution(
        self,
        hidden_states: torch.Tensor,
        eos_mask: torch.Tensor,
        histories: list[list[int]],
    ) -> torch.Tensor:
        logits = self.head(hidden_states).float()
        if self.temperature > 0:
            logits = logits / self.temperature
        eos_mask = eos_mask.to(device=logits.device, dtype=torch.bool).flatten()
        logits[eos_mask, self.eos_code] = -torch.inf

        if self.repetition_penalty != 1.0:
            counts = self._repetition_counts(histories, logits.shape[-1]).to(logits.dtype)
            alpha = torch.pow(
                torch.tensor(self.repetition_penalty, device=logits.device), counts
            )
            logits = torch.where(logits < 0, logits * alpha, logits / alpha)

        if self.temperature <= 0:
            probabilities = torch.zeros_like(logits)
            probabilities.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return probabilities
        if 0 < self.top_k < logits.shape[-1]:
            threshold = torch.topk(logits, self.top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
            sorted_probs = sorted_logits.softmax(dim=-1)
            remove = sorted_probs.cumsum(dim=-1) > self.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove[..., : self.min_tokens_to_keep] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            logits = torch.full_like(logits, -torch.inf)
            logits.scatter_(-1, sorted_indices, sorted_logits)
        return logits.softmax(dim=-1)

    def sample(self, probabilities: torch.Tensor) -> torch.Tensor:
        samples = [_sample_row(row, self.generator) for row in probabilities]
        return torch.tensor(samples, dtype=torch.long, device=probabilities.device)

    def crop_cache(self, length: int) -> None:
        if hasattr(self.past, "crop"):
            self.past.crop(length)
            return
        # Compatibility with legacy tuple caches.
        self.past = tuple(
            (layer[0][:, :, :length, :], layer[1][:, :, :length, :])
            + tuple(layer[2:])
            for layer in self.past
        )


class DFlashDecoder:
    """DFlash decoder with textbook or explicitly lenient acceptance."""

    def __init__(
        self,
        tts: Any,
        draft: DFlashDraftModel,
        device: torch.device | str,
        seed: int = 42,
        verbose: bool = True,
        sampling: DFlashSamplingConfig | None = None,
    ) -> None:
        self.tts = tts
        self.draft = draft
        self.device = torch.device(device)
        self.verbose = verbose
        self.sampling = sampling or DFlashSamplingConfig()
        self.runner = TalkerRunner(tts, draft.aux_layer_ids, self.device, seed)
        self.draft_generator = torch.Generator(device=self.device)
        # llama.cpp-omni keeps a separate default-constructed mt19937 for q.
        # A separate generator likewise prevents proposal draws from shifting
        # the verifier's random stream.
        self.draft_generator.manual_seed(5489)
        self.block_tokens = draft.block_size - 1
        self.aux_history: torch.Tensor | None = None
        self.position = 0
        self.last_stats = DecodeStats()
        self.last_tokens: list[int] = []
        self.total = DecodeStats()

    def reset(self) -> None:
        self.runner.past = None
        self.aux_history = None
        self.position = 0

    def _configure_sampling(
        self,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        repetition_window: int,
        repetition_by_frequency: bool,
        min_tokens_to_keep: int,
    ) -> None:
        self.runner.temperature = float(temperature)
        self.runner.top_k = int(top_k)
        self.runner.top_p = float(top_p)
        self.runner.repetition_penalty = float(repetition_penalty)
        self.runner.repetition_window = int(repetition_window)
        self.runner.repetition_by_frequency = bool(repetition_by_frequency)
        self.runner.min_tokens_to_keep = int(min_tokens_to_keep)

    @torch.inference_mode()
    def _decode(
        self,
        condition: torch.Tensor,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        force_no_stop: bool,
        continue_turn: bool,
        per_second: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        if not continue_turn:
            self.reset()
        dtype = condition.dtype
        start = time.perf_counter()
        stats = DecodeStats()

        auxiliary, last_hidden = self.runner.forward(condition[0], self.position)
        self.aux_history = (
            auxiliary
            if self.aux_history is None
            else torch.cat((self.aux_history, auxiliary), dim=0)
        )
        self.position += condition.shape[1]

        first_eos_mask = torch.tensor(
            [force_no_stop or min_new_tokens > 0], device=self.device
        )
        first_probs = self.runner.distribution(last_hidden[-1:], first_eos_mask, [[]])
        pending = int(self.runner.sample(first_probs).item())
        output: list[int] = []
        eos_code = self.runner.eos_code
        if pending == eos_code:
            return self._finish(output, stats, start, per_second=per_second), self.runner.past

        while True:
            output.append(pending)
            # MiniCPMTTS.generate_chunk leaves the most recent sampled token out
            # of its return at the final iteration. Returning at most N-1 keeps
            # the duplex caller's KV/position bookkeeping identical.
            if len(output) + 1 >= max_new_tokens:
                stats.truncated = True
                embedding = self.runner.embed_code(
                    torch.tensor([pending], dtype=torch.long, device=self.device)
                )
                new_aux, _ = self.runner.forward(embedding.to(dtype), self.position)
                self.aux_history = torch.cat((self.aux_history, new_aux), dim=0)
                self.position += 1
                break

            stats.rounds += 1
            anchor = self.runner.embed_code(
                torch.tensor([pending], dtype=torch.long, device=self.device)
            )[0]
            lenient_accept = (
                self.runner.temperature > 0
                and (
                    self.sampling.accept_top_k > 0
                    or self.sampling.accept_min_prob > 0.0
                )
            )
            draft_temperature = 0.0
            if self.runner.temperature > 0 and not lenient_accept:
                draft_temperature = (
                    self.runner.temperature
                    if self.sampling.draft_temperature < 0
                    else self.sampling.draft_temperature
                )
            proposal = self.draft.propose(
                self.aux_history,
                anchor,
                self.position,
                temperature=draft_temperature,
                generator=self.draft_generator,
            )
            candidates = proposal.tokens
            stats.proposed += self.block_tokens
            verify_ids = torch.cat(
                (
                    torch.tensor([pending], dtype=torch.long, device=self.device),
                    candidates,
                )
            )
            verify_embeddings = self.runner.embed_code(verify_ids)
            new_aux, new_hidden = self.runner.forward(
                verify_embeddings.to(dtype), self.position
            )
            eos_mask = (
                force_no_stop
                | (
                    torch.arange(new_hidden.shape[0], device=self.device) + len(output)
                    < min_new_tokens
                )
            )
            candidate_list = candidates.tolist()
            histories = [output + candidate_list[:row] for row in range(new_hidden.shape[0])]
            verifier_probs = self.runner.distribution(new_hidden, eos_mask, histories)

            accepted = 0
            stopped = False
            eos_matched = False
            accepted_tokens: list[int] = []
            next_pending: int | None = None
            for index in range(self.block_tokens):
                if len(output) + accepted + 1 >= max_new_tokens:
                    stats.truncated = True
                    stopped = True
                    break
                candidate = int(candidates[index].item())
                target_probabilities = verifier_probs[index]
                if lenient_accept:
                    # The llama.cpp-omni lenient path consumes a verifier draw
                    # on every inspected row and reuses it as the next bonus
                    # if the approximation rejects the candidate.
                    drawn = _sample_row(target_probabilities, self.runner.generator)
                    candidate_probability = float(target_probabilities[candidate].item())
                    if self.sampling.accept_min_prob > 0.0:
                        is_accepted = (
                            candidate_probability >= self.sampling.accept_min_prob
                        )
                    else:
                        top_k = min(
                            self.sampling.accept_top_k,
                            target_probabilities.numel(),
                        )
                        top_tokens = torch.topk(target_probabilities, top_k).indices
                        is_accepted = (
                            candidate_probability > 1e-6
                            and bool((top_tokens == candidate).any().item())
                        )
                    emitted = candidate if is_accepted else drawn
                elif self.runner.temperature <= 0:
                    emitted = int(target_probabilities.argmax().item())
                    is_accepted = emitted == candidate
                else:
                    q_row = (
                        proposal.probabilities[index]
                        if proposal.probabilities is not None
                        else None
                    )
                    emitted, is_accepted = speculative_accept_sample(
                        target_probabilities,
                        candidate,
                        self.runner.generator,
                        q_row,
                    )

                if not is_accepted:
                    next_pending = emitted
                    if emitted == eos_code:
                        stopped = True
                    break
                accepted += 1
                if candidate == eos_code:
                    stopped = True
                    eos_matched = True
                    break
                accepted_tokens.append(candidate)

            # Row B-1 predicts the bonus after all B-1 draft tokens.
            if not stopped and next_pending is None and accepted == self.block_tokens:
                next_pending = _sample_row(
                    verifier_probs[accepted], self.runner.generator
                )

            keep_rows = accepted if eos_matched else accepted + 1
            self.runner.crop_cache(self.position + keep_rows)
            self.aux_history = torch.cat(
                (self.aux_history, new_aux[:keep_rows]), dim=0
            )
            self.position += keep_rows
            stats.accepted += accepted
            output.extend(accepted_tokens)
            if stopped:
                break
            if next_pending is None:
                raise RuntimeError("DFlash round ended without a verifier bonus token")
            pending = next_pending
            if pending == eos_code:
                break

        return self._finish(output, stats, start, per_second=per_second), self.runner.past

    def _finish(
        self,
        output: list[int],
        stats: DecodeStats,
        start: float,
        per_second: bool = False,
    ) -> torch.Tensor:
        stats.output_tokens = len(output)
        stats.seconds = time.perf_counter() - start
        self.last_stats = stats
        self.last_tokens = list(output)
        self.total.rounds += stats.rounds
        self.total.proposed += stats.proposed
        self.total.accepted += stats.accepted
        self.total.output_tokens += stats.output_tokens
        self.total.seconds += stats.seconds
        if self.verbose and stats.proposed:
            # Duplex chunks run once per second, so the per-round yield of
            # emitted codec tokens is more meaningful there than EAL.
            round_metric = (
                f"tokens/round={stats.tokens_per_round:.2f}"
                if per_second
                else f"EAL={stats.accepted_per_round:.2f}"
            )
            print(
                f"[DFlash] accepted {stats.accepted}/{stats.proposed} "
                f"({stats.acceptance_rate:.3f}), {round_metric}, "
                f"{stats.output_tokens} codec tokens in {stats.seconds:.2f}s"
            )
        if output:
            return torch.tensor(output, dtype=torch.long, device=self.device).view(1, -1, 1)
        return torch.zeros((1, 0, 1), dtype=torch.long, device=self.device)

    def generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        temperature: torch.Tensor | float,
        repetition_penalty: float,
        eos_token: Any,
        force_no_stop: bool = False,
        max_new_token: int = 500,
        min_new_tokens: int = 0,
        past_key_values: Any = None,
        logits_processors: Any = None,
        text_start_pos: int | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        del eos_token, logits_processors
        continue_turn = past_key_values is not None
        if continue_turn and self.runner.past is not past_key_values:
            raise RuntimeError("Talker KV cache is out of sync with the duplex session")
        if continue_turn and text_start_pos != self.position:
            raise RuntimeError(
                f"Talker position mismatch: caller={text_start_pos}, decoder={self.position}"
            )
        scalar_temperature = float(
            temperature.flatten()[0].item() if isinstance(temperature, torch.Tensor) else temperature
        )
        self._configure_sampling(
            temperature=scalar_temperature,
            top_k=25,
            top_p=0.85,
            repetition_penalty=repetition_penalty,
            repetition_window=16,
            repetition_by_frequency=True,
            min_tokens_to_keep=3,
        )
        return self._decode(
            inputs_embeds,
            max_new_tokens=max_new_token,
            min_new_tokens=min_new_tokens,
            force_no_stop=force_no_stop,
            continue_turn=continue_turn,
            per_second=True,
        )

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Any,
        force_no_stop: bool = False,
        min_new_token: int = 50,
        max_new_token: int = 2048,
        sampling_params: Any = None,
        **_: Any,
    ) -> SimpleNamespace:
        del eos_token
        # top_k/top_p are intentionally not read from sampling_params: the
        # Talker uses full-vocabulary multinomial sampling by design (see
        # README), so _configure_sampling hard-codes top_k=0, top_p=1.0.
        params = sampling_params or SimpleNamespace(
            temperature=0.8, repetition_penalty=1.05
        )
        self._configure_sampling(
            temperature=float(params.temperature),
            top_k=0,
            top_p=1.0,
            repetition_penalty=float(params.repetition_penalty),
            repetition_window=8,
            repetition_by_frequency=False,
            min_tokens_to_keep=1,
        )
        tokens, _ = self._decode(
            inputs_embeds,
            max_new_tokens=max_new_token,
            min_new_tokens=min_new_token,
            force_no_stop=force_no_stop,
            continue_turn=False,
        )
        return SimpleNamespace(
            new_ids=tokens,
            audio_input_ids=None,
            past_key_values=None,
            past_input_ids=None,
            finished=not self.last_stats.truncated,
        )

    def report(self) -> str:
        stats = self.total
        return (
            f"DFlash total: rounds={stats.rounds}, proposed={stats.proposed}, "
            f"accepted={stats.accepted}, acceptance={stats.acceptance_rate:.4f}, "
            f"EAL={stats.accepted_per_round:.3f}, "
            f"tokens/round={stats.tokens_per_round:.3f}, "
            f"output={stats.output_tokens}, "
            f"decode={stats.seconds:.2f}s"
        )
