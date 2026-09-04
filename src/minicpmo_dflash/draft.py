"""Inference-only DFlash network implemented with plain PyTorch.

The module intentionally does not import ``speculators``.  It mirrors the
Qwen3-style DFlash layers used by the training checkpoint and loads the saved
``model.safetensors`` directly, including its vocabulary mappings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .runtime import resolve_model_path


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _as_attr(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _as_attr(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_attr(item) for item in value]
    return value


@dataclass
class DFlashProposal:
    """Draft candidates and the distributions used to sample them.

    ``probabilities`` is ``None`` for an argmax proposal.  Otherwise it is in
    the Talker codec-token vocabulary, so it can be used directly by the
    textbook speculative-sampling acceptance rule.
    """

    tokens: torch.Tensor
    probabilities: torch.Tensor | None


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.float()
    output = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (output * weight.float()).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        return _rms_norm(x, self.weight, eps)


class DFlashAttention(nn.Module):
    """Attention with verifier context injected into every layer's K/V."""

    def __init__(self, config: AttrDict, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        layer_types = config.get("layer_types") or []
        self.sliding_window = (
            config.get("sliding_window")
            if layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        context_len = target_hidden.shape[1]
        query = self.q_proj(hidden_states).view(
            batch_size, query_len, self.num_heads, self.head_dim
        )
        query = self.q_norm(query, self.eps).transpose(1, 2)

        key = torch.cat((self.k_proj(target_hidden), self.k_proj(hidden_states)), dim=1)
        value = torch.cat((self.v_proj(target_hidden), self.v_proj(hidden_states)), dim=1)
        key = key.view(
            batch_size, context_len + query_len, self.num_kv_heads, self.head_dim
        )
        value = value.view(
            batch_size, context_len + query_len, self.num_kv_heads, self.head_dim
        )
        key = self.k_norm(key, self.eps).transpose(1, 2)
        value = value.transpose(1, 2)

        query = query * cos[-query_len:] + _rotate_half(query) * sin[-query_len:]
        key = key * cos + _rotate_half(key) * sin
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(batch_size, query_len, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: AttrDict, layer_idx: int):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.self_attn = DFlashAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, self.eps)
        hidden_states = residual + self.self_attn(
            hidden_states, target_hidden, cos, sin, attention_mask
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, self.eps)
        hidden_states = self.mlp.down_proj(
            F.silu(self.mlp.gate_proj(hidden_states)) * self.mlp.up_proj(hidden_states)
        )
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    """One-pass anchored-block DFlash drafter."""

    def __init__(self, config: AttrDict):
        super().__init__()
        transformer = config.transformer_layer_config
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.eps = float(transformer.rms_norm_eps)
        self.aux_layer_ids = tuple(int(index) for index in config.aux_hidden_state_layer_ids)
        self.sliding_window = transformer.get("sliding_window")
        self.sliding_window_non_causal = bool(config.get("sliding_window_non_causal", False))
        layer_types = transformer.get("layer_types") or []
        self.sliding_window_indices = {
            index for index, kind in enumerate(layer_types) if kind == "sliding_attention"
        }

        self.embed_tokens = nn.Embedding(transformer.vocab_size, transformer.hidden_size)
        self.fc = nn.Linear(
            len(self.aux_layer_ids) * transformer.hidden_size,
            transformer.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(transformer.hidden_size)
        self.layers = nn.ModuleList(
            DFlashDecoderLayer(transformer, index)
            for index in range(transformer.num_hidden_layers)
        )
        self.norm = RMSNorm(transformer.hidden_size)
        self.lm_head = nn.Linear(
            transformer.hidden_size, config.draft_vocab_size, bias=False
        )
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(transformer.vocab_size, dtype=torch.bool))

        rope = transformer.get("rope_parameters") or {}
        self.rope_theta = float(rope.get("rope_theta", transformer.get("rope_theta", 10000.0)))
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, transformer.head_dim, 2).float() / transformer.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # DFlash is a single-sequence drafter; positions always has batch=1.
        assert positions.shape[0] == 1, "_rotary expects batch_size=1"
        frequencies = torch.outer(positions[0].float(), self.inv_freq.to(positions.device))
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos(), embedding.sin()

    @staticmethod
    def _attention_mask(
        block_size: int,
        context_len: int,
        dtype: torch.dtype,
        device: torch.device,
        sliding_window: int | None,
        block_non_causal: bool,
    ) -> torch.Tensor:
        negative = torch.finfo(dtype).min
        mask = torch.full(
            (block_size, context_len + block_size), negative, dtype=dtype, device=device
        )
        context_start = max(0, context_len - sliding_window) if sliding_window else 0
        for slot in range(block_size):
            mask[slot, context_start:context_len] = 0
            block_end = block_size if block_non_causal else slot + 1
            mask[slot, context_len : context_len + block_end] = 0
        return mask.unsqueeze(0).unsqueeze(0)

    def _forward_block(
        self, aux_history: torch.Tensor, anchor_embedding: torch.Tensor, position: int
    ) -> torch.Tensor:
        dtype = self.lm_head.weight.dtype
        device = self.lm_head.weight.device
        target_hidden = self.hidden_norm(
            self.fc(aux_history.unsqueeze(0).to(dtype)), self.eps
        )
        mask_ids = torch.full(
            (1, self.block_size - 1), self.mask_token_id, dtype=torch.long, device=device
        )
        hidden_states = torch.cat(
            (
                anchor_embedding.to(dtype).reshape(1, 1, -1),
                self.embed_tokens(mask_ids),
            ),
            dim=1,
        )
        positions = torch.cat(
            (
                torch.arange(position, device=device),
                torch.arange(position, position + self.block_size, device=device),
            )
        ).unsqueeze(0)
        cos, sin = self._rotary(positions)
        full_mask = self._attention_mask(
            self.block_size, position, dtype, device, None, True
        )
        sliding_mask = self._attention_mask(
            self.block_size,
            position,
            dtype,
            device,
            self.sliding_window,
            self.sliding_window_non_causal,
        )
        for index, layer in enumerate(self.layers):
            layer_mask = sliding_mask if index in self.sliding_window_indices else full_mask
            hidden_states = layer(hidden_states, target_hidden, cos, sin, layer_mask)
        return self.lm_head(self.norm(hidden_states, self.eps))

    @torch.inference_mode()
    def propose(
        self,
        aux_history: torch.Tensor,
        anchor_embedding: torch.Tensor,
        position: int,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> DFlashProposal:
        logits = self._forward_block(aux_history, anchor_embedding, position)
        candidate_logits = logits[0, 1 : self.block_size].float()
        if temperature > 0.0:
            draft_probabilities = F.softmax(candidate_logits / temperature, dim=-1)
            draft_ids = torch.multinomial(
                draft_probabilities, 1, generator=generator
            ).squeeze(-1)

            # ``d2t`` stores the offset from compressed draft ids to target
            # ids.  The supplied TTS checkpoint is identity-mapped, but doing
            # the scatter keeps the acceptance formula correct for a future
            # vocabulary-pruned checkpoint as well.
            target_ids = torch.arange(
                self.d2t.numel(), device=draft_probabilities.device
            ) + self.d2t
            target_vocab_size = int(target_ids[-1].item()) + 1
            probabilities = torch.zeros(
                (draft_probabilities.shape[0], target_vocab_size),
                dtype=draft_probabilities.dtype,
                device=draft_probabilities.device,
            )
            probabilities.scatter_(
                -1,
                target_ids.unsqueeze(0).expand(draft_probabilities.shape[0], -1),
                draft_probabilities,
            )
        else:
            draft_ids = candidate_logits.argmax(dim=-1)
            probabilities = None
        return DFlashProposal(
            tokens=draft_ids + self.d2t[draft_ids], probabilities=probabilities
        )


def load_draft_model(
    checkpoint: str | Path, device: torch.device | str, dtype: torch.dtype = torch.bfloat16
) -> DFlashDraftModel:
    """Load a DFlash drafter from a local directory or a HF repo id."""
    checkpoint = Path(resolve_model_path(checkpoint))
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Invalid DFlash checkpoint directory: {checkpoint}")
    config = _as_attr(json.loads(config_path.read_text(encoding="utf-8")))
    if config.get("speculators_model_type") != "dflash":
        raise ValueError(f"Checkpoint is not DFlash: {checkpoint}")
    if config.get("sample_from_anchor", False):
        raise ValueError("Only sample_from_anchor=false DFlash checkpoints are supported")

    model = DFlashDraftModel(config)
    load_result = model.load_state_dict(load_file(str(weights_path)), strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected DFlash state dict result: {load_result}")
    selected = torch.nonzero(model.t2d, as_tuple=False).flatten()
    expected_offsets = selected - torch.arange(selected.numel())
    if selected.numel() != model.d2t.numel() or not torch.equal(
        model.d2t.cpu(), expected_offsets.cpu()
    ):
        raise ValueError("The checkpoint's t2d/d2t vocabulary mappings are inconsistent")
    return model.to(device=device, dtype=dtype).eval()
