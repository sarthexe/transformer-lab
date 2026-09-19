"""Model hyper-parameters for TRANSFORMER LAB.

The config is frozen: a trace is only meaningful next to the exact geometry it
was produced with, so the config travels with every trace and every checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Geometry of the toy GPT.

    Architecture decisions baked into `transformer.ToyGPT` (not configurable):
    pre-LayerNorm, learned absolute positional embeddings, GELU MLP, no weight
    tying between the token embedding and the LM head, a final LayerNorm before
    the LM head, and bias=True on every Linear.
    """

    n_layer: int = 2
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    block_size: int = 64
    vocab_size: int = 4096
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        """Width of a single attention head: d_model // n_head (32 by default)."""
        return self.d_model // self.n_head

    def to_dict(self) -> dict[str, Any]:
        """Serializable view. `head_dim` is derived but included for consumers
        (the frontend) that should never have to recompute it."""
        return {**asdict(self), "head_dim": self.head_dim}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Rebuild from `to_dict()` output, ignoring derived/unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
