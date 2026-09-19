"""Derived fields: small numbers computed from a trace that drive the UI.

Everything here is a function of the captured tensors, so an ablated run gets
ablated summaries for free. Nothing here hardcodes an interpretability claim --
in particular `head_summary` labels are read off the actual attention matrix,
so an untrained model honestly reports diffuse heads.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from model.config import ModelConfig
from model.sites import layer_site

#: Displayed numbers are rounded server-side, the same rule the tensor
#: envelopes follow, so a delta the UI shows is a delta of numbers it displays.
ROUND_DP: int = 4

#: A head is called "prev-token"/"attention sink" when more than this fraction
#: of its average attention mass sits on the sub-diagonal / on column 0.
MASS_THRESHOLD: float = 0.5

#: A head is called "diffuse" when its mean row entropy exceeds this fraction
#: of log(T), the entropy of a uniform distribution over T positions.
ENTROPY_FRACTION: float = 0.8


class Derived(BaseModel):
    """Summary statistics attached to every trace."""

    #: L2 norm of what each component writes into the residual stream.
    #: Keys: "L{n}.attn", "L{n}.mlp".
    write_norm: dict[str, float] = Field(default_factory=dict)

    #: L2 norm of the residual stream at each position, per layer.
    #: Keys: "L{n}.resid_post" -> list of length T.
    resid_norm: dict[str, list[float]] = Field(default_factory=dict)

    #: A short label per head, derived from its attention matrix.
    #: Keys: "L{n}.head_{h}".
    head_summary: dict[str, str] = Field(default_factory=dict)


def _round(value: float) -> float:
    return round(float(value), ROUND_DP)


def head_label(weights: np.ndarray, token_texts: list[str]) -> str:
    """Label one head from its (T, T) attention matrix.

    Rules, applied in order, averaging over query positions:

      * >50% of mass on the first sub-diagonal -> "prev-token"
      * >50% of mass on column 0               -> "attention sink"
      * mean row entropy > 0.8 * log(T)        -> "diffuse"
      * otherwise                              -> '-> "<argmax column token>"'
    """
    seq_len = weights.shape[0]

    # Sub-diagonal mass: sum of W[i, i-1] averaged over ALL T query positions.
    # Row 0 has no previous token and contributes 0, so a perfect prev-token
    # head scores (T-1)/T.
    subdiag_mass = float(np.trace(weights, offset=-1)) / seq_len
    if subdiag_mass > MASS_THRESHOLD:
        return "prev-token"

    sink_mass = float(weights[:, 0].mean())
    if sink_mass > MASS_THRESHOLD:
        return "attention sink"

    # Row entropy in nats, with the 0 * log(0) = 0 convention.
    safe = np.where(weights > 0.0, weights, 1.0)
    row_entropy = -np.sum(np.where(weights > 0.0, weights * np.log(safe), 0.0), axis=-1)
    if float(row_entropy.mean()) > ENTROPY_FRACTION * math.log(seq_len):
        return "diffuse"

    column = int(weights.mean(axis=0).argmax())
    text = token_texts[column] if column < len(token_texts) else f"<{column}>"
    return f'-> "{text}"'


def compute_write_norm(raw_sites: dict[str, Tensor], cfg: ModelConfig) -> dict[str, float]:
    """L2 norm of each component's contribution to the residual stream."""
    norms: dict[str, float] = {}
    for layer in range(cfg.n_layer):
        norms[f"L{layer}.attn"] = _round(
            torch.linalg.vector_norm(raw_sites[layer_site(layer, "attn.out")]).item()
        )
        norms[f"L{layer}.mlp"] = _round(
            torch.linalg.vector_norm(raw_sites[layer_site(layer, "mlp.out")]).item()
        )
    return norms


def compute_resid_norm(
    raw_sites: dict[str, Tensor], cfg: ModelConfig
) -> dict[str, list[float]]:
    """Per-position L2 norm of the residual stream after each layer."""
    norms: dict[str, list[float]] = {}
    for layer in range(cfg.n_layer):
        site = layer_site(layer, "resid_post")
        per_position = torch.linalg.vector_norm(raw_sites[site], dim=-1)
        norms[site] = [_round(v) for v in per_position.tolist()]
    return norms


def compute_head_summary(
    raw_sites: dict[str, Tensor], cfg: ModelConfig, token_texts: list[str]
) -> dict[str, str]:
    """A short, data-derived label for every head."""
    summary: dict[str, str] = {}
    for layer in range(cfg.n_layer):
        weights = raw_sites[layer_site(layer, "attn.weights")].to(torch.float64).numpy()
        for head in range(cfg.n_head):
            summary[f"L{layer}.head_{head}"] = head_label(weights[head], token_texts)
    return summary


def compute_derived(
    raw_sites: dict[str, Tensor], cfg: ModelConfig, token_texts: list[str]
) -> Derived:
    """All derived fields for one traced forward pass."""
    return Derived(
        write_norm=compute_write_norm(raw_sites, cfg),
        resid_norm=compute_resid_norm(raw_sites, cfg),
        head_summary=compute_head_summary(raw_sites, cfg, token_texts),
    )
