"""The trace wire schema, and the recorder that fills it.

Every tensor ships in the same envelope: shape, min, max, and a flat row-major
`data` list rounded to 4 decimal places. Rounding happens here, server-side, and
never in the browser -- a delta the UI draws must be a delta of the numbers the
UI shows, which is only true if both sides see identical values.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from model.config import ModelConfig
from model.sites import MLP_HIDDEN_MARKER, SITE_LOGITS, SITE_PROBS

from .interventions import Intervention, InterventionPlan
from .summaries import Derived, compute_derived

SCHEMA_VERSION: str = "1.0"

#: Decimal places for every number that crosses the wire.
ROUND_DP: int = 4

#: Probabilities span several orders of magnitude, so the top-k list keeps more
#: places than the tensor envelopes; 4dp would floor most of a 4096-way tail.
PROB_ROUND_DP: int = 6

DEFAULT_TOP_K: int = 20

#: Payload groups that are excluded by default and opted into via
#: `Trace.to_dict(include=...)`.
GROUP_PROBS: str = "probs"
GROUP_MLP_HIDDEN: str = "mlp_hidden"
GROUP_ALL: str = "all"


# --- tensors ----------------------------------------------------------------


class TensorEnvelope(BaseModel):
    """One tensor, flattened row-major.

    `min`/`max` are taken over the finite entries only, so they are usable
    directly as a colour-scale domain.

    A `None` in `data` marks a non-finite entry. In practice the only source is
    the causal mask, which writes exactly -inf into `attn.scores_masked`; JSON
    has no representation for it, and null is the honest one. `to_numpy()`
    restores those entries as -inf.
    """

    shape: list[int]
    min: float
    max: float
    data: list[float | None]

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> TensorEnvelope:
        array = tensor.detach().to(device="cpu", dtype=torch.float64).numpy()
        finite = np.isfinite(array)

        if finite.all():
            low, high = float(array.min()), float(array.max())
            data: list[float | None] = np.round(array, ROUND_DP).reshape(-1).tolist()
        elif finite.any():
            finite_values = array[finite]
            low, high = float(finite_values.min()), float(finite_values.max())
            flat = np.round(array, ROUND_DP).reshape(-1).tolist()
            data = [value if math.isfinite(value) else None for value in flat]
        else:
            low = high = 0.0
            data = [None] * array.size

        return cls(
            shape=list(tensor.shape),
            min=round(low, ROUND_DP),
            max=round(high, ROUND_DP),
            data=data,
        )

    def to_numpy(self) -> np.ndarray:
        """Rebuild the tensor from the shipped numbers (nulls become -inf).

        This is the reconstruction entry point: anything recomputed from here
        is recomputed from exactly what the UI displays.
        """
        flat = np.array(
            [-np.inf if value is None else value for value in self.data],
            dtype=np.float64,
        )
        return flat.reshape(self.shape)


# --- trace ------------------------------------------------------------------


class Token(BaseModel):
    id: int
    text: str
    pos: int


class TopToken(BaseModel):
    id: int
    text: str
    logit: float
    prob: float


class Output(BaseModel):
    """The model's prediction at one position.

    `position` is the resolved absolute index into the sequence: a caller
    asking for the default -1 gets back T-1, so the frontend never has to
    normalize it.
    """

    position: int = -1
    top_k: list[TopToken] = Field(default_factory=list)


class Trace(BaseModel):
    """Everything one instrumented forward pass produced."""

    schema_version: str = SCHEMA_VERSION
    input: str
    tokens: list[Token]
    config: dict[str, Any]
    interventions: list[Intervention] = Field(default_factory=list)
    sites: dict[str, TensorEnvelope]
    derived: Derived
    output: Output
    timing_ms: float

    def optional_sites(self) -> set[str]:
        """Sites excluded from the payload unless explicitly included."""
        return {site for site in self.sites if MLP_HIDDEN_MARKER in site} | (
            {SITE_PROBS} if SITE_PROBS in self.sites else set()
        )

    def to_dict(self, include: Iterable[str] = ()) -> dict[str, Any]:
        """Serialize for the wire.

        Excluded by default because they dwarf everything else: `probs`
        (T x vocab_size) and every `L{n}.mlp.hidden_*` site (T x d_ff).

        `include` accepts the group names "probs", "mlp_hidden" and "all", or
        any exact site name.
        """
        requested = set(include)
        payload: dict[str, Any] = self.model_dump(mode="json")

        if GROUP_ALL in requested:
            return payload

        optional = self.optional_sites()
        keep: set[str] = set()
        for name in requested:
            if name == GROUP_PROBS:
                keep |= {site for site in optional if site == SITE_PROBS}
            elif name == GROUP_MLP_HIDDEN:
                keep |= {site for site in optional if MLP_HIDDEN_MARKER in site}
            elif name in self.sites:
                keep.add(name)
            else:
                raise ValueError(
                    f"unknown include {name!r}; expected a site name or one of "
                    f"{sorted((GROUP_PROBS, GROUP_MLP_HIDDEN, GROUP_ALL))}"
                )

        for site in optional - keep:
            payload["sites"].pop(site, None)
        return payload


# --- recorder ---------------------------------------------------------------


class TraceRecorder:
    """Captures every site of one forward pass, applying interventions inline.

    Implements `model.transformer.Recorder`. Tracing forces batch size 1, so
    each captured tensor is stored with its batch dimension squeezed out.
    """

    active: bool = True

    def __init__(self, plan: InterventionPlan) -> None:
        self.plan = plan
        self.sites: dict[str, Tensor] = {}

    @property
    def mask_enabled(self) -> bool:
        return not self.plan.causal_mask_disabled

    def keeps_residual(self, layer: int, around: str) -> bool:
        return self.plan.keeps_residual(layer, around)

    def capture(self, site: str, tensor: Tensor) -> Tensor:
        # Interventions are applied here, at the site they name, and the
        # returned value is what the forward pass carries onward.
        tensor = self.plan.apply(site, tensor)
        assert tensor.shape[0] == 1, (
            f"site {site!r} has batch dimension {tensor.shape[0]}, expected 1"
        )
        self.sites[site] = tensor.detach()[0]
        return tensor


# --- assembly ---------------------------------------------------------------


def _top_k(
    logits_row: Tensor, probs_row: Tensor, token_texts: Sequence[str], k: int
) -> list[TopToken]:
    k = max(1, min(k, logits_row.numel()))
    values, indices = torch.topk(logits_row, k)
    top: list[TopToken] = []
    for logit, index in zip(values.tolist(), indices.tolist()):
        top.append(
            TopToken(
                id=int(index),
                text=token_texts[index] if index < len(token_texts) else f"<{index}>",
                logit=round(float(logit), ROUND_DP),
                prob=round(float(probs_row[index].item()), PROB_ROUND_DP),
            )
        )
    return top


def build_trace(
    *,
    cfg: ModelConfig,
    token_ids: Sequence[int],
    token_texts: Sequence[str],
    raw_sites: dict[str, Tensor],
    interventions: Sequence[Intervention],
    input_text: str | None,
    position: int,
    top_k: int,
    timing_ms: float,
    vocab_texts: Sequence[str] | None = None,
) -> Trace:
    """Wrap the raw captured tensors into the wire schema.

    `vocab_texts` resolves top-k token ids; it defaults to `token_texts`, which
    only covers ids present in the input, so a model with no vocab attached
    still produces a well-formed (if less readable) trace.
    """
    seq_len = len(token_ids)
    if seq_len == 0:
        raise ValueError("cannot build a trace for an empty sequence")
    if not -seq_len <= position < seq_len:
        raise ValueError(
            f"position {position} out of range for a sequence of length {seq_len}"
        )
    resolved_position = position % seq_len

    logits = raw_sites[SITE_LOGITS]
    probs = raw_sites[SITE_PROBS]
    if vocab_texts is not None:
        lookup = list(vocab_texts)
    else:
        # token_texts is indexed by position, not by id. Build an id-indexed
        # view so top-k labels are still right for tokens present in the input.
        lookup = [f"<{i}>" for i in range(cfg.vocab_size)]
        for token_id, text in zip(token_ids, token_texts):
            lookup[token_id] = text

    tokens = [
        Token(id=int(token_id), text=text, pos=pos)
        for pos, (token_id, text) in enumerate(zip(token_ids, token_texts))
    ]

    return Trace(
        input=input_text if input_text is not None else " ".join(token_texts),
        tokens=tokens,
        config=cfg.to_dict(),
        interventions=list(interventions),
        sites={
            name: TensorEnvelope.from_tensor(tensor)
            for name, tensor in raw_sites.items()
        },
        derived=compute_derived(raw_sites, cfg, list(token_texts)),
        output=Output(
            position=resolved_position,
            top_k=_top_k(logits[resolved_position], probs[resolved_position], lookup, top_k),
        ),
        timing_ms=round(timing_ms, ROUND_DP),
    )
