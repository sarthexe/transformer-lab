"""Runtime ablations.

An intervention set is a LIST of typed ops, never a flat dict of booleans: a
dict cannot express "head 3 of layer 1 only". The list is ordered, and ops that
target the same site compose in list order.

Every op is applied AT CAPTURE TIME, at the site it names. There is no post-hoc
patching of a finished forward pass: `InterventionPlan.apply` is called by the
recorder as each site is captured, and the value it returns is the value that
flows onward. That is what makes an ablated run and a baseline run the same
code path.

Two ops are structural rather than value-level -- they change which arithmetic
happens, not the value at a site -- so the model asks the plan about them
instead: `causal_mask_disabled` and `keeps_residual`.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Iterable, Literal, Sequence, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from torch import Tensor

from model.config import ModelConfig
from model.sites import SITE_EMBED_POS, layer_site, site_names

# --- op schema --------------------------------------------------------------


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AblateHead(_Op):
    """Silence one attention head.

    The head's slice of `attn.z` is zeroed BEFORE the output projection. This
    matters: `out_proj` mixes all heads into a single d_model vector, so zeroing
    the *projected* output would delete every head's contribution, not this
    head's. `z` is the last point at which heads are still separable.
    """

    op: Literal["ablate_head"]
    layer: int
    head: int
    mode: Literal["zero", "mean"] = "zero"


class DisablePositional(_Op):
    """Zero `embed.pos` before it is added to `embed.tok`.

    The site is still captured (as zeros) so the frontend renders a zeroed
    heatmap rather than a missing node.
    """

    op: Literal["disable_positional"]


class DisableCausalMask(_Op):
    """Skip the causal mask entirely at `scores_masked`.

    With n_layer=2 this changes the final prediction: position t < T-1 in layer
    0 now attends to future tokens, and layer 1's last position reads from those
    contaminated earlier positions, so future information leaks forward. With
    n_layer=1 it would be a no-op at the last position, which already attends to
    everything.
    """

    op: Literal["disable_causal_mask"]


class ZeroMLP(_Op):
    """Zero a layer's `mlp.out`, so the MLP writes nothing into the residual."""

    op: Literal["zero_mlp"]
    layer: int


class RemoveResidual(_Op):
    """Drop one skip connection.

    around="attn": resid_mid = attn_out only.
    around="mlp":  resid_post = mlp_out only.

    There is deliberately no "both" -- removing two skips is two ops, so the UI
    can show them as two independently toggleable edges.
    """

    op: Literal["remove_residual"]
    layer: int
    around: Literal["attn", "mlp"]


class ScaleSite(_Op):
    """Multiply a site's tensor by `factor` at capture time."""

    op: Literal["scale_site"]
    site: str
    factor: float


Intervention = Annotated[
    Union[
        AblateHead,
        DisablePositional,
        DisableCausalMask,
        ZeroMLP,
        RemoveResidual,
        ScaleSite,
    ],
    Field(discriminator="op"),
]

INTERVENTION_ADAPTER: TypeAdapter[list[Intervention]] = TypeAdapter(list[Intervention])

#: Accepted by `parse_interventions`: already-parsed ops, or raw dicts.
InterventionInput = Union[Intervention, dict[str, Any]]


def parse_interventions(
    interventions: Iterable[InterventionInput] | None,
) -> list[Intervention]:
    """Validate a heterogeneous list of ops/dicts into typed ops."""
    if not interventions:
        return []
    return INTERVENTION_ADAPTER.validate_python(list(interventions))


# --- compiled plan ----------------------------------------------------------

_SiteHook = Callable[[Tensor], Tensor]


def _scale_hook(factor: float) -> _SiteHook:
    return lambda t: t * factor


def _zero_hook() -> _SiteHook:
    return torch.zeros_like


def _zero_head_hook(head: int) -> _SiteHook:
    def hook(t: Tensor) -> Tensor:
        # t is (B, n_head, T, head_dim): the batched, pre-merge view of z.
        out = t.clone()
        out[:, head] = 0.0
        return out

    return hook


class InterventionPlan:
    """An intervention list compiled against a `ModelConfig`.

    Validation happens here, once, up front: out-of-range layer/head indices and
    unknown site names raise `ValueError` before any tensor is touched.
    """

    def __init__(self, interventions: Sequence[Intervention], cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.interventions: list[Intervention] = list(interventions)

        self._site_hooks: dict[str, list[_SiteHook]] = {}
        self.causal_mask_disabled: bool = False
        self._removed_residuals: set[tuple[int, str]] = set()

        valid_sites = frozenset(site_names(cfg))

        for index, op in enumerate(self.interventions):
            where = f"interventions[{index}] ({op.op})"

            if isinstance(op, AblateHead):
                self._check_layer(op.layer, where)
                self._check_head(op.head, where)
                if op.mode == "mean":
                    raise NotImplementedError(
                        f"{where}: ablate_head mode='mean' is accepted by the schema "
                        "but not implemented yet. It needs a mean head output "
                        "estimated over a corpus, which the engine does not compute. "
                        "Use mode='zero'."
                    )
                self._add(layer_site(op.layer, "attn.z"), _zero_head_hook(op.head))

            elif isinstance(op, DisablePositional):
                self._add(SITE_EMBED_POS, _zero_hook())

            elif isinstance(op, DisableCausalMask):
                self.causal_mask_disabled = True

            elif isinstance(op, ZeroMLP):
                self._check_layer(op.layer, where)
                self._add(layer_site(op.layer, "mlp.out"), _zero_hook())

            elif isinstance(op, RemoveResidual):
                self._check_layer(op.layer, where)
                self._removed_residuals.add((op.layer, op.around))

            elif isinstance(op, ScaleSite):
                if op.site not in valid_sites:
                    raise ValueError(
                        f"{where}: unknown site {op.site!r}. "
                        f"Valid sites for this config: {sorted(valid_sites)}"
                    )
                self._add(op.site, _scale_hook(op.factor))

            else:  # pragma: no cover - the discriminated union is exhaustive
                raise ValueError(f"{where}: unhandled op type {type(op).__name__}")

    # -- construction helpers

    def _add(self, site: str, hook: _SiteHook) -> None:
        self._site_hooks.setdefault(site, []).append(hook)

    def _check_layer(self, layer: int, where: str) -> None:
        if not 0 <= layer < self.cfg.n_layer:
            raise ValueError(
                f"{where}: layer {layer} out of range for n_layer={self.cfg.n_layer} "
                f"(valid: 0..{self.cfg.n_layer - 1})"
            )

    def _check_head(self, head: int, where: str) -> None:
        if not 0 <= head < self.cfg.n_head:
            raise ValueError(
                f"{where}: head {head} out of range for n_head={self.cfg.n_head} "
                f"(valid: 0..{self.cfg.n_head - 1})"
            )

    # -- runtime queries, called from inside the forward pass

    def __bool__(self) -> bool:
        return bool(self.interventions)

    def __len__(self) -> int:
        return len(self.interventions)

    def apply(self, site: str, tensor: Tensor) -> Tensor:
        """Value-level ops for `site`, folded in list order.

        Returns `tensor` unchanged (the same object, no clone) when nothing
        targets the site, so an empty plan is exactly the identity.
        """
        hooks = self._site_hooks.get(site)
        if not hooks:
            return tensor
        for hook in hooks:
            tensor = hook(tensor)
        return tensor

    def keeps_residual(self, layer: int, around: str) -> bool:
        """False when a `remove_residual` op drops this skip connection."""
        return (layer, around) not in self._removed_residuals

    def targets(self) -> frozenset[str]:
        """Sites this plan modifies at capture time (for UI highlighting)."""
        return frozenset(self._site_hooks)


def compile_interventions(
    interventions: Iterable[InterventionInput] | None, cfg: ModelConfig
) -> tuple[list[Intervention], InterventionPlan]:
    """Parse, validate against `cfg`, and compile. Returns (typed ops, plan)."""
    ops = parse_interventions(interventions)
    return ops, InterventionPlan(ops, cfg)
