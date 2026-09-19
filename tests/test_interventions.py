"""Intervention schema, validation, and semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from engine.interventions import (
    AblateHead,
    InterventionPlan,
    ScaleSite,
    compile_interventions,
    parse_interventions,
)
from engine.summaries import head_label
from model.config import ModelConfig
from model.sites import layer_site
from model.transformer import ToyGPT


def logits_of(model: ToyGPT, tokens, interventions=None) -> torch.Tensor:
    sites, _ops, _plan, _ms = model.capture_sites(tokens, interventions)
    return sites["logits"]


def sites_of(model: ToyGPT, tokens, interventions=None) -> dict[str, torch.Tensor]:
    sites, _ops, _plan, _ms = model.capture_sites(tokens, interventions)
    return sites


# --- schema ------------------------------------------------------------------


def test_ops_parse_from_dicts() -> None:
    ops = parse_interventions(
        [
            {"op": "ablate_head", "layer": 1, "head": 3},
            {"op": "disable_positional"},
            {"op": "disable_causal_mask"},
            {"op": "zero_mlp", "layer": 0},
            {"op": "remove_residual", "layer": 0, "around": "attn"},
            {"op": "scale_site", "site": "L0.attn.out", "factor": 2.0},
        ]
    )
    assert [op.op for op in ops] == [
        "ablate_head",
        "disable_positional",
        "disable_causal_mask",
        "zero_mlp",
        "remove_residual",
        "scale_site",
    ]
    assert isinstance(ops[0], AblateHead) and ops[0].mode == "zero"


def test_an_intervention_set_is_a_list_not_a_flag_dict() -> None:
    """A dict of booleans cannot say 'head 3 of layer 1 only'. A list can."""
    ops = parse_interventions(
        [
            {"op": "ablate_head", "layer": 1, "head": 3},
            {"op": "ablate_head", "layer": 0, "head": 1},
        ]
    )
    assert [(op.layer, op.head) for op in ops] == [(1, 3), (0, 1)]


def test_remove_residual_requires_around() -> None:
    with pytest.raises(ValidationError):
        parse_interventions([{"op": "remove_residual", "layer": 0}])
    with pytest.raises(ValidationError):
        parse_interventions([{"op": "remove_residual", "layer": 0, "around": "both"}])


def test_unknown_op_is_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_interventions([{"op": "delete_everything"}])


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        {"op": "ablate_head", "layer": 2, "head": 0},
        {"op": "ablate_head", "layer": -1, "head": 0},
        {"op": "zero_mlp", "layer": 5},
        {"op": "remove_residual", "layer": 9, "around": "mlp"},
    ],
)
def test_out_of_range_layer_raises(cfg: ModelConfig, op) -> None:
    with pytest.raises(ValueError, match="layer .* out of range"):
        compile_interventions([op], cfg)


@pytest.mark.parametrize("head", [4, 17, -1])
def test_out_of_range_head_raises(cfg: ModelConfig, head: int) -> None:
    with pytest.raises(ValueError, match="head .* out of range"):
        compile_interventions([{"op": "ablate_head", "layer": 0, "head": head}], cfg)


def test_unknown_scale_site_raises(cfg: ModelConfig) -> None:
    with pytest.raises(ValueError, match="unknown site"):
        compile_interventions(
            [{"op": "scale_site", "site": "L0.attn.scores", "factor": 2.0}], cfg
        )


def test_scale_site_accepts_every_site_in_the_manifest(cfg: ModelConfig) -> None:
    from model.sites import site_names

    for site in site_names(cfg):
        InterventionPlan([ScaleSite(op="scale_site", site=site, factor=1.5)], cfg)


def test_mean_head_ablation_is_not_implemented_yet(cfg: ModelConfig) -> None:
    with pytest.raises(NotImplementedError, match="mode='mean'"):
        compile_interventions(
            [{"op": "ablate_head", "layer": 0, "head": 0, "mode": "mean"}], cfg
        )


# --- 7. every op measurably changes the logits -------------------------------

OPS = [
    {"op": "ablate_head", "layer": 0, "head": 0},
    {"op": "ablate_head", "layer": 1, "head": 3},
    {"op": "disable_positional"},
    {"op": "disable_causal_mask"},
    {"op": "zero_mlp", "layer": 0},
    {"op": "zero_mlp", "layer": 1},
    {"op": "remove_residual", "layer": 0, "around": "attn"},
    {"op": "remove_residual", "layer": 0, "around": "mlp"},
    {"op": "remove_residual", "layer": 1, "around": "attn"},
    {"op": "remove_residual", "layer": 1, "around": "mlp"},
    {"op": "scale_site", "site": "L0.attn.out", "factor": 3.0},
    {"op": "scale_site", "site": "L1.v", "factor": 0.0},
    {"op": "scale_site", "site": "final.ln", "factor": 1.5},
]


@pytest.mark.parametrize("op", OPS, ids=lambda o: str(sorted(o.items())))
def test_each_op_measurably_changes_the_logits(model: ToyGPT, idx, op) -> None:
    baseline = logits_of(model, idx)
    ablated = logits_of(model, idx, [op])
    assert (baseline - ablated).abs().max().item() > 1e-4


def test_scaling_probs_is_the_documented_terminal_no_op(model: ToyGPT, idx) -> None:
    """`probs` is the one site downstream of everything, so scaling it cannot
    move the logits -- only the probs site and the top-k probabilities."""
    baseline = sites_of(model, idx)
    scaled = sites_of(model, idx, [{"op": "scale_site", "site": "probs", "factor": 2.0}])
    assert torch.equal(baseline["logits"], scaled["logits"])
    assert torch.allclose(scaled["probs"], baseline["probs"] * 2.0)


def test_interventions_are_echoed_back_in_the_trace(model: ToyGPT, idx) -> None:
    ops = [{"op": "ablate_head", "layer": 1, "head": 2}, {"op": "zero_mlp", "layer": 0}]
    trace = model.forward_with_trace(idx, ops)
    assert [op.model_dump() for op in trace.interventions] == [
        {"op": "ablate_head", "layer": 1, "head": 2, "mode": "zero"},
        {"op": "zero_mlp", "layer": 0},
    ]


# --- semantics: ablate_head ---------------------------------------------------


def test_ablate_head_zeroes_z_before_the_output_projection(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    layer, head = 1, 2
    baseline = sites_of(model, idx)
    ablated = sites_of(model, idx, [{"op": "ablate_head", "layer": layer, "head": head}])

    z_site = layer_site(layer, "attn.z")
    assert torch.all(ablated[z_site][head] == 0.0)

    # Only that head's slice is touched; the rest of z is untouched.
    for other in range(cfg.n_head):
        if other != head:
            assert torch.equal(ablated[z_site][other], baseline[z_site][other])

    # The projected output is NOT zeroed -- out_proj mixes all heads, so the
    # other three heads still write into the residual stream.
    out_site = layer_site(layer, "attn.out")
    assert not torch.all(ablated[out_site] == 0.0)
    assert not torch.equal(ablated[out_site], baseline[out_site])

    # Everything upstream of z is unchanged.
    for suffix in ("q", "k", "v", "attn.weights", "attn.scores_masked"):
        assert torch.equal(
            ablated[layer_site(layer, suffix)], baseline[layer_site(layer, suffix)]
        )


def test_ablating_every_head_zeroes_the_attention_contribution(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    ops = [{"op": "ablate_head", "layer": 0, "head": h} for h in range(cfg.n_head)]
    ablated = sites_of(model, idx, ops)
    assert torch.all(ablated[layer_site(0, "attn.z")] == 0.0)
    # out_proj's bias is all that remains.
    expected = model.blocks[0].attn.out_proj.bias.expand_as(ablated["L0.attn.out"])
    assert torch.allclose(ablated["L0.attn.out"], expected)


# --- semantics: disable_positional -------------------------------------------


def test_disable_positional_zeroes_the_site_but_still_captures_it(
    model: ToyGPT, idx
) -> None:
    trace = model.forward_with_trace(idx, [{"op": "disable_positional"}])

    # Present, so the frontend renders a zeroed heatmap rather than a hole.
    assert "embed.pos" in trace.sites
    pos = trace.sites["embed.pos"]
    assert np.all(pos.to_numpy() == 0.0)
    assert (pos.min, pos.max) == (0.0, 0.0)

    sites = sites_of(model, idx, [{"op": "disable_positional"}])
    assert torch.equal(sites["resid.0"], sites["embed.tok"])


def test_disable_positional_makes_repeated_tokens_identical(
    model: ToyGPT, cfg: ModelConfig
) -> None:
    """Without positions, two copies of the same token are indistinguishable
    to the embedding, so resid.0 repeats exactly."""
    tokens = torch.tensor([[7, 7, 7, 7]])
    sites = sites_of(model, tokens, [{"op": "disable_positional"}])
    resid = sites["resid.0"]
    for pos in range(1, tokens.shape[1]):
        assert torch.equal(resid[0], resid[pos])


# --- semantics: disable_causal_mask -------------------------------------------


def test_disable_causal_mask_removes_the_mask_entirely(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    sites = sites_of(model, idx, [{"op": "disable_causal_mask"}])
    for layer in range(cfg.n_layer):
        scaled = sites[layer_site(layer, "attn.scores_scaled")]
        masked = sites[layer_site(layer, "attn.scores_masked")]
        assert torch.equal(scaled, masked)
        assert torch.isfinite(masked).all()
        # Every cell is now reachable, so no weight is exactly zero.
        assert (sites[layer_site(layer, "attn.weights")] > 0.0).all()


def test_disable_causal_mask_lets_a_position_see_its_future(model: ToyGPT) -> None:
    tokens = torch.tensor([[11, 12, 13, 14]])
    future = tokens.clone()
    future[0, 3] = 99  # a token strictly after position 2
    ops = [{"op": "disable_causal_mask"}]

    assert torch.equal(
        logits_of(model, tokens)[2], logits_of(model, future)[2]
    ), "the mask must block the future"
    assert not torch.equal(
        logits_of(model, tokens, ops)[2], logits_of(model, future, ops)[2]
    ), "without the mask, position 2 must see position 3"


def test_causal_mask_removal_reaches_the_last_position_only_with_depth(
    cfg: ModelConfig, vocab: list[str]
) -> None:
    """The documented depth note, checked.

    With n_layer=1 the last position already attends to everything, so dropping
    the mask cannot change its prediction. With n_layer=2 it can: layer 0 lets
    earlier positions see the future, and layer 1's last position reads from
    those now-contaminated positions.
    """
    tokens = torch.tensor([[11, 12, 13, 14]])
    ops = [{"op": "disable_causal_mask"}]

    torch.manual_seed(0)
    shallow = ToyGPT(ModelConfig.from_dict({**cfg.to_dict(), "n_layer": 1}), vocab)
    shallow.eval()
    assert torch.equal(
        logits_of(shallow, tokens)[-1], logits_of(shallow, tokens, ops)[-1]
    )
    # ...but earlier positions do change, even at depth 1.
    assert not torch.equal(
        logits_of(shallow, tokens)[0], logits_of(shallow, tokens, ops)[0]
    )

    torch.manual_seed(0)
    deep = ToyGPT(ModelConfig.from_dict({**cfg.to_dict(), "n_layer": 2}), vocab)
    deep.eval()
    assert not torch.equal(logits_of(deep, tokens)[-1], logits_of(deep, tokens, ops)[-1])


# --- semantics: zero_mlp and remove_residual ----------------------------------


@pytest.mark.parametrize("layer", [0, 1])
def test_zero_mlp_removes_the_mlp_write(model: ToyGPT, idx, layer: int) -> None:
    sites = sites_of(model, idx, [{"op": "zero_mlp", "layer": layer}])
    assert torch.all(sites[layer_site(layer, "mlp.out")] == 0.0)
    assert torch.equal(
        sites[layer_site(layer, "resid_post")], sites[layer_site(layer, "resid_mid")]
    )
    # The hidden activations are still computed and captured.
    assert not torch.all(sites[layer_site(layer, "mlp.hidden_post")] == 0.0)


@pytest.mark.parametrize("layer", [0, 1])
def test_remove_residual_around_attn(model: ToyGPT, idx, layer: int) -> None:
    sites = sites_of(
        model, idx, [{"op": "remove_residual", "layer": layer, "around": "attn"}]
    )
    assert torch.equal(
        sites[layer_site(layer, "resid_mid")], sites[layer_site(layer, "attn.out")]
    )
    # The mlp skip is untouched.
    assert not torch.equal(
        sites[layer_site(layer, "resid_post")], sites[layer_site(layer, "mlp.out")]
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_remove_residual_around_mlp(model: ToyGPT, idx, layer: int) -> None:
    sites = sites_of(
        model, idx, [{"op": "remove_residual", "layer": layer, "around": "mlp"}]
    )
    assert torch.equal(
        sites[layer_site(layer, "resid_post")], sites[layer_site(layer, "mlp.out")]
    )


def test_two_remove_residual_ops_compose(model: ToyGPT, idx) -> None:
    """There is no around='both'; removing two skips is two ops."""
    sites = sites_of(
        model,
        idx,
        [
            {"op": "remove_residual", "layer": 0, "around": "attn"},
            {"op": "remove_residual", "layer": 0, "around": "mlp"},
        ],
    )
    assert torch.equal(sites["L0.resid_mid"], sites["L0.attn.out"])
    assert torch.equal(sites["L0.resid_post"], sites["L0.mlp.out"])


# --- semantics: scale_site ----------------------------------------------------


@pytest.mark.parametrize(
    "site", ["embed.tok", "L0.q", "L0.attn.weights", "L1.mlp.hidden_post", "final.ln"]
)
def test_scale_site_multiplies_at_capture(model: ToyGPT, idx, site: str) -> None:
    baseline = sites_of(model, idx)
    scaled = sites_of(model, idx, [{"op": "scale_site", "site": site, "factor": 2.0}])
    assert torch.allclose(scaled[site], baseline[site] * 2.0, atol=1e-6)


def test_scale_site_by_one_is_a_no_op(model: ToyGPT, idx) -> None:
    baseline = logits_of(model, idx)
    scaled = logits_of(model, idx, [{"op": "scale_site", "site": "L0.v", "factor": 1.0}])
    assert torch.allclose(baseline, scaled, atol=1e-6)


def test_scale_site_ops_compose_in_list_order(model: ToyGPT, idx) -> None:
    baseline = sites_of(model, idx)
    scaled = sites_of(
        model,
        idx,
        [
            {"op": "scale_site", "site": "L0.k", "factor": 2.0},
            {"op": "scale_site", "site": "L0.k", "factor": 3.0},
        ],
    )
    assert torch.allclose(scaled["L0.k"], baseline["L0.k"] * 6.0, atol=1e-6)


# --- plan bookkeeping ---------------------------------------------------------


def test_plan_reports_its_targets(cfg: ModelConfig) -> None:
    _ops, plan = compile_interventions(
        [
            {"op": "ablate_head", "layer": 1, "head": 0},
            {"op": "zero_mlp", "layer": 0},
            {"op": "disable_causal_mask"},
            {"op": "remove_residual", "layer": 1, "around": "mlp"},
        ],
        cfg,
    )
    assert plan.targets() == {"L1.attn.z", "L0.mlp.out"}
    assert plan.causal_mask_disabled
    assert not plan.keeps_residual(1, "mlp")
    assert plan.keeps_residual(1, "attn")
    assert plan.keeps_residual(0, "mlp")


def test_empty_plan_is_the_identity(cfg: ModelConfig) -> None:
    plan = InterventionPlan([], cfg)
    tensor = torch.randn(1, 4, 8)
    assert plan.apply("L0.attn.z", tensor) is tensor
    assert not plan.causal_mask_disabled
    assert plan.keeps_residual(0, "attn") and plan.keeps_residual(0, "mlp")
    assert plan.targets() == frozenset()
    assert not plan and len(plan) == 0


# --- head labels are read off the data ----------------------------------------


def test_head_label_detects_a_prev_token_head() -> None:
    weights = np.eye(8, k=-1)
    weights[0, 0] = 1.0
    assert head_label(weights, [f"t{i}" for i in range(8)]) == "prev-token"


def test_head_label_detects_an_attention_sink() -> None:
    weights = np.zeros((8, 8))
    weights[:, 0] = 1.0
    assert head_label(weights, [f"t{i}" for i in range(8)]) == "attention sink"


def test_head_label_detects_a_diffuse_head() -> None:
    # Uniform over all 8 positions: entropy log(8) > 0.8 * log(8), and no
    # single column or the sub-diagonal carries enough mass to win first.
    weights = np.full((8, 8), 1.0 / 8.0)
    assert head_label(weights, [f"t{i}" for i in range(8)]) == "diffuse"


def test_head_label_falls_back_to_the_argmax_column() -> None:
    weights = np.zeros((4, 4))
    weights[:, 2] = 0.45
    weights[:, 3] = 0.3
    weights[:, 1] = 0.25
    labels = ["alpha", "beta", "gamma", "delta"]
    assert head_label(weights, labels) == '-> "gamma"'


def test_head_labels_are_not_hardcoded(model: ToyGPT, idx) -> None:
    """An ablation that changes the attention pattern changes the labels."""
    baseline = model.forward_with_trace(idx).derived.head_summary
    assert not all(label == "diffuse" for label in baseline.values())

    # Flat scores with no causal mask is a uniform distribution over all T
    # positions, whose row entropy is log(T) -- the diffuse rule, by
    # construction. Every head must now report it.
    flattened = model.forward_with_trace(
        idx,
        [
            {"op": "scale_site", "site": "L0.attn.scores_scaled", "factor": 0.0},
            {"op": "scale_site", "site": "L1.attn.scores_scaled", "factor": 0.0},
            {"op": "disable_causal_mask"},
        ],
    ).derived.head_summary

    assert set(baseline) == set(flattened)
    assert all(label == "diffuse" for label in flattened.values())
