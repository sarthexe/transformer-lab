"""Model and site-manifest invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from engine.trace import TraceRecorder
from model.config import ModelConfig
from model.sites import layer_site, site_names, site_shape
from model.transformer import ToyGPT

SEQ_LENGTHS = [1, 2, 8, 64]


# --- config -----------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ModelConfig()
    assert (cfg.n_layer, cfg.n_head, cfg.d_model, cfg.d_ff) == (2, 4, 128, 512)
    assert (cfg.block_size, cfg.vocab_size, cfg.dropout) == (64, 4096, 0.0)
    assert cfg.head_dim == 32


def test_config_is_frozen(cfg: ModelConfig) -> None:
    with pytest.raises(Exception):
        cfg.n_layer = 3  # type: ignore[misc]


def test_config_round_trips_through_dict(cfg: ModelConfig) -> None:
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_indivisible_d_model() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=130, n_head=4)


# --- architecture -----------------------------------------------------------


def test_positional_embedding_is_learned(model: ToyGPT, cfg: ModelConfig) -> None:
    assert isinstance(model.wpe, torch.nn.Embedding)
    assert model.wpe.weight.shape == (cfg.block_size, cfg.d_model)
    assert model.wpe.weight.requires_grad


def test_lm_head_is_not_tied_to_the_token_embedding(model: ToyGPT) -> None:
    assert model.lm_head.weight is not model.wte.weight
    assert not torch.equal(model.lm_head.weight, model.wte.weight)


def test_every_linear_has_a_bias(model: ToyGPT) -> None:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is not None, f"{name} has no bias"


def test_sequence_longer_than_block_size_is_rejected(model: ToyGPT, cfg: ModelConfig) -> None:
    too_long = torch.zeros((1, cfg.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="block_size"):
        model(too_long)


# --- the site manifest ------------------------------------------------------


def test_site_manifest_matches_the_spec(cfg: ModelConfig) -> None:
    names = site_names(cfg)
    assert len(names) == len(set(names)), "site names must be unique"
    assert set(names) == {
        "embed.tok",
        "embed.pos",
        "resid.0",
        "final.ln",
        "logits",
        "probs",
    } | {
        f"L{n}.{suffix}"
        for n in range(cfg.n_layer)
        for suffix in (
            "resid_pre",
            "ln1",
            "q",
            "k",
            "v",
            "attn.scores_raw",
            "attn.scores_scaled",
            "attn.scores_masked",
            "attn.weights",
            "attn.z",
            "attn.out",
            "resid_mid",
            "ln2",
            "mlp.hidden_pre",
            "mlp.hidden_post",
            "mlp.out",
            "resid_post",
        )
    }


def test_trace_captures_exactly_the_manifest(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    trace = model.forward_with_trace(idx)
    assert set(trace.sites) == set(site_names(cfg))


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_shapes_match_the_manifest(model: ToyGPT, cfg: ModelConfig, seq_len: int) -> None:
    generator = torch.Generator().manual_seed(seq_len)
    tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=generator)
    trace = model.forward_with_trace(tokens)

    for name in site_names(cfg):
        expected = list(site_shape(name, cfg, seq_len))
        assert trace.sites[name].shape == expected, name
        assert len(trace.sites[name].data) == math.prod(expected), name


def test_tracing_requires_batch_size_one(model: ToyGPT, cfg: ModelConfig) -> None:
    batched = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(AssertionError, match="batch size 1"):
        model.forward_with_trace(batched)


# --- attention invariants ---------------------------------------------------


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_attention_rows_sum_to_one(model: ToyGPT, cfg: ModelConfig, seq_len: int) -> None:
    generator = torch.Generator().manual_seed(seq_len)
    tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=generator)
    sites, _ops, _plan, _ms = model.capture_sites(tokens)

    for layer in range(cfg.n_layer):
        weights = sites[layer_site(layer, "attn.weights")]
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_masked_entries_are_exactly_minus_inf_and_exactly_zero(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    sites, _ops, _plan, _ms = model.capture_sites(idx)
    seq_len = idx.shape[1]
    masked_out = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    for layer in range(cfg.n_layer):
        scores = sites[layer_site(layer, "attn.scores_masked")]
        weights = sites[layer_site(layer, "attn.weights")]
        for head in range(cfg.n_head):
            assert torch.all(scores[head][masked_out] == float("-inf"))
            assert torch.all(torch.isfinite(scores[head][~masked_out]))
            # Exactly 0.0, not 1e-9: the mask is an exact statement about
            # reachability and stays exact all the way to the UI.
            assert torch.all(weights[head][masked_out] == 0.0)


def test_masked_entries_survive_serialization(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    trace = model.forward_with_trace(idx)
    seq_len = idx.shape[1]
    masked_out = ~np.tril(np.ones((seq_len, seq_len), dtype=bool))

    for layer in range(cfg.n_layer):
        scores = trace.sites[layer_site(layer, "attn.scores_masked")]
        weights = trace.sites[layer_site(layer, "attn.weights")].to_numpy()
        # -inf has no JSON spelling, so it ships as null and restores as -inf.
        shipped = np.array(scores.data, dtype=object).reshape(scores.shape)
        assert all(value is None for value in shipped[0][masked_out])
        assert np.isneginf(scores.to_numpy()[0][masked_out]).all()
        assert np.isfinite(scores.min) and np.isfinite(scores.max)
        assert np.all(weights[0][masked_out] == 0.0)


def test_scores_scaled_is_scores_raw_over_sqrt_head_dim(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    sites, _ops, _plan, _ms = model.capture_sites(idx)
    for layer in range(cfg.n_layer):
        raw = sites[layer_site(layer, "attn.scores_raw")]
        scaled = sites[layer_site(layer, "attn.scores_scaled")]
        assert torch.allclose(scaled, raw / math.sqrt(cfg.head_dim), atol=1e-6)


# --- causality --------------------------------------------------------------


@pytest.mark.parametrize("changed_pos", [1, 3, 7])
def test_changing_a_token_leaves_earlier_positions_bit_identical(
    model: ToyGPT, cfg: ModelConfig, idx, changed_pos: int
) -> None:
    other = idx.clone()
    other[0, changed_pos] = (int(other[0, changed_pos]) + 1) % cfg.vocab_size
    assert not torch.equal(idx, other)

    base, _o, _p, _m = model.capture_sites(idx)
    perturbed, _o, _p, _m = model.capture_sites(other)

    for layer in range(cfg.n_layer):
        site = layer_site(layer, "resid_post")
        assert torch.equal(
            base[site][:changed_pos], perturbed[site][:changed_pos]
        ), f"{site} changed at a position before {changed_pos}"

    assert torch.equal(base["logits"][:changed_pos], perturbed["logits"][:changed_pos])
    # And the change does reach position `changed_pos` itself.
    assert not torch.equal(
        base[layer_site(cfg.n_layer - 1, "resid_post")][changed_pos],
        perturbed[layer_site(cfg.n_layer - 1, "resid_post")][changed_pos],
    )


# --- recorder ---------------------------------------------------------------


def test_recorder_is_the_only_instrumentation_path(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    """Both entry points go through the one `_forward_impl`."""
    from engine.interventions import InterventionPlan

    calls: list[str] = []
    original = ToyGPT._forward_impl

    def counting(self, tokens, rec):  # type: ignore[no-untyped-def]
        calls.append(type(rec).__name__)
        return original(self, tokens, rec)

    ToyGPT._forward_impl = counting  # type: ignore[method-assign]
    try:
        model(idx)
        model.forward_with_trace(idx)
    finally:
        ToyGPT._forward_impl = original  # type: ignore[method-assign]

    assert calls == ["NullRecorder", "TraceRecorder"]
    assert isinstance(TraceRecorder(InterventionPlan([], cfg)).sites, dict)
