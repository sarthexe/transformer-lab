"""Trace schema, payload control, and the reconstruction test.

The reconstruction test is the point of the whole thing: it takes the numbers
the trace actually ships -- rounded to 4dp, exactly what the UI renders -- and
recomputes the forward pass from them in pure NumPy using only the model
weights. If the logits come back out, "every number is real" is a verifiable
claim rather than a slogan.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from engine.trace import SCHEMA_VERSION, TensorEnvelope
from model.config import ModelConfig
from model.sites import layer_site
from model.transformer import ToyGPT

#: Tolerance for the logits, the number the spec pins down.
ATOL = 1e-4

#: Tolerance for the site-by-site sweep, set by the rounding budget rather than
#: by arithmetic error. Each input was rounded to 4dp (5e-5) before the step and
#: the output is rounded again, and a site that contracts over d inputs -- like
#: `scores_raw`, a 32-term dot product -- accumulates about 5e-5 * sqrt(d) * |x|
#: on top. A genuine implementation error (a wrong transpose, the wrong weight)
#: shows up at the scale of the values themselves, orders of magnitude above.
SITE_ATOL = 5e-4

#: LayerNorm divides by the standard deviation of its input, so its gain is
#: 1/std. At initialization the residual stream has std ~0.03, which turns a
#: 5e-5 rounding of the input into a ~2e-3 error at the output. It is the only
#: operation in the graph with gain far above 1, and the amplification does not
#: propagate: every site downstream reads the recorded LayerNorm output, not
#: this recomputed one.
LAYER_NORM_ATOL = 5e-3


def tolerance_for(site: str) -> float:
    is_layer_norm = site == "final.ln" or site.endswith((".ln1", ".ln2"))
    return LAYER_NORM_ATOL if is_layer_norm else SITE_ATOL


# --- a pure-NumPy transformer ------------------------------------------------

_erf = np.vectorize(math.erf)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # biased, as torch does
    return (x - mean) / np.sqrt(var + 1e-5) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf GELU, matching nn.GELU (not the tanh approximation)."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def numpy_forward(
    state: dict[str, np.ndarray],
    cfg: ModelConfig,
    trace,
    *,
    chain: bool,
) -> dict[str, np.ndarray]:
    """Recompute every site of `trace` in pure NumPy from the model weights.

    Two modes, differing only in where each step reads its inputs:

    `chain=False` (step-wise) reads the RECORDED value of every input, so each
    site is checked as one operation applied to the numbers the trace actually
    ships. This verifies every edge of the graph independently.

    `chain=True` (composed) reads only `resid.0` from the trace and carries its
    own output forward, rebuilding the whole forward pass from one shipped
    tensor.
    """
    recomputed: dict[str, np.ndarray] = {}

    def recorded(name: str) -> np.ndarray:
        return trace.sites[name].to_numpy()

    def emit(name: str, value: np.ndarray) -> np.ndarray:
        """Record the recomputed value; return what the next step consumes."""
        recomputed[name] = value
        return value if chain else recorded(name)

    seq_len = len(trace.tokens)
    n_head, head_dim = cfg.n_head, cfg.head_dim
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))[None, :, :]

    def linear(name: str, x: np.ndarray) -> np.ndarray:
        return x @ state[f"{name}.weight"].T + state[f"{name}.bias"]

    def split(x: np.ndarray) -> np.ndarray:
        return x.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)

    # The two embedding lookups are the only sites with no upstream site to
    # recompute from, so they come from the weights and the token ids.
    token_ids = [token.id for token in trace.tokens]
    emit("embed.tok", state["wte.weight"][token_ids])
    emit("embed.pos", state["wpe.weight"][:seq_len])
    x = emit("resid.0", recorded("embed.tok") + recorded("embed.pos"))
    if chain:
        x = recorded("resid.0")

    for layer in range(cfg.n_layer):
        block = f"blocks.{layer}"

        def site(suffix: str, value: np.ndarray, _layer: int = layer) -> np.ndarray:
            return emit(layer_site(_layer, suffix), value)

        resid_pre = site("resid_pre", x)
        h = site(
            "ln1",
            _layer_norm(resid_pre, state[f"{block}.ln1.weight"], state[f"{block}.ln1.bias"]),
        )

        q = site("q", split(linear(f"{block}.attn.q_proj", h)))
        k = site("k", split(linear(f"{block}.attn.k_proj", h)))
        v = site("v", split(linear(f"{block}.attn.v_proj", h)))

        scores = site("attn.scores_raw", q @ k.transpose(0, 2, 1))
        scores = site("attn.scores_scaled", scores / math.sqrt(head_dim))
        scores = site("attn.scores_masked", np.where(causal, scores, -np.inf))
        weights = site("attn.weights", _softmax(scores))

        z = site("attn.z", weights @ v)
        merged = z.transpose(1, 0, 2).reshape(seq_len, cfg.d_model)
        attn_out = site("attn.out", linear(f"{block}.attn.out_proj", merged))

        resid_mid = site("resid_mid", resid_pre + attn_out)
        h2 = site(
            "ln2",
            _layer_norm(resid_mid, state[f"{block}.ln2.weight"], state[f"{block}.ln2.bias"]),
        )

        hidden = site("mlp.hidden_pre", linear(f"{block}.mlp.fc", h2))
        activated = site("mlp.hidden_post", _gelu(hidden))
        mlp_out = site("mlp.out", linear(f"{block}.mlp.proj", activated))

        x = site("resid_post", resid_mid + mlp_out)

    final = emit("final.ln", _layer_norm(x, state["ln_f.weight"], state["ln_f.bias"]))
    logits = emit("logits", linear("lm_head", final))
    emit("probs", _softmax(logits))
    return recomputed


@pytest.fixture(scope="module")
def numpy_state(model: ToyGPT) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().astype(np.float64)
        for name, tensor in model.state_dict().items()
    }


def assert_sites_match(
    recomputed: dict[str, np.ndarray],
    trace,
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    for name, value in recomputed.items():
        if name in skip:
            continue
        shipped = trace.sites[name].to_numpy()
        assert shipped.shape == value.shape, name
        finite = np.isfinite(value)
        assert np.array_equal(finite, np.isfinite(shipped)), name
        atol = tolerance_for(name)
        assert np.allclose(value[finite], shipped[finite], atol=atol), (
            f"{name}: max |diff| = {np.abs(value[finite] - shipped[finite]).max():.3e} "
            f"(atol {atol:.0e})"
        )


# --- 1. trace reconstruction -------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 2, 8, 64])
def test_trace_reconstructs_the_forward_pass(
    model: ToyGPT, cfg: ModelConfig, numpy_state: dict[str, np.ndarray], seq_len: int
) -> None:
    """Every site, recomputed in NumPy from the numbers the trace ships."""
    generator = torch.Generator().manual_seed(seq_len)
    tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=generator)
    trace = model.forward_with_trace(tokens)

    recomputed = numpy_forward(numpy_state, cfg, trace, chain=False)
    assert set(recomputed) == set(trace.sites), "the recompute must cover the manifest"
    assert_sites_match(recomputed, trace)

    # The headline claim: the recomputed logits are the model's logits.
    with torch.no_grad():
        reference = model(tokens)[0].numpy().astype(np.float64)
    assert np.allclose(recomputed["logits"], reference, atol=ATOL)
    assert np.allclose(recomputed["logits"], trace.sites["logits"].to_numpy(), atol=ATOL)


@pytest.mark.parametrize("seq_len", [1, 8, 64])
def test_trace_composes_end_to_end_from_a_single_shipped_tensor(
    model: ToyGPT, cfg: ModelConfig, numpy_state: dict[str, np.ndarray], seq_len: int
) -> None:
    """The same recompute, chained: only `resid.0` is read from the trace.

    Looser tolerance, and the reason is worth stating. At initialization the
    residual stream has entries of order 0.03, so rounding to 4dp leaves about
    two and a half significant digits. The first LayerNorm rescales to unit
    variance and turns that into a ~2e-3 relative error which rides through the
    rest of the network. This bounds accumulated rounding, not an arithmetic
    disagreement -- the step-wise test above is the exact one.
    """
    generator = torch.Generator().manual_seed(seq_len)
    tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=generator)
    trace = model.forward_with_trace(tokens)

    recomputed = numpy_forward(numpy_state, cfg, trace, chain=True)
    with torch.no_grad():
        reference = model(tokens)[0].numpy().astype(np.float64)

    assert np.allclose(recomputed["logits"], reference, atol=5e-3)
    assert (
        np.argmax(recomputed["logits"], axis=-1).tolist()
        == np.argmax(reference, axis=-1).tolist()
    )


@pytest.mark.parametrize(
    ("ops", "diverging"),
    [
        ([{"op": "zero_mlp", "layer": 0}], {"L0.mlp.out"}),
        ([{"op": "ablate_head", "layer": 1, "head": 2}], {"L1.attn.z"}),
        ([{"op": "disable_positional"}], {"embed.pos"}),
        ([{"op": "scale_site", "site": "L0.attn.out", "factor": 3.0}], {"L0.attn.out"}),
        ([{"op": "remove_residual", "layer": 1, "around": "mlp"}], {"L1.resid_post"}),
    ],
    ids=["zero_mlp", "ablate_head", "disable_positional", "scale_site", "remove_residual"],
)
def test_an_ablated_trace_is_internally_consistent(
    model: ToyGPT,
    cfg: ModelConfig,
    numpy_state: dict[str, np.ndarray],
    idx,
    ops,
    diverging,
) -> None:
    """An ablated run reconstructs too, everywhere the ablation did not reach.

    Exactly the sites the intervention overwrites diverge from what the plain
    arithmetic would give -- that is what an intervention is -- and every other
    site, logits included, still follows from its recorded inputs. That is what
    makes an ablated trace as trustworthy as a baseline one.
    """
    trace = model.forward_with_trace(idx, ops)
    recomputed = numpy_forward(numpy_state, cfg, trace, chain=False)

    assert_sites_match(recomputed, trace, skip=frozenset(diverging))
    for site in diverging:
        assert not np.allclose(
            recomputed[site], trace.sites[site].to_numpy(), atol=SITE_ATOL
        ), f"{site} was expected to diverge but did not"


# --- 2. empty interventions are the baseline ---------------------------------


@pytest.mark.parametrize("interventions", [None, []])
def test_empty_interventions_are_bit_identical_to_plain_forward(
    model: ToyGPT, idx, interventions
) -> None:
    """There is no separate vanilla forward: the baseline IS the traced path
    with an empty intervention list, so the logits must match bit for bit."""
    with torch.no_grad():
        plain = model(idx)[0]
    sites, ops, plan, _ms = model.capture_sites(idx, interventions)

    assert ops == []
    assert len(plan) == 0
    assert torch.equal(sites["logits"], plain)


def test_trace_of_an_empty_plan_matches_the_plain_logits(model: ToyGPT, idx) -> None:
    with torch.no_grad():
        plain = model(idx)[0].to(torch.float64).numpy()
    trace = model.forward_with_trace(idx)
    assert np.allclose(trace.sites["logits"].to_numpy(), plain, atol=5e-5)


# --- envelopes ---------------------------------------------------------------


def test_envelope_rounds_to_four_places_and_bounds_finite_values() -> None:
    tensor = torch.tensor([[[1.234567, -2.0], [0.000049, 3.5]]])
    envelope = TensorEnvelope.from_tensor(tensor)
    assert envelope.shape == [1, 2, 2]
    assert envelope.data == [1.2346, -2.0, 0.0, 3.5]
    assert envelope.min == -2.0
    assert envelope.max == 3.5


def test_envelope_ships_non_finite_entries_as_null() -> None:
    envelope = TensorEnvelope.from_tensor(torch.tensor([[1.0, float("-inf")]]))
    assert envelope.data == [1.0, None]
    assert (envelope.min, envelope.max) == (1.0, 1.0)
    assert np.isneginf(envelope.to_numpy()[0, 1])


def test_every_shipped_number_is_already_rounded(model: ToyGPT, idx) -> None:
    trace = model.forward_with_trace(idx)
    for name, envelope in trace.sites.items():
        for value in envelope.data[:256]:
            if value is None:
                continue
            assert value == round(value, 4), f"{name} ships an unrounded value"


# --- payload control ---------------------------------------------------------


def test_heavy_sites_are_excluded_by_default(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    payload = model.forward_with_trace(idx).to_dict()
    sites = payload["sites"]

    assert "probs" not in sites
    for layer in range(cfg.n_layer):
        assert layer_site(layer, "mlp.hidden_pre") not in sites
        assert layer_site(layer, "mlp.hidden_post") not in sites

    # Everything else ships.
    assert "logits" in sites and "L0.attn.weights" in sites and "final.ln" in sites
    assert len(sites) == 40 - 1 - 2 * cfg.n_layer


@pytest.mark.parametrize(
    ("include", "expected"),
    [
        (("probs",), {"probs"}),
        (("mlp_hidden",), {"L0.mlp.hidden_pre", "L0.mlp.hidden_post",
                           "L1.mlp.hidden_pre", "L1.mlp.hidden_post"}),
        (("L1.mlp.hidden_post",), {"L1.mlp.hidden_post"}),
        (("probs", "mlp_hidden"), {"probs", "L0.mlp.hidden_pre", "L0.mlp.hidden_post",
                                   "L1.mlp.hidden_pre", "L1.mlp.hidden_post"}),
    ],
)
def test_optional_sites_are_opt_in(model: ToyGPT, idx, include, expected) -> None:
    trace = model.forward_with_trace(idx)
    sites = trace.to_dict(include=include)["sites"]
    assert trace.optional_sites() & set(sites) == expected


def test_include_all_ships_everything(model: ToyGPT, idx) -> None:
    trace = model.forward_with_trace(idx)
    assert set(trace.to_dict(include=("all",))["sites"]) == set(trace.sites)


def test_unknown_include_is_rejected(model: ToyGPT, idx) -> None:
    trace = model.forward_with_trace(idx)
    with pytest.raises(ValueError, match="unknown include"):
        trace.to_dict(include=("L9.attn.z",))


def test_payload_is_json_safe(model: ToyGPT, idx) -> None:
    import json

    payload = model.forward_with_trace(idx).to_dict(include=("all",))
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION


# --- trace contents ----------------------------------------------------------


def test_trace_metadata(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    trace = model.forward_with_trace(idx, input_text="hello world")

    assert trace.schema_version == "1.0"
    assert trace.input == "hello world"
    assert trace.config == cfg.to_dict()
    assert trace.interventions == []
    assert trace.timing_ms > 0.0

    assert [t.pos for t in trace.tokens] == list(range(idx.shape[1]))
    assert [t.id for t in trace.tokens] == idx[0].tolist()
    assert all(t.text == f"tok{t.id}" for t in trace.tokens)


def test_output_resolves_the_queried_position(model: ToyGPT, idx) -> None:
    seq_len = idx.shape[1]
    last = model.forward_with_trace(idx)
    assert last.output.position == seq_len - 1
    assert len(last.output.top_k) == 20

    explicit = model.forward_with_trace(idx, position=2)
    assert explicit.output.position == 2
    assert explicit.output.top_k != last.output.top_k

    with pytest.raises(ValueError, match="out of range"):
        model.forward_with_trace(idx, position=seq_len)


def test_top_k_is_sorted_and_agrees_with_the_logits(model: ToyGPT, idx) -> None:
    trace = model.forward_with_trace(idx)
    logits = trace.sites["logits"].to_numpy()[trace.output.position]

    ranked = [t.logit for t in trace.output.top_k]
    assert ranked == sorted(ranked, reverse=True)
    assert trace.output.top_k[0].id == int(np.argmax(logits))
    for top in trace.output.top_k:
        assert top.logit == pytest.approx(logits[top.id], abs=ATOL)
        assert top.text == f"tok{top.id}"
    assert sum(t.prob for t in trace.output.top_k) <= 1.0 + 1e-6


# --- derived fields ----------------------------------------------------------


def test_derived_write_norm_matches_the_captured_contributions(
    model: ToyGPT, cfg: ModelConfig, idx
) -> None:
    sites, _o, _p, _m = model.capture_sites(idx)
    derived = model.forward_with_trace(idx).derived

    for layer in range(cfg.n_layer):
        for component, site in (("attn", "attn.out"), ("mlp", "mlp.out")):
            expected = float(torch.linalg.vector_norm(sites[layer_site(layer, site)]))
            assert derived.write_norm[f"L{layer}.{component}"] == pytest.approx(
                expected, abs=ATOL
            )


def test_derived_resid_norm_is_per_position(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    derived = model.forward_with_trace(idx).derived
    assert set(derived.resid_norm) == {
        layer_site(n, "resid_post") for n in range(cfg.n_layer)
    }
    for values in derived.resid_norm.values():
        assert len(values) == idx.shape[1]
        assert all(v > 0.0 for v in values)


def test_derived_head_summary_covers_every_head(model: ToyGPT, cfg: ModelConfig, idx) -> None:
    derived = model.forward_with_trace(idx).derived
    assert set(derived.head_summary) == {
        f"L{n}.head_{h}" for n in range(cfg.n_layer) for h in range(cfg.n_head)
    }
    assert all(label for label in derived.head_summary.values())
