"""The canonical hook-site namespace.

This namespace is the contract. A site name is simultaneously:

  * the key under which a tensor is captured in a `Trace`,
  * the address an intervention targets (`scale_site`),
  * the node id the frontend renders.

It must not drift. Everything that names a site imports it from here; nothing
re-spells a site name as a string literal.

Site order is execution order, so iterating `site_names(cfg)` walks the forward
pass from the embedding to the output distribution.
"""

from __future__ import annotations

from typing import Final

from .config import ModelConfig

# --- global sites -----------------------------------------------------------

SITE_EMBED_TOK: Final[str] = "embed.tok"
SITE_EMBED_POS: Final[str] = "embed.pos"
SITE_RESID_0: Final[str] = "resid.0"
SITE_FINAL_LN: Final[str] = "final.ln"
SITE_LOGITS: Final[str] = "logits"
SITE_PROBS: Final[str] = "probs"

#: Captured before the first block.
PRE_LAYER_SITES: Final[tuple[str, ...]] = (SITE_EMBED_TOK, SITE_EMBED_POS, SITE_RESID_0)
#: Captured after the last block.
POST_LAYER_SITES: Final[tuple[str, ...]] = (SITE_FINAL_LN, SITE_LOGITS, SITE_PROBS)
GLOBAL_SITES: Final[tuple[str, ...]] = PRE_LAYER_SITES + POST_LAYER_SITES

# --- per-layer sites --------------------------------------------------------

#: Suffixes appended to the `L{n}.` prefix, in execution order.
#:
#: The four attention stages are captured separately on purpose: each one is a
#: distinct, inspectable object in the UI. `scores_raw` is the bare q @ k^T,
#: `scores_scaled` divides by sqrt(head_dim), `scores_masked` applies the causal
#: mask (masked entries are exactly -inf), `weights` is the softmax. Collapsing
#: them would hide exactly the steps the lab exists to show.
LAYER_SITE_SUFFIXES: Final[tuple[str, ...]] = (
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

#: Suffixes of the wide (T, d_ff) MLP sites, excluded from the wire payload by
#: default. See `engine.trace.Trace.to_dict`.
MLP_HIDDEN_SUFFIXES: Final[tuple[str, ...]] = ("mlp.hidden_pre", "mlp.hidden_post")

#: Substring that identifies an MLP hidden site by name alone.
MLP_HIDDEN_MARKER: Final[str] = ".mlp.hidden_"


def layer_prefix(layer: int) -> str:
    """`"L0."`, `"L1."`, ... — the prefix every per-layer site carries."""
    return f"L{layer}."


def layer_site(layer: int, suffix: str) -> str:
    """Compose a per-layer site name, e.g. `layer_site(1, "attn.z") -> "L1.attn.z"`."""
    return f"L{layer}.{suffix}"


def layer_site_names(layer: int) -> list[str]:
    """Every site belonging to `layer`, in execution order."""
    return [layer_site(layer, suffix) for suffix in LAYER_SITE_SUFFIXES]


def site_names(cfg: ModelConfig) -> list[str]:
    """The full site manifest for `cfg`, in execution order.

    This is the authoritative list: `forward_with_trace` captures exactly these
    keys, and an intervention naming anything outside it is a `ValueError`.
    """
    names: list[str] = list(PRE_LAYER_SITES)
    for layer in range(cfg.n_layer):
        names.extend(layer_site_names(layer))
    names.extend(POST_LAYER_SITES)
    return names


def mlp_hidden_sites(cfg: ModelConfig) -> list[str]:
    """The `L{n}.mlp.hidden_*` sites, for payload filtering."""
    return [
        layer_site(layer, suffix)
        for layer in range(cfg.n_layer)
        for suffix in MLP_HIDDEN_SUFFIXES
    ]


#: Shape of each site given (T, cfg), after the batch dimension is squeezed out.
_RESIDUAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"resid_pre", "ln1", "attn.out", "resid_mid", "ln2", "mlp.out", "resid_post"}
)
_HEAD_SUFFIXES: Final[frozenset[str]] = frozenset({"q", "k", "v", "attn.z"})
_SCORE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"attn.scores_raw", "attn.scores_scaled", "attn.scores_masked", "attn.weights"}
)


def site_shape(site: str, cfg: ModelConfig, seq_len: int) -> tuple[int, ...]:
    """Expected shape of `site` for a sequence of length `seq_len`.

    Batch is always squeezed out (tracing forces batch size 1), so:

        embed.*, resid.*, ln*, attn.out, mlp.out -> (T, d_model)
        q, k, v, attn.z                          -> (n_head, T, head_dim)
        attn.scores_*, attn.weights              -> (n_head, T, T)
        mlp.hidden_*                             -> (T, d_ff)
        logits, probs                            -> (T, vocab_size)
    """
    if site in (SITE_EMBED_TOK, SITE_EMBED_POS, SITE_RESID_0, SITE_FINAL_LN):
        return (seq_len, cfg.d_model)
    if site in (SITE_LOGITS, SITE_PROBS):
        return (seq_len, cfg.vocab_size)

    layer, _, suffix = site.partition(".")
    if not (layer.startswith("L") and layer[1:].isdigit()):
        raise ValueError(f"unknown site {site!r}")

    if suffix in _RESIDUAL_SUFFIXES:
        return (seq_len, cfg.d_model)
    if suffix in _HEAD_SUFFIXES:
        return (cfg.n_head, seq_len, cfg.head_dim)
    if suffix in _SCORE_SUFFIXES:
        return (cfg.n_head, seq_len, seq_len)
    if suffix in MLP_HIDDEN_SUFFIXES:
        return (seq_len, cfg.d_ff)
    raise ValueError(f"unknown site {site!r}")
