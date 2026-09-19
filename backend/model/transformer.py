"""A small GPT with an instrumented forward pass.

There is exactly ONE forward implementation, `ToyGPT._forward_impl`. Training
runs it with a `NullRecorder` (pure identity, zero overhead beyond a dict miss);
tracing runs it with a `TraceRecorder`. The baseline trace is that same code
path with an empty intervention list -- there is deliberately no separate
"vanilla" forward to drift out of sync with the instrumented one.

Interventions are not bolted on after the fact. `Recorder.capture` returns the
value that flows onward, so an op registered against a site is applied exactly
where that site is produced.

Architecture (all fixed, see `config.ModelConfig`): pre-LayerNorm, learned
absolute positional embeddings, GELU MLP, no weight tying, final LayerNorm
before the LM head, bias=True on every Linear.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .sites import (
    SITE_EMBED_POS,
    SITE_EMBED_TOK,
    SITE_FINAL_LN,
    SITE_LOGITS,
    SITE_PROBS,
    SITE_RESID_0,
    layer_prefix,
)

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from engine.interventions import InterventionInput, InterventionPlan
    from engine.trace import Trace


# --- recorder interface -----------------------------------------------------


class Recorder(Protocol):
    """What the forward pass needs from its instrumentation.

    `engine.trace.TraceRecorder` is the real implementation. Defined here, as a
    structural protocol, so `model` never has to import `engine` at module
    level.
    """

    #: False for the null recorder, letting the forward pass skip work whose
    #: only purpose is being captured (the `probs` site).
    active: bool

    @property
    def mask_enabled(self) -> bool:
        """False when a `disable_causal_mask` op is in effect."""

    def capture(self, site: str, tensor: Tensor) -> Tensor:
        """Record `tensor` at `site` and return the value that flows onward
        (which differs from the input when an intervention targets the site)."""

    def keeps_residual(self, layer: int, around: str) -> bool:
        """False when a `remove_residual` op drops this skip connection."""


class NullRecorder:
    """The no-op recorder used by the plain training forward pass."""

    __slots__ = ()

    active: bool = False

    @property
    def mask_enabled(self) -> bool:
        return True

    def capture(self, site: str, tensor: Tensor) -> Tensor:
        return tensor

    def keeps_residual(self, layer: int, around: str) -> bool:
        return True


NULL_RECORDER: Recorder = NullRecorder()


# --- modules ----------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with every stage exposed as a site.

    q/k/v are separate projections rather than one fused `c_attn`, so the trace
    can address them individually and so a NumPy reconstruction of the forward
    pass reads straight off the state dict.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.scale = math.sqrt(cfg.head_dim)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Not persistent: it is derived from block_size, so checkpoints stay
        # clean and a config change cannot be shadowed by a stale buffer.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def _split_heads(self, t: Tensor) -> Tensor:
        """(B, T, d_model) -> (B, n_head, T, head_dim)."""
        batch, seq_len, _ = t.shape
        return t.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: Tensor) -> Tensor:
        """(B, n_head, T, head_dim) -> (B, T, d_model)."""
        batch, n_head, seq_len, head_dim = t.shape
        return t.transpose(1, 2).contiguous().view(batch, seq_len, n_head * head_dim)

    def forward(self, x: Tensor, rec: Recorder) -> Tensor:
        seq_len = x.shape[1]
        p = layer_prefix(self.layer_idx)

        q = rec.capture(f"{p}q", self._split_heads(self.q_proj(x)))
        k = rec.capture(f"{p}k", self._split_heads(self.k_proj(x)))
        v = rec.capture(f"{p}v", self._split_heads(self.v_proj(x)))

        # Four separate stages, because each is a distinct thing to look at:
        # the raw dot products, the variance correction, the causal structure,
        # and the normalized distribution.
        scores = rec.capture(f"{p}attn.scores_raw", q @ k.transpose(-2, -1))
        scores = rec.capture(f"{p}attn.scores_scaled", scores / self.scale)

        if rec.mask_enabled:
            # Exactly -inf, not a large negative number: softmax then produces
            # exactly 0.0 for masked entries, so "is this cell reachable?" stays
            # an exact question all the way to the UI.
            scores = scores.masked_fill(
                ~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf")
            )
        # else: disable_causal_mask -- the mask is skipped entirely, so
        # scores_masked is just scores_scaled passed through.
        scores = rec.capture(f"{p}attn.scores_masked", scores)

        weights = rec.capture(f"{p}attn.weights", torch.softmax(scores, dim=-1))

        # Head ablation lands on z, the last point at which heads are still
        # separable: out_proj below mixes all of them into one d_model vector.
        # Dropout is identity in eval mode, the only mode tracing runs in, so
        # the captured `weights` site is exactly what produces z.
        z = rec.capture(f"{p}attn.z", self.attn_dropout(weights) @ v)

        out = self.resid_dropout(self.out_proj(self._merge_heads(z)))
        return rec.capture(f"{p}attn.out", out)


class MLP(nn.Module):
    """Position-wise feed-forward block: d_model -> d_ff -> GELU -> d_model."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.act = nn.GELU()  # exact erf GELU, not the tanh approximation
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, rec: Recorder) -> Tensor:
        p = layer_prefix(self.layer_idx)
        h = rec.capture(f"{p}mlp.hidden_pre", self.fc(x))
        h = rec.capture(f"{p}mlp.hidden_post", self.act(h))
        return rec.capture(f"{p}mlp.out", self.dropout(self.proj(h)))


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, layer_idx)

    def forward(self, x: Tensor, rec: Recorder) -> Tensor:
        n = self.layer_idx
        p = layer_prefix(n)

        resid_pre = rec.capture(f"{p}resid_pre", x)
        attn_out = self.attn(rec.capture(f"{p}ln1", self.ln1(resid_pre)), rec)
        resid_mid = resid_pre + attn_out if rec.keeps_residual(n, "attn") else attn_out
        resid_mid = rec.capture(f"{p}resid_mid", resid_mid)

        mlp_out = self.mlp(rec.capture(f"{p}ln2", self.ln2(resid_mid)), rec)
        resid_post = resid_mid + mlp_out if rec.keeps_residual(n, "mlp") else mlp_out
        return rec.capture(f"{p}resid_post", resid_post)


# --- the model --------------------------------------------------------------


class ToyGPT(nn.Module):
    """The instrumented toy GPT.

    `vocab` (an id -> token-text list) is optional and only affects the human
    readable parts of a trace: token text, top-k text, and the argmax-column
    head labels. Checkpoints bundle it so inference needs no external files.
    """

    def __init__(self, cfg: ModelConfig, vocab: Sequence[str] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab: list[str] | None = list(vocab) if vocab is not None else None

        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)  # learned, absolute
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, n) for n in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # No weight tying: lm_head.weight is independent of wte.weight.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

        self.apply(self._init_weights)
        # Scale down the two projections that write into the residual stream, so
        # its variance does not grow with depth (GPT-2 initialization).
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("attn.out_proj.weight", "mlp.proj.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- vocabulary

    def set_vocab(self, vocab: Sequence[str] | None) -> None:
        self.vocab = list(vocab) if vocab is not None else None

    def token_text(self, token_id: int) -> str:
        """Human-readable text for a token id, or `<id>` when no vocab is set."""
        if self.vocab is not None and 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return f"<{token_id}>"

    # -- the single forward implementation

    def _forward_impl(self, idx: Tensor, rec: Recorder) -> Tensor:
        batch, seq_len = idx.shape
        if seq_len > self.cfg.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.cfg.block_size}"
            )

        tok = rec.capture(SITE_EMBED_TOK, self.wte(idx))
        positions = torch.arange(seq_len, device=idx.device)
        # Kept as (1, T, d_model) so every captured tensor has a batch dim to
        # squeeze; it broadcasts over the batch exactly as a (T, d_model) would.
        pos = rec.capture(SITE_EMBED_POS, self.wpe(positions).unsqueeze(0))

        x = rec.capture(SITE_RESID_0, self.drop(tok + pos))
        for block in self.blocks:
            x = block(x, rec)

        x = rec.capture(SITE_FINAL_LN, self.ln_f(x))
        logits = rec.capture(SITE_LOGITS, self.lm_head(x))

        if rec.active:
            # Only computed when something will look at it; it feeds nothing.
            rec.capture(SITE_PROBS, torch.softmax(logits, dim=-1))
        return logits

    def forward(self, idx: Tensor) -> Tensor:
        """Plain forward for training. (B, T) token ids -> (B, T, vocab_size)."""
        return self._forward_impl(idx, NULL_RECORDER)

    # -- tracing

    def capture_sites(
        self,
        idx: Tensor,
        interventions: Iterable[InterventionInput] | None = None,
    ) -> tuple[dict[str, Tensor], list[object], InterventionPlan, float]:
        """Run the instrumented forward pass and return the raw captured tensors.

        Returns `(sites, ops, plan, timing_ms)` where `sites` maps every site in
        the manifest to its tensor with the batch dimension squeezed out. These
        are the unrounded tensors; `forward_with_trace` wraps them into the
        rounded wire envelopes.
        """
        # Local import: `engine` imports `model`, so importing it at module
        # level would close an import cycle.
        from engine.interventions import compile_interventions
        from engine.trace import TraceRecorder

        if idx.dim() != 2:
            raise ValueError(f"expected idx of shape (1, T), got {tuple(idx.shape)}")
        assert idx.shape[0] == 1, (
            f"tracing forces batch size 1, got batch {idx.shape[0]}; "
            "every captured tensor has its batch dimension squeezed out"
        )

        ops, plan = compile_interventions(interventions, self.cfg)
        rec = TraceRecorder(plan)

        was_training = self.training
        self.eval()
        try:
            start = time.perf_counter()
            with torch.no_grad():
                self._forward_impl(idx, rec)
            timing_ms = (time.perf_counter() - start) * 1000.0
        finally:
            self.train(was_training)

        return rec.sites, ops, plan, timing_ms

    def forward_with_trace(
        self,
        idx: Tensor,
        interventions: Iterable[InterventionInput] | None = None,
        *,
        input_text: str | None = None,
        position: int = -1,
        top_k: int = 20,
    ) -> Trace:
        """Instrumented forward pass over a single sequence.

        `interventions=None` or `[]` is the baseline, produced by this exact
        code path -- not by a separate uninstrumented forward.
        """
        from engine.trace import build_trace

        sites, ops, _plan, timing_ms = self.capture_sites(idx, interventions)

        token_ids = [int(t) for t in idx[0].tolist()]
        return build_trace(
            cfg=self.cfg,
            token_ids=token_ids,
            token_texts=[self.token_text(t) for t in token_ids],
            raw_sites=sites,
            interventions=ops,
            input_text=input_text,
            position=position,
            top_k=top_k,
            timing_ms=timing_ms,
            vocab_texts=self.vocab,
        )
