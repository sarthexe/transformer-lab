"""Train the toy GPT on the corpus produced by `backend/data/prepare.py`.

Checkpointing every 500 steps is not optional: this is a hackathon model and
the run has to survive being killed. Each checkpoint is self-contained -- model
weights, the `ModelConfig` that shaped them, and the vocabulary -- so inference
and tracing need no external files.

    python backend/train.py
    python backend/train.py --steps 500 --batch 16
    python backend/train.py --resume checkpoints/step_2000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from model.config import ModelConfig
from model.transformer import ToyGPT

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
CHECKPOINT_DIR = BACKEND_DIR.parent / "checkpoints"

LEARNING_RATE: float = 3e-4
WARMUP_STEPS: int = 100
TOTAL_STEPS: int = 4000
BATCH_SIZE: int = 64
WEIGHT_DECAY: float = 0.1
GRAD_CLIP: float = 1.0

EVAL_EVERY: int = 250
EVAL_BATCHES: int = 50
CHECKPOINT_EVERY: int = 500

#: The cosine schedule decays to this fraction of the peak learning rate rather
#: than to zero; a small floor keeps the last few hundred steps useful.
MIN_LR_RATIO: float = 0.1

SEED: int = 1337


# --- data -------------------------------------------------------------------


def require_data(data_dir: Path) -> tuple[Path, Path, Path]:
    """Locate the prepared corpus, or exit with an actionable message."""
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    vocab_json = data_dir / "vocab.json"

    missing = [p.name for p in (train_bin, val_bin, vocab_json) if not p.exists()]
    if missing:
        sys.exit(
            f"missing {', '.join(missing)} in {data_dir}\n"
            "Run the corpus preparation script first:\n"
            "    python backend/data/prepare.py"
        )
    return train_bin, val_bin, vocab_json


def load_split(path: Path, block_size: int) -> np.ndarray:
    """Memory-map a flat uint16 token stream."""
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    if tokens.size < block_size + 1:
        sys.exit(
            f"{path.name} holds only {tokens.size} tokens, need at least "
            f"{block_size + 1} for a single {block_size}-token window"
        )
    return tokens


def load_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return list(payload["id_to_token"])
    except (TypeError, KeyError):
        sys.exit(f"{path} has no 'id_to_token' list; regenerate it with prepare.py")


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample `batch_size` random windows; y is x shifted one position left."""
    starts = generator.integers(0, len(tokens) - block_size - 1, size=batch_size)
    x = np.stack([tokens[i : i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([tokens[i + 1 : i + 1 + block_size] for i in starts]).astype(np.int64)

    inputs = torch.from_numpy(x).to(device, non_blocking=True)
    targets = torch.from_numpy(y).to(device, non_blocking=True)
    return inputs, targets


# --- training ---------------------------------------------------------------


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: ToyGPT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Decay the matrices, not the biases and LayerNorm gains.

    Those are 1-D parameters whose scale is meaningful; shrinking them toward
    zero is regularization of the wrong thing.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def lr_multiplier(step: int, total_steps: int) -> float:
    """Linear warmup for `WARMUP_STEPS`, then cosine decay to MIN_LR_RATIO."""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    span = max(1, total_steps - WARMUP_STEPS)
    progress = min(1.0, (step - WARMUP_STEPS) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def loss_for(model: ToyGPT, inputs: Tensor, targets: Tensor) -> Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: ToyGPT,
    splits: dict[str, np.ndarray],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Mean loss over `EVAL_BATCHES` fresh batches per split.

    Uses its own fixed-seed generator so the eval windows are the same at every
    evaluation, which makes the printed curve comparable step to step.
    """
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    for name, tokens in splits.items():
        generator = np.random.default_rng(SEED)
        total = 0.0
        for _ in range(EVAL_BATCHES):
            inputs, targets = get_batch(tokens, batch_size, block_size, device, generator)
            total += loss_for(model, inputs, targets).item()
        results[name] = total / EVAL_BATCHES
    model.train(was_training)
    return results


def save_checkpoint(
    path: Path,
    model: ToyGPT,
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    vocab: list[str],
    step: int,
    losses: dict[str, float],
) -> None:
    """Write a self-contained checkpoint: weights, config, and vocabulary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "losses": losses,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"no checkpoint at {path}")
    return torch.load(path, map_location=device, weights_only=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="training steps")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="peak learning rate")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="prepared corpus dir")
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR, help="checkpoint dir")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(SEED)

    device = pick_device()
    print(f"device              {device.type}")

    train_bin, val_bin, vocab_json = require_data(args.data)
    vocab = load_vocab(vocab_json)
    cfg = ModelConfig(vocab_size=len(vocab))

    splits = {
        "train": load_split(train_bin, cfg.block_size),
        "val": load_split(val_bin, cfg.block_size),
    }
    print(f"vocab               {cfg.vocab_size:,}")
    print(f"train tokens        {splits['train'].size:,}")
    print(f"val tokens          {splits['val'].size:,}")

    model = ToyGPT(cfg, vocab=vocab).to(device)
    optimizer = build_optimizer(model, args.lr, WEIGHT_DECAY)
    print(f"parameters          {model.num_parameters():,}")
    print(f"steps               {args.steps:,}  batch {args.batch}  block {cfg.block_size}")

    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        saved_cfg = ModelConfig.from_dict(checkpoint["config"])
        if saved_cfg != cfg:
            sys.exit(
                f"checkpoint config {saved_cfg} does not match the current config {cfg}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        print(f"resumed             {args.resume} at step {start_step:,}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, args.steps), last_epoch=start_step - 1
    )

    generator = np.random.default_rng(SEED + start_step)
    model.train()
    losses: dict[str, float] = {}
    started = time.perf_counter()
    print("-" * 58)

    for step in range(start_step, args.steps):
        inputs, targets = get_batch(
            splits["train"], args.batch, cfg.block_size, device, generator
        )
        loss = loss_for(model, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        done = step + 1
        if done % EVAL_EVERY == 0 or done == args.steps:
            losses = estimate_loss(model, splits, args.batch, cfg.block_size, device)
            elapsed = time.perf_counter() - started
            print(
                f"step {done:>5}/{args.steps}  "
                f"train {losses['train']:.4f}  val {losses['val']:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {elapsed:6.1f}s"
            )

        if done % CHECKPOINT_EVERY == 0 or done == args.steps:
            path = args.out / f"step_{done}.pt"
            save_checkpoint(path, model, optimizer, cfg, vocab, done, losses)
            print(f"                     saved {path}")

    print("-" * 58)
    print(f"done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
