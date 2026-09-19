"""Shared fixtures.

`backend/` is the import root: `model` and `engine` are sibling top-level
packages, which is how `backend/train.py` imports them too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from model.config import ModelConfig  # noqa: E402
from model.transformer import ToyGPT  # noqa: E402

SEED: int = 1234


@pytest.fixture(scope="session")
def cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture(scope="session")
def vocab(cfg: ModelConfig) -> list[str]:
    return [f"tok{i}" for i in range(cfg.vocab_size)]


@pytest.fixture(scope="session")
def model(cfg: ModelConfig, vocab: list[str]) -> ToyGPT:
    """A seeded, eval-mode model. Session-scoped: no test mutates its weights."""
    torch.manual_seed(SEED)
    net = ToyGPT(cfg, vocab=vocab)
    net.eval()
    return net


@pytest.fixture(scope="session")
def idx(cfg: ModelConfig) -> torch.Tensor:
    """A fixed (1, 8) input sequence."""
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, cfg.vocab_size, (1, 8), generator=generator)
