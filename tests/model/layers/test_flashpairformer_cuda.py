"""FlashPairformer kernels vs cuequivariance on a GPU.

These need an Ampere or newer NVIDIA GPU with cuequivariance and opt_core
installed, and are excluded from CI; run them manually:

    pytest tests/model/layers/test_flashpairformer_cuda.py -m cuda -v

- ``exact`` must be bit-identical to the cuequivariance op.
- ``fast`` must be about as close to an fp32 PyTorch reference as the
  cuequivariance op is.
"""

import importlib.util

import pytest
import torch

from boltz.model.layers import flashpairformer
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() < (8, 0)
        or importlib.util.find_spec("opt_core") is None
        or importlib.util.find_spec("cuequivariance_torch") is None,
        reason="needs an Ampere+ GPU with opt_core and cuequivariance",
    ),
]

LAYERS = [
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
]


def build(layer_cls: type, dim: int) -> torch.nn.Module:
    """Build a layer with random weights, including its zero-initialised outputs."""
    torch.manual_seed(0)
    if issubclass(
        layer_cls, (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
    ):
        layer = layer_cls(dim=dim)
    else:
        layer = layer_cls(c_in=dim, c_hidden=32, no_heads=4)
    with torch.no_grad():
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=dim**-0.5)
    return layer.cuda().eval()


def run(
    monkeypatch: pytest.MonkeyPatch,
    layer: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    tier: str,
) -> torch.Tensor:
    """Run the layer twice with kernels on at a FlashPairformer tier, in bf16 autocast.

    The second call goes through every cache the first one filled; both must
    agree and neither may turn the route off.
    """
    monkeypatch.setenv(flashpairformer.ENV_VAR, tier)
    flashpairformer.tier.cache_clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x, mask, use_kernels=True)
        again = layer(x, mask, use_kernels=True)
    assert flashpairformer._kernel_failure["reason"] is None  # noqa: SLF001
    assert torch.equal(out, again)
    return out.float()


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the route's per-process state around each test."""
    monkeypatch.setitem(flashpairformer._kernel_failure, "reason", None)  # noqa: SLF001
    yield
    flashpairformer.tier.cache_clear()


def inputs(num_tokens: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random pair features and a mask with the last 7 tokens padded out."""
    torch.manual_seed(1)
    x = torch.randn(1, num_tokens, num_tokens, dim, device="cuda")
    mask = torch.ones(1, num_tokens, num_tokens, device="cuda")
    mask[:, num_tokens - 7 :, :] = 0
    mask[:, :, num_tokens - 7 :] = 0
    return x, mask


@pytest.mark.parametrize("num_tokens", [256, 512])
@pytest.mark.parametrize("layer_cls", LAYERS)
def test_exact_matches_cuequivariance(
    monkeypatch: pytest.MonkeyPatch, layer_cls: type, num_tokens: int
) -> None:
    """The exact tier is bit-identical to the cuequivariance op."""
    layer = build(layer_cls, dim=128)
    x, mask = inputs(num_tokens, dim=128)

    cueq = run(monkeypatch, layer, x, mask, "off")
    exact = run(monkeypatch, layer, x, mask, "exact")

    assert torch.equal(exact, cueq)


@pytest.mark.parametrize("num_tokens", [256, 512])
@pytest.mark.parametrize("layer_cls", LAYERS)
def test_fast_is_as_close_as_cuequivariance(
    monkeypatch: pytest.MonkeyPatch, layer_cls: type, num_tokens: int
) -> None:
    """The fast tier's error vs fp32 PyTorch is within 2x the cuequivariance op's."""
    layer = build(layer_cls, dim=128)
    x, mask = inputs(num_tokens, dim=128)
    with torch.no_grad():
        reference = layer(x, mask, use_kernels=False)

    cueq = run(monkeypatch, layer, x, mask, "off")
    fast = run(monkeypatch, layer, x, mask, "fast")

    def rel_error(out: torch.Tensor) -> float:
        return ((out - reference).norm() / reference.norm()).item()

    assert rel_error(fast) <= 2 * rel_error(cueq) + 1e-3
