"""Tests for the FlashPairformer (opt_core) kernel route in the triangle layers.

These run anywhere: on CPU the route never reaches opt_core, so they cover the
tier setting, the error policy and how the layers use a FlashPairformer result.
The kernels themselves are checked against cuequivariance on a GPU in
test_flashpairformer_cuda.py.
"""

import importlib

import pytest
import torch

from boltz.model.layers import flashpairformer
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the route's per-process state around each test."""
    monkeypatch.setitem(flashpairformer._kernel_failure, "reason", None)  # noqa: SLF001
    flashpairformer.tier.cache_clear()
    yield
    flashpairformer.tier.cache_clear()


@pytest.fixture
def dim() -> int:
    """Pair dimension for the layers under test."""
    return 32


class TestTier:
    """Tests for the BOLTZ_FLASHPAIRFORMER setting."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, "fast"), ("exact", "exact"), ("OFF", "off"), ("0", "off")],
    )
    def test_env(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: str
    ) -> None:
        """The tier defaults to fast and accepts the documented spellings."""
        if value is None:
            monkeypatch.delenv(flashpairformer.ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(flashpairformer.ENV_VAR, value)
        assert flashpairformer.tier() == expected

    def test_unknown_value_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown tier warns and uses fast."""
        monkeypatch.setenv(flashpairformer.ENV_VAR, "turbo")
        with pytest.warns(RuntimeWarning, match="turbo"):
            assert flashpairformer.tier() == "fast"


class TestRoute:
    """Tests for when FlashPairformer is tried and what happens on errors."""

    def test_cpu_tensors_are_not_routed(self) -> None:
        """On CPU, both ops return None without calling the stock op."""
        q = torch.randn(1, 4, 2, 4, 8)

        def stock(*_args: object, **_kwargs: object) -> None:
            message = "the stock op should not be called"
            raise AssertionError(message)

        assert (
            flashpairformer.triangle_attention(
                q, q, q, torch.randn(1, 1, 2, 4, 4), None, 0.5, stock=stock
            )
            is None
        )
        layer = TriangleMultiplicationOutgoing(dim=16)
        assert (
            flashpairformer.triangle_multiplication(
                x=torch.randn(1, 4, 4, 16),
                direction="outgoing",
                mask=torch.ones(1, 4, 4),
                norm_in_weight=layer.norm_in.weight,
                norm_in_bias=layer.norm_in.bias,
                p_in_weight=layer.p_in.weight,
                g_in_weight=layer.g_in.weight,
                norm_out_weight=layer.norm_out.weight,
                norm_out_bias=layer.norm_out.bias,
                p_out_weight=layer.p_out.weight,
                g_out_weight=layer.g_out.weight,
                eps=1e-5,
            )
            is None
        )

    def test_kernel_error_disables_once(self) -> None:
        """A kernel error warns once and turns the route off for the process."""
        with pytest.warns(RuntimeWarning, match="triton compile failed"):
            flashpairformer._on_kernel_error(RuntimeError("triton compile failed"))  # noqa: SLF001
        assert "triton compile failed" in flashpairformer._kernel_failure["reason"]  # noqa: SLF001

    def test_out_of_memory_propagates(self) -> None:
        """Out-of-memory errors are raised, not turned into a fallback."""
        with pytest.raises(torch.cuda.OutOfMemoryError):
            flashpairformer._on_kernel_error(  # noqa: SLF001
                torch.cuda.OutOfMemoryError("CUDA out of memory")
            )
        assert flashpairformer._kernel_failure["reason"] is None  # noqa: SLF001


class TestTriangleMultiplicationCache:
    """Tests for the per-layer cache handed to opt_core."""

    def test_same_weight_same_cache(self) -> None:
        """Repeated calls with one layer's weights reuse one cache."""
        weight = torch.randn(64, 32)
        first = flashpairformer._trimul_cache(weight)  # noqa: SLF001
        assert flashpairformer._trimul_cache(weight) is first  # noqa: SLF001
        assert flashpairformer._trimul_cache(torch.randn(64, 32)) is not first  # noqa: SLF001

    def test_cache_is_dropped_with_the_weight(self) -> None:
        """A layer's cache goes away when its weights do."""
        weight = torch.randn(64, 32)
        key = id(weight)
        flashpairformer._trimul_cache(weight)  # noqa: SLF001
        del weight
        assert key not in flashpairformer._trimul_caches  # noqa: SLF001


class TestOptCoreInterface:
    """Tests that the installed opt_core has what the route calls (skipped without it)."""

    @pytest.mark.parametrize(
        ("module", "names"),
        [
            ("triattn", ["Refusal", "triangle_attention"]),
            (
                "trimul",
                [
                    "Refusal",
                    "STOCK_ROWS",
                    "call_precision",
                    "select",
                    "stack_word",
                    "triangle_multiplication",
                ],
            ),
        ],
    )
    def test_kernel_modules(self, module: str, names: list[str]) -> None:
        """The opt_core kernel modules import by name and expose the functions used."""
        pytest.importorskip("opt_core")
        kernels = importlib.import_module(f"opt_core.kernels.{module}")
        assert [n for n in names if not hasattr(kernels, n)] == []


class TestLayers:
    """Tests that the layers return a FlashPairformer result when there is one."""

    def test_triangle_attention_uses_result(
        self, monkeypatch: pytest.MonkeyPatch, dim: int
    ) -> None:
        """Triangle attention uses the FlashPairformer output when there is one."""
        layer = TriangleAttention(c_in=dim, c_hidden=8, no_heads=2)
        layer.eval()
        monkeypatch.setattr(
            flashpairformer,
            "triangle_attention",
            lambda q, *_args, **_kwargs: torch.zeros_like(q),
        )

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            torch.nn.init.normal_(layer.mha.linear_o.weight, std=0.1)
            out = layer(x, mask, use_kernels=True)
            reference = layer(x, mask, use_kernels=False)

        # A zero attention output projects to zero; the PyTorch path does not.
        assert torch.equal(out, torch.zeros_like(out))
        assert not torch.allclose(reference, torch.zeros_like(reference))

    @pytest.mark.parametrize(
        "layer_cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming]
    )
    def test_triangle_multiplication_uses_result(
        self, monkeypatch: pytest.MonkeyPatch, dim: int, layer_cls: type
    ) -> None:
        """Triangle multiplication returns the FlashPairformer output unchanged."""
        layer = layer_cls(dim=dim)
        layer.eval()
        monkeypatch.setattr(
            flashpairformer,
            "triangle_multiplication",
            lambda x, **_kwargs: torch.full_like(x, 3.0),
        )

        batch, num_tokens = 1, 6
        x = torch.randn(batch, num_tokens, num_tokens, dim)
        mask = torch.ones(batch, num_tokens, num_tokens)

        with torch.no_grad():
            out = layer(x, mask, use_kernels=True)

        assert torch.equal(out, torch.full_like(x, 3.0))
