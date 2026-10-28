"""Tests for the FlashPairformer (opt_core) kernel route in the triangle layers.

These run anywhere: on CPU the route never reaches opt_core, so they cover the
tier setting, the error policy and how the layers use a FlashPairformer result.
The kernels themselves are checked against cuequivariance on a GPU in
test_flashpairformer_cuda.py.
"""

import gc
import importlib
import sys
import weakref
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

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
    flashpairformer._prepare_kernel_environment.cache_clear()  # noqa: SLF001
    flashpairformer.clear_weight_caches()
    yield
    flashpairformer._prepare_kernel_environment.cache_clear()  # noqa: SLF001
    flashpairformer.clear_weight_caches()


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

    def test_environment_changes_apply_without_reset(self, monkeypatch) -> None:
        for value in ("fast", "off", "exact", "fast"):
            monkeypatch.setenv(flashpairformer.ENV_VAR, value)
            assert flashpairformer.tier() == value


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

    @pytest.mark.parametrize(
        "error",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            RuntimeError("CUDA OUT OF MEMORY"),
            RuntimeError("Out Of Memory in kernel"),
        ],
    )
    def test_out_of_memory_propagates(self, error: Exception) -> None:
        """Out-of-memory errors are raised, not turned into a fallback."""
        with pytest.raises(type(error)) as caught:
            flashpairformer._on_kernel_error(error)  # noqa: SLF001
        assert caught.value is error
        assert flashpairformer._kernel_failure["reason"] is None  # noqa: SLF001


@pytest.fixture
def cuda_dispatch(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Exercise admission without requiring a CUDA device or importing opt_core."""
    backend = SimpleNamespace()
    importer = Mock(return_value=backend)
    monkeypatch.setattr(
        flashpairformer, "importlib", SimpleNamespace(import_module=importer)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: device == "cuda")
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.bfloat16)
    return SimpleNamespace(
        tensor=SimpleNamespace(
            is_cuda=True, device=torch.device("cuda"), dtype=torch.float32
        ),
        backend=backend,
        importer=importer,
    )


class TestInferenceAdmission:
    """Only the measured Boltz-2 inference scope may reach the optional package."""

    @pytest.mark.parametrize("name", ["triattn", "trimul"])
    @pytest.mark.parametrize(
        ("scope", "grad", "autocast", "dtype", "cuda", "capability", "accepted"),
        [
            (False, False, True, torch.bfloat16, True, (9, 0), False),
            (True, True, True, torch.bfloat16, True, (9, 0), False),
            (True, False, False, torch.bfloat16, True, (9, 0), False),
            (True, False, True, torch.float16, True, (9, 0), False),
            (True, False, True, torch.float32, True, (9, 0), False),
            (True, False, True, torch.bfloat16, False, (9, 0), False),
            (True, False, True, torch.bfloat16, True, (7, 5), False),
            (True, False, True, torch.bfloat16, True, (9, 0), True),
        ],
    )
    def test_scope_grad_and_precision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cuda_dispatch: SimpleNamespace,
        name: str,
        scope: bool,
        grad: bool,
        autocast: bool,
        dtype: torch.dtype,
        cuda: bool,
        capability: tuple[int, int],
        accepted: bool,
    ) -> None:
        monkeypatch.setattr(
            torch,
            "is_autocast_enabled",
            lambda device: autocast if device == "cuda" else False,
        )
        monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: dtype)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device: capability
        )
        cuda_dispatch.tensor.is_cuda = cuda
        with flashpairformer.inference_scope(scope), torch.set_grad_enabled(grad):
            result = flashpairformer._kernels(cuda_dispatch.tensor, name)
        if accepted:
            assert result is cuda_dispatch.backend
            cuda_dispatch.importer.assert_called_once_with(f"opt_core.kernels.{name}")
        else:
            assert result is None
            cuda_dispatch.importer.assert_not_called()

    def test_boltz2_eval_enables_and_restores_scope(
        self, cuda_dispatch: SimpleNamespace
    ) -> None:
        class Model(torch.nn.Module):
            @flashpairformer.boltz2_inference
            def forward(self) -> object:
                return flashpairformer._kernels(cuda_dispatch.tensor, "triattn")

        with torch.no_grad():
            assert Model().eval()() is cuda_dispatch.backend
            # An unrelated caller, including Boltz-1, is still excluded.
            assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None

    def test_training_no_grad_recycling_stays_excluded(
        self, cuda_dispatch: SimpleNamespace
    ) -> None:
        class Model(torch.nn.Module):
            @flashpairformer.boltz2_inference
            def forward(self) -> object:
                return flashpairformer._kernels(cuda_dispatch.tensor, "trimul")

        with torch.no_grad(), flashpairformer.inference_scope():
            # Model.train() must override even an enclosing evaluation scope.
            assert Model().train()() is None
            assert (
                flashpairformer._kernels(cuda_dispatch.tensor, "trimul")
                is cuda_dispatch.backend
            )

    def test_exception_restores_scope(self, cuda_dispatch: SimpleNamespace) -> None:
        class Model(torch.nn.Module):
            @flashpairformer.boltz2_inference
            def forward(self) -> object:
                raise RuntimeError("forward failed")

        with torch.no_grad():
            with pytest.raises(RuntimeError, match="forward failed"):
                Model().eval()()
            assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None
        cuda_dispatch.importer.assert_not_called()

    def test_tier_changes_control_dispatch_without_reset(
        self, monkeypatch, cuda_dispatch
    ):
        with torch.no_grad(), flashpairformer.inference_scope():
            monkeypatch.setenv(flashpairformer.ENV_VAR, "off")
            assert flashpairformer._kernels(cuda_dispatch.tensor, "trimul") is None
            monkeypatch.setenv(flashpairformer.ENV_VAR, "fast")
            assert (
                flashpairformer._kernels(cuda_dispatch.tensor, "trimul")
                is cuda_dispatch.backend
            )
            monkeypatch.setenv(flashpairformer.ENV_VAR, "off")
            assert flashpairformer._kernels(cuda_dispatch.tensor, "trimul") is None
        cuda_dispatch.importer.assert_called_once()

    def test_census_setup_is_lazy_and_runs_once(self, monkeypatch, cuda_dispatch):
        monkeypatch.delenv("OPT_CORE_CELL_CENSUS_PRINT", raising=False)
        setter = Mock(wraps=flashpairformer.os.environ.setdefault)
        monkeypatch.setattr(flashpairformer.os.environ, "setdefault", setter)
        with torch.no_grad():
            assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None
            setter.assert_not_called()
            with flashpairformer.inference_scope():
                flashpairformer._kernels(cuda_dispatch.tensor, "triattn")
                flashpairformer._kernels(cuda_dispatch.tensor, "trimul")
        setter.assert_called_once_with("OPT_CORE_CELL_CENSUS_PRINT", "0")

    def test_failure_stays_latched_across_tier_changes(
        self, monkeypatch, cuda_dispatch
    ):
        with pytest.warns(RuntimeWarning):
            flashpairformer._on_kernel_error(RuntimeError("kernel failed"))
        with torch.no_grad(), flashpairformer.inference_scope():
            for value in ("off", "fast", "exact"):
                monkeypatch.setenv(flashpairformer.ENV_VAR, value)
                assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None
        cuda_dispatch.importer.assert_not_called()

    def test_cpu_autocast_does_not_admit_cuda(self, monkeypatch):
        importer = Mock()
        monkeypatch.setattr(
            flashpairformer, "importlib", SimpleNamespace(import_module=importer)
        )
        tensor = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
        with (
            torch.no_grad(),
            flashpairformer.inference_scope(),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            assert torch.is_autocast_enabled("cpu")
            assert flashpairformer._kernels(tensor, "trimul") is None
        importer.assert_not_called()

    def test_excluded_callers_do_not_read_invalid_tier(
        self, monkeypatch, cuda_dispatch
    ):
        monkeypatch.setenv(flashpairformer.ENV_VAR, "invalid")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None
            with flashpairformer.inference_scope(), torch.enable_grad():
                assert flashpairformer._kernels(cuda_dispatch.tensor, "triattn") is None
        assert not caught
        cuda_dispatch.importer.assert_not_called()


@pytest.fixture
def packed_trimul(monkeypatch):
    """A backend that retains detached sources and packs them only on cache miss."""

    calls = []
    state = SimpleNamespace(address_keyed=False)
    shared = {(0, 4, 16, 16): {}, (0, 8, 16, 16): {"workspace": object()}}
    shared_packs = shared[(0, 4, 16, 16)]
    payload_cache = Mock(side_effect=lambda *key: shared[key])
    native = SimpleNamespace(
        shared_geometries=lambda: [(*key, 1234) for key in shared],
        payload_cache=payload_cache,
    )
    monkeypatch.setitem(sys.modules, "opt_core.kernels.trimul.native", native)

    def kernel(x, mask, *, weights, cache, eps, **kwargs):
        calls.append(weights)
        if state.address_keyed:
            # Native payloads share a pack cache outside the caller dictionary.
            key = tuple(
                (k, w.data_ptr(), tuple(w.shape), w.dtype) for k, w in weights.items()
            )
            if key not in shared_packs:
                shared_packs[key] = (
                    {k: w.clone() for k, w in weights.items()},
                    weights,
                )
            return x.new_tensor(
                sum(w.sum().item() for w in shared_packs[key][0].values())
            )
        if "test_pack" not in cache:
            cache["test_pack"] = {key: value.clone() for key, value in weights.items()}
            cache["test_sources"] = weights
        return x.new_tensor(
            sum(value.sum().item() for value in cache["test_pack"].values())
        )

    backend = SimpleNamespace(
        STOCK_ROWS=(),
        Refusal=RuntimeError,
        triangle_multiplication=kernel,
        call_precision=lambda x: ("bf16", None),
    )
    monkeypatch.setattr(flashpairformer, "_kernels", lambda *args: backend)
    monkeypatch.setattr(flashpairformer, "_trimul_stack", lambda device: "test")
    monkeypatch.setattr(
        flashpairformer,
        "_trimul_selection",
        lambda *args, **kwargs: SimpleNamespace(row="test"),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def run(layer):
        with torch.no_grad():
            return flashpairformer.triangle_multiplication(
                x=torch.ones(1, 4, 4, 16),
                mask=torch.ones(1, 4, 4),
                direction="outgoing",
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

    run.calls = calls
    run.state = state
    run.shared = shared
    run.payload_cache = payload_cache
    return run


class TestTriangleMultiplicationCache:
    """Tests for the per-layer cache handed to opt_core."""

    def test_same_weight_same_cache(self) -> None:
        """Repeated calls with one layer's weights reuse one cache."""
        weight = torch.randn(64, 32)
        first = flashpairformer._trimul_cache(weight)  # noqa: SLF001
        assert flashpairformer._trimul_cache(weight) is first  # noqa: SLF001
        assert flashpairformer._trimul_cache(torch.randn(64, 32)) is not first  # noqa: SLF001

    def test_unchanged_weights_keep_packs(self, packed_trimul):
        packed_trimul.state.address_keyed = True
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        packs = dict(packed_trimul.shared[(0, 4, 16, 16)])
        assert torch.equal(before, packed_trimul(layer))
        assert packed_trimul.shared[(0, 4, 16, 16)] == packs
        packed_trimul.payload_cache.assert_not_called()

    def test_clear_does_not_import_native(self, monkeypatch):
        monkeypatch.delitem(
            sys.modules, "opt_core.kernels.trimul.native", raising=False
        )
        with monkeypatch.context() as patch:
            patch.setattr(flashpairformer.importlib, "import_module", Mock())
            flashpairformer.clear_weight_caches()
            flashpairformer.importlib.import_module.assert_not_called()
        assert "opt_core.kernels.trimul.native" not in sys.modules

    def test_clear_drops_all_loaded_shared_geometries(self, packed_trimul):
        flashpairformer.clear_weight_caches()
        assert all(not cache for cache in packed_trimul.shared.values())
        assert packed_trimul.payload_cache.call_count == 2

    def test_cache_is_dropped_with_the_weight(self) -> None:
        """A layer's cache goes away when its weights do."""
        weight = torch.randn(64, 32)
        key = id(weight)
        flashpairformer._trimul_cache(weight)  # noqa: SLF001
        del weight
        assert key not in flashpairformer._trimul_caches  # noqa: SLF001

    @pytest.mark.parametrize(
        "parameter",
        [
            "norm_in.weight",
            "norm_in.bias",
            "p_in.weight",
            "g_in.weight",
            "norm_out.weight",
            "norm_out.bias",
            "p_out.weight",
            "g_out.weight",
        ],
    )
    def test_tracked_update_matches_fresh_pack(self, packed_trimul, parameter):
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        with torch.no_grad():
            dict(layer.named_parameters())[parameter].add_(1)
        updated = packed_trimul(layer)
        flashpairformer.clear_weight_caches()
        assert torch.equal(updated, packed_trimul(layer))
        assert not torch.equal(before, updated)

    @pytest.mark.parametrize(
        "parameter",
        [
            "norm_in.weight",
            "norm_in.bias",
            "p_in.weight",
            "g_in.weight",
            "norm_out.weight",
            "norm_out.bias",
            "p_out.weight",
            "g_out.weight",
        ],
    )
    def test_shared_address_cache_observes_updates(self, packed_trimul, parameter):
        packed_trimul.state.address_keyed = True
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        with torch.no_grad():
            dict(layer.named_parameters())[parameter].add_(1)
        updated = packed_trimul(layer)
        assert not torch.equal(before, updated)
        assert packed_trimul.payload_cache.call_count == 2
        assert not packed_trimul.shared[(0, 8, 16, 16)]
        flashpairformer.clear_weight_caches()
        assert torch.equal(updated, packed_trimul(layer))

    def test_shared_address_cache_observes_untracked_write(self, packed_trimul):
        packed_trimul.state.address_keyed = True
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        layer.g_out.weight.data.add_(1)
        flashpairformer.clear_weight_caches()
        assert not torch.equal(before, packed_trimul(layer))

    @pytest.mark.parametrize("parameter", ["p_in.weight", "g_out.weight"])
    def test_unversioned_weights_skip_backend(self, packed_trimul, parameter):
        layer = TriangleMultiplicationOutgoing(dim=16)
        module_name, parameter_name = parameter.split(".")
        module = getattr(layer, module_name)
        with torch.inference_mode():
            unversioned = torch.nn.Parameter(getattr(module, parameter_name).clone())
        setattr(module, parameter_name, unversioned)
        assert packed_trimul(layer) is None
        assert not packed_trimul.calls
        assert id(layer.p_in.weight) not in flashpairformer._trimul_caches

    def test_storage_replacement_invalidates_pack(self, packed_trimul):
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        # Replacing .data does not advance the parameter's version counter.
        layer.g_in.weight.data = layer.g_in.weight.detach().clone() + 1
        assert not torch.equal(before, packed_trimul(layer))

    def test_untracked_write_can_clear_packs(self, packed_trimul):
        layer = TriangleMultiplicationOutgoing(dim=16)
        before = packed_trimul(layer)
        layer.p_in.weight.data.add_(1)
        flashpairformer.clear_weight_caches()
        assert not torch.equal(before, packed_trimul(layer))

    def test_normalization_change_invalidates_pack(self):
        weight = torch.randn(32, 16)
        cache = flashpairformer._trimul_cache(weight, eps=1e-5)
        cache["test_pack"] = object()
        assert "test_pack" not in flashpairformer._trimul_cache(weight, eps=1e-3)

    def test_retained_sources_do_not_prevent_cleanup(self, packed_trimul):
        layer = TriangleMultiplicationOutgoing(dim=16)
        owner = weakref.ref(layer.p_in.weight)
        key = id(layer.p_in.weight)
        packed_trimul(layer)
        assert "test_sources" in flashpairformer._trimul_caches[key]
        del layer
        gc.collect()
        assert owner() is None
        assert key not in flashpairformer._trimul_caches


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
            ("trimul.native", ["shared_geometries", "payload_cache"]),
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
