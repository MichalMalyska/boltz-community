"""Forward-local MSA work reuse, with exact CUDA comparisons to the original path."""

from types import SimpleNamespace

import pytest
import torch

from boltz.model.layers import msa_inference
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.bfloat16)
    return SimpleNamespace(is_cuda=True)


def test_scope_admission_and_exception_cleanup(fake_cuda):
    with torch.no_grad():
        with msa_inference.inference_scope(fake_cuda, training=False):
            outer = msa_inference._mask_counts.get()
            assert outer == {}
            with msa_inference.inference_scope(fake_cuda, training=True):
                assert msa_inference._mask_counts.get() is None
            assert msa_inference._mask_counts.get() is outer
            with (
                pytest.raises(ValueError),
                msa_inference.inference_scope(fake_cuda, training=False),
            ):
                raise ValueError("failure inside forward")
            assert msa_inference._mask_counts.get() is outer
    assert msa_inference._mask_counts.get() is None
    with torch.enable_grad(), msa_inference.inference_scope(fake_cuda, training=False):
        assert msa_inference._mask_counts.get() is None


def test_cpu_and_unscoped_calls_are_unchanged():
    x = torch.randn(2, 3)
    with torch.no_grad(), msa_inference.inference_scope(x, training=False):
        assert not msa_inference.enabled(x)
        assert msa_inference.cast_activation(x) is x
    assert msa_inference.cast_activation(x) is x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_other_precision_is_excluded(fake_cuda, monkeypatch, dtype):
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: dtype)
    with torch.no_grad(), msa_inference.inference_scope(fake_cuda, training=False):
        assert msa_inference._mask_counts.get() is None


def test_cpu_autocast_is_excluded():
    x = torch.randn(2, 3)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        with msa_inference.inference_scope(x, training=False):
            assert not msa_inference.enabled(x)
            assert msa_inference.cast_activation(x) is x


@pytest.fixture
def scoped_counts(monkeypatch):
    # Exercise ownership/arithmetic on CPU; these mocks are not CUDA parity proof.
    token = msa_inference._mask_counts.set({})
    monkeypatch.setattr(msa_inference, "enabled", lambda x: True)
    yield
    msa_inference._mask_counts.reset(token)


def test_counts_reuse_and_tracked_mutation(scoped_counts):
    mask = torch.ones(1, 65, 9)
    expanded = mask.unsqueeze(-1)
    first = msa_inference.mask_counts(mask, expanded)
    assert torch.equal(first, torch.full((1, 9, 9, 1), 65.0))
    assert msa_inference.mask_counts(mask, expanded) is first
    mask[:, :5] = 0
    updated = msa_inference.mask_counts(mask, expanded)
    assert torch.equal(updated, torch.full_like(first, 60.0))
    assert updated is not first


def test_mask_identity_dtype_and_empty_columns(scoped_counts):
    mask = torch.ones(2, 67, 9)
    mask[:, :, -1] = 0
    first = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
    assert (first[:, -1] == 1).all()
    assert (first[:, :, -1] == 1).all()
    copy = mask.clone()
    assert msa_inference.mask_counts(copy, copy.unsqueeze(-1)) is not first
    low = msa_inference.mask_counts(mask, mask.unsqueeze(-1).bfloat16())
    assert low.dtype == torch.bfloat16
    assert low.float().equal(first)


def test_inference_tensors_refresh_between_forwards(fake_cuda, monkeypatch):
    monkeypatch.setattr(msa_inference, "enabled", lambda x: True)
    with torch.inference_mode():
        mask = torch.ones(1, 65, 9)
        with msa_inference.inference_scope(fake_cuda, training=False):
            first = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
            assert msa_inference.mask_counts(mask, mask.unsqueeze(-1)) is first
        mask[:, :5] = 0
        with msa_inference.inference_scope(fake_cuda, training=False):
            updated = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
        assert torch.equal(updated, torch.full_like(first, 60.0))
    assert msa_inference._mask_counts.get() is None


def test_decorator_restores_scope(fake_cuda):
    class Module:
        training = False

        @msa_inference.msa_forward
        def forward(self, z, *, fail=False):
            assert msa_inference._mask_counts.get() == {}
            if fail:
                raise ValueError("forward failed")
            return z

    with torch.no_grad():
        module = Module()
        assert module.forward(fake_cuda) is fake_cuda
        with pytest.raises(ValueError):
            module.forward(z=fake_cuda, fail=True)
    assert msa_inference._mask_counts.get() is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA bf16 autocast")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["transition", "pwa", "opm"])
def test_cuda_matches_original_chunked_path(kind, input_dtype):
    torch.manual_seed(7)
    if kind == "transition":
        layer = Transition(dim=64, hidden=257).cuda().eval()
        args = [torch.randn(2, 67, 17, 64, device="cuda", dtype=input_dtype)]
        kwargs = {"chunk_size": 32}
    elif kind == "pwa":
        layer = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
        args = [
            torch.randn(2, 67, 17, 64, device="cuda", dtype=input_dtype),
            torch.randn(2, 17, 17, 128, device="cuda", dtype=input_dtype),
            torch.ones(2, 17, 17, device="cuda"),
        ]
        args[-1][:, -2:] = 0
        kwargs = {"chunk_heads": True}
    else:
        layer = OuterProductMean(64, 32, 128).cuda().eval()
        args = [
            torch.randn(2, 67, 17, 64, device="cuda", dtype=input_dtype),
            torch.randint(0, 2, (2, 67, 17), device="cuda").float(),
        ]
        args[-1][:, :, -2:] = 0
        kwargs = {"chunk_size": 4}
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = torch.cuda.get_rng_state()
            reference = layer(*args, **kwargs)
            with msa_inference.inference_scope(args[0], training=False):
                actual = layer(*args, **kwargs)
            assert torch.equal(actual, reference)
            assert actual.dtype == reference.dtype
            assert torch.equal(state, torch.cuda.get_rng_state())


def test_compiled_training_does_not_enter_reuse_scope():
    class Module(torch.nn.Module):
        @msa_inference.msa_forward
        def forward(self, z):
            return z.sin() + z.square()

    module = Module().train()
    x = torch.randn(5, requires_grad=True)
    with torch.enable_grad():
        compiled = torch.compile(module, backend="eager", fullgraph=True)
        out = compiled(x)
        assert torch.equal(out, module(x))
        out.sum().backward()
        assert x.grad is not None
    assert msa_inference._mask_counts.get() is None


@pytest.mark.parametrize("kind", ["transition", "pwa", "opm"])
def test_compiled_eval_chunked_layers_keep_existing_path(kind):
    if kind == "transition":
        layer = Transition(8, 17).eval()
        args, kwargs = [torch.randn(1, 5, 8)], {"chunk_size": 4}
    elif kind == "pwa":
        layer = PairWeightedAveraging(8, 8, 4, 2).eval()
        args, kwargs = (
            [torch.randn(1, 3, 5, 8), torch.randn(1, 5, 5, 8), torch.ones(1, 5, 5)],
            {"chunk_heads": True},
        )
    else:
        layer = OuterProductMean(8, 4, 8).eval()
        args, kwargs = [torch.randn(1, 3, 5, 8), torch.ones(1, 3, 5)], {"chunk_size": 2}
    with torch.no_grad():
        reference = layer(*args, **kwargs)
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        assert torch.equal(compiled(*args, **kwargs), reference)
