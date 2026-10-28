"""Forward-local MSA work reuse, with exact CUDA comparisons to the original path."""

from types import SimpleNamespace
import weakref

import pytest
import torch

from boltz.model.layers import msa_inference
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ],
    ),
]


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


def test_unversioned_masks_never_reuse_stale_counts(scoped_counts):
    with torch.inference_mode():
        mask = torch.ones(1, 65, 9)
        first = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
        mask[:, :5] = 0
        updated = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
        assert torch.equal(updated, torch.full_like(first, 60.0))
        assert (first == 65).all()
        assert msa_inference._mask_counts.get() == {}


def test_owned_inference_mask_tracks_edits_within_forward(scoped_counts):
    with torch.inference_mode():
        original = torch.ones(1, 65, 9)
        mask = msa_inference.versioned_mask(original)
        assert not torch.is_inference(mask)
        first = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
        assert msa_inference.mask_counts(mask, mask.unsqueeze(-1)) is first
        mask[:, :5] = 0
        updated = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
        assert updated is not first
        assert (updated == 60).all()
        assert msa_inference.mask_counts(mask, mask.unsqueeze(-1)) is updated
        assert (original == 1).all()


def test_versioned_mask_keeps_existing_inputs_and_unscoped_inference_tensors():
    tracked = torch.ones(1, 3, 5)
    assert msa_inference.versioned_mask(tracked) is tracked
    with torch.inference_mode():
        mask = torch.ones(1, 3, 5)
        assert msa_inference.versioned_mask(mask) is mask


def test_empty_msa_denominator_and_chunked_opm():
    mask = torch.empty(2, 0, 5)
    count = msa_inference.mask_counts(mask, mask.unsqueeze(-1))
    assert torch.equal(count, torch.ones(2, 5, 5, 1))
    layer = OuterProductMean(8, 4, 8).eval()
    with torch.no_grad():
        layer.proj_o.bias.fill_(0.25)
        result = layer(torch.empty(2, 0, 5, 8), mask, chunk_size=2)
    assert torch.equal(result, torch.full((2, 5, 5, 8), 0.25))


@pytest.mark.parametrize("device", DEVICES)
def test_opm_releases_fp32_projections_before_chunk_loop(
    scoped_counts, monkeypatch, device
):
    refs = []
    chunk_refs = []
    original_cast = msa_inference.cast_activation
    original_einsum = torch.einsum
    original_div = torch.div

    def cast(x):
        result = original_cast(x)
        if x.shape[-1] == 4 and result is not x:
            refs.append(weakref.ref(x))
        return result

    def einsum(*args, **kwargs):
        assert len(refs) == 2
        assert all(ref() is None for ref in refs)
        assert all(ref() is None for ref in chunk_refs)
        return original_einsum(*args, **kwargs)

    def divide(*args, **kwargs):
        chunk_refs.append(weakref.ref(kwargs["out"]))
        return original_div(*args, **kwargs)

    monkeypatch.setattr(msa_inference, "cast_activation", cast)
    monkeypatch.setattr(torch, "einsum", einsum)
    monkeypatch.setattr(torch, "div", divide)
    layer = OuterProductMean(8, 4, 8).to(device).eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        layer(
            torch.randn(1, 3, 5, 8, device=device),
            torch.ones(1, 3, 5, device=device),
            chunk_size=2,
        )
    assert all(ref() is None for ref in chunk_refs)


@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
def test_opm_division_matches_standard_promotion_under_autocast(mask_dtype, device):
    z = torch.randn(2, 4, 3, 3, 5, device=device).bfloat16().permute(0, 2, 3, 1, 4)
    count = torch.rand(2, 3, 3, 1, device=device).to(mask_dtype).clamp(min=0.1)
    with torch.autocast(device, dtype=torch.bfloat16):
        reference = z.reshape(2, 3, 3, -1) / count
        divided = torch.empty(z.shape, dtype=torch.result_type(z, count), device=device)
        torch.div(z, count.unsqueeze(-1), out=divided)
    actual = divided.reshape(2, 3, 3, -1)
    assert actual.dtype == reference.dtype
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reuse", [False, True])
def test_opm_releases_normalized_input_before_chunk_loop(
    scoped_counts, monkeypatch, device, reuse
):
    if not reuse:
        monkeypatch.setattr(msa_inference, "enabled", lambda x: False)
    refs = []
    layer = OuterProductMean(8, 4, 8).to(device).eval()
    layer.norm.register_forward_hook(
        lambda module, args, output: refs.append(weakref.ref(output))
    )
    original_einsum = torch.einsum

    def einsum(*args, **kwargs):
        assert len(refs) == 1
        assert refs[0]() is None
        return original_einsum(*args, **kwargs)

    monkeypatch.setattr(torch, "einsum", einsum)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        layer(
            torch.randn(1, 3, 5, 8, device=device),
            torch.ones(1, 3, 5, device=device),
            chunk_size=2,
        )


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


def _small_msa_inputs(device):
    n = 385 if device == "cuda" else 5
    return {
        "msa": torch.zeros(1, 3, n, dtype=torch.long, device=device),
        "has_deletion": torch.zeros(1, 3, n, device=device),
        "deletion_value": torch.zeros(1, 3, n, device=device),
        "msa_paired": torch.ones(1, 3, n, device=device),
        "msa_mask": torch.ones(1, 3, n, device=device),
        "token_pad_mask": torch.ones(1, n, device=device),
        "embedding": torch.randn(1, n, 8, device=device),
        "pair_features": torch.zeros(1, n, n, 8, device=device),
        "token_bonds": torch.zeros(1, n, n, 1, device=device),
    }


def _msa_args():
    return dict(
        msa_s=8,
        msa_blocks=3,
        msa_dropout=0.15,
        z_dropout=0.25,
        pairwise_head_width=4,
        pairwise_num_heads=2,
    )


def _probe_real_scope(monkeypatch, device):
    from boltz.data import const

    # Small geometry exercises real chunked modules. CPU probes mock admission
    # only; the CUDA variants exercise the production admission criteria.
    if device == "cpu":
        monkeypatch.setattr(const, "chunk_size_threshold", 4)

        def admitted(x, *, training=False):
            return (
                not torch.compiler.is_compiling()
                and not training
                and not torch.is_grad_enabled()
            )

        monkeypatch.setattr(msa_inference, "_admitted", admitted)
    observed = []
    original = msa_inference.mask_counts

    def counts(raw, expanded):
        value = original(raw, expanded)
        observed.append((raw, value, msa_inference._mask_counts.get()))
        return value

    monkeypatch.setattr(msa_inference, "mask_counts", counts)
    return observed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mutate_mask", [False, True])
def test_real_msa_forward_reuses_and_invalidates_counts(
    monkeypatch, device, mutate_mask
):
    from boltz.model.modules.trunkv2 import MSAModule

    observed = _probe_real_scope(monkeypatch, device)
    module = MSAModule(token_z=8, token_s=8, **_msa_args()).to(device).eval()
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.1)
    if mutate_mask:

        def mutate(layer, args):
            args[1][:, :1] = 0

        module.layers[1].outer_product_mean.register_forward_pre_hook(mutate)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        feats = _small_msa_inputs(device)
        z = torch.randn_like(feats["pair_features"])
        if not mutate_mask:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
            with monkeypatch.context() as unscoped:
                unscoped.setattr(msa_inference, "enabled", lambda x: False)
                reference = MSAModule.forward.__wrapped__(
                    module, z, feats["embedding"], feats
                )
            reference_cpu_rng = torch.get_rng_state()
            reference_cuda_rng = (
                torch.cuda.get_rng_state() if device == "cuda" else None
            )
            observed.clear()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        actual = module(z, feats["embedding"], feats)
        assert msa_inference._mask_counts.get() is None
        assert len(observed) == 3
        assert all(
            raw is observed[0][0] and cache is observed[0][2]
            for raw, _, cache in observed
        )
        assert observed[0][2] is not None
        assert not torch.is_inference(observed[0][0])
        assert (feats["msa_mask"] == 1).all()
        if mutate_mask:
            assert (observed[0][1] == 3).all()
            assert (observed[1][1] == 2).all()
            assert observed[1][1] is observed[2][1]
        else:
            assert all(value is observed[0][1] for _, value, _ in observed)
            assert actual.dtype == reference.dtype
            assert torch.equal(actual, reference)
            assert torch.equal(reference_cpu_rng, torch.get_rng_state())
            if cuda_rng is not None:
                assert torch.equal(reference_cuda_rng, torch.cuda.get_rng_state())
        first_cache = observed[0][2]
        module(z, feats["embedding"], feats)
        assert observed[3][2] is not first_cache
        assert msa_inference._mask_counts.get() is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compile_msa", [False, True])
def test_boltz2_real_msa_compile_original_path(monkeypatch, device, compile_msa):
    from torch import nn
    from boltz.model.models import boltz2

    class Feature(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, feats):
            return feats["pair_features"]

    class Embed(Feature):
        def forward(self, feats):
            return feats["embedding"]

    class Pair(Feature):
        def forward(self, s, z, **kwargs):
            return s, z

    # Keep the real Boltz2 constructor/forward and three real MSA layers; bypass
    # atom embedding, diffusion and the surrounding pairformer for this test.
    for name, cls in [
        ("InputEmbedder", Embed),
        ("RelativePositionEncoder", Feature),
        ("ContactConditioning", Feature),
        ("PairformerModule", Pair),
        ("DiffusionConditioning", Feature),
        ("AtomDiffusion", Feature),
    ]:
        monkeypatch.setattr(boltz2, name, cls)
    compile_fn = torch.compile
    monkeypatch.setattr(
        torch, "compile", lambda module, **kw: compile_fn(module, backend="eager", **kw)
    )
    observed = _probe_real_scope(monkeypatch, device)
    model = (
        boltz2.Boltz2(
            atom_s=8,
            atom_z=8,
            token_s=8,
            token_z=8,
            num_bins=4,
            training_args={},
            validation_args={},
            embedder_args={},
            msa_args=_msa_args(),
            pairformer_args={},
            diffusion_process_args={},
            diffusion_loss_args={},
            score_model_args=dict.fromkeys(
                [
                    "atom_encoder_depth",
                    "atom_encoder_heads",
                    "token_transformer_depth",
                    "token_transformer_heads",
                    "atom_decoder_depth",
                    "atom_decoder_heads",
                    "conditioning_transition_layers",
                ],
                1,
            ),
            compile_msa=compile_msa,
            confidence_prediction=False,
            skip_run_structure=True,
            validate_structure=False,
        )
        .to(device)
        .eval()
    )
    assert model.is_msa_compiled is compile_msa
    if compile_msa:
        from boltz.model.modules.trunkv2 import MSAModule

        assert isinstance(model.msa_module._orig_mod, MSAModule)

        def forbidden(*args, **kwargs):
            raise AssertionError("Boltz2 eval must call _orig_mod")

        monkeypatch.setattr(model.msa_module, "forward", forbidden)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        feats = _small_msa_inputs(device)
        output = model(feats, recycling_steps=1)
    assert output["z"].shape == feats["pair_features"].shape
    assert len(observed) == 6
    for start in [0, 3]:
        assert observed[start][2] is not None
        assert all(
            observed[i][1] is observed[start][1] for i in range(start, start + 3)
        )
    assert observed[0][2] is not observed[3][2]
    assert msa_inference._mask_counts.get() is None


def test_real_msa_fullgraph_keeps_existing_compiled_path():
    from boltz.model.modules.trunkv2 import MSAModule

    module = MSAModule(token_z=8, token_s=8, **_msa_args()).eval()
    with torch.no_grad():
        feats = _small_msa_inputs("cpu")
        z = torch.randn(1, 5, 5, 8)
        reference = module(z, feats["embedding"], feats)
        compiled = torch.compile(module, backend="eager", fullgraph=True)
        assert torch.equal(compiled(z, feats["embedding"], feats), reference)
    assert msa_inference._mask_counts.get() is None
