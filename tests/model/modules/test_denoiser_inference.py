"""Synthetic denoiser cache checks; no sequences, checkpoints or predictions."""

from collections import Counter
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from boltz.model.layers import denoiser_inference as caching
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys
from boltz.model.modules.transformersv2 import AdaLN

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


def _case(device, batch=1):
    args = dict(
        token_s=8,
        atom_s=8,
        sigma_data=1,
        dim_fourier=8,
        atoms_per_window_queries=4,
        atoms_per_window_keys=8,
        atom_encoder_depth=2,
        atom_encoder_heads=2,
        atom_decoder_depth=2,
        atom_decoder_heads=2,
        token_transformer_depth=2,
        token_transformer_heads=2,
        conditioning_transition_layers=1,
    )
    sampler = (
        AtomDiffusion(
            score_model_args=args,
            num_sampling_steps=3,
            sigma_data=1,
            sigma_min=0.1,
            sigma_max=1,
            gamma_0=0.2,
            gamma_min=0.5,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for p in sampler.parameters():
            p.normal_(std=0.1)
    atom_to_token = torch.nn.functional.one_hot(
        torch.arange(8, device=device) % 3, 3
    ).float()
    mask = torch.ones(batch, 8, device=device)
    mask[:, -1] = 0
    matrix = get_indexing_matrix(2, 4, 8, device)
    kwargs = dict(
        s_inputs=torch.randn(batch, 3, 8, device=device),
        s_trunk=torch.randn(batch, 3, 8, device=device),
        feats={
            "ref_pos": torch.randn(batch, 8, 3, device=device),
            "atom_to_token": atom_to_token.unsqueeze(0).repeat(batch, 1, 1),
            "atom_pad_mask": mask,
            "token_pad_mask": torch.ones(batch, 3, device=device),
        },
        diffusion_conditioning={
            "q": torch.randn(batch, 8, 8, device=device),
            "c": torch.randn(batch, 8, 8, device=device),
            "atom_enc_bias": torch.randn(batch, 2, 4, 8, 4, device=device),
            "atom_dec_bias": torch.randn(batch, 2, 4, 8, 4, device=device),
            "token_trans_bias": torch.randn(batch, 3, 3, 4, device=device),
            "to_keys": partial(single_to_keys, indexing_matrix=matrix, W=4, H=8),
        },
    )
    return sampler, mask, kwargs


def _probe(monkeypatch, device):
    # CPU cases test the real eager modules with an explicit admission mock.
    # CUDA cases keep the production admission predicate intact.
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
    if device == "cpu":
        monkeypatch.setattr(
            caching,
            "_eligible",
            lambda m, x: not m.training and not torch.is_grad_enabled(),
        )
    seen = []
    base = caching.SamplingCache

    class Recorded(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

        def clear(self):
            self.saved_roles = Counter(role[0] for role in self.entries)
            super().clear()

    monkeypatch.setattr(caching, "SamplingCache", Recorded)
    return seen


def _state(device):
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if device == "cuda" else None,
    )


def _restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1])


def _assert_state(left, right):
    assert torch.equal(left[0], right[0])
    if left[1] is not None:
        assert torch.equal(left[1], right[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch,multiplicity,cap", [(1, 1, 1), (2, 11, 5)])
@pytest.mark.parametrize("amp", [False, True])
def test_real_sampler_parity_reuse_and_cleanup(
    monkeypatch, device, batch, multiplicity, cap, amp
):
    sampler, mask, kwargs = _case(device, batch)
    if amp:
        kwargs["s_inputs"] = kwargs["s_inputs"].bfloat16()
        kwargs["diffusion_conditioning"]["c"] = kwargs["diffusion_conditioning"][
            "c"
        ].bfloat16()
    seen = _probe(monkeypatch, device)
    calls = Counter()
    for name in ["norm_single", "single_embed", "fourier_embed"]:
        getattr(sampler.score_model.single_conditioner, name).register_forward_hook(
            lambda module, args, out, key=name: calls.update([key])
        )
    atom_norm = sampler.score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[
        0
    ].adaln.s_norm
    token_norm = sampler.score_model.token_transformer.layers[0].adaln.s_norm
    atom_norm.register_forward_hook(lambda *args: calls.update(["atom_norm"]))
    token_norm.register_forward_hook(lambda *args: calls.update(["token_norm"]))
    original_inputs = {k: kwargs[k].clone() for k in ["s_inputs", "s_trunk"]}
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=amp),
    ):
        start = _state(device)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
        baseline = sampler.sample(
            mask, multiplicity=multiplicity, max_parallel_samples=cap, **kwargs
        )
        end = _state(device)
        baseline_calls = calls.copy()
        calls.clear()
        _restore(start)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
        actual = sampler.sample(
            mask, multiplicity=multiplicity, max_parallel_samples=cap, **kwargs
        )
        _assert_state(end, _state(device))
    assert actual["sample_atom_coords"].dtype == baseline["sample_atom_coords"].dtype
    assert torch.equal(actual["sample_atom_coords"], baseline["sample_atom_coords"])
    shapes = 1 if multiplicity <= cap or multiplicity % cap == 0 else 2
    assert calls["norm_single"] == calls["single_embed"] == calls["atom_norm"] == shapes
    assert calls["fourier_embed"] == baseline_calls["fourier_embed"]
    assert calls["token_norm"] == baseline_calls["token_norm"]
    assert baseline_calls["norm_single"] == 3 * ((multiplicity + cap - 1) // cap)
    assert len(seen) == 1
    cache = seen[0]
    assert cache.hits > 0 and cache.builds > 0 and cache.peak_bytes <= cache.max_bytes
    assert cache.closed and cache.bytes == 0 and not cache.entries and not cache.roots
    assert cache.saved_roles["single"] == shapes
    assert cache.saved_roles["atom"] == shapes
    for k in original_inputs:
        assert torch.equal(original_inputs[k], kwargs[k])


@pytest.mark.parametrize("device", DEVICES)
def test_new_sampler_call_uses_new_conditioning(monkeypatch, device):
    sampler, mask, kwargs = _case(device)
    seen = _probe(monkeypatch, device)
    with torch.inference_mode():
        initial = _state(device)
        first = sampler.sample(mask, multiplicity=2, **kwargs)
        kwargs["s_trunk"].add_(0.2)
        _restore(initial)
        start = _state(device)
        second = sampler.sample(mask, multiplicity=2, **kwargs)
        _restore(start)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
        baseline = sampler.sample(mask, multiplicity=2, **kwargs)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert all(c.closed and c.bytes == 0 for c in seen)
    assert torch.equal(second["sample_atom_coords"], baseline["sample_atom_coords"])
    assert not torch.equal(first["sample_atom_coords"], second["sample_atom_coords"])


def test_cache_invalidates_inputs_weights_configuration_and_values():
    module = AdaLN(8, 8).eval()
    cache = caching.SamplingCache()
    a, s = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    with torch.no_grad():
        for edit in [
            lambda: None,
            lambda: s.add_(0.2),
            lambda: module.s_scale.weight.add_(0.1),
            lambda: setattr(module.s_norm, "eps", 0.1),
            lambda: setattr(
                module.s_scale,
                "weight",
                torch.nn.Parameter(module.s_scale.weight.clone()),
            ),
        ]:
            edit()
            cached = module(a, s, conditioning_cache=cache)
            assert torch.equal(cached, module(a, s))
        scale, bias = cache.modulation(module, s)
        scale.add_(1)
        bias.add_(1)
        assert torch.equal(module(a, s, conditioning_cache=cache), module(a, s))
    assert cache.builds == 6


def test_cache_counts_storage_aliases_and_declines_over_budget():
    x = torch.randn(3, 8)
    cache = caching.SamplingCache(max_bytes=x.numel() * x.element_size())
    with torch.no_grad():
        cache.get("a", (x,), (), lambda: x[:, :4])
        cache.get("b", (x,), (), lambda: x[:, 4:])
    assert cache.bytes == x.numel() * x.element_size()
    cache.clear()
    assert cache.closed and cache.bytes == 0 and not cache.entries
    cache = caching.SamplingCache(max_bytes=1)
    with torch.no_grad():
        result = cache.get("a", (x,), (), lambda: x.square())
    assert torch.equal(result, x.square())
    assert cache.declines == 1 and cache.bytes == 0 and not cache.entries


def test_unversioned_tensors_are_not_cached():
    module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Sigmoid()).eval()
    cache = caching.SamplingCache()
    with torch.inference_mode():
        x = torch.randn(2, 3, 8)
        first = cache.gate(module, x)
        x.add_(1)
        second = cache.gate(module, x)
        assert torch.equal(second, module(x))
        assert not torch.equal(first, second)
    assert not cache.entries


def test_owned_inputs_are_versioned_isolated_and_budgeted():
    sampler, mask, kwargs = _case("cpu")
    del sampler, mask
    cache = caching.SamplingCache()
    with torch.inference_mode():
        kwargs["s_inputs"] = kwargs["s_inputs"].clone()
        owned = cache.own_inputs(kwargs)
        assert not torch.is_inference(owned["s_inputs"])
        expected = owned["s_inputs"].clone()
        kwargs["s_inputs"].add_(1)
        assert torch.equal(owned["s_inputs"], expected)
    assert caching.SamplingCache(max_bytes=1).own_inputs(kwargs) is None


@pytest.mark.parametrize("device", DEVICES)
def test_budget_fallback_and_exception_cleanup(monkeypatch, device):
    sampler, mask, kwargs = _case(device)
    seen = _probe(monkeypatch, device)
    monkeypatch.setattr(caching, "MAX_CACHE_BYTES", 0)
    with torch.no_grad():
        sampler.sample(mask, **kwargs)
    assert len(seen) == 1 and seen[0].closed and seen[0].builds == 0
    monkeypatch.setattr(caching, "MAX_CACHE_BYTES", 128 * 1024**2)

    def failure(**kwargs):
        assert kwargs["_inference_cache"].roots
        raise RuntimeError("injected score failure")

    monkeypatch.setattr(sampler.score_model, "forward", failure)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected"):
        sampler.sample(mask, **kwargs)
    assert seen[-1].closed and seen[-1].bytes == 0 and not seen[-1].entries


def test_cpu_training_grad_and_custom_models_keep_original_path(monkeypatch):
    sampler, mask, kwargs = _case("cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("cache should be bypassed")

    monkeypatch.setattr(caching, "SamplingCache", forbidden)
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
    with torch.no_grad():
        sampler.sample(mask, **kwargs)  # actual CPU admission
    monkeypatch.setattr(
        caching,
        "_eligible",
        lambda m, x: not m.training and not torch.is_grad_enabled(),
    )
    sampler.train()
    with torch.no_grad():
        sampler.sample(mask, **kwargs)
    sampler.eval()
    with torch.enable_grad():
        sampler.sample(mask, **kwargs)
    sampler.score_model = torch.nn.Linear(3, 3)
    decorated = caching.cache_sampling(lambda self, atom_mask: "unchanged")
    with torch.no_grad():
        assert decorated(sampler, mask) == "unchanged"


def test_compiled_score_bypasses_cache(monkeypatch):
    sampler, mask, kwargs = _case("cpu")
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
    monkeypatch.setattr(caching, "_eligible", lambda m, x: True)
    compile_fn = torch.compile
    sampler.score_model = compile_fn(
        sampler.score_model, backend="eager", fullgraph=False
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled score must bypass the inference cache")

    monkeypatch.setattr(caching, "SamplingCache", forbidden)
    with torch.no_grad():
        output = sampler.sample(mask, **kwargs)
    assert output["sample_atom_coords"].shape == (1, 8, 3)


def test_direct_gradient_path_is_not_cached():
    module = AdaLN(8, 8).eval()
    cache = caching.SamplingCache()
    a, s = (
        torch.randn(2, 3, 8, requires_grad=True),
        torch.randn(2, 3, 8, requires_grad=True),
    )
    with torch.enable_grad():
        module(a, s, conditioning_cache=cache).sum().backward()
    assert a.grad is not None and s.grad is not None and not cache.entries


@pytest.mark.parametrize("kind", ["single", "adaln"])
def test_fullgraph_components_keep_original_path(kind):
    from boltz.model.modules.encodersv2 import SingleConditioning

    if kind == "single":
        module = SingleConditioning(1, 8, dim_fourier=8, num_transitions=1).eval()
        args = (torch.rand(2), torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    else:
        module = AdaLN(8, 8).eval()
        args = (torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    with torch.no_grad():
        reference = module(*args)
        actual = torch.compile(module, backend="eager", fullgraph=True)(*args)
    if kind == "single":
        assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    else:
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("device", DEVICES)
def test_particle_resampling_guidance_and_alignment_parity(monkeypatch, device):
    from boltz.model.modules import diffusionv2

    class Potential:
        def compute_parameters(self, t):
            return dict(resampling_weight=1.0, guidance_weight=0.1, guidance_interval=1)

        def compute(self, coords, feats, params):
            return coords.square().sum(dim=(-1, -2))

        def compute_gradient(self, coords, feats, params):
            return coords * 0.01

    sampler, mask, kwargs = _case(device, batch=2)
    sampler.alignment_reverse_diff = True
    sampler.gamma_min = 0  # Exercise particle resampling before the final step.
    seen = _probe(monkeypatch, device)
    monkeypatch.setattr(
        diffusionv2, "get_potentials", lambda *args, **kwargs: [Potential()]
    )
    settings = dict(
        fk_steering=True,
        num_particles=2,
        fk_lambda=0.1,
        physical_guidance_update=True,
        contact_guidance_update=True,
    )
    with torch.inference_mode():
        start = _state(device)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
        baseline = sampler.sample(
            mask,
            multiplicity=2,
            max_parallel_samples=3,
            steering_args=settings,
            **kwargs,
        )
        end = _state(device)
        _restore(start)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
        actual = sampler.sample(
            mask,
            multiplicity=2,
            max_parallel_samples=3,
            steering_args=settings,
            **kwargs,
        )
        _assert_state(end, _state(device))
    assert torch.equal(actual["sample_atom_coords"], baseline["sample_atom_coords"])
    assert actual["sample_atom_coords"].shape == (4, 8, 3)
    assert seen[0].hits > 0 and seen[0].closed


@pytest.mark.parametrize(
    "training,grad,compiled,amp,dtype,expected",
    [
        (False, False, False, False, torch.bfloat16, True),
        (False, False, False, True, torch.bfloat16, True),
        (False, False, False, True, torch.float16, False),
        (True, False, False, False, torch.bfloat16, False),
        (False, True, False, False, torch.bfloat16, False),
        (False, False, True, False, torch.bfloat16, False),
    ],
)
def test_sampler_admission_matrix(
    monkeypatch, training, grad, compiled, amp, dtype, expected
):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiled)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: amp)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: dtype)
    with torch.set_grad_enabled(grad):
        assert (
            caching._eligible(
                SimpleNamespace(training=training), SimpleNamespace(is_cuda=True)
            )
            is expected
        )


def test_precision_changes_do_not_reuse_old_modulation():
    module = AdaLN(8, 8).eval()
    cache = caching.SamplingCache()
    a, s = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    for amp in [False, True, False]:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            assert torch.equal(module(a, s, conditioning_cache=cache), module(a, s))
    assert cache.builds == 3


@pytest.mark.parametrize(
    "value,enabled",
    [
        ("off", False),
        ("0", False),
        ("false", False),
        ("on", True),
        ("1", True),
        ("true", True),
    ],
)
def test_switch_boolean_aliases(monkeypatch, value, enabled):
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", value)
    assert caching.cache_enabled() is enabled


def test_bad_switch_rejected_before_device_admission(monkeypatch):
    monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "typo")
    monkeypatch.setattr(
        caching, "_eligible", lambda *args: pytest.fail("device admission reached")
    )
    call = caching.cache_sampling(lambda *args: pytest.fail("sampler reached"))
    with pytest.raises(ValueError, match="BOLTZ_DENOISER_CACHE"):
        call(None, None)


def test_metadata_plan_reused_without_weakening_invalidation(monkeypatch):
    module = AdaLN(8, 8).eval()
    count = Counter()
    for name in ["parameters", "modules"]:
        original = getattr(module, name)

        def observed(*args, _name=name, _fn=original, **kwargs):
            count[_name] += 1
            return _fn(*args, **kwargs)

        monkeypatch.setattr(module, name, observed)
    cache = caching.SamplingCache()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        for _ in range(5):
            cache.modulation(module, x)
        assert count == {"parameters": 1, "modules": 1}
        module.s_scale.weight.add_(0.1)
        cache.modulation(module, x)
        assert cache.builds == 2 and cache.plan_builds == 1
        module.s_norm.eps = 0.1
        cache.modulation(module, x)
        assert count == {"parameters": 2, "modules": 2}
        module.s_scale.weight = torch.nn.Parameter(module.s_scale.weight.clone())
        cache.modulation(module, x)
        assert cache.plan_builds == 3


def test_realistic_atom_width_saturates_budget_and_remembers_declines():
    from boltz.model.modules.transformersv2 import DiffusionTransformerLayer

    cache = caching.SamplingCache()
    layers = [DiffusionTransformerLayer(4, 128, 128).eval() for _ in range(6)]
    conditioning = torch.randn(1, 2048, 128)
    with torch.inference_mode():
        expanded = cache.repeat_conditioning(conditioning, 5).view(-1, 32, 128)
        for layer in layers:
            cache.modulation(layer.adaln, expanded)
            cache.modulation(layer.transition.adaln, expanded)
            cache.gate(layer.output_projection, expanded)
            cache.gate(layer.transition.output_projection, expanded)
        assert 0 < cache.peak_bytes <= 128 * 1024**2
        assert cache.declines > 0
        counters = cache.builds, cache.plan_builds, cache.declines
        refused = tuple(cache.declined)
        for role in refused:
            if role[0] == "adaln":
                cache.modulation(role[1], expanded)
            else:
                cache.gate(role[1], expanded)
        assert counters == (cache.builds, cache.plan_builds, cache.declines)
        assert cache.declined_calls == len(refused)
    cache.clear()
    assert not cache.declined and cache.bytes == 0


@pytest.mark.parametrize("device", DEVICES)
def test_sampler_partial_budget_parity(monkeypatch, device):
    sampler, mask, kwargs = _case(device)
    seen = _probe(monkeypatch, device)
    monkeypatch.setattr(caching, "MAX_CACHE_BYTES", 2048)
    with torch.inference_mode():
        initial = _state(device)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "off")
        baseline = sampler.sample(mask, **kwargs)
        end = _state(device)
        _restore(initial)
        monkeypatch.setenv("BOLTZ_DENOISER_CACHE", "on")
        actual = sampler.sample(mask, **kwargs)
        _assert_state(end, _state(device))
    assert torch.equal(actual["sample_atom_coords"], baseline["sample_atom_coords"])
    assert seen[0].hits > 0 and seen[0].declines > 0 and seen[0].declined_calls > 0
    assert seen[0].peak_bytes <= 2048
