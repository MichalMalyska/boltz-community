"""Skipping the Boltz-2 template stack when no template slot is live."""

import pytest
import torch

from boltz.data.feature.featurizerv2 import load_dummy_templates_features
from boltz.model.modules.trunkv2 import TemplateV2Module

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
TOKEN_Z = 32


@pytest.fixture
def module():
    torch.manual_seed(0)
    module = TemplateV2Module(TOKEN_Z, template_dim=16, template_blocks=2)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.2)
    return module.eval()


def _inputs(device, batch=2, tokens=12, slots=2):
    feats = [load_dummy_templates_features(slots, tokens) for _ in range(batch)]
    feats = {k: torch.stack([f[k] for f in feats]).to(device) for k in feats[0]}
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(batch, tokens, tokens, TOKEN_Z, generator=generator).to(device)
    z[:, 0, 0] = -0.0
    pair_mask = torch.ones(batch, tokens, tokens, device=device)
    return z, feats, pair_mask


def _make_live(feats, record=0):
    feats["template_mask"][record, 0, :5] = 1
    feats["template_mask_cb"][record, 0, :5] = 1
    feats["template_mask_frame"][record, 0, :5] = 1
    feats["template_cb"][record, 0, :5] = torch.randn(5, 3, device=feats["template_cb"].device)
    feats["template_frame_rot"][record, 0, :5] = torch.eye(3, device=feats["template_cb"].device)


def _seeded_update(module, z, feats, pair_mask, grad):
    torch.manual_seed(123)
    with torch.set_grad_enabled(grad):
        update = module(z, feats, pair_mask)
    state = torch.cuda.get_rng_state() if z.is_cuda else torch.get_rng_state()
    return (z + update).detach(), update, state


class TestDummyTemplateSkip:
    """Inference with only dummy template slots skips the stack, bit for bit."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_dummy_slots_match_full_pass_and_rng(self, module, device):
        """The pair update and the random state after the call match the full pass."""
        module = module.to(device)
        z, feats, pair_mask = _inputs(device)
        full, full_update, full_rng = _seeded_update(module, z, feats, pair_mask, grad=True)
        skipped, skipped_update, skipped_rng = _seeded_update(
            module, z, feats, pair_mask, grad=False
        )
        assert torch.equal(full_update, torch.zeros_like(full_update))
        assert torch.equal(skipped.view(torch.int32), full.view(torch.int32))
        assert torch.equal(skipped_rng, full_rng)
        assert skipped_update.dtype == full_update.dtype

    @pytest.mark.parametrize("device", DEVICES)
    def test_autocast_dtype_matches_full_pass(self, module, device):
        """Under autocast the skipped update has the projection's autocast dtype."""
        module = module.to(device)
        z, feats, pair_mask = _inputs(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            _, full_update, _ = _seeded_update(module, z, feats, pair_mask, grad=True)
            _, skipped_update, _ = _seeded_update(module, z, feats, pair_mask, grad=False)
        assert skipped_update.dtype == full_update.dtype == torch.bfloat16

    @pytest.mark.parametrize("device", DEVICES)
    def test_any_live_slot_runs_the_stack(self, module, device):
        """One live template in the batch runs the stack for every record."""
        module = module.to(device)
        z, feats, pair_mask = _inputs(device)
        _make_live(feats, record=1)
        full, _, full_rng = _seeded_update(module, z, feats, pair_mask, grad=True)
        inference, update, inference_rng = _seeded_update(
            module, z, feats, pair_mask, grad=False
        )
        assert update[1].abs().sum() > 0
        assert torch.equal(inference, full)
        assert torch.equal(inference_rng, full_rng)
