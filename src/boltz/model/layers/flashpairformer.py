"""FlashPairformer kernels for triangle attention and triangle multiplication.

Anthropic's ``opt_core`` package (Apache-2.0, from
https://github.com/anthropics/uplifting-biomolecular-modeling, ``common/opt_core``)
carries the FlashPairformer kernels: Triton and prebuilt CUDA implementations of
the Pairformer's triangle attention and triangle multiplication, chosen per GPU
and input size from a measured table.

When ``opt_core`` is installed and kernels are enabled (the default, i.e. no
``--no_kernels``), the layers' ``_kernel_or_none`` helpers try these functions
before the cuequivariance op.  Each returns ``None`` when it does not serve the
call, and the helper then runs the cuequivariance op (and, failing that, the
PyTorch path) exactly as before.

``BOLTZ_FLASHPAIRFORMER`` selects the tier:

- ``fast`` (default): the fastest measured kernel for the call; outputs differ
  from the cuequivariance op by bf16 rounding.
- ``exact``: only kernels that are bit-identical to the cuequivariance op on a
  measured stack; everything else runs the cuequivariance op.
- ``off``: never use ``opt_core``.
"""

from __future__ import annotations

import functools
import importlib
import os
import warnings
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Callable, Optional

import torch
from torch import Tensor

ENV_VAR = "BOLTZ_FLASHPAIRFORMER"
TIERS = ("fast", "exact", "off")

# Once a FlashPairformer kernel fails, skip opt_core for the rest of the process,
# like the cuequivariance helpers' _kernel_failure.
_kernel_failure = {"reason": None}

# opt_core's per-layer packed-weight caches, keyed by id() because tensors compare
# elementwise, and dropped with the layer's weights.
_trimul_caches: dict[int, dict] = {}

# The triangle layers are shared with Boltz-1. Opt in only while evaluating a
# Boltz-2 forward, rather than changing every caller that enables CUDA kernels.
_inference_scope: ContextVar[bool] = ContextVar("boltz2_flashpairformer", default=False)


@contextmanager
def inference_scope(enabled: bool = True) -> Iterator[None]:
    """Scope the route to a Boltz-2 evaluation (also used by layer-only tests)."""
    token = _inference_scope.set(enabled)
    try:
        yield
    finally:
        _inference_scope.reset(token)


def boltz2_inference(function: Callable) -> Callable:
    """Enable the route for evaluation, including correct nesting and cleanup.

    Checking the model's mode excludes training's no-grad recycling and
    checkpoint passes too; checking grad mode alone would not exclude them.
    """

    @functools.wraps(function)
    def scoped_forward(
        model: torch.nn.Module, *args: object, **kwargs: object
    ) -> object:
        with inference_scope(enabled=not model.training):
            return function(model, *args, **kwargs)

    return scoped_forward


def tier() -> str:
    """Read the current tier, including environment changes between calls."""
    value = os.environ.get(ENV_VAR, "fast").strip().lower() or "fast"
    if value in ("0", "false", "no", "none"):
        return "off"
    if value not in TIERS:
        warnings.warn(
            f"{ENV_VAR}={value!r} is not one of {TIERS}; using 'fast'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "fast"
    return value


@functools.cache
def _prepare_kernel_environment() -> None:
    """Quiet opt_core's census once, only when the optional route is admitted."""
    os.environ.setdefault("OPT_CORE_CELL_CENSUS_PRINT", "0")


def _kernels(x: Tensor, name: str) -> Optional[ModuleType]:
    """Return the ``opt_core.kernels`` module to try for a call on ``x``, if any."""
    if _kernel_failure["reason"] is not None or tier() == "off":
        return None
    if not _inference_scope.get() or torch.is_grad_enabled():
        return None
    # The measured route is CUDA bf16 autocast only. In particular, affinity's
    # explicitly disabled autocast must keep the existing fp32 implementation.
    if not x.is_cuda or not torch.is_autocast_enabled("cuda"):
        return None
    if torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return None
    if torch.cuda.get_device_capability(x.device) < (8, 0):
        return None
    _prepare_kernel_environment()
    try:
        return importlib.import_module(f"opt_core.kernels.{name}")
    except ImportError:
        return None


def _on_kernel_error(error: Exception) -> None:
    """Disable opt_core after a kernel error; out-of-memory errors propagate."""
    if (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(error).casefold()
    ):
        raise error
    _kernel_failure["reason"] = f"{type(error).__name__}: {error}"
    warnings.warn(
        "FlashPairformer kernels are unavailable; falling back to the "
        f"cuequivariance / PyTorch path. Kernel error: {_kernel_failure['reason']}",
        RuntimeWarning,
        stacklevel=4,
    )


@torch.compiler.disable
def triangle_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tri_bias: Tensor,
    mask: Tensor,
    scale: float,
    stock: Callable[..., Tensor],
) -> Optional[Tensor]:
    """Triangle attention in the cuequivariance calling convention.

    ``stock`` is the raw cuequivariance op, which opt_core runs itself for
    calls where it is the measured winner and builds the ``exact`` tier's
    per-head rows on.
    """
    triattn = _kernels(q, "triattn")
    if triattn is None:
        return None
    try:
        return triattn.triangle_attention(
            q, k, v, tri_bias, mask, scale, word=tier(), stock=stock
        )
    except triattn.Refusal:
        # A named refusal for this call's shape or GPU, not an error.
        return None
    except Exception as error:  # noqa: BLE001
        _on_kernel_error(error)
        return None


def clear_weight_caches() -> None:
    """Drop packed weights after writes that bypass PyTorch version tracking."""
    # Keep each live owner's empty slot and its existing finalizer.
    for cache in tuple(_trimul_caches.values()):
        cache.clear()


def _trimul_cache(weight: Tensor, *other_weights: Tensor, eps: float = 1e-5) -> dict:
    """Reuse packs only while all source parameters and normalization agree."""
    key = id(weight)
    try:
        signature = (
            eps,
            tuple(
                (
                    id(tensor),
                    tensor._version,  # noqa: SLF001
                    tensor.data_ptr(),
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.dtype,
                    tensor.device,
                )
                for tensor in (weight, *other_weights)
            ),
        )
    except RuntimeError:
        # Parameters created in inference_mode have no version counter. Their
        # writes cannot be detected, so never reuse packed weights for them.
        cache = _trimul_caches.get(key)
        if cache is not None:
            cache.clear()
        return {}
    cache = _trimul_caches.get(key)
    if cache is None:
        cache = _trimul_caches[key] = {}
        weakref.finalize(weight, _trimul_caches.pop, key, None)
    if cache.get("_boltz_weight_signature") != signature:
        cache.clear()
        cache["_boltz_weight_signature"] = signature
    return cache


@functools.cache
def _trimul_stack(device: torch.device) -> str:
    from opt_core.kernels import trimul

    return trimul.stack_word(torch.empty(0, device=device))


@functools.lru_cache(maxsize=256)
def _trimul_selection(
    cc: tuple[int, int],
    prec: str,
    stack: str,
    shape: tuple[int, ...],
    c_hidden: int,
    direction: str,
    backward: bool,
    word: str,
) -> object:
    from opt_core.kernels import trimul

    dtype = prec.rsplit("_", maxsplit=1)[-1] if prec.startswith("f32z_") else prec
    return trimul.select(
        cc,
        "fp32" if dtype == "tf32" else dtype,
        shape[-1],
        c_hidden,
        shape[-2],
        direction,
        word=word,
        residency="fp32" if prec.startswith("f32z_") else None,
        backward=backward,
        stack=stack,
        tf32=prec == "tf32",
        has_cueq=True,
        batch=shape[0],
    )


@torch.compiler.disable
def triangle_multiplication(
    x: Tensor,
    direction: str,
    mask: Tensor,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    eps: float,
) -> Optional[Tensor]:
    """Triangle multiplication in the cuequivariance calling convention.

    Returns ``None`` when opt_core's measured winner for the call is the
    library op itself, which the caller then runs.
    """
    trimul = _kernels(x, "trimul")
    if trimul is None:
        return None
    # Boltz packs the a and b projections as the two halves of p_in / g_in.
    d = p_in_weight.shape[0] // 2
    # opt_core retains source tensors in its packs. Detach before slicing so
    # retained views cannot keep the owning Parameter alive and defeat cleanup.
    p_in = p_in_weight.detach()
    g_in = g_in_weight.detach()
    weights = {
        "ln_in_w": norm_in_weight.detach(),
        "ln_in_b": norm_in_bias.detach(),
        "w_ap": p_in[:d],
        "w_ag": g_in[:d],
        "w_bp": p_in[d:],
        "w_bg": g_in[d:],
        "ln_out_w": norm_out_weight.detach(),
        "ln_out_b": norm_out_bias.detach(),
        "w_o": p_out_weight.detach(),
        "w_og": g_out_weight.detach(),
    }
    try:
        z = x if x.dim() == 4 else x[None]
        selection = _trimul_selection(
            torch.cuda.get_device_capability(z.device),
            trimul.call_precision(z)[0],
            _trimul_stack(z.device),
            tuple(z.shape),
            d,
            direction,
            backward=False,  # _kernels admits only gradient-disabled inference.
            word=tier(),
        )
        if selection.row in trimul.STOCK_ROWS:
            return None
        return trimul.triangle_multiplication(
            x,
            mask,
            direction=direction,
            weights=weights,
            selection=selection,
            cache=_trimul_cache(
                p_in_weight,
                norm_in_weight,
                norm_in_bias,
                g_in_weight,
                norm_out_weight,
                norm_out_bias,
                p_out_weight,
                g_out_weight,
                eps=eps,
            ),
            eps=eps,
        )
    except trimul.Refusal:
        return None
    except Exception as error:  # noqa: BLE001
        _on_kernel_error(error)
        return None
