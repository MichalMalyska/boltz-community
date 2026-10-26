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


@functools.cache
def tier() -> str:
    """Return the configured tier (``fast``, ``exact`` or ``off``)."""
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


def _kernels(x: Tensor, name: str) -> Optional[ModuleType]:
    """Return the ``opt_core.kernels`` module to try for a call on ``x``, if any."""
    if _kernel_failure["reason"] is not None or tier() == "off":
        return None
    if not x.is_cuda or torch.cuda.get_device_capability(x.device) < (8, 0):
        return None
    # opt_core prints a per-process kernel census at exit unless asked not to.
    os.environ.setdefault("OPT_CORE_CELL_CENSUS_PRINT", "0")
    try:
        return importlib.import_module(f"opt_core.kernels.{name}")
    except ImportError:
        return None


def _on_kernel_error(error: Exception) -> None:
    """Disable opt_core after a kernel error; out-of-memory errors propagate."""
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error):
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


def _trimul_cache(weight: Tensor) -> dict:
    """Return opt_core's packed-weight cache for the layer that owns ``weight``."""
    key = id(weight)
    cache = _trimul_caches.get(key)
    if cache is None:
        cache = _trimul_caches[key] = {}
        weakref.finalize(weight, _trimul_caches.pop, key, None)
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
    weights = {
        "ln_in_w": norm_in_weight,
        "ln_in_b": norm_in_bias,
        "w_ap": p_in_weight[:d],
        "w_ag": g_in_weight[:d],
        "w_bp": p_in_weight[d:],
        "w_bg": g_in_weight[d:],
        "ln_out_w": norm_out_weight,
        "ln_out_b": norm_out_bias,
        "w_o": p_out_weight,
        "w_og": g_out_weight,
    }
    try:
        z = x if x.dim() == 4 else x[None]
        backward = torch.is_grad_enabled() and (
            x.requires_grad or any(w.requires_grad for w in weights.values())
        )
        selection = _trimul_selection(
            torch.cuda.get_device_capability(z.device),
            trimul.call_precision(z)[0],
            _trimul_stack(z.device),
            tuple(z.shape),
            d,
            direction,
            backward,
            tier(),
        )
        if selection.row in trimul.STOCK_ROWS:
            return None
        return trimul.triangle_multiplication(
            x,
            mask,
            direction=direction,
            weights=weights,
            selection=selection,
            cache=_trimul_cache(p_in_weight),
            eps=eps,
        )
    except trimul.Refusal:
        return None
    except Exception as error:  # noqa: BLE001
        _on_kernel_error(error)
        return None
