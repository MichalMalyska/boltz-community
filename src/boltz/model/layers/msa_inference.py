"""Reuse invariant work inside one Boltz-2 MSA inference call.

The mask counts are forward-local intermediates, never a process-wide tensor
cache. Features stay unchanged inside MSAModule.forward; a later call (including
one using mutated inference-mode tensors) always starts with empty state.
"""

import functools
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor

_mask_counts: ContextVar[dict | None] = ContextVar("msa_mask_counts", default=None)


def enabled(x: Tensor) -> bool:
    """Whether this call is scoped, gradient-free CUDA bf16 inference."""
    if torch.compiler.is_compiling():
        return False
    return (
        _mask_counts.get() is not None
        and not torch.is_grad_enabled()
        and x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


@contextmanager
def inference_scope(x: Tensor, *, training: bool):
    """Bound reuse to one MSA forward and restore nested scopes on all exits."""
    admitted = (
        not training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    token = _mask_counts.set({} if admitted else None)
    try:
        yield
    finally:
        _mask_counts.reset(token)


def msa_forward(function):
    """Enable reuse only in the Boltz-2 MSA module's inference forward."""

    @functools.wraps(function)
    def scoped(module, z, *args, **kwargs):
        # Keep ContextVar operations out of compiled training/CPU forwards.
        if (
            torch.compiler.is_compiling()
            or module.training
            or torch.is_grad_enabled()
            or not z.is_cuda
        ):
            return function(module, z, *args, **kwargs)
        with inference_scope(z, training=module.training):
            return function(module, z, *args, **kwargs)

    return scoped


def cast_activation(x: Tensor) -> Tensor:
    """Perform autocast's fp32-to-bf16 conversion once before a chunk loop."""
    if enabled(x) and x.dtype == torch.float32:
        return x.to(torch.bfloat16)
    return x


def mask_counts(raw_mask: Tensor, mask: Tensor) -> Tensor:
    """Compute the original chunked denominator, reusing it within this forward."""
    cache = _mask_counts.get() if enabled(mask) else None
    key = None
    if cache is not None:
        try:
            version = raw_mask._version
        except RuntimeError:
            version = None  # Inference tensors are fixed inputs within this scope.
        key = (
            id(raw_mask),
            version,
            raw_mask.data_ptr(),
            tuple(raw_mask.shape),
            raw_mask.stride(),
            mask.dtype,
            mask.device,
        )
        if key in cache:
            return cache[key][1]
    for i in range(0, mask.shape[1], 64):
        part = (mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]).sum(1)
        if i == 0:
            count = part
        else:
            count += part
    count = count.clamp(min=1)
    if cache is not None:
        cache[key] = (raw_mask, count)  # Keep the input alive until scope cleanup.
    return count
