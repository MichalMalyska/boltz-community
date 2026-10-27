# MSA inference work reuse

The Boltz-2 MSA module reuses repeated work during gradient-free CUDA inference
with bf16 autocast. Chunked transitions and pair-weighted averaging cast their
normalized fp32 activations once before the chunk/head loop. Chunked
outer-product mean also reuses its pair-mask denominator across MSA layers and
combines division with writing a contiguous output, avoiding a separate reshape
copy. The original chunk sizes, accumulation order and output dtypes are kept.

The scope belongs to one `MSAModule.forward` call. It is discarded on return or
exception; later recycling passes and predictions compute fresh intermediates.
This includes masks created in inference mode, which have no version counter.
For chunked inference, the MSA module takes one versioned copy of such a mask.
An in-place edit between layers then invalidates its cached denominator. Direct
calls to `mask_counts` with unversioned masks recompute rather than cache.
Empty MSAs produce a denominator of one, matching zero-count clamping.
Boltz-1 callers, training, gradient-enabled calls, CPU, compiled forwards and
non-bf16 contexts use the existing arithmetic. The optimization needs no `opt_core` installation.

The chunked OPM path releases its fp32 projections after casting and releases
each divided chunk after its output projection. It does not retain fp32 and
bf16 projection copies throughout the loop or retain a divided chunk into the
next iteration.

These changes adapt the work-elimination ideas in Anthropic's
[Boltz-2 optimization bundle](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/opt/forward/waste/boltz_waste_levers.py).
They are intended to preserve outputs; CUDA equality and measured performance
must be checked on the supported runtime before rollout. CPU checks alone do
not establish CUDA equality or a prediction speed-up.

Run the focused checks with:

```bash
python -m pytest tests/model/layers/test_msa_inference.py -v
```

The CUDA cases compare the scoped path with the unscoped chunked path on the
same nonzero weights, inputs and masked columns, and check output dtype and RNG
state. Real `MSAModule` forwards at N=385 test automatic scope admission, reuse
across layers, cache reset across forwards, and in-place mask edits. Tests through
`Boltz2.forward` exercise both `compile_msa` settings and require the eval path to
call `_orig_mod` when compiled. Other components in this wiring test are replaced
by small stand-ins; it is not an end-to-end scientific prediction.

The compiled-module checks use Dynamo's `eager` backend, including a full-graph
real MSA forward on CPU. They test graph capture and dispatch, not Inductor
performance. The CPU admission probes mock CUDA eligibility explicitly; actual
CUDA variants exercise the production admission checks. Real MSA RNG comparisons
check the post-forward state against the baseline, because eval dropout still
draws random masks even when its effective dropout rate is zero.

Validation on 26 October 2026 used an A10G with PyTorch 2.9.1: 16 synthetic
CUDA comparisons against the original classes passed bitwise output, dtype and
RNG checks (N=17/385, batch=2, fp32/bf16 inputs), along with mask mutation and
scope cleanup checks. The minimum supported PyTorch 2.4.1 CPU suite passed
422 tests, with 40 skipped, 11 deselected and 9 subtests passed.

For the initial implementation (`3d5b2b7`), three paired synthetic timing blocks
at N=1344, S=64 measured approximately
31% less MSA-transition wall time, 50% less pair-weighted-averaging time and
13% less outer-product-mean time. At S=1, OPM showed no gain. These are short
layer microbenchmarks; no whole-prediction speed-up has been established for
this patch. The separate ten-case review-branch prediction benchmark excluded
these MSA changes.

Review follow-ups on 27 October 2026 passed 433 tests on the minimum PyTorch
2.4.1 CPU runtime (47 skipped, 11 deselected) and 13 CUDA cases on RTX PRO 6000
Blackwell with PyTorch 2.14.0+cu130. Weak-reference regressions verify that fp32
projections and previous divided chunks have been released before the next
einsum. The retained-projection, unversioned-mask mutation and empty-MSA tests
fail on `3d5b2b7`. Division-promotion comparisons pass against both versions on
CPU and also pass with CUDA autocast. The follow-up CUDA process was capped at 4% of GPU memory and
used about 222 MiB peak allocated memory. It shared the device with a running
scientific process, so these runs establish correctness rather than timing.
The review fixes' end-to-end performance has not been measured.
