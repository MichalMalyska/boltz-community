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
Boltz-1 callers, training, gradient-enabled calls, CPU, compiled forwards and
non-bf16 contexts use the existing arithmetic. The optimization needs no `opt_core` installation.

These changes adapt the work-elimination ideas in Anthropic's
[Boltz-2 optimization bundle](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/opt/forward/waste/boltz_waste_levers.py).
They are intended to preserve outputs; CUDA equality and measured performance
must be checked on the supported runtime before rollout. CPU checks alone do
not establish CUDA equality or a prediction speed-up.

Run the focused checks with:

```bash
python -m pytest tests/model/layers/test_msa_inference.py -v
```

The CUDA cases compare the scoped path with the original unscoped chunked path
on the same nonzero weights, inputs and masked columns, and check output dtype
and RNG state. The compiled-training check ensures the wrapper does not insert
scope bookkeeping into training graphs.

Validation on 26 October 2026 used an A10G with PyTorch 2.9.1: 16 synthetic
CUDA comparisons against the original classes passed bitwise output, dtype and
RNG checks (N=17/385, batch=2, fp32/bf16 inputs), along with mask mutation and
scope cleanup checks. The minimum supported PyTorch 2.4.1 CPU suite passed
422 tests, with 40 skipped, 11 deselected and 9 subtests passed.

Three paired synthetic timing blocks at N=1344, S=64 measured approximately
31% less MSA-transition wall time, 50% less pair-weighted-averaging time and
13% less outer-product-mean time. At S=1, OPM showed no gain. These are short
layer microbenchmarks; no whole-prediction speed-up has been established for
this patch. The separate ten-case review-branch prediction benchmark excluded
these MSA changes.
