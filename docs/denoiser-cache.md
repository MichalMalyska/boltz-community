# Denoiser invariant caching

Eager Boltz-2 sampling reuses timestep-independent conditioning work within one
`AtomDiffusion.sample` invocation. Automatic caching requires CUDA, eval mode,
gradients disabled, and either fp32 execution or CUDA bf16 autocast. Compiled or
custom score models retain their original call path. CPU, training, Boltz-1 and
direct denoiser calls do not automatically create a cache.

Set `BOLTZ_DENOISER_CACHE=off` to use the existing computation. The default is
`on`; `0` and `1` are also accepted. The setting is read once per eligible
sampling invocation. A direct `sample` caller should use `eval()` and
`torch.no_grad()` or `torch.inference_mode()`, as the regular prediction path does.

The cache reuses:

- The SingleConditioning prefix: concatenation of the expanded trunk and input
  embeddings, LayerNorm and the single embedding projection.
- The fp32 atom-conditioning tensor and its sample-multiplicity expansion.
- Atom-layer AdaLN conditioning normalization, sigmoid scale and bias, and the
  attention/transition output gates derived from that conditioning.

The original sample expansion and matrix shapes are preserved. Uneven sample
chunks have separate entries. The Fourier/time contribution, subsequent single
conditioning transitions, token-transformer modulation, coordinate-dependent
atom activations, attention and output projections still run each denoiser call.
Main diffusion conditioning was already computed once and is unchanged.

Each invocation owns versioned snapshots of its trunk/input embeddings and atom
conditioning. This isolates them from caller edits and lets internal in-place
edits invalidate entries even under Lightning's inference mode. Entry signatures
include tensor geometry, version counters, weight identity/version, normalization
settings and execution precision. Unversioned inputs or parameters are not used
for reusable entries. Changes to cached result tensors also invalidate entries.
The cache is passed explicitly through atom layers and is not stored on a model
or in global tensor state. Checkpoint keys are unchanged.

Retained tensor storage is bounded by the smaller of 128 MiB and one eighth of
GPU free memory at admission. Accounting counts shared storage once, including
retained source/weight references. A value that does not fit is computed normally
without being retained; if input snapshots do not fit, the invocation uses the
original path. Large quadratic pair-bias expansions are not cached. All entries
are cleared on return or exception. Exceptions, including OOM, propagate normally.
Debug logging for `boltz.model.layers.denoiser_inference` reports hits, builds,
budget declines and peak retained storage once per sampling invocation.

This ports the invariant-work approach from Anthropic's pinned
[`dit_hoist` level 1](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/opt/forward/dit_hoist/src/boltz_dit_hoist.py)
into the existing sampler. It does not replace sampling, guidance or indexing
with the bundle's stock-version-specific implementation and requires no
`opt_core` installation.

Run the synthetic checks with:

```bash
python -m pytest tests/model/modules/test_denoiser_inference.py -v
python -m pytest tests/model/modules/test_denoiser_inference.py -m cuda -v -rs
```

The tests use artificial tensors and randomized nonzero model parameters, with
no sequence input or pretrained checkpoint. They compare complete sampler
outputs and final RNG states, exercise batched records and uneven chunks, and
cover resampling, guidance, alignment, precision changes, mutation invalidation,
memory-budget fallback and exception cleanup. Compiled-score tests use Dynamo's
`eager` backend and the existing `fullgraph=False` configuration; component
full-graph tests cover SingleConditioning and AdaLN. CPU admission probes are
explicitly mocked; CUDA cases exercise the production admission predicate.

On a three-step, two-record synthetic case with 11 samples and a cap of five,
the fixed prefix and each observed atom normalization run twice, for the two
chunk shapes, instead of nine times. Fourier and token normalization still run
nine times. This establishes eliminated work, not an end-to-end speed-up.
Full pretrained-prediction timing and accuracy validation remain outstanding.

Validation on 27 October 2026: 458 tests passed on the minimum PyTorch 2.4.1
CPU runtime (56 skipped, 11 deselected). Seven new denoiser CUDA cases and all
15 existing MSA CUDA cases passed on RTX PRO 6000 Blackwell with PyTorch
2.14.0+cu130. Sampling tests include fp32 and bf16 autocast with mixed-precision
conditioning inputs. The GPU process shared the device with an existing run,
so these results establish tested correctness rather than performance.
