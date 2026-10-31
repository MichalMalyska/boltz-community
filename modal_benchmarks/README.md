# Boltz torch/CUDA benchmark on Modal

Benchmarks `boltz predict` end-to-end across a matrix of (torch version, CUDA wheel,
GPU type), to see how much a torch/CUDA upgrade actually changes wall-clock time and
peak GPU memory for this repo's inference path.

Everything here is a plain Modal app — it does nothing on its own. You run the
`modal` CLI commands below yourself, with your own Modal account/credentials.

## What it does

- `torch_cuda_benchmark.py` defines one Modal `Image` + `Function` per matrix entry.
  Each image pins a specific torch build (installed with `--index-url
  https://download.pytorch.org/whl/<cuda-tag>`), installs cuEquivariance and
  Anthropic's FlashPairformer kernels (`opt_core` — see below), and bakes in this
  repo's source (`pip install --no-deps -e .`), so every config is a fully
  isolated environment — a broken combo only fails that one entry.
- Each run executes `boltz predict examples/ligand.yaml ...` — the repo's own
  846-token documented benchmark assembly (see docs/denoiser-cache.md) — with
  fixed settings (seed, recycling/sampling steps, `--flash_attn` — see
  `PREDICT_EXTRA_ARGS`) so the **only** things that vary between configs are torch
  version, CUDA build, and GPU type. The input uses a precomputed MSA
  (`examples/msa/seq1.a3m`), so there's no MMseqs2-server network variance either.
  (An earlier, much smaller single-chain input was tried first and came back flat
  across every pin — wall time was all fixed per-call overhead, no compute signal.)
- **FlashPairformer / cuEquivariance kernels are on, not a variable.** This
  benchmark intentionally reflects the repo's recommended/production kernel
  configuration (`BOLTZ_FLASHPAIRFORMER` defaults to `fast`) on every config, not
  a kernel-free baseline — see `build_image()`'s `opt_core` install step. If you
  want kernels-on vs kernels-off as a separate axis, that's a different, narrower
  comparison; `scripts/benchmark_flash_attn.py` already does an isolated version
  of that independent of this tool.
- Each config runs `WARMUP_RUNS` (1) discarded warmup call followed by
  `TIMED_RUNS` (5) timed calls of `boltz predict`, all in the same
  container/GPU — so one-time costs (Triton JIT compilation, opt_core's
  measured-fastest-kernel pick, cuDNN/cuBLAS autotuning) only pay once per
  container rather than once per timed call, and don't bleed into the reported
  numbers. A10 and H100 configs are always separate containers, so their kernels
  are naturally compiled separately per architecture too — nothing extra needed
  for that. The result records every individual run's timing plus
  mean/median/stdev/min/max across the timed ones.
- GPU memory is sampled via `nvidia-smi` every 200ms during each run (peak over
  the successful timed runs is reported); each run's wall time wraps its
  `boltz predict` subprocess only (not image build / weight staging / the
  warmup call).
- A failure inside a run (driver issue, OOM, `boltz predict` crashing, etc.) is
  caught inside the container and returned as a plain dict (`exit_code: -1`,
  `error`, `traceback` strings) rather than raised — a raw exception referencing
  a torch type can't be unpickled back in the local process, since torch isn't
  installed there, so letting one escape uncaught would kill the whole run with
  an opaque "module not available in the local environment" error instead of
  telling you what actually went wrong.
- Every result (full run and `--smoke-test`) carries
  `torch_version_matches_requested`: a direct check that the torch actually
  importable in the container is the one the config asked for, not something
  a later dependency silently swapped it for. When false, `exit_code` is
  forced to -1 and both summary printers print an explicit `torch mismatch`
  line — this is not hypothetical, see `CUEQUIVARIANCE_PIN` below for the bug
  this check exists to catch.

### FlashPairformer kernels (`opt_core`)

`build_image()` fetches the `common/opt_core` subtree of
[`anthropics/uplifting-biomolecular-modeling`](https://github.com/anthropics/uplifting-biomolecular-modeling)
at the commit pinned in `OPT_CORE_COMMIT` (matching README.md's own
"FlashPairformer kernels" setup section) and does an editable install of it,
plus the repo's `[project.optional-dependencies] cuda` extra (cuEquivariance)
beforehand. It's not on PyPI and the fetch is a public, unauthenticated GitHub
archive download — if that repo is ever made private or the pinned commit
goes stale, every config's image build fails identically and clearly, not
subtly per-config.

The fetch (`_fetch_opt_core.py`) downloads the pinned commit as a GitHub
archive tarball via plain `urllib` + `tarfile`, deliberately *not* `git clone`
— some Modal workspaces are still on an older/legacy Image Builder version
pinned to an EOL Debian snapshot whose `bullseye-security` packages (`git`
included) have rotated out of the mirror and 404 on `apt-get install`. If you
hit that same 404 anywhere else, upgrade your workspace's Image Builder
version at <https://modal.com/settings/image-config>. It extracts only the
`common/opt_core` subtree (streamed, member by member, with the repo-root
prefix stripped) rather than the whole repo — the full monorepo is ~1.1GB at
this commit; `common/opt_core` alone, the only part anything here uses, is
~300MB.

Checked directly against the pinned commit's actual source (not just the
summary in README.md's setup section) before relying on any of this:
`common/opt_core/pyproject.toml` declares `build-backend =
"setuptools.build_meta"` with no runtime dependencies and ships its CUDA
kernels as prebuilt package-data binaries, so the editable install is a plain
setuptools operation with nothing to compile — no `git`, `nvcc`, or `gcc`
needed in the image at all, beyond what's already there. Those binaries are
*not* Git LFS (no `.gitattributes` under `common/opt_core`; its `.cubin` files
are genuine multi-hundred-KB binary blobs in the tree, not LFS pointer
stubs), so the archive-tarball fetch gets their real content, not placeholder
text. `opt_core/__init__.py` imports nothing eagerly (PEP 562 lazy
`__getattr__` throughout), so `import opt_core` alone — what `--smoke-test`
checks — is cheap even without a GPU present. And its GPU-capability gate
(`opt_core/arch.py`) labels only A100/H100/H200/B200/B300 by name in its
`SM_CLASSES` table, but that table is explicitly documented as a reporting
label only, not a gate (`"card names are labels for tables and reports
only — no gate here ... derives a ... capability from a name"`); an unlisted
card like A10 (sm_86) that's still above a lever's floor runs "uncertified"
(engaged, just not on Anthropic's own tested-card list) rather than being
skipped.

cuEquivariance's packages are named `*_cu12` upstream, but that's attempted on
every pin including the `cu130` ones — boltz's own fallback chain (`opt_core` →
cuEquivariance → plain PyTorch, logged as a `RuntimeWarning` per stage) means a
`cu130` config that can't actually load cuEquivariance just degrades one step
further, which is itself a valid, informative result rather than something to
special-case around.

## Before running

**Model weights**: expects a Modal Volume named `boltz-weights-cache` to already
have the Boltz checkpoint(s) cached (so timed runs never pay a download). It's
mounted at `/cache` and `BOLTZ_CACHE` is set to that path. If that volume doesn't
exist yet or is empty, the first run per config will download the checkpoint into
it (slow, and that time will be included in `wall_time_s` for that run) — populate
it first with a throwaway run if you want clean numbers from run 1.

**Version matrix**: `MATRIX` in `torch_cuda_benchmark.py` is pinned to
(torch, CUDA-tag) pairs that were confirmed to exist on
`download.pytorch.org/whl/<tag>/torch/` as of 2026-10-31. PyTorch's CUDA-tag support
shifts with every release (old tags stop getting new torch builds; new tags appear).
Before a big run, re-check with:

```bash
./modal_benchmarks/list_torch_cuda_wheels.sh cu130   # or cu121, cu124, cu126, cu128, cu132, ...
```

and adjust `MATRIX` if a pin has gone stale.

**Modal setup**: `modal token new` (or `modal setup`) once, if you haven't already.

**Container Python version**: fixed at 3.12 in the script (`PYTHON_VERSION`),
independent of whatever local Python runs `modal` itself. The per-config
Functions are registered as plain (non-serialized) Functions via a bare
module-level loop, so there's no local/remote Python-version coupling to worry
about — 3.12 just needs to (a) be within your Modal workspace's Image Builder
version's supported Python range and (b) have wheels for every `MATRIX` pin,
both true as of 2026-10-31. If your workspace's Image Builder is unusually old,
bump it in the Modal dashboard rather than lowering `PYTHON_VERSION` below 3.10
(the repo's own minimum).

## Running it

```bash
# list the configured matrix, run nothing
modal run modal_benchmarks/torch_cuda_benchmark.py
```

```bash
# cheap: build one config's image and check torch/cuEquivariance/opt_core/
# flashpairformer all import — seconds, no boltz predict call. Use this first
# when validating an image-build or kernel-install change.
modal run modal_benchmarks/torch_cuda_benchmark.py --config-id torch2.6.0-cu126-A10 --smoke-test
```

```bash
# run a single config's full benchmark (warmup + timed runs)
modal run modal_benchmarks/torch_cuda_benchmark.py --config-id torch2.6.0-cu126-A10
```

```bash
# run the entire matrix (12 configs in the default MATRIX)
modal run modal_benchmarks/torch_cuda_benchmark.py --run-all
```

```bash
# or smoke-test the entire matrix at once (same --run-all GPU-grouped
# parallelism, just the cheap import check instead of the full benchmark)
modal run modal_benchmarks/torch_cuda_benchmark.py --run-all --smoke-test
```

Each run prints its JSON result to stdout as it finishes, and a summary table at the
end if more than one config ran. Results are also written to the
`boltz-benchmark-results` Volume as `<config_id>.json`, so they survive across
separate invocations. Pull them all down with:

```bash
modal volume get boltz-benchmark-results / ./results
```

Then compare them with `analyze_results.py` — prints a table (sorted by GPU then
wall time) and, if matplotlib is available, writes a bar-chart PNG next to the
results directory comparing wall time and peak GPU memory across configs:

```bash
python modal_benchmarks/analyze_results.py ./results
# or, without installing matplotlib into your main environment:
uv run --with matplotlib modal_benchmarks/analyze_results.py ./results
```

## Notes / things you may want to change

- **Cost & time**: `--run-all` as shipped runs 12 configs (6 torch/CUDA pairs × 2
  GPU types: A10 and H100, using `H100!` to pin the exact SKU and avoid Modal's
  silent H100→H200 auto-upgrade) in two sequential batches — all 6 A10 configs in
  parallel, then all 6 H100 configs in parallel — so it never asks for more than 6
  concurrent GPUs. If your account's concurrent-GPU limit is lower than 6, either
  trim `MATRIX` or split a batch further by calling `_run_group()` on a smaller
  list from `main()`. Per config: 1 warmup + 5 timed `boltz predict` calls on the
  846-token assembly, each ~57s on an A10 per the repo's own documented
  measurement (faster on H100) — so each config is roughly 6×that, plus a
  one-time image-build cost per *image* (torch, rdkit, cuEquivariance, cloning
  and building `opt_core`). Budget accordingly; this is a materially more
  expensive sweep than a toy single-chain input would be.
- **GPU string naming**: Modal's current supported GPU strings are things like `T4`,
  `L4`, `A10`, `L40S`, `A100-40GB`, `A100-80GB`, `H100`/`H100!`, `H200`, `B200`. There
  is no `A10G` — that's an AWS EC2 instance-family name, not a Modal GPU spec; `A10`
  is the equivalent here.
- **Editing the workload**: `BENCHMARK_INPUT_REL_PATH`, `PREDICT_EXTRA_ARGS`,
  `WARMUP_RUNS` and `TIMED_RUNS` at the top of `torch_cuda_benchmark.py` control
  what gets benchmarked and how many repeats. Swap in a different complex from
  `examples/` if you want a different workload — just make sure whatever you pick
  doesn't need `--use_msa_server` if you want network-variance-free timing.
- **Dependency pins**: `BOLTZ_DEPS` in the script is a copy of this repo's
  `[project.dependencies]` from `pyproject.toml` with `torch` removed (torch is
  pinned per-config instead). If that list in `pyproject.toml` changes, update
  `BOLTZ_DEPS` to match.
- **Why cuEquivariance is exact-pinned, not floor-matched to `pyproject.toml`'s
  unbounded `cuda` extra**: `cuequivariance_ops_torch_cu12` made `torch` a hard
  runtime dependency (`torch>=2.11`) starting at release 0.11.1 — below that
  (through 0.10.0) it only needed torch under its `test` extra, which this
  install never pulls in. An unbounded floor let pip's resolver silently
  upgrade an already-exactly-pinned older torch build to satisfy that newer
  release's requirement — using the *default* PyPI index, not the CUDA-tag
  index that torch was originally installed from — which is exactly the kind
  of cross-config contamination this benchmark exists to prevent (caught via
  `--smoke-test`: `torch_version_actual` didn't match `torch_version_requested`).
  `CUEQUIVARIANCE_PIN` fixes all four cuequivariance packages to `0.10.0`
  across every config, which also means cuEquivariance's own version — not one
  of the things being compared — stays constant throughout the matrix instead
  of drifting per torch pin.
