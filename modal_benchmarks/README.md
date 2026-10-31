# Boltz torch/CUDA benchmark on Modal

Benchmarks `boltz predict` end-to-end across a matrix of (torch version, CUDA wheel,
GPU type), to see how much a torch/CUDA upgrade actually changes wall-clock time and
peak GPU memory for this repo's inference path.

Everything here is a plain Modal app — it does nothing on its own. You run the
`modal` CLI commands below yourself, with your own Modal account/credentials.

## What it does

- `torch_cuda_benchmark.py` defines one Modal `Image` + `Function` per matrix entry.
  Each image pins a specific torch build (installed with `--index-url
  https://download.pytorch.org/whl/<cuda-tag>`) and bakes in this repo's source
  (`pip install --no-deps -e .`), so every config is a fully isolated environment —
  a broken combo only fails that one entry.
- Each run executes `boltz predict examples/prot_custom_msa.yaml ...` with fixed
  settings (seed, recycling/sampling steps, diffusion samples — see
  `PREDICT_EXTRA_ARGS`) so the **only** things that vary between runs are torch
  version, CUDA build, and GPU type. The input uses a precomputed MSA
  (`examples/msa/seq2.a3m`), so there's no MMseqs2-server network variance either.
- Deliberately *not* varying `--flash_attn` in this matrix — that's a separate
  attention-kernel toggle, orthogonal to the torch/CUDA comparison. Add a parallel
  matrix for that if you want it compared too (`scripts/benchmark_flash_attn.py`
  already does a narrower version of that, independent of this tool).
- GPU memory is sampled via `nvidia-smi` every 200ms during the run (peak reported);
  wall time wraps the `boltz predict` subprocess only (not image build / weight
  staging).

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

**Local Python version**: each image is registered with `serialized=True` (needed
because the per-config Functions are created dynamically in a loop, not as
module-level definitions), which requires the image's Python to exactly match
whatever *local* interpreter is running `modal run` itself. `PYTHON_VERSION` in
the script is therefore derived from `sys.version_info`, not hardcoded. All pins
in the default `MATRIX` currently publish wheels for cp310–cp313+, so this should
just work — but if your local Python is unusually new/old and a pin 404s at
build time, that's the first thing to check.

## Running it

```bash
# list the configured matrix, run nothing
modal run modal_benchmarks/torch_cuda_benchmark.py
```

```bash
# run a single config (fast way to sanity-check one combo / cost)
modal run modal_benchmarks/torch_cuda_benchmark.py --config-id torch2.6.0-cu126-A10
```

```bash
# run the entire matrix (12 configs in the default MATRIX)
modal run modal_benchmarks/torch_cuda_benchmark.py --run-all
```

Each run prints its JSON result to stdout as it finishes, and a summary table at the
end if more than one config ran. Results are also written to the
`boltz-benchmark-results` Volume as `<config_id>.json`, so they survive across
separate invocations. Pull them all down with:

```bash
modal volume get boltz-benchmark-results / ./results
```

## Notes / things you may want to change

- **Cost & time**: `--run-all` as shipped launches 12 runs sequentially (6 torch/CUDA
  pairs × 2 GPU types: A10 and H100, using `H100!` to pin the exact SKU and avoid
  Modal's silent H100→H200 auto-upgrade). First run per *image* also pays a one-time
  build cost (installing torch + rdkit + friends). Runs are sequential by design
  (predictable cost/ordering); if you want them in parallel, change the loop in
  `main()` to `.spawn()` all of them first, then `.get()` each.
- **GPU string naming**: Modal's current supported GPU strings are things like `T4`,
  `L4`, `A10`, `L40S`, `A100-40GB`, `A100-80GB`, `H100`/`H100!`, `H200`, `B200`. There
  is no `A10G` — that's an AWS EC2 instance-family name, not a Modal GPU spec; `A10`
  is the equivalent here.
- **Editing the workload**: `BENCHMARK_INPUT_REL_PATH` and `PREDICT_EXTRA_ARGS` at the
  top of `torch_cuda_benchmark.py` control what gets benchmarked (currently a single
  118-residue chain, boltz2, 200 sampling steps, 1 diffusion sample). Swap in a bigger
  complex from `examples/` for a heavier/more realistic workload — just make sure
  whatever you pick doesn't need `--use_msa_server` if you want network-variance-free
  timing.
- **Dependency pins**: `BOLTZ_DEPS` in the script is a copy of this repo's
  `[project.dependencies]` from `pyproject.toml` with `torch` removed (torch is
  pinned per-config instead). If `pyproject.toml`'s dependency list changes, update
  `BOLTZ_DEPS` to match.
