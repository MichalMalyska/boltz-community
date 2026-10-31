"""Benchmark boltz-community `predict` across torch/CUDA/GPU combinations on Modal.

Each matrix entry gets its own Modal Image (pinned torch wheel from the matching
download.pytorch.org CUDA index, plus Anthropic's FlashPairformer kernels —
`opt_core` — and cuEquivariance, both on by default per the repo's own docs) and
its own Modal Function (fixed GPU type), so every run is isolated: a broken
combination only fails that one entry, and GPU type is never conflated with
torch/CUDA version in a single result.

Each config runs one discarded warmup `boltz predict` call followed by
TIMED_RUNS timed calls, all in the same container/GPU — so Triton JIT
compilation, opt_core's measured-fastest-kernel pick, and cuDNN/cuBLAS
autotuning only pay their one-time cost once per container (and, since A10 and
H100 configs are always separate containers, those are naturally compiled
separately per GPU architecture too), not once per timed call.

Model weights are expected to already exist in the `boltz-weights-cache` Modal Volume
(so no checkpoint download happens during timed runs) — see README.md if that volume
needs populating first.

Usage (run locally; Modal CLI must be set up with `modal token new` / `modal setup`):

    modal run modal_benchmarks/torch_cuda_benchmark.py                   # list configs
    modal run modal_benchmarks/torch_cuda_benchmark.py --config-id <id>  # run one
    modal run modal_benchmarks/torch_cuda_benchmark.py --run-all         # run matrix
    modal run modal_benchmarks/torch_cuda_benchmark.py --config-id <id> --smoke-test
        # ^ cheap image/kernel-import check only, no boltz predict call

Results are printed to stdout and persisted as JSON in the `boltz-benchmark-results`
Volume (one file per config id). Pull them down afterwards with:

    modal volume get boltz-benchmark-results / ./results
"""

import json
import pathlib
import re

import modal

APP_NAME = "boltz-torch-cuda-benchmark"

WEIGHTS_VOLUME_NAME = "boltz-weights-cache"
RESULTS_VOLUME_NAME = "boltz-benchmark-results"
WEIGHTS_MOUNT = "/cache"
RESULTS_MOUNT = "/results"

# Container Python version. Kept independent of whatever Python runs `modal`
# locally (the benchmark Functions are plain, non-serialized Functions — see the
# MATRIX registration loop below — so there's no local/remote Python-version
# coupling to worry about). 3.12 is within every Modal Image Builder version's
# supported range as of 2026-10-31 and has wheels for every pin in MATRIX.
PYTHON_VERSION = "3.12"

# Fixed input + predict settings, identical across every config, so the only things
# that vary run-to-run are torch version / CUDA build / GPU type.
#
# examples/ligand.yaml is the repo's own documented 846-token benchmark assembly
# (see docs/denoiser-cache.md's "27 October 2026" real-prediction comparison,
# ~57s/run on an A10G with this exact methodology) — heavy enough that wall time
# is dominated by actual GPU compute rather than fixed per-call overhead (a tiny
# single-chain input was tried first and came back flat across every torch/CUDA
# pin — all overhead, no signal). It ships a precomputed MSA
# (examples/msa/seq1.a3m), so there's no MMseqs2-server network dependency.
BENCHMARK_INPUT_REL_PATH = "examples/ligand.yaml"
PREDICT_EXTRA_ARGS = [
    "--model", "boltz2",
    "--accelerator", "gpu",
    "--recycling_steps", "3",
    "--sampling_steps", "200",
    "--diffusion_samples", "1",
    "--seed", "0",
    "--override",
    "--flash_attn",
    # FlashPairformer (opt_core) and cuEquivariance are installed in every image
    # below and used by default (BOLTZ_FLASHPAIRFORMER defaults to "fast") — this
    # benchmark intentionally reflects the repo's recommended/production kernel
    # configuration, not a kernel-free baseline. See build_image().
]

# A single discarded warmup call, then this many timed calls, all in the same
# container — see the module docstring for why.
WARMUP_RUNS = 1
TIMED_RUNS = 5

def _cfg(torch_version: str, cuda_tag: str, gpu: str) -> dict:
    return {
        "id": f"torch{torch_version}-{cuda_tag}-{gpu.rstrip('!')}",
        "torch": torch_version,
        "cuda": cuda_tag,
        "gpu": gpu,
    }


# torch -> CUDA-wheel-index pairings, each confirmed to exist by actually listing
# https://download.pytorch.org/whl/<tag>/torch/ (see list_torch_cuda_wheels.sh) on
# 2026-10-31. PyTorch's CUDA-tag support shifts with every release, so if a build
# below 404s by the time you run this, re-check with that script and fix the pin.
#
# The spread: oldest-still-available CUDA 12.1 build, through the two CUDA-12
# transition points (12.4/12.6 — same torch 2.6.0 on both, to isolate the CUDA-only
# effect — then 12.8), to CUDA 13's first wheel and the latest torch available today.
MATRIX = [
    cfg
    for torch_version, cuda_tag in [
        ("2.5.1", "cu121"),
        ("2.6.0", "cu124"),
        ("2.6.0", "cu126"),
        ("2.7.1", "cu128"),
        ("2.9.0", "cu130"),
        ("2.14.1", "cu130"),
    ]
    for gpu in ("A10", "H100!")
    for cfg in [_cfg(torch_version, cuda_tag, gpu)]
]

# boltz-community's own dependencies, minus torch (torch is pinned separately per
# config below). Keep this in sync with [project.dependencies] in pyproject.toml.
BOLTZ_DEPS = [
    "numpy>=1.26",
    "hydra-core>=1.3.2",
    "pytorch-lightning>=2.5.0",
    "rdkit>=2024.3.2",
    "dm-tree>=0.1.8",
    "requests>=2.32.3",
    "pandas>=2.2.2",
    "types-requests",
    "einops>=0.8.0",
    "einx>=0.3.0",
    "mashumaro>=3.14",
    "modelcif>=1.2",
    "wandb>=0.18.7",
    "click>=8.1.7",
    "pyyaml>=6.0.2",
    "biopython>=1.84",
    "scipy>=1.13.1",
    "numba>=0.61.0",
    "gemmi>=0.6.5",
    "scikit-learn>=1.6.1",
    "chembl_structure_pipeline>=1.2.2",
]

# The repo's own [project.optional-dependencies] "cuda" extra — cuEquivariance
# kernels for triangle attention/multiplication. Named *_cu12 upstream; boltz's
# own fallback chain (opt_core -> cuEquivariance -> plain PyTorch, logged as a
# RuntimeWarning when a stage is unavailable) means this is safe to attempt on
# every pin, cu130 included — a cu130 config just degrades one step further if
# cuEquivariance itself turns out not to load there, which is itself a valid,
# informative benchmark result rather than something to special-case around.
CUDA_EXTRA_DEPS = [
    "cuequivariance_ops_cu12>=0.5.0",
    "cuequivariance_ops_torch_cu12>=0.5.0",
    "cuequivariance_torch>=0.5.0",
]

# Anthropic's FlashPairformer kernels (opt_core), per README.md's "FlashPairformer
# kernels" setup section. Not on PyPI; installed editable from a pinned commit so
# its prebuilt CUDA kernels are found (a plain `pip install` of the subdirectory
# leaves some out, falling back to slower Triton kernels). Needs compute
# capability 8.0+ (verified against opt_core's own arch.py gating logic: a card
# not in its SM_CLASSES label table, like A10, is not excluded by that absence —
# SM_CLASSES is a reporting label, not a gate; a lever runs "uncertified" on an
# unlisted-but-above-floor card rather than being skipped). Both A10 (sm_86) and
# H100 (sm_90) qualify.
#
# Fetched via _fetch_opt_core.py (plain urllib+tarfile, a GitHub archive
# tarball, extracting only the common/opt_core subtree) rather than `git
# clone`, deliberately avoiding any dependency on a `git` binary — some Modal
# workspaces are still on an older/legacy Image Builder version pinned to an
# EOL Debian snapshot whose bullseye-security packages (git included) have
# rotated out of the mirror and 404 on `apt-get install`. If that bites you
# elsewhere too, upgrade your workspace's Image Builder version at
# https://modal.com/settings/image-config. (The full
# uplifting-biomolecular-modeling repo is ~1.1GB at this commit; common/opt_core
# alone, everything actually needed, is ~300MB — extracting just that subtree
# keeps every built image smaller.) opt_core's own pyproject.toml declares
# build-backend = "setuptools.build_meta" with no runtime dependencies and ships
# its CUDA kernels as prebuilt package-data binaries (not git-lfs — verified:
# no .gitattributes under common/opt_core, and its "cubin" files are
# multi-hundred-KB binary blobs, not LFS pointer stubs), so this is a plain
# editable setuptools install with no compiler needed at install time.
OPT_CORE_COMMIT = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
OPT_CORE_SUBDIR = "common/opt_core"
OPT_CORE_INSTALL_DIR = "/root/opt_core"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_MOUNT = "/root/boltz-community"
FETCH_OPT_CORE_SCRIPT = pathlib.Path(__file__).resolve().parent / "_fetch_opt_core.py"
FETCH_OPT_CORE_MOUNT = "/root/_fetch_opt_core.py"


def _ignore_from_repo(path: pathlib.Path) -> bool:
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "modal_benchmarks"}
    return any(part in skip_dirs for part in path.parts) or path.suffix == ".pyc"


def build_image(torch_version: str, cuda_tag: str) -> modal.Image:
    torch_index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(f"torch=={torch_version}", index_url=torch_index_url)
        .pip_install(*BOLTZ_DEPS)
        .pip_install(*CUDA_EXTRA_DEPS)
        .add_local_dir(
            REPO_ROOT, remote_path=REPO_MOUNT, copy=True, ignore=_ignore_from_repo
        )
        .run_commands(f"pip install --no-deps -e {REPO_MOUNT}")
        .add_local_file(FETCH_OPT_CORE_SCRIPT, FETCH_OPT_CORE_MOUNT, copy=True)
        .run_commands(
            f"python3 {FETCH_OPT_CORE_MOUNT} {OPT_CORE_COMMIT} "
            f"{OPT_CORE_SUBDIR} {OPT_CORE_INSTALL_DIR}",
            f"pip install -e {OPT_CORE_INSTALL_DIR}",
        )
        .env({"BOLTZ_CACHE": WEIGHTS_MOUNT, "TOKENIZERS_PARALLELISM": "false"})
    )


app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _run_predict_once(cmd: list[str]) -> dict:
    """Run one `boltz predict` subprocess, timed, with peak-GPU-memory sampling."""
    import subprocess
    import threading
    import time

    mem_samples_mb = []
    stop = threading.Event()

    def poll_gpu_mem() -> None:
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                mem_samples_mb.append(max(int(v) for v in out.strip().splitlines()))
            except Exception:
                pass
            stop.wait(0.2)

    poller = threading.Thread(target=poll_gpu_mem, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_MOUNT, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0
    stop.set()
    poller.join(timeout=2)

    return {
        "wall_time_s": round(elapsed_s, 2),
        "peak_gpu_mem_mb": max(mem_samples_mb) if mem_samples_mb else None,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_config(cfg: dict) -> dict:
    """Run inside the Modal container: warmup + TIMED_RUNS timed `boltz predict`s.

    Everything is wrapped in a broad try/except that only ever stores plain
    str/int/float/None values into `result`. A raw exception propagating out of a
    Modal Function gets cloudpickled back to the *local* process to be re-raised
    there; if that exception (or its traceback frames) holds a reference to a
    torch type — e.g. a CUDA driver/OOM error — unpickling it locally fails with
    "the 'torch' module is not available in the local environment", since torch
    was never installed locally. Reporting failures as plain data instead avoids
    that regardless of what actually went wrong inside the container.
    """
    import statistics
    import traceback

    result: dict = {
        "config_id": cfg["id"],
        "torch_version_requested": cfg["torch"],
        "cuda_tag_requested": cfg["cuda"],
        "gpu_requested": cfg["gpu"],
    }

    try:
        import torch

        result["torch_version_actual"] = torch.__version__
        result["cuda_version_actual"] = torch.version.cuda
        try:
            result["gpu_name_actual"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
        except Exception as e:
            result["gpu_name_actual"] = f"<error: {e}>"

        cmd = [
            "boltz", "predict", f"{REPO_MOUNT}/{BENCHMARK_INPUT_REL_PATH}",
            "--out_dir", "/tmp/boltz_out",
            "--cache", WEIGHTS_MOUNT,
            *PREDICT_EXTRA_ARGS,
        ]

        warmup_runs = [_run_predict_once(cmd) for _ in range(WARMUP_RUNS)]
        result["warmup_runs"] = warmup_runs

        timed_runs = [_run_predict_once(cmd) for _ in range(TIMED_RUNS)]
        result["timed_runs"] = timed_runs

        ok_times = [r["wall_time_s"] for r in timed_runs if r["exit_code"] == 0]
        ok_mem = [
            r["peak_gpu_mem_mb"]
            for r in timed_runs
            if r["exit_code"] == 0 and r["peak_gpu_mem_mb"] is not None
        ]
        result.update(
            {
                "timed_runs_ok": len(ok_times),
                "timed_runs_total": len(timed_runs),
                "wall_time_s_mean": round(statistics.mean(ok_times), 2)
                if ok_times
                else None,
                "wall_time_s_median": round(statistics.median(ok_times), 2)
                if ok_times
                else None,
                "wall_time_s_stdev": round(statistics.stdev(ok_times), 2)
                if len(ok_times) > 1
                else None,
                "wall_time_s_min": min(ok_times) if ok_times else None,
                "wall_time_s_max": max(ok_times) if ok_times else None,
                "peak_gpu_mem_mb": max(ok_mem) if ok_mem else None,
                # exit_code reflects the run as a whole: 0 only if every timed
                # run succeeded, so a sweep summary can tell "fully clean" apart
                # from "some repeats failed" at a glance.
                "exit_code": 0 if len(ok_times) == len(timed_runs) else -1,
            }
        )
    except Exception as e:
        result["exit_code"] = -1
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    with open(f"{RESULTS_MOUNT}/{cfg['id']}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


def run_smoke_test(cfg: dict) -> dict:
    """Cheap image/kernel check: imports only, no `boltz predict` call.

    Seconds of image-build-plus-import-check instead of minutes of warmup+timed
    predict runs — use this to validate that a torch/CUDA/kernel combination
    actually builds and imports before spending a full run_config() cycle on it.
    """
    import traceback

    result: dict = {
        "config_id": cfg["id"],
        "torch_version_requested": cfg["torch"],
        "cuda_tag_requested": cfg["cuda"],
        "gpu_requested": cfg["gpu"],
    }

    try:
        import torch

        result["torch_version_actual"] = torch.__version__
        result["cuda_version_actual"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        try:
            result["gpu_name_actual"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
        except Exception as e:
            result["gpu_name_actual"] = f"<error: {e}>"

        for module_name, result_key in (
            ("cuequivariance_torch", "cuequivariance_torch"),
            ("opt_core", "opt_core"),
            ("boltz.model.layers.flashpairformer", "boltz_flashpairformer"),
        ):
            try:
                mod = __import__(module_name, fromlist=["_"])
                result[f"{result_key}_import_ok"] = True
                result[f"{result_key}_version"] = getattr(mod, "__version__", "unknown")
            except Exception as e:
                result[f"{result_key}_import_ok"] = False
                result[f"{result_key}_error"] = f"{type(e).__name__}: {e}"

        result["exit_code"] = 0
    except Exception as e:
        result["exit_code"] = -1
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# Registered as a bare module-level loop (not inside a helper function) so each
# generated function def is textually at module scope — a `for` loop doesn't
# introduce a new scope in Python, so this satisfies Modal's "@app.function must
# apply to functions in global scope" rule without needing serialized=True. That
# in turn means these Functions are looked up by import path inside their
# container, not pickled by value, so there's no local/remote Python-version
# coupling.
#
# Each config MUST get a genuinely distinct top-level function name (not just a
# distinct Python object bound to a reused name like `_run`). A Modal container
# re-imports this whole module to resolve which function to actually execute,
# and it does that by name — so reusing one `def _run(...)` across every loop
# iteration left only the *last* iteration's function reachable under that name
# once the module finished importing. Every container ended up running the last
# config in MATRIX regardless of which image/GPU it was actually provisioned
# with (confirmed empirically: torch/GPU were correct — the container-level
# config — but cfg itself, and therefore the result's reported config_id, was
# always the last MATRIX entry). exec() with a per-config name builds a real,
# distinctly-named top-level `def`, exactly as if each were hand-written.
BENCHMARK_FNS: dict[str, modal.Function] = {}
SMOKE_FNS: dict[str, modal.Function] = {}
for _cfg_entry in MATRIX:
    _image = build_image(_cfg_entry["torch"], _cfg_entry["cuda"])
    _id_safe = re.sub(r"[^0-9a-zA-Z_]", "_", _cfg_entry["id"])
    _fn_name = "_bench_" + _id_safe
    _smoke_fn_name = "_smoke_" + _id_safe

    exec(
        f"""
@app.function(
    image=_image,
    gpu=_cfg_entry["gpu"],
    volumes={{WEIGHTS_MOUNT: weights_volume, RESULTS_MOUNT: results_volume}},
    # 1 warmup + {TIMED_RUNS} timed runs on a ~57s/run (A10, per the repo's own
    # docs/denoiser-cache.md measurement) heavier input, plus opt_core/
    # cuEquivariance build time on first use — generous headroom for slower
    # torch/CUDA pins.
    timeout=60 * 60,
    name="bench_{_cfg_entry['id']}",
)
def {_fn_name}(cfg: dict = _cfg_entry) -> dict:
    return run_config(cfg)


@app.function(
    image=_image,
    gpu=_cfg_entry["gpu"],
    timeout=15 * 60,
    name="smoke_{_cfg_entry['id']}",
)
def {_smoke_fn_name}(cfg: dict = _cfg_entry) -> dict:
    return run_smoke_test(cfg)
""",
        globals(),
    )

    BENCHMARK_FNS[_cfg_entry["id"]] = globals()[_fn_name]
    SMOKE_FNS[_cfg_entry["id"]] = globals()[_smoke_fn_name]


def _print_matrix() -> None:
    print("Available configs:")
    for cfg in MATRIX:
        row = (
            f"  {cfg['id']:28s} torch={cfg['torch']:<8} "
            f"cuda={cfg['cuda']:<7} gpu={cfg['gpu']}"
        )
        print(row)
    print(
        "\nRun one:    modal run modal_benchmarks/torch_cuda_benchmark.py"
        " --config-id <id>"
        "\nRun all:    modal run modal_benchmarks/torch_cuda_benchmark.py --run-all"
        "\nSmoke test: modal run modal_benchmarks/torch_cuda_benchmark.py"
        " --config-id <id> --smoke-test"
    )


def _run_group(
    cids: list[str], fns: dict[str, modal.Function] = BENCHMARK_FNS
) -> list[dict]:
    """Launch every id in `cids` concurrently and collect results as they finish.

    One Modal container/GPU per id. Concurrency here is bounded by len(cids), so
    callers that need to stay under an account-wide concurrent-GPU cap should
    pass groups sized accordingly (MATRIX is split one group per GPU type below,
    6 configs each, so --run-all never asks for more than 6 GPUs at once).
    """
    handles = {}
    for cid in cids:
        fn = fns.get(cid)
        if fn is None:
            known = ", ".join(fns)
            print(f"Unknown config id: {cid!r}. Known ids: {known}")
            continue
        handles[cid] = fn.spawn()

    results = []
    for cid, handle in handles.items():
        try:
            result = handle.get()
        except Exception as e:
            print(f"FAILED to run {cid}: {e}")
            continue
        print(f"\n=== {cid} ===")
        print(json.dumps(result, indent=2))
        results.append(result)
    return results


def _print_summary(results: list[dict]) -> None:
    if len(results) <= 1:
        return
    print("\n=== Summary ===")
    header = (
        f"{'config':28s} {'median_s':>9s} {'mean_s':>9s} {'stdev_s':>8s} "
        f"{'peak_mem_mb':>12s} {'ok/total':>9s}"
    )
    print(header)
    for r in results:
        median = r.get("wall_time_s_median")
        mean = r.get("wall_time_s_mean")
        stdev = r.get("wall_time_s_stdev")
        median_s = f"{median:.1f}" if median is not None else "n/a"
        mean_s = f"{mean:.1f}" if mean is not None else "n/a"
        stdev_s = f"{stdev:.1f}" if stdev is not None else "n/a"
        ok_total = f"{r.get('timed_runs_ok', 0)}/{r.get('timed_runs_total', 0)}"
        print(
            f"{r.get('config_id', '?'):28s} {median_s:>9s} {mean_s:>9s} "
            f"{stdev_s:>8s} {r.get('peak_gpu_mem_mb')!s:>12s} {ok_total:>9s}"
        )


def _print_smoke_summary(results: list[dict]) -> None:
    if not results:
        return
    print("\n=== Smoke test summary ===")
    checks = ("cuequivariance_torch", "opt_core", "boltz_flashpairformer")
    header = f"{'config':28s} {'torch':12s}" + "".join(f" {c:>14s}" for c in checks)
    print(header)
    for r in results:
        row = f"{r.get('config_id', '?'):28s} {r.get('torch_version_actual', '?'):12s}"
        for c in checks:
            ok = r.get(f"{c}_import_ok")
            row += f" {'ok' if ok else 'FAILED':>14s}"
        print(row)
        for c in checks:
            if r.get(f"{c}_import_ok") is False:
                print(f"    {c}: {r.get(f'{c}_error')}")
        if r.get("exit_code") != 0 and "error" in r:
            print(f"    top-level error: {r['error']}")


@app.local_entrypoint()
def main(
    config_id: str = "", run_all: bool = False, smoke_test: bool = False
) -> None:
    """Entrypoint for `modal run modal_benchmarks/torch_cuda_benchmark.py`.

    --smoke-test runs run_smoke_test() instead of the full benchmark: builds
    the same per-config image, then just checks that torch/cuEquivariance/
    opt_core/flashpairformer all import — seconds, not minutes, so it's a cheap
    way to validate an image-build or kernel-install change before spending a
    full warmup+timed-runs cycle on it.
    """
    fns = SMOKE_FNS if smoke_test else BENCHMARK_FNS
    print_results = _print_smoke_summary if smoke_test else _print_summary

    if not config_id and not run_all:
        _print_matrix()
        return

    if config_id and not run_all:
        results = _run_group([config_id], fns)
        print_results(results)
        return

    # --run-all: one group per GPU type, run concurrently within each group, so
    # this never requests more than max(len(group)) GPUs at once (6, for the
    # default MATRIX — A10 batch first, then H100).
    results = []
    present_gpus = {cfg["gpu"] for cfg in MATRIX}
    gpu_order = [g for g in ("A10", "H100!") if g in present_gpus]
    gpu_order += sorted(present_gpus - set(gpu_order))  # any other GPU types, if added
    for gpu_label in gpu_order:
        group_ids = [cfg["id"] for cfg in MATRIX if cfg["gpu"] == gpu_label]
        print(f"\n### {gpu_label} group: {len(group_ids)} configs in parallel ###")
        results.extend(_run_group(group_ids, fns))

    print_results(results)
