"""Benchmark boltz-community `predict` across torch/CUDA/GPU combinations on Modal.

Each matrix entry gets its own Modal Image (pinned torch wheel from the matching
download.pytorch.org CUDA index) and its own Modal Function (fixed GPU type), so every
run is isolated: a broken combination only fails that one entry, and GPU type is never
conflated with torch/CUDA version in a single result.

Model weights are expected to already exist in the `boltz-weights-cache` Modal Volume
(so no checkpoint download happens during timed runs) — see README.md if that volume
needs populating first.

Usage (run locally; Modal CLI must be set up with `modal token new` / `modal setup`):

    modal run modal_benchmarks/torch_cuda_benchmark.py                   # list configs
    modal run modal_benchmarks/torch_cuda_benchmark.py --config-id <id>  # run one
    modal run modal_benchmarks/torch_cuda_benchmark.py --run-all         # run matrix

Results are printed to stdout and persisted as JSON in the `boltz-benchmark-results`
Volume (one file per config id). Pull them down afterwards with:

    modal volume get boltz-benchmark-results / ./results
"""

import json
import pathlib
import sys

import modal

APP_NAME = "boltz-torch-cuda-benchmark"

WEIGHTS_VOLUME_NAME = "boltz-weights-cache"
RESULTS_VOLUME_NAME = "boltz-benchmark-results"
WEIGHTS_MOUNT = "/cache"
RESULTS_MOUNT = "/results"

# Each benchmark Function below is registered with `serialized=True` (see
# register_benchmark), which pickles the function by value rather than looking it
# up by import path in the container. Modal requires the image's Python to exactly
# match the *local* interpreter running this script in that case, so this is tied
# to sys.version_info rather than hardcoded — whatever Python invokes `modal run`
# is what each image gets built with. All torch pins in MATRIX currently publish
# wheels for cp310 through at least cp313, so this should resolve on any
# reasonably current local Python; if yours is newer than that and a pin 404s,
# either pin an older local Python for this one invocation or drop that pin.
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# Fixed input + predict settings, identical across every config, so the only things
# that vary run-to-run are torch version / CUDA build / GPU type.
#
# examples/prot_custom_msa.yaml ships a precomputed MSA, so this has no network
# dependency on the MMseqs2 server and no run-to-run MSA variance.
BENCHMARK_INPUT_REL_PATH = "examples/prot_custom_msa.yaml"
PREDICT_EXTRA_ARGS = [
    "--model", "boltz2",
    "--accelerator", "gpu",
    "--recycling_steps", "3",
    "--sampling_steps", "200",
    "--diffusion_samples", "1",
    "--seed", "42",
    "--override",
    # Deliberately NOT using --flash_attn here: it's a third variable (a separate
    # kernel/attention backend) orthogonal to the torch/CUDA comparison this is
    # trying to isolate. Add a parallel matrix for that if you want it compared too.
]

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

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_MOUNT = "/root/boltz-community"


def _ignore_from_repo(path: pathlib.Path) -> bool:
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "modal_benchmarks"}
    return any(part in skip_dirs for part in path.parts) or path.suffix == ".pyc"


def build_image(torch_version: str, cuda_tag: str) -> modal.Image:
    torch_index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(f"torch=={torch_version}", index_url=torch_index_url)
        .pip_install(*BOLTZ_DEPS)
        .add_local_dir(
            REPO_ROOT, remote_path=REPO_MOUNT, copy=True, ignore=_ignore_from_repo
        )
        .run_commands(f"pip install --no-deps -e {REPO_MOUNT}")
        .env({"BOLTZ_CACHE": WEIGHTS_MOUNT, "TOKENIZERS_PARALLELISM": "false"})
    )


app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def run_config(cfg: dict) -> dict:
    """Run inside the Modal container: time `boltz predict`, sample peak GPU memory."""
    import subprocess
    import threading
    import time

    import torch

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

    out_dir = "/tmp/boltz_out"
    cmd = [
        "boltz", "predict", f"{REPO_MOUNT}/{BENCHMARK_INPUT_REL_PATH}",
        "--out_dir", out_dir,
        "--cache", WEIGHTS_MOUNT,
        *PREDICT_EXTRA_ARGS,
    ]

    poller = threading.Thread(target=poll_gpu_mem, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_MOUNT, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0
    stop.set()
    poller.join(timeout=2)

    result = {
        "config_id": cfg["id"],
        "torch_version_requested": cfg["torch"],
        "cuda_tag_requested": cfg["cuda"],
        "gpu_requested": cfg["gpu"],
        "torch_version_actual": torch.__version__,
        "cuda_version_actual": torch.version.cuda,
        "gpu_name_actual": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "wall_time_s": round(elapsed_s, 2),
        "peak_gpu_mem_mb": max(mem_samples_mb) if mem_samples_mb else None,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }

    with open(f"{RESULTS_MOUNT}/{cfg['id']}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


def register_benchmark(cfg: dict) -> modal.Function:
    image = build_image(cfg["torch"], cfg["cuda"])

    @app.function(
        image=image,
        gpu=cfg["gpu"],
        volumes={WEIGHTS_MOUNT: weights_volume, RESULTS_MOUNT: results_volume},
        timeout=30 * 60,
        name=f"bench_{cfg['id']}",
        serialized=True,  # created dynamically in a loop, not at module scope
    )
    def _run() -> dict:
        return run_config(cfg)

    return _run


BENCHMARK_FNS = {cfg["id"]: register_benchmark(cfg) for cfg in MATRIX}


def _print_matrix() -> None:
    print("Available configs:")
    for cfg in MATRIX:
        row = (
            f"  {cfg['id']:28s} torch={cfg['torch']:<8} "
            f"cuda={cfg['cuda']:<7} gpu={cfg['gpu']}"
        )
        print(row)
    print(
        "\nRun one:  modal run modal_benchmarks/torch_cuda_benchmark.py"
        " --config-id <id>"
        "\nRun all:  modal run modal_benchmarks/torch_cuda_benchmark.py --run-all"
    )


@app.local_entrypoint()
def main(config_id: str = "", run_all: bool = False) -> None:
    if not config_id and not run_all:
        _print_matrix()
        return

    target_ids = [cfg["id"] for cfg in MATRIX] if run_all else [config_id]
    results = []
    for cid in target_ids:
        fn = BENCHMARK_FNS.get(cid)
        if fn is None:
            known = ", ".join(BENCHMARK_FNS)
            print(f"Unknown config id: {cid!r}. Known ids: {known}")
            continue
        print(f"\n=== Running {cid} ===")
        try:
            result = fn.remote()
        except Exception as e:
            print(f"FAILED to run {cid}: {e}")
            continue
        print(json.dumps(result, indent=2))
        results.append(result)

    if len(results) > 1:
        print("\n=== Summary ===")
        header = (
            f"{'config':28s} {'wall_time_s':>12s} {'peak_gpu_mem_mb':>15s} {'exit':>5s}"
        )
        print(header)
        for r in results:
            print(
                f"{r['config_id']:28s} {r['wall_time_s']:12.1f} "
                f"{r['peak_gpu_mem_mb']!s:>15s} {r['exit_code']:5d}"
            )
