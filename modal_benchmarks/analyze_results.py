"""Compare torch_cuda_benchmark.py results pulled from the Modal results Volume.

Usage:

    modal volume get boltz-benchmark-results / ./results   # one-time pull per sweep
    python modal_benchmarks/analyze_results.py ./results
    # or, for the chart, with matplotlib available ad hoc:
    uv run --with matplotlib modal_benchmarks/analyze_results.py ./results

Prints a table sorted by GPU type then wall time, and (if matplotlib is
importable) writes a bar-chart PNG comparing wall time and peak GPU memory
across configs next to the results directory.
"""

import argparse
import json
import pathlib
import sys


def load_results(results_dir: pathlib.Path) -> list[dict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            results.append(json.load(f))
    return results


def print_table(results: list[dict]) -> None:
    def sort_key(r: dict) -> tuple:
        return (r.get("gpu_requested", ""), r.get("wall_time_s") or float("inf"))

    rows = sorted(results, key=sort_key)
    header = (
        f"{'config':28s} {'gpu':7s} {'torch':10s} {'cuda':6s} "
        f"{'wall_time_s':>12s} {'peak_mem_mb':>12s} {'exit':>5s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        wall = r.get("wall_time_s")
        wall_s = f"{wall:.1f}" if wall is not None else "n/a"
        mem = r.get("peak_gpu_mem_mb")
        mem_s = str(mem) if mem is not None else "n/a"
        torch_v = r.get("torch_version_actual", "?")
        cuda_v = r.get("cuda_version_actual", "?")
        print(
            f"{r.get('config_id', '?'):28s} {r.get('gpu_requested', '?'):7s} "
            f"{torch_v:10s} {cuda_v:6s} {wall_s:>12s} {mem_s:>12s} "
            f"{r.get('exit_code', -1):5d}"
        )
        if r.get("exit_code") != 0:
            print(f"    error: {r.get('error', '(see stderr_tail in the JSON file)')}")


def plot(results: list[dict], out_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    ok = [
        r
        for r in results
        if r.get("exit_code") == 0 and r.get("wall_time_s") is not None
    ]
    ok.sort(key=lambda r: (r.get("gpu_requested", ""), r["wall_time_s"]))
    if not ok:
        print("\nNo successful runs to plot.")
        return

    labels = [r["config_id"] for r in ok]
    wall_times = [r["wall_time_s"] for r in ok]
    mem = [r.get("peak_gpu_mem_mb") or 0 for r in ok]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.6), 8))
    ax1.bar(labels, wall_times, color="steelblue")
    ax1.set_ylabel("wall time (s)")
    ax1.set_title("boltz predict wall time by torch/CUDA/GPU config")
    ax1.tick_params(axis="x", rotation=60)

    ax2.bar(labels, mem, color="indianred")
    ax2.set_ylabel("peak GPU memory (MB)")
    ax2.tick_params(axis="x", rotation=60)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare torch/CUDA benchmark results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory of pulled JSON files",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the PNG chart")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"No such directory: {results_dir}", file=sys.stderr)
        print(
            "Pull results first: modal volume get boltz-benchmark-results / ./results",
            file=sys.stderr,
        )
        raise SystemExit(1)

    results = load_results(results_dir)
    if not results:
        print(f"No *.json result files found in {results_dir}", file=sys.stderr)
        raise SystemExit(1)

    print_table(results)

    if not args.no_plot:
        try:
            plot(results, results_dir.parent / "benchmark_comparison.png")
        except ImportError:
            print(
                "\n(matplotlib not installed - skipping chart; table above still"
                " works.\nRun with: uv run --with matplotlib"
                " modal_benchmarks/analyze_results.py <dir>)"
            )


if __name__ == "__main__":
    main()
