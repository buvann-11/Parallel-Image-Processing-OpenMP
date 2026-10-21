#!/usr/bin/env python3
"""Summarise the chunk-size experiment as a table and a figure.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Reads results/chunk_default.csv and results/chunk_large.csv (produced by
scripts/run_schedule_experiment.sh) and shows, for every kernel, the speedup
under `schedule(dynamic)` with the default chunk of 1 versus an explicit
large chunk.

The point of the comparison: the right chunk size depends on how much work
one loop ITERATION represents, and getting it wrong is expensive in both
directions.

Usage:  python3 bench/compare_chunks.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"


def load(path: Path, schedule: str) -> dict[str, float]:
    if not path.is_file():
        raise SystemExit(f"error: {path} not found.  "
                         "Run scripts/run_schedule_experiment.sh first.")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"error: {path} is empty.")
    max_threads = max(int(r["threads"]) for r in rows)
    return {r["kernel"]: float(r["speedup"]) for r in rows
            if r["schedule"] == schedule and int(r["threads"]) == max_threads}


def main() -> int:
    default = load(ROOT / "results" / "chunk_default.csv", "dynamic")
    large = load(ROOT / "results" / "chunk_large.csv", "dynamic")
    static = load(ROOT / "results" / "chunk_default.csv", "static")

    kernels = [k for k in default if k in large]

    print()
    print("=" * 78)
    print(" schedule(dynamic): the cost of the default chunk size")
    print("=" * 78)
    print(f"  {'kernel':<18}{'static':>9}{'dyn chunk=1':>14}"
          f"{'dyn chunk=8192':>16}   {'verdict':<20}")
    print("  " + "-" * 74)

    for kernel in kernels:
        d1, dl, st = default[kernel], large[kernel], static.get(kernel, 0.0)
        if d1 < 0.5 * st and dl > 0.8 * st:
            verdict = "chunk=1 catastrophic"
        elif dl < 0.6 * d1:
            verdict = "chunk too coarse"
        else:
            verdict = "insensitive"
        print(f"  {kernel:<18}{st:>9.2f}{d1:>14.2f}{dl:>16.2f}   {verdict:<20}")

    print()
    print("  Flat per-element loops need a LARGE chunk: with chunk = 1 the")
    print("  runtime's dispatch cost dwarfs one multiply-add per pixel.")
    print("  Row-based loops need a SMALL chunk: an image has only ~1000 rows,")
    print("  so a chunk of 8192 hands the whole loop to a single thread.")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (install matplotlib for the figure)")
        return 0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    positions = range(len(kernels))
    width = 0.27
    ax.bar([p - width for p in positions], [static.get(k, 0) for k in kernels],
           width, label="static")
    ax.bar(list(positions), [default[k] for k in kernels], width,
           label="dynamic, chunk = 1 (default)")
    ax.bar([p + width for p in positions], [large[k] for k in kernels],
           width, label="dynamic, chunk = 8192")
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(kernels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("speedup at the full thread count")
    ax.set_title("Chunk size is not a detail: dynamic scheduling with the\n"
                 "default chunk of 1 collapses fine-grained loops",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chunk_size.png", dpi=140)
    plt.close(fig)
    print(f"  Figure written to {FIG_DIR / 'chunk_size.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
