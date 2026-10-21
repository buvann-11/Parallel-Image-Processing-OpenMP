#!/usr/bin/env python3
"""Turn results/scaling.csv into the figures used in the report.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Produces four figures in report/figures/:

  1. speedup.png       S(p) for every kernel, against the ideal S = p line.
  2. efficiency.png    E(p) = S(p)/p.  Where a curve falls away shows where
                       the kernel stops being worth more cores.
  3. schedules.png     static vs dynamic vs guided at the full thread count.
  4. karp_flatt.png    the experimentally determined serial fraction.  A flat
                       line means Amdahl serial work; a rising line means
                       parallel overhead.  They need different fixes.

Usage:
    make bench                      # writes results/scaling.csv
    python3 bench/plot_results.py   # writes the figures

Needs matplotlib.  If it is missing, the script prints a text summary
instead of failing, so the numbers are still available.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "scaling.csv"
FIG_DIR = ROOT / "report" / "figures"


def load(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"error: {path} not found.  Run `make bench` first.")
        raise SystemExit(1)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["threads"] = int(row["threads"])
            for key in ("serial_ms", "omp_ms", "speedup", "efficiency",
                        "karp_flatt"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def text_summary(rows: list[dict]) -> None:
    print("\nBest speedup per kernel (static schedule):")
    best: dict[str, tuple[int, float]] = {}
    for row in rows:
        if row["schedule"] != "static":
            continue
        kernel = row["kernel"]
        if kernel not in best or row["speedup"] > best[kernel][1]:
            best[kernel] = (row["threads"], row["speedup"])
    for kernel, (threads, speedup) in sorted(
            best.items(), key=lambda kv: -kv[1][1]):
        print(f"  {kernel:<18} {speedup:5.2f}x at {threads} threads")


def main() -> int:
    rows = load(CSV_PATH)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; printing a text summary instead.")
        print("  pip install matplotlib   to get the figures")
        text_summary(rows)
        return 0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    static = [r for r in rows if r["schedule"] == "static"]
    kernels = sorted({r["kernel"] for r in static})
    max_threads = max(r["threads"] for r in static)

    series: dict[str, list[dict]] = defaultdict(list)
    for row in static:
        series[row["kernel"]].append(row)
    for values in series.values():
        values.sort(key=lambda r: r["threads"])

    cmap = plt.get_cmap("tab10")
    colours = {k: cmap(i % 10) for i, k in enumerate(kernels)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    marks = {k: markers[i % len(markers)] for i, k in enumerate(kernels)}

    # --- 1. speedup -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ideal = list(range(1, max_threads + 1))
    ax.plot(ideal, ideal, "k--", linewidth=1.2, label="ideal (S = p)",
            zorder=1)
    for kernel in kernels:
        data = series[kernel]
        ax.plot([r["threads"] for r in data], [r["speedup"] for r in data],
                marker=marks[kernel], color=colours[kernel], linewidth=1.6,
                markersize=5, label=kernel)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup  S(p) = T_serial / T(p)")
    ax.set_title("Strong scaling: speedup", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speedup.png", dpi=140)
    plt.close(fig)

    # --- 2. efficiency --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.2,
               label="ideal (E = 1)")
    for kernel in kernels:
        data = series[kernel]
        ax.plot([r["threads"] for r in data],
                [r["efficiency"] for r in data],
                marker=marks[kernel], color=colours[kernel], linewidth=1.6,
                markersize=5, label=kernel)
    ax.set_xlabel("threads")
    ax.set_ylabel("efficiency  E(p) = S(p) / p")
    ax.set_title("Strong scaling: parallel efficiency", fontweight="bold")
    ax.set_ylim(0, max(1.35, max(r["efficiency"] for r in static) * 1.1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "efficiency.png", dpi=140)
    plt.close(fig)

    # --- 3. schedule comparison -----------------------------------------
    schedules = sorted({r["schedule"] for r in rows})
    if len(schedules) > 1:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        width = 0.8 / len(schedules)
        positions = range(len(kernels))
        for si, sched in enumerate(schedules):
            values = []
            for kernel in kernels:
                match = [r["speedup"] for r in rows
                         if r["kernel"] == kernel and r["schedule"] == sched
                         and r["threads"] == max_threads]
                values.append(match[0] if match else 0.0)
            ax.bar([p + si * width for p in positions], values, width,
                   label=sched)
        ax.set_xticks([p + width * (len(schedules) - 1) / 2
                       for p in positions])
        ax.set_xticklabels(kernels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"speedup at {max_threads} threads")
        ax.set_title("Effect of the OpenMP loop schedule", fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "schedules.png", dpi=140)
        plt.close(fig)

    # --- 4. Karp-Flatt --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for kernel in kernels:
        data = [r for r in series[kernel] if r["threads"] > 1]
        if not data:
            continue
        ax.plot([r["threads"] for r in data],
                [r["karp_flatt"] for r in data],
                marker=marks[kernel], color=colours[kernel], linewidth=1.6,
                markersize=5, label=kernel)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("threads")
    ax.set_ylabel("experimentally determined serial fraction  e")
    ax.set_title("Karp-Flatt metric\n"
                 "flat = genuine serial work   |   rising = parallel overhead",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "karp_flatt.png", dpi=140)
    plt.close(fig)

    print(f"Figures written to {FIG_DIR}")
    text_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
