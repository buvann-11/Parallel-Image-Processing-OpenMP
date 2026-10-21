#!/usr/bin/env python3
"""Figures for the two individual reports (Part 1 and Part 2).

Writes, into reports/figures/:
  part1_gallery.png   padding, convolution and pooling on the sample photo
  part2_gallery.png   greyscale, normalisation and resize on the sample photo
  part1_speedup.png   serial vs 2-thread time and speedup, Part 1 kernels
  part2_speedup.png   the same for Part 2 kernels

Needs bin/imgproc (build first), NumPy and Matplotlib.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_against_reference import read_netpbm  # noqa: E402

BIN = ROOT / "bin" / "imgproc"
SAMPLE = ROOT / "data" / "samples" / "astronaut.png"
FIG = ROOT / "reports" / "figures"

PART1_OPS = [
    ("Input (512×512)", None),
    ("Padding — reflect, 64 px", ["pad", "--amount", "64", "--mode", "reflect"]),
    ("Padding — circular, 64 px", ["pad", "--amount", "64", "--mode", "circular"]),
    ("Convolution — Gaussian 5×5", ["conv", "--kernel", "gauss"]),
    ("Convolution — Sobel x", ["conv", "--kernel", "sobelx"]),
    ("Convolution — emboss", ["conv", "--kernel", "emboss"]),
    ("Max-pool 4×4, stride 4", ["pool", "--type", "max", "--size", "4", "--stride", "4"]),
    ("Avg-pool 4×4, stride 4", ["pool", "--type", "avg", "--size", "4", "--stride", "4"]),
    ("Convolution — Laplacian", ["conv", "--kernel", "laplacian"]),
]
PART2_OPS = [
    ("Input (512×512)", None),
    ("Greyscale — BT.601", ["grayscale", "--method", "bt601"]),
    ("Greyscale — linear luminance", ["grayscale", "--method", "luminance"]),
    ("Normalise — min–max", ["normalize", "--method", "minmax"]),
    ("Normalise — z-score", ["normalize", "--method", "zscore"]),
    ("Normalise — equalisation", ["normalize", "--method", "equalize"]),
    ("Resize — nearest 512→96", ["resize", "--size", "96", "96", "--method", "nearest"]),
    ("Resize — bilinear 512→96", ["resize", "--size", "96", "96", "--method", "bilinear"]),
    ("Resize — bicubic 512→96", ["resize", "--size", "96", "96", "--method", "bicubic"]),
]
PART1_KERNELS = ["pad_reflect", "conv_3x3", "conv_5x5", "conv_cnn_16ch", "pool_max_2x2"]
PART2_KERNELS = ["grayscale", "grayscale_gamma", "norm_minmax", "norm_equalize",
                 "resize_bicubic"]


def gallery(ops, out: Path, tmp: Path) -> None:
    src = tmp / "src.ppm"
    subprocess.run([str(BIN), "convert", str(SAMPLE), str(src), "--quiet"], check=True)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.2), dpi=200)
    for ax, (title, args) in zip(axes.flat, ops):
        if args is None:
            img = read_netpbm(src)
        else:
            dst = tmp / "out.ppm"
            subprocess.run([str(BIN), args[0], str(src), str(dst), *args[1:],
                            "--repeat", "1", "--quiet"], check=True)
            img = read_netpbm(dst)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  interpolation="nearest", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12, pad=6)
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def speedup(kernels, title, out: Path) -> None:
    rows = {r["kernel"]: r for r in csv.DictReader(open(ROOT / "results" / "scaling.csv"))
            if r["threads"] == "2" and r["schedule"] == "static"}
    ser = [float(rows[k]["serial_ms"]) for k in kernels]
    par = [float(rows[k]["omp_ms"]) for k in kernels]
    spd = [float(rows[k]["speedup"]) for k in kernels]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200,
                               gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(kernels))
    a.bar(x - 0.2, ser, 0.4, label="serial", color="#9aa5b1")
    a.bar(x + 0.2, par, 0.4, label="OpenMP, 2 threads", color="#2f6fb3")
    a.set_yscale("log")
    a.set_ylabel("time (ms, log scale)")
    a.set_xticks(x, kernels, rotation=25, ha="right", fontsize=10)
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
             ncol=2, fontsize=10)
    a.set_title("Execution time, 1024×1024×3", fontsize=12, pad=26)
    a.grid(axis="y", alpha=0.3)

    colours = ["#c0392b" if s > 2.2 else "#2f6fb3" for s in spd]
    b.bar(x, spd, 0.6, color=colours)
    b.axhline(2.0, color="#333", ls="--", lw=1)
    b.text(-0.45, 2.0, " ideal = 2 ", ha="left", va="center", fontsize=9,
           bbox=dict(fc="white", ec="none", pad=1))
    for xi, s in zip(x, spd):
        b.text(xi, s + 0.05, f"{s:.2f}×", ha="center", va="bottom", fontsize=10,
               bbox=dict(fc="white", ec="none", pad=0.5))
    b.set_xticks(x, kernels, rotation=25, ha="right", fontsize=10)
    b.set_ylabel("speedup")
    b.set_ylim(0, max(spd) * 1.15)
    b.set_title("Speedup on 2 threads", fontsize=12, pad=26)
    b.grid(axis="y", alpha=0.3)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> int:
    if not BIN.exists():
        print("build first: make  (or python3 run_project.py build)")
        return 1
    FIG.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        gallery(PART1_OPS, FIG / "part1_gallery.png", Path(d))
        gallery(PART2_OPS, FIG / "part2_gallery.png", Path(d))
    speedup(PART1_KERNELS, "Part 1 — CNN primitives", FIG / "part1_speedup.png")
    speedup(PART2_KERNELS, "Part 2 — image processing", FIG / "part2_speedup.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
