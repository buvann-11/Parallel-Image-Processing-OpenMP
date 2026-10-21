#!/usr/bin/env python3
"""Assemble the demo outputs into two contact sheets for the report.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Run `make demo` first to produce data/demo/, then:

    python3 tools/make_contact_sheets.py

Writes reports/figures/results_part1.png and results_part2.png.  Needs
matplotlib; the PPM reader is borrowed from the verification tool so this
script adds no new image dependency.
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, sys
sys.path.insert(0, "tools")
from verify_against_reference import read_netpbm
from pathlib import Path

def sheet(items, title, path, cols):
    rows = (len(items)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.9*cols, 3.25*rows))
    axes = np.atleast_1d(axes).ravel()
    for ax,(label,f) in zip(axes, items):
        im = read_netpbm(Path("data/demo")/f)
        ax.imshow(im, cmap=None if im.ndim==3 else "gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(items):]: ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,0.95), h_pad=1.6, w_pad=1.0)
    fig.savefig(path, dpi=130); plt.close(fig)

sheet([("input","00_input.ppm"),
       ("pad: zeros","10_pad_zeros.ppm"),
       ("pad: reflect","11_pad_reflect.ppm"),
       ("pad: replicate","12_pad_replicate.ppm"),
       ("pad: circular","13_pad_circular.ppm"),
       ("max pool 4x4","14_pool_max.ppm"),
       ("avg pool 4x4","15_pool_avg.ppm"),
       ("conv: gaussian","16_conv_gauss.ppm"),
       ("conv: sobel x","17_conv_sobelx.ppm"),
       ("conv: laplacian","18_conv_laplacian.ppm"),
       ("conv: sharpen","19_conv_sharpen.ppm")],
      "Part 1 -- CNN primitives: padding, pooling, convolution",
      "reports/figures/results_part1.png", 4)

sheet([("input","00_input.ppm"),
       ("grey: BT.601","20_gray_bt601.pgm"),
       ("grey: BT.709","21_gray_bt709.pgm"),
       ("grey: average","22_gray_average.pgm"),
       ("grey: luminance (gamma)","23_gray_lum.pgm"),
       ("norm: min-max","30_norm_minmax.ppm"),
       ("norm: z-score","31_norm_zscore.ppm"),
       ("norm: equalised","32_norm_equalize.ppm"),
       ("resize: nearest","40_resize_nearest.ppm"),
       ("resize: bilinear","41_resize_bilinear.ppm"),
       ("resize: bicubic","42_resize_bicubic.ppm"),
       ("resize: NO anti-alias","43_resize_noaa.ppm")],
      "Part 2 -- Image processing: greyscale, normalisation, resizing",
      "reports/figures/results_part2.png", 4)
print("contact sheets written")
