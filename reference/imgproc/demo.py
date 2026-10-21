"""End-to-end demonstration: every operation, rendered as a figure.

``run_demo()`` writes one PNG per topic plus a short console report.  It uses
Matplotlib when it is available and falls back to writing plain images with
Pillow otherwise, so the demo degrades gracefully rather than crashing.

Run it with::

    python examples/demo.py            # or:  imgproc demo --outdir results/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .color import GRAYSCALE_WEIGHTS, to_grayscale
from .convolution import conv2d, convolve_image
from .io_utils import checkerboard, circle, gradient_ramp, sample_rgb_image, save_image
from .kernels import box_blur, gaussian_blur, laplacian, sharpen, sobel
from .layers import Conv2D, Flatten, MaxPool2D, ReLU, Sequential
from .normalization import (
    contrast_stretch,
    histogram_equalize,
    min_max_normalize,
    to_unit_range,
)
from .padding import pad_image
from .pooling import avg_pool2d, max_pool2d
from .resize import resize
from .validation import hwc_to_nchw

__all__ = ["run_demo"]


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")          # headless backend: no display required
        import matplotlib.pyplot as plt

        return plt
    except ImportError:  # pragma: no cover - environment dependent
        return None


def _grid(plt, images: list[tuple[str, np.ndarray]], title: str, path: Path,
          columns: int = 4, cmap: str | None = None) -> Path:
    """Render a labelled grid of images to ``path``."""
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.95 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (label, image) in zip(axes, images):
        display = image
        if display.ndim == 2:
            axis.imshow(display, cmap=cmap or "gray", vmin=None, vmax=None)
        else:
            axis.imshow(np.clip(display, 0, 255).astype(np.uint8))
        axis.set_title(label, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(images):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.8, w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _fallback(images: list[tuple[str, np.ndarray]], prefix: str, outdir: Path) -> list[Path]:
    """Write individual images when Matplotlib is unavailable."""
    written = []
    for label, image in images:
        safe = label.lower().replace(" ", "_").replace("/", "-").replace("=", "")
        written.append(save_image(image, outdir / f"{prefix}_{safe}.png"))
    return written


# --------------------------------------------------------------------------
# Individual demonstrations
# --------------------------------------------------------------------------
def _demo_padding(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    amount = source.shape[0] // 3
    modes = ["zeros", "constant", "reflect", "symmetric", "replicate", "circular"]
    return [("original", source)] + [
        (mode, pad_image(source, amount, mode=mode, value=128.0)) for mode in modes
    ]


def _demo_convolution(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    def magnitude(kernel: np.ndarray) -> np.ndarray:
        out = convolve_image(source, kernel, padding_mode="replicate", preserve_dtype=False)
        return np.clip(np.abs(out), 0, 255).astype(np.uint8)

    gx = convolve_image(source, sobel("x"), mode="correlate", preserve_dtype=False)
    gy = convolve_image(source, sobel("y"), mode="correlate", preserve_dtype=False)
    gradient = np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)

    return [
        ("original", source),
        ("box blur 5x5", convolve_image(source, box_blur(5), padding_mode="replicate")),
        ("gaussian sigma=2", convolve_image(source, gaussian_blur(2.0), padding_mode="replicate")),
        ("sharpen", convolve_image(source, sharpen(1.0), padding_mode="replicate")),
        ("|sobel x|", magnitude(sobel("x"))),
        ("|sobel y|", magnitude(sobel("y"))),
        ("gradient magnitude", gradient),
        ("|laplacian|", magnitude(laplacian(diagonal=True))),
    ]


def _demo_pooling(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    tensor = hwc_to_nchw(source)
    entries: list[tuple[str, np.ndarray]] = [(f"original {source.shape[:2]}", source)]
    for size in (2, 4, 8):
        for name, function in (("max", max_pool2d), ("avg", avg_pool2d)):
            pooled = function(tensor, size)[0, 0]
            entries.append((f"{name} pool {size}x{size} -> {pooled.shape}", pooled))
    return entries


def _demo_grayscale(source_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    entries: list[tuple[str, np.ndarray]] = [("original RGB", source_rgb)]
    for method in ("bt601", "bt709", "bt2020", "average", "lightness", "value"):
        entries.append((method, to_grayscale(source_rgb, method)))
    entries.append(
        ("bt709 + gamma", to_grayscale(source_rgb, "bt709", gamma_correct=True))
    )
    return entries


def _demo_normalization(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    low_contrast = (source.astype(np.float64) * 0.25 + 100.0).astype(np.uint8)
    unit = to_unit_range(low_contrast)
    return [
        ("low-contrast input", low_contrast),
        ("unit range x255", (unit * 255).astype(np.uint8)),
        ("min-max", (min_max_normalize(low_contrast) * 255).astype(np.uint8)),
        ("percentile stretch", contrast_stretch(low_contrast)),
        ("histogram equalised", histogram_equalize(low_contrast)),
        ("original for reference", source),
    ]


def _demo_resize(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    small = (source.shape[0] // 8, source.shape[1] // 8)
    entries: list[tuple[str, np.ndarray]] = [(f"original {source.shape[:2]}", source)]
    for method in ("nearest", "box", "bilinear", "bicubic", "lanczos"):
        shrunk = resize(source, small, method=method)
        entries.append((f"{method} -> {small}", resize(shrunk, source.shape[:2], method="nearest")))
    return entries


def _demo_aliasing() -> list[tuple[str, np.ndarray]]:
    pattern = checkerboard(256, 3)
    target = (32, 32)
    aliased = resize(pattern, target, method="bilinear", antialias=False)
    filtered = resize(pattern, target, method="bilinear", antialias=True)
    upscale = (256, 256)
    return [
        ("fine checkerboard (256px)", pattern),
        ("downscaled, no antialias", resize(aliased, upscale, method="nearest")),
        ("downscaled, antialiased", resize(filtered, upscale, method="nearest")),
    ]


def _demo_feature_maps(source: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Show what a CNN block actually does to an image, layer by layer."""
    gray = to_grayscale(source)
    tensor = hwc_to_nchw(to_unit_range(gray))

    # Hand-set the first layer to classical filters, so the output is
    # interpretable instead of random.
    layer = Conv2D(1, 4, 3, padding="same", bias=False, seed=0)
    layer.weight = np.stack(
        [
            sobel("x")[None, :, :],
            sobel("y")[None, :, :],
            laplacian(diagonal=True)[None, :, :],
            gaussian_blur(1.0, 3)[None, :, :],
        ]
    )
    names = ["sobel x", "sobel y", "laplacian", "gaussian"]

    features = layer.forward(tensor)
    activated = ReLU().forward(features)
    pooled = MaxPool2D(2).forward(activated)

    entries: list[tuple[str, np.ndarray]] = [("input (grayscale)", gray)]
    for index, name in enumerate(names):
        entries.append((f"conv: {name}", _to_display(features[0, index])))
    for index, name in enumerate(names):
        entries.append((f"relu+pool: {name}", _to_display(pooled[0, index])))
    return entries


def _to_display(plane: np.ndarray) -> np.ndarray:
    return (min_max_normalize(plane) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Console report
# --------------------------------------------------------------------------
def _report() -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("=" * 74)
    add("imgproc -- verification report")
    add("=" * 74)

    # Network summary
    net = Sequential(
        Conv2D(3, 16, 3, padding="same", seed=0),
        ReLU(),
        MaxPool2D(2),
        Conv2D(16, 32, 3, padding="same", seed=1),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
    )
    add()
    add("A small CNN built from these primitives (input 1x3x32x32):")
    add(net.summary((1, 3, 32, 32)))

    # Numerical identities
    add()
    add("Numerical identities")
    add("-" * 74)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 1, 16, 16))
    identity_kernel = np.zeros((1, 1, 3, 3))
    identity_kernel[0, 0, 1, 1] = 1.0
    error = np.abs(conv2d(x, identity_kernel, padding="same") - x).max()
    add(f"  conv with identity kernel reproduces the input : max error {error:.2e}")

    constant = np.full((1, 1, 16, 16), 5.0)
    error = np.abs(avg_pool2d(constant, 2) - 5.0).max()
    add(f"  average pooling preserves a constant map       : max error {error:.2e}")

    const_image = np.full((32, 32), 7.0)
    error = max(
        np.abs(resize(const_image, (13, 19), method=method, preserve_dtype=False) - 7.0).max()
        for method in ("nearest", "box", "bilinear", "bicubic", "lanczos")
    )
    add(f"  every resize kernel is a partition of unity    : max error {error:.2e}")

    gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
    error = max(
        abs(int(to_grayscale(gray, m)[0, 0]) - 128) for m in GRAYSCALE_WEIGHTS
    )
    add(f"  grayscale leaves neutral gray unchanged        : max error {error}")

    # Timing
    add()
    add("Throughput (im2col matrix-multiplication backend)")
    add("-" * 74)
    for shape, kernel_shape in (
        ((8, 3, 64, 64), (16, 3, 3, 3)),
        ((8, 16, 64, 64), (32, 16, 3, 3)),
        ((1, 3, 512, 512), (8, 3, 5, 5)),
    ):
        data = rng.normal(size=shape)
        weight = rng.normal(size=kernel_shape)
        start = time.perf_counter()
        out = conv2d(data, weight, padding="same")
        elapsed = time.perf_counter() - start
        macs = out.size * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
        add(
            f"  conv2d {str(shape):>18} x {str(kernel_shape):>14} : "
            f"{elapsed * 1e3:7.1f} ms  ({macs / max(elapsed, 1e-9) / 1e9:5.2f} GMAC/s)"
        )

    add()
    add("=" * 74)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_demo(outdir: Path | str = "examples/outputs", *, verbose: bool = True) -> list[Path]:
    """Generate every demonstration figure and return the paths written."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()

    rgb = sample_rgb_image(128)
    gray = to_grayscale(rgb)
    disc = circle(128, 40)
    ramp = gradient_ramp(128, 128, direction="diagonal")
    structured = np.clip(
        0.6 * disc.astype(np.float64) + 0.4 * ramp.astype(np.float64), 0, 255
    ).astype(np.uint8)

    sections: list[tuple[str, str, list[tuple[str, np.ndarray]], int]] = [
        ("01_padding", "Part 1 - Padding modes (CNN border handling)",
         _demo_padding(structured), 4),
        ("02_convolution", "Part 1 - Convolution with classical kernels",
         _demo_convolution(gray), 4),
        ("03_pooling", "Part 1 - Max vs. average pooling",
         _demo_pooling(gray), 4),
        ("04_feature_maps", "Part 1 - A CNN block: convolution -> ReLU -> max pool",
         _demo_feature_maps(rgb), 5),
        ("05_grayscale", "Part 2 - Grayscale conversion methods",
         _demo_grayscale(rgb), 4),
        ("06_normalization", "Part 2 - Intensity normalization",
         _demo_normalization(gray), 3),
        ("07_resize", "Part 2 - Resampling methods (8x down, then nearest up)",
         _demo_resize(gray), 3),
        ("08_antialiasing", "Part 2 - Why anti-aliasing matters when downscaling",
         _demo_aliasing(), 3),
    ]

    written: list[Path] = []
    for stem, title, images, columns in sections:
        if plt is not None:
            written.append(_grid(plt, images, title, out_path / f"{stem}.png", columns))
        else:  # pragma: no cover - only when Matplotlib is absent
            written.extend(_fallback(images, stem, out_path))
        if verbose:
            print(f"  {title}")

    report = _report()
    report_path = out_path / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    written.append(report_path)

    if verbose:
        print()
        print(report)
        print()
        print(f"Figures written to: {out_path.resolve()}")
    return written
