#!/usr/bin/env python3
"""Cross-validate the C/OpenMP implementation against an independent oracle.

MTCS-103P High Performance Computing Laboratory -- Mini Project

Why this exists
---------------
The C test-suite (``make test``) proves internal consistency: that the code
does what we *think* we asked it to do, and that the parallel version agrees
with the serial one.  It cannot catch a shared misunderstanding -- if the
convolution and its test encode the same wrong idea of what convolution is,
both agree and both are wrong.

This script closes that gap.  ``reference/imgproc`` is a completely separate
implementation of the same nine operations, written in NumPy, developed
independently of the C code, and itself validated against SciPy, Pillow,
OpenCV and scikit-image.  Agreement between two implementations that share no
source code is strong evidence that the algorithms themselves are right, not
merely self-consistent.

Method
------
For each operation: generate a deterministic input, run the C binary through
its command-line interface, read the PPM it wrote, compute the same result
with NumPy, and compare.

Tolerances
----------
Exact equality is not expected, and demanding it would be dishonest:

  * The C code computes in ``float32``; the reference uses ``float64``.
  * Both quantise to 8-bit on the way out, but round differently at exact
    halves -- C rounds half away from zero, NumPy's ``rint`` rounds half to
    even.  That alone permits a one-level difference on any pixel.

So each operation declares a tolerance in grey levels (out of 255), and the
script reports the measured maximum and mean difference so the margin is
visible rather than hidden behind a pass/fail.

Usage:  python3 tools/verify_against_reference.py [--binary bin/imgproc]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

try:
    import imgproc as ref
except ImportError as exc:  # pragma: no cover
    print(f"error: cannot import the NumPy reference: {exc}")
    print("       (it lives in reference/imgproc and needs NumPy)")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Netpbm I/O -- deliberately not using Pillow, so the comparison does not
# depend on a third library agreeing about file formats.
# ---------------------------------------------------------------------------
def read_netpbm(path: Path) -> np.ndarray:
    """Read a binary PGM (P5) or PPM (P6) into a uint8 array."""
    data = path.read_bytes()

    def token(pos: int) -> tuple[bytes, int]:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos : pos + 1] != b"\n":
                pos += 1
            return token(pos)
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        return data[start:pos], pos

    magic, pos = token(0)
    if magic not in (b"P5", b"P6"):
        raise ValueError(f"{path}: not a binary Netpbm file (magic {magic!r})")
    channels = 3 if magic == b"P6" else 1
    width, pos = token(pos)
    height, pos = token(pos)
    maxval, pos = token(pos)
    pos += 1  # the single whitespace byte before the raster

    w, h = int(width), int(height)
    if int(maxval) != 255:
        raise ValueError(f"{path}: only maxval 255 is supported")
    count = w * h * channels
    raw = np.frombuffer(data[pos : pos + count], dtype=np.uint8)
    if raw.size != count:
        raise ValueError(f"{path}: truncated raster ({raw.size} of {count})")
    return raw.reshape((h, w, channels)) if channels == 3 else raw.reshape((h, w))


def write_netpbm(array: np.ndarray, path: Path) -> None:
    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 2:
        header = f"P5\n{array.shape[1]} {array.shape[0]}\n255\n"
    else:
        header = f"P6\n{array.shape[1]} {array.shape[0]}\n255\n"
    path.write_bytes(header.encode("ascii") + array.tobytes())


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
class Result:
    def __init__(self, name: str, tol: float, max_diff: float,
                 mean_diff: float, note: str = "") -> None:
        self.name = name
        self.tol = tol
        self.max_diff = max_diff
        self.mean_diff = mean_diff
        self.note = note
        self.passed = max_diff <= tol

    def line(self) -> str:
        status = "ok  " if self.passed else "FAIL"
        return (f"  {status}  {self.name:<34} "
                f"max {self.max_diff:6.2f}  mean {self.mean_diff:6.3f}  "
                f"(tol {self.tol:.1f}) {self.note}")


def compare(name: str, produced: np.ndarray, expected: np.ndarray,
            tol: float, note: str = "") -> Result:
    if produced.shape != expected.shape:
        return Result(name, tol, float("inf"), float("inf"),
                      f"SHAPE {produced.shape} vs {expected.shape}")
    diff = np.abs(produced.astype(np.float64) - expected.astype(np.float64))
    return Result(name, tol, float(diff.max()), float(diff.mean()), note)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def run(binary: Path, tmp: Path) -> list[Result]:
    results: list[Result] = []

    def cli(*args: str) -> None:
        cmd = [str(binary), *args, "--quiet"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")

    # The C binary's own generator produces the input, so both sides start
    # from byte-identical data.
    source = tmp / "input.ppm"
    cli("gen", str(source), "--size", "256")
    rgb = read_netpbm(source)

    # --- Part 2: greyscale -------------------------------------------------
    for method in ("bt601", "bt709", "average"):
        out = tmp / f"gray_{method}.pgm"
        cli("grayscale", str(source), str(out), "--method", method)
        produced = read_netpbm(out)
        expected = ref.to_grayscale(rgb, method)
        results.append(compare(f"grayscale {method}", produced, expected, 1.0))

    out = tmp / "gray_lum.pgm"
    cli("grayscale", str(source), str(out), "--method", "luminance")
    produced = read_netpbm(out)
    expected = ref.to_grayscale(rgb, "bt709", gamma_correct=True)
    results.append(compare("grayscale luminance (gamma)", produced, expected, 2.0,
                           "powf in float32"))

    # --- Part 2: normalisation --------------------------------------------
    out = tmp / "norm_minmax.ppm"
    cli("normalize", str(source), str(out), "--method", "minmax")
    produced = read_netpbm(out)
    expected = (ref.min_max_normalize(rgb) * 255.0)
    expected = np.clip(np.rint(expected), 0, 255).astype(np.uint8)
    results.append(compare("normalize minmax", produced, expected, 1.0))

    out = tmp / "norm_eq.ppm"
    cli("normalize", str(source), str(out), "--method", "equalize")
    produced = read_netpbm(out)
    # `reference.histogram_equalize` uses a continuous-CDF variant, whereas
    # the C code implements the classic 256-bin discrete formulation (the one
    # in Gonzalez & Woods, and what OpenCV's equalizeHist does).  Comparing
    # those two would be comparing different algorithms, so the discrete
    # formula is written out here in NumPy and compared pixel for pixel.
    # Eight lines, and it makes the check exact instead of approximate.
    hist = np.bincount(rgb.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    nonzero = hist[hist > 0]
    cdf_min = nonzero[0] if nonzero.size else 0.0
    total = float(rgb.size)
    denom = max(total - cdf_min, 1e-9)
    lut = np.clip(np.rint((cdf - cdf_min) * 255.0 / denom), 0, 255).astype(np.uint8)
    expected = lut[rgb]
    results.append(compare("normalize equalize", produced, expected, 1.0,
                           "discrete 256-bin CDF"))

    # Equalisation is a CDF mapping, so it must preserve pixel ORDER.
    order_ok = True
    flat_in, flat_out = rgb.ravel()[::37], produced.ravel()[::37]
    idx = np.argsort(flat_in, kind="stable")
    if np.any(np.diff(flat_out[idx].astype(np.int32)) < 0):
        order_ok = False
    results.append(Result("normalize equalize (monotone)", 0.0,
                          0.0 if order_ok else 1.0, 0.0 if order_ok else 1.0,
                          "pixel order preserved"))

    # --- Part 2: resize ----------------------------------------------------
    for method in ("nearest", "bilinear", "bicubic"):
        out = tmp / f"resize_{method}.ppm"
        cli("resize", str(source), str(out), "--size", "97", "141",
            "--method", method)
        produced = read_netpbm(out)
        expected = ref.resize(rgb, (97, 141), method=method, antialias=True)
        results.append(compare(f"resize {method} (downscale)",
                               produced, expected, 2.0))

    out = tmp / "resize_up.ppm"
    cli("resize", str(source), str(out), "--size", "400", "400",
        "--method", "bilinear")
    produced = read_netpbm(out)
    expected = ref.resize(rgb, (400, 400), method="bilinear", antialias=True)
    results.append(compare("resize bilinear (upscale)", produced, expected, 2.0))

    # --- Part 1: padding ---------------------------------------------------
    for mode in ("zeros", "reflect", "replicate", "circular"):
        out = tmp / f"pad_{mode}.ppm"
        cli("pad", str(source), str(out), "--amount", "37", "--mode", mode)
        produced = read_netpbm(out)
        expected = ref.pad_image(rgb, 37, mode=mode)
        results.append(compare(f"pad {mode}", produced, expected, 0.0,
                               "must be exact"))

    # --- Part 1: pooling ---------------------------------------------------
    for pool in ("max", "avg"):
        out = tmp / f"pool_{pool}.ppm"
        cli("pool", str(source), str(out), "--type", pool,
            "--size", "4", "--stride", "4")
        produced = read_netpbm(out)
        tensor = ref.hwc_to_nchw(rgb)
        if pool == "max":
            pooled = ref.max_pool2d(tensor, 4, 4)
        else:
            pooled = ref.avg_pool2d(tensor, 4, 4)
        expected = ref.nchw_to_hwc(pooled)
        expected = np.clip(np.rint(expected), 0, 255).astype(np.uint8)
        results.append(compare(f"pool {pool}", produced, expected, 1.0))

    # --- Part 1: convolution ----------------------------------------------
    kernels = {
        "box": ref.box_blur(3),
        "sobelx": np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),
        "laplacian": ref.laplacian(diagonal=True),
    }
    for name, kernel in kernels.items():
        out = tmp / f"conv_{name}.ppm"
        cli("conv", str(source), str(out), "--kernel", name, "--mode", "reflect")
        produced = read_netpbm(out)
        expected = ref.convolve_image(rgb, kernel, padding="same",
                                      padding_mode="reflect", mode="convolve")
        results.append(compare(f"conv {name}", produced, expected, 1.0))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validate the C/OpenMP build against the NumPy "
                    "reference implementation.")
    parser.add_argument("--binary", default=str(ROOT / "bin" / "imgproc"),
                        help="path to the compiled imgproc binary")
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        print(f"error: {binary} not found.  Run `make` first.")
        return 1

    print()
    print("=" * 74)
    print(" Cross-validation: C/OpenMP  vs  independent NumPy reference")
    print("=" * 74)
    print(f"  binary    : {binary}")
    print(f"  reference : {ROOT / 'reference' / 'imgproc'} (v{ref.__version__})")
    print(f"  numpy     : {np.__version__}")
    print()
    print("  Differences are in grey levels out of 255.  A one-level margin is")
    print("  expected wherever both sides quantise to 8 bits: the C code")
    print("  computes in float32 and rounds halves away from zero, the")
    print("  reference computes in float64 and rounds halves to even.")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            results = run(binary, Path(tmpdir))
        except Exception as exc:
            print(f"  error: {exc}")
            return 1

    for result in results:
        print(result.line())

    failures = [r for r in results if not r.passed]
    print()
    print("-" * 74)
    if failures:
        print(f"  {len(failures)} of {len(results)} checks FAILED")
    else:
        print(f"  all {len(results)} checks passed -- the C implementation "
              "agrees with an")
        print("  independently written reference across both parts of the project")
    print("-" * 74)
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
