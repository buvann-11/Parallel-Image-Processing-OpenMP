"""Command-line interface: ``imgproc <command> ...``.

A library that can only be driven from Python is half a tool.  This CLI exposes
every operation in the package so that results can be reproduced from a shell,
scripted in a Makefile, or checked by a grader without writing any code::

    imgproc info
    imgproc grayscale photo.png gray.png --method bt709
    imgproc resize photo.png small.png --size 128 128 --method bicubic
    imgproc filter photo.png edges.png --kernel sobel_x
    imgproc pool photo.png pooled.png --type max --size 2
    imgproc pad photo.png padded.png --amount 16 --mode reflect
    imgproc normalize photo.png fixed.png --scheme equalize
    imgproc demo --outdir results/

Exit codes: ``0`` success, ``1`` a handled error (bad argument, missing file),
``2`` argparse usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .color import GRAYSCALE_METHODS, to_grayscale
from .convolution import convolve_image
from .exceptions import ImgProcError
from .io_utils import load_image, save_image
from .kernels import KERNELS
from .normalization import (
    contrast_stretch,
    histogram_equalize,
    min_max_normalize,
    to_unit_range,
)
from .padding import PAD_MODES, pad_image
from .pooling import avg_pool2d, max_pool2d
from .resize import RESIZE_METHODS, resize
from .validation import hwc_to_nchw, nchw_to_hwc, restore_dtype

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser (exposed so the tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="imgproc",
        description=(
            "CNN primitives (padding, convolution, pooling) and classical image "
            "processing (grayscale, normalization, resize), implemented from "
            "scratch with NumPy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'imgproc <command> --help' for command-specific options.",
    )
    parser.add_argument("--version", action="version", version=f"imgproc {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # -- info ---------------------------------------------------------------
    info = subparsers.add_parser("info", help="print package and capability information")
    info.add_argument("image", nargs="?", help="optional image to report on")

    # -- grayscale ----------------------------------------------------------
    gray = subparsers.add_parser("grayscale", help="convert an image to grayscale")
    _add_io(gray)
    gray.add_argument(
        "--method", default="bt601", choices=list(GRAYSCALE_METHODS),
        help="luma / lightness formula (default: bt601)",
    )
    gray.add_argument(
        "--gamma-correct", action="store_true",
        help="linearise sRGB first, for true relative luminance",
    )

    # -- resize -------------------------------------------------------------
    rsz = subparsers.add_parser("resize", help="resize an image")
    _add_io(rsz)
    rsz.add_argument(
        "--size", nargs=2, type=int, required=True, metavar=("HEIGHT", "WIDTH"),
        help="target size in pixels",
    )
    rsz.add_argument(
        "--method", default="bilinear", choices=list(RESIZE_METHODS),
        help="resampling kernel (default: bilinear)",
    )
    rsz.add_argument(
        "--no-antialias", action="store_true",
        help="disable kernel widening when downscaling (demonstrates aliasing)",
    )
    rsz.add_argument(
        "--keep-aspect", action="store_true",
        help="letterbox instead of stretching",
    )

    # -- normalize ----------------------------------------------------------
    norm = subparsers.add_parser("normalize", help="adjust intensity distribution")
    _add_io(norm)
    norm.add_argument(
        "--scheme", default="minmax", choices=["minmax", "stretch", "equalize", "unit"],
        help="minmax | stretch (percentile) | equalize (histogram) | unit ([0,1] scaling)",
    )
    norm.add_argument(
        "--per-channel", action="store_true", help="treat each channel independently"
    )

    # -- filter -------------------------------------------------------------
    filt = subparsers.add_parser("filter", help="convolve an image with a named kernel")
    _add_io(filt)
    filt.add_argument(
        "--kernel", default="gaussian_blur", choices=sorted(KERNELS),
        help="kernel from imgproc.kernels (default: gaussian_blur)",
    )
    filt.add_argument(
        "--padding-mode", default="reflect", choices=list(PAD_MODES),
        help="border handling (default: reflect)",
    )

    # -- pool ---------------------------------------------------------------
    pool = subparsers.add_parser("pool", help="downsample an image by pooling")
    _add_io(pool)
    pool.add_argument("--type", default="max", choices=["max", "avg"], dest="pool_type")
    pool.add_argument("--size", type=int, default=2, help="window size (default: 2)")
    pool.add_argument("--stride", type=int, default=None, help="defaults to --size")

    # -- pad ----------------------------------------------------------------
    pad = subparsers.add_parser("pad", help="pad an image border")
    _add_io(pad)
    pad.add_argument("--amount", type=int, default=8, help="pixels on every side")
    pad.add_argument("--mode", default="zeros", choices=list(PAD_MODES))
    pad.add_argument("--value", type=float, default=0.0, help="fill for --mode constant")

    # -- demo ---------------------------------------------------------------
    demo = subparsers.add_parser("demo", help="run the full demonstration suite")
    demo.add_argument(
        "--outdir", default="examples/outputs", help="where to write the figures"
    )

    return parser


def _add_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def _cmd_info(args: argparse.Namespace) -> int:
    print(f"imgproc {__version__}")
    print(f"numpy   {np.__version__}")
    print()
    print("Part 1 -- CNN primitives : padding, convolution, pooling, layers")
    print("Part 2 -- Image processing: grayscale, normalization, resize, pipeline")
    print()
    print(f"padding modes  : {', '.join(PAD_MODES)}")
    print(f"resize methods : {', '.join(RESIZE_METHODS)}")
    print(f"gray methods   : {', '.join(GRAYSCALE_METHODS)}")
    print(f"kernels        : {', '.join(sorted(KERNELS))}")

    if args.image:
        image = load_image(args.image)
        print()
        print(f"{args.image}:")
        print(f"  shape  : {image.shape}")
        print(f"  dtype  : {image.dtype}")
        print(f"  range  : [{image.min()}, {image.max()}]")
        print(f"  mean   : {image.mean():.3f}")
    return 0


def _cmd_grayscale(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    result = to_grayscale(image, args.method, gamma_correct=args.gamma_correct)
    return _write(result, args.output, f"grayscale ({args.method})")


def _cmd_resize(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    size = (args.size[0], args.size[1])
    if args.keep_aspect:
        from .resize import resize_preserve_aspect

        result = resize_preserve_aspect(
            image, size, method=args.method, antialias=not args.no_antialias
        )
    else:
        result = resize(
            image, size, method=args.method, antialias=not args.no_antialias
        )
    return _write(result, args.output, f"resize to {size} ({args.method})")


def _cmd_normalize(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    if args.scheme == "minmax":
        unit = min_max_normalize(image, per_channel=args.per_channel)
        result = restore_dtype(unit * 255.0, np.uint8)
    elif args.scheme == "unit":
        result = restore_dtype(to_unit_range(image) * 255.0, np.uint8)
    elif args.scheme == "stretch":
        result = contrast_stretch(image, per_channel=args.per_channel)
    else:
        result = histogram_equalize(image, per_channel=args.per_channel)
    return _write(result, args.output, f"normalize ({args.scheme})")


def _cmd_filter(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    kernel = KERNELS[args.kernel]()
    filtered = convolve_image(
        image, kernel, padding_mode=args.padding_mode, preserve_dtype=False
    )
    # Derivative kernels produce signed output; show its magnitude.
    if abs(float(kernel.sum())) < 1e-9:
        filtered = np.abs(filtered)
    result = restore_dtype(np.clip(filtered, 0, 255), np.uint8)
    return _write(result, args.output, f"filter ({args.kernel})")


def _cmd_pool(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    tensor = hwc_to_nchw(image)
    if args.pool_type == "max":
        pooled = max_pool2d(tensor, args.size, args.stride)
    else:
        pooled = avg_pool2d(tensor, args.size, args.stride)
    result = restore_dtype(nchw_to_hwc(pooled), image.dtype)
    return _write(result, args.output, f"{args.pool_type} pool {args.size}x{args.size}")


def _cmd_pad(args: argparse.Namespace) -> int:
    image = load_image(args.input)
    result = pad_image(image, args.amount, mode=args.mode, value=args.value)
    return _write(result, args.output, f"pad {args.amount}px ({args.mode})")


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    written = run_demo(Path(args.outdir))
    print(f"Wrote {len(written)} file(s) to {args.outdir}")
    return 0


def _write(image: np.ndarray, path: str, description: str) -> int:
    written = save_image(image, path)
    print(f"{description}: {tuple(image.shape)} -> {written}")
    return 0


_COMMANDS = {
    "info": _cmd_info,
    "grayscale": _cmd_grayscale,
    "resize": _cmd_resize,
    "normalize": _cmd_normalize,
    "filter": _cmd_filter,
    "pool": _cmd_pool,
    "pad": _cmd_pad,
    "demo": _cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code rather than calling ``sys.exit``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        return _COMMANDS[args.command](args)
    except (ImgProcError, ValueError, OSError) as error:
        # ImgProcError covers the library's own failures; ValueError/OSError
        # cover what Pillow raises for unreadable or unwritable paths.  The
        # user gets one clear line, not a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
