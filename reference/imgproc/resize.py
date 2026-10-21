r"""Image resizing by separable resampling.

The idea
--------
Resizing is *resampling*: the output grid is a set of positions in the input's
coordinate system, and each output pixel is a weighted average of nearby input
pixels,

.. math:: y[p] \;=\; \sum_{x} k\!\left(\frac{x + 0.5 - c_p}{\sigma}\right) x[k],
          \qquad c_p = (p + 0.5)\,\frac{L_{in}}{L_{out}}

where :math:`k` is a reconstruction kernel and :math:`c_p` is where output
pixel *p* lands in the input.  Different kernels give the familiar methods:

=============  =======  ===============================================
Method         Support  Character
=============  =======  ===============================================
``nearest``    --       No interpolation.  Blocky, but exactly preserves
                        label values -- the only correct choice for
                        segmentation masks.
``box``        0.5      Averages the source pixels that fall in the
                        output footprint.  Identical to OpenCV's
                        ``INTER_AREA`` for downscaling.
``bilinear``   1.0      Triangle kernel; smooth and cheap.  The default.
``bicubic``    2.0      Keys cubic (a = -0.5); sharper edges, slight
                        overshoot (ringing) at strong transitions.
``lanczos``    3.0      Windowed sinc; sharpest, strongest ringing.
=============  =======  ===============================================

Anti-aliasing: the part that is usually wrong
----------------------------------------------
Shrinking an image is *decimation*.  Sample a 1000-pixel row at 100 positions
with a kernel one pixel wide and 90% of the input is simply never read --
fine detail folds back as moiré.  The fix is to widen the kernel by the
downscaling factor before sampling, so every input pixel contributes:

.. math:: \sigma = \max\left(1, \frac{L_{in}}{L_{out}}\right)

That is what ``antialias=True`` (the default here) does; it is what Pillow has
always done and what ``torch.nn.functional.interpolate`` does *not* do unless
you ask.  ``antialias=False`` reproduces the naive GPU-style behaviour and is
kept for comparison.

Separability
------------
All these kernels are separable, so a 2-D resize is two 1-D passes -- rows,
then columns.  For an output of :math:`P \times Q` pixels with a kernel of
support :math:`T`, the cost drops from :math:`O(PQT^2)` to :math:`O(PQT)`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .exceptions import ParameterError, ShapeError
from .padding import pad_image
from .validation import (
    as_float_array,
    check_choice,
    check_image,
    check_positive_int,
    restore_dtype,
)

__all__ = [
    "RESIZE_METHODS",
    "resize",
    "resize_preserve_aspect",
    "center_crop",
    "rescale",
]


# --------------------------------------------------------------------------
# Reconstruction kernels
# --------------------------------------------------------------------------
def _box(t: np.ndarray) -> np.ndarray:
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def _triangle(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a < 1.0, 1.0 - a, 0.0)


def _keys_cubic(t: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys' cubic convolution kernel; ``a = -0.5`` is the Catmull-Rom choice
    used by Pillow, OpenCV and PyTorch."""
    x = np.abs(t)
    x2 = x * x
    x3 = x2 * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))


def _sinc(x: np.ndarray) -> np.ndarray:
    # np.sinc(x) is sin(pi x) / (pi x), already handling x == 0.
    return np.sinc(x)


def _lanczos3(t: np.ndarray) -> np.ndarray:
    return np.where(np.abs(t) < 3.0, _sinc(t) * _sinc(t / 3.0), 0.0)


#: method name -> (kernel, support radius in output-normalised units)
_FILTERS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float]] = {
    "box": (_box, 0.5),
    "bilinear": (_triangle, 1.0),
    "bicubic": (_keys_cubic, 2.0),
    "lanczos": (_lanczos3, 3.0),
}

#: Every accepted value of ``resize(method=...)``.
RESIZE_METHODS: tuple[str, ...] = ("nearest", "box", "area", "bilinear", "bicubic", "lanczos")

_METHOD_ALIASES = {"area": "box", "linear": "bilinear", "cubic": "bicubic", "lanczos3": "lanczos"}


def _contributions(
    in_size: int,
    out_size: int,
    method: str,
    antialias: bool,
    align_corners: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sampling indices and weights for one axis.

    Returns
    -------
    (indices, weights)
        Both of shape ``(out_size, taps)``.  Indices are already clamped into
        ``[0, in_size - 1]`` and each row of weights sums to exactly 1, so the
        transform preserves a constant image (the *partition of unity*
        property -- without it, resizing would change the mean brightness).
    """
    if align_corners and out_size > 1:
        scale = (in_size - 1) / (out_size - 1)
        centers = np.arange(out_size, dtype=np.float64) * scale + 0.5
        step = max(scale, 1.0)
    else:
        scale = in_size / out_size
        centers = (np.arange(out_size, dtype=np.float64) + 0.5) * scale
        step = scale
        if align_corners:  # out_size == 1: sample the middle of the image
            centers = np.full(1, in_size / 2.0)

    if method == "nearest":
        idx = np.clip(np.floor(centers).astype(np.intp), 0, in_size - 1)
        return idx[:, None], np.ones((out_size, 1), dtype=np.float64)

    kernel, support = _FILTERS[method]
    filter_scale = max(step, 1.0) if antialias else 1.0
    scaled_support = support * filter_scale

    taps = int(np.ceil(scaled_support * 2)) + 1
    taps = min(max(taps, 1), in_size + 2 * int(np.ceil(scaled_support)) + 1)

    left = np.floor(centers - scaled_support + 0.5).astype(np.intp)
    offsets = np.arange(taps, dtype=np.intp)[None, :]
    positions = left[:, None] + offsets  # (out, taps), may fall outside

    # Kernel argument: distance from the sample centre, in filter units.
    t = (positions + 0.5 - centers[:, None]) / filter_scale
    weights = kernel(t)

    # Positions outside the image contribute nothing; the remaining weights are
    # renormalised, which is equivalent to Pillow's border handling.
    inside = (positions >= 0) & (positions <= in_size - 1)
    weights = np.where(inside, weights, 0.0)

    totals = weights.sum(axis=1, keepdims=True)
    # A degenerate row (all taps outside, possible only for pathological
    # sizes) falls back to nearest-neighbour so the output stays defined.
    degenerate = np.abs(totals[:, 0]) < 1e-12
    if np.any(degenerate):
        fallback = np.clip(np.round(centers - 0.5).astype(np.intp), 0, in_size - 1)
        for row in np.nonzero(degenerate)[0]:
            weights[row] = 0.0
            positions[row] = fallback[row]
            weights[row, 0] = 1.0
        totals = weights.sum(axis=1, keepdims=True)

    weights = weights / totals
    indices = np.clip(positions, 0, in_size - 1)
    return indices, weights


def _resample_axis(array: np.ndarray, axis: int, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply a pre-computed 1-D resampling along ``axis``."""
    out_size, taps = indices.shape
    taken = np.take(array, indices.ravel(), axis=axis)
    new_shape = list(taken.shape)
    new_shape[axis : axis + 1] = [out_size, taps]
    taken = taken.reshape(new_shape)

    wshape = [1] * taken.ndim
    wshape[axis] = out_size
    wshape[axis + 1] = taps
    return (taken * weights.reshape(wshape)).sum(axis=axis + 1)


def resize(
    image: np.ndarray,
    size: Sequence[int],
    *,
    method: str = "bilinear",
    antialias: bool = True,
    align_corners: bool = False,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize an ``HW`` / ``HWC`` image to ``size = (height, width)``.

    Parameters
    ----------
    image
        Input image, any numeric dtype.
    size
        Target ``(height, width)``.  Both must be >= 1.
    method
        One of :data:`RESIZE_METHODS`.  ``"area"`` is an alias of ``"box"``.
    antialias
        Widen the kernel when downscaling so no input pixel is skipped.
        Default ``True``; turn it off only to reproduce naive GPU resizing.
    align_corners
        ``False`` (default) treats pixels as unit squares and aligns their
        *centres* -- the half-pixel convention used by Pillow, OpenCV and
        modern PyTorch.  ``True`` aligns the outermost pixel centres exactly,
        matching legacy TensorFlow ``align_corners=True`` behaviour.
    preserve_dtype
        Cast back to the input dtype, rounding and clipping.  Clipping matters
        for ``bicubic``/``lanczos``, whose ringing can overshoot the valid
        range.

    Returns
    -------
    numpy.ndarray
        Resized image with the input's layout and (by default) dtype.

    Examples
    --------
    Exact 2x nearest-neighbour upscale:

    >>> x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    >>> resize(x, (4, 4), method="nearest").tolist()
    [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    A constant image stays constant under every method (partition of unity):

    >>> const = np.full((9, 9), 7.0)
    >>> bool(np.allclose(resize(const, (4, 5), method="bicubic"), 7.0))
    True
    """
    arr = check_image(image, name="image")
    out_h, out_w = check_positive_int(size, name="size")

    key = check_choice(method, RESIZE_METHODS, name="method")
    key = _METHOD_ALIASES.get(key, key)

    in_h, in_w = arr.shape[0], arr.shape[1]
    work = as_float_array(arr, name="image")

    if (in_h, in_w) == (out_h, out_w):
        result = work.copy()
    else:
        row_idx, row_w = _contributions(in_h, out_h, key, antialias, align_corners)
        col_idx, col_w = _contributions(in_w, out_w, key, antialias, align_corners)
        result = _resample_axis(work, 0, row_idx, row_w)
        result = _resample_axis(result, 1, col_idx, col_w)

    result = np.ascontiguousarray(result)
    return restore_dtype(result, arr.dtype) if preserve_dtype else result


def rescale(
    image: np.ndarray,
    factor: float | Sequence[float],
    **kwargs,
) -> np.ndarray:
    """Resize by a multiplicative ``factor`` instead of an absolute size.

    ``factor`` may be a scalar or ``(fy, fx)``.  The output size is rounded to
    the nearest integer and clamped to at least one pixel.

    >>> rescale(np.zeros((10, 20)), 0.5).shape
    (5, 10)
    """
    arr = check_image(image, name="image")
    try:
        factors = tuple(float(f) for f in factor)  # type: ignore[union-attr]
    except TypeError:
        factors = (float(factor), float(factor))  # type: ignore[arg-type]
    if len(factors) != 2:
        raise ParameterError(f"'factor' must be a scalar or 2 values, got {factor!r}.")
    if any((not np.isfinite(f)) or f <= 0 for f in factors):
        raise ParameterError(f"'factor' entries must be positive and finite, got {factor!r}.")

    out_h = max(1, int(round(arr.shape[0] * factors[0])))
    out_w = max(1, int(round(arr.shape[1] * factors[1])))
    return resize(image, (out_h, out_w), **kwargs)


def resize_preserve_aspect(
    image: np.ndarray,
    size: Sequence[int],
    *,
    method: str = "bilinear",
    antialias: bool = True,
    pad_value: float = 0.0,
    pad_mode: str = "constant",
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Fit an image into ``size`` without distorting it ("letterboxing").

    The image is scaled by ``min(target_h / h, target_w / w)`` and the leftover
    margin is padded.  This is the standard preprocessing step for detectors
    such as YOLO: stretching the image instead would change every object's
    aspect ratio and hurt accuracy.

    Returns
    -------
    numpy.ndarray
        An image of exactly ``size``, with the original centred inside it.

    Examples
    --------
    >>> out = resize_preserve_aspect(np.ones((10, 20), dtype=np.uint8), (10, 10))
    >>> out.shape, int(out[0, 0]), int(out[5, 5])
    ((10, 10), 0, 1)
    """
    arr = check_image(image, name="image")
    target_h, target_w = check_positive_int(size, name="size")
    in_h, in_w = arr.shape[0], arr.shape[1]

    ratio = min(target_h / in_h, target_w / in_w)
    new_h = max(1, min(target_h, int(round(in_h * ratio))))
    new_w = max(1, min(target_w, int(round(in_w * ratio))))

    scaled = resize(
        arr, (new_h, new_w), method=method, antialias=antialias,
        preserve_dtype=preserve_dtype,
    )

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    if pad_top == pad_bottom == pad_left == pad_right == 0:
        return scaled
    return pad_image(
        scaled,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode=pad_mode,
        value=pad_value,
    )


def center_crop(image: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """Crop the central ``size = (height, width)`` region.

    Combined with :func:`resize`, this is the canonical "resize the short side
    to 256, then crop 224" evaluation transform.

    Raises
    ------
    ShapeError
        If the requested crop is larger than the image.
    """
    arr = check_image(image, name="image")
    crop_h, crop_w = check_positive_int(size, name="size")
    in_h, in_w = arr.shape[0], arr.shape[1]
    if crop_h > in_h or crop_w > in_w:
        raise ShapeError(
            f"Crop size {(crop_h, crop_w)} exceeds the image size {(in_h, in_w)}; "
            "resize first, or use resize_preserve_aspect()."
        )
    top = (in_h - crop_h) // 2
    left = (in_w - crop_w) // 2
    return np.ascontiguousarray(arr[top : top + crop_h, left : left + crop_w])
