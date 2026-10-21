"""Input validation helpers shared by every module in :mod:`imgproc`.

Every public function in this package validates its inputs *before* doing any
numerical work.  Failing fast with a precise, actionable message is what
separates a library you can debug from one you cannot.

Conventions used throughout the package
---------------------------------------
* **Feature maps / mini-batches** use the ``NCHW`` layout
  ``(batch, channels, height, width)`` -- the layout used by PyTorch, cuDNN and
  most high-performance kernels.
* **Images** use the ``HWC`` layout ``(height, width, channels)`` or ``HW`` for
  single-channel images -- the layout used by Pillow, OpenCV and Matplotlib.
* Helpers :func:`hwc_to_nchw` and :func:`nchw_to_hwc` convert between the two.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .exceptions import DTypeError, ParameterError, ShapeError

__all__ = [
    "as_float_array",
    "check_ndim",
    "check_tensor_nchw",
    "check_image",
    "to_pair",
    "check_positive_int",
    "check_choice",
    "hwc_to_nchw",
    "nchw_to_hwc",
    "restore_dtype",
    "dtype_value_range",
]

#: dtypes that carry a fixed, documented value range.
_INTEGER_RANGES: dict[str, tuple[float, float]] = {
    "uint8": (0.0, 255.0),
    "uint16": (0.0, 65535.0),
    "int8": (-128.0, 127.0),
    "int16": (-32768.0, 32767.0),
}


def as_float_array(
    array: Any,
    *,
    name: str = "array",
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Convert ``array`` to a contiguous floating-point :class:`numpy.ndarray`.

    Parameters
    ----------
    array
        Anything array-like (``list``, ``tuple``, :class:`numpy.ndarray`, ...).
    name
        Name used in error messages, so the caller knows *which* argument
        was wrong.
    dtype
        Target floating dtype.  Defaults to ``float64`` for maximum numerical
        accuracy; pass ``np.float32`` when memory matters.

    Returns
    -------
    numpy.ndarray
        A C-contiguous array of the requested floating dtype.

    Raises
    ------
    DTypeError
        If the input cannot be interpreted as a numeric array, or contains
        complex values.
    """
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ParameterError(f"dtype must be a floating-point type, got {dtype!r}")
    try:
        out = np.asarray(array)
    except Exception as exc:  # pragma: no cover - defensive
        raise DTypeError(f"{name!r} could not be converted to an array: {exc}") from exc

    if out.dtype == object:
        raise DTypeError(
            f"{name!r} has dtype=object; it is probably a ragged nested sequence."
        )
    if np.issubdtype(out.dtype, np.complexfloating):
        raise DTypeError(f"{name!r} is complex; only real-valued input is supported.")
    if not (
        np.issubdtype(out.dtype, np.number) or np.issubdtype(out.dtype, np.bool_)
    ):
        raise DTypeError(f"{name!r} has non-numeric dtype {out.dtype}.")

    return np.ascontiguousarray(out, dtype=dtype)


def check_ndim(array: np.ndarray, ndim: int | Sequence[int], *, name: str) -> None:
    """Assert that ``array`` has one of the permitted ranks."""
    allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if array.ndim not in allowed:
        expected = " or ".join(f"{d}-D" for d in allowed)
        raise ShapeError(
            f"{name!r} must be {expected}, got {array.ndim}-D array "
            f"with shape {array.shape}."
        )


def check_tensor_nchw(tensor: Any, *, name: str = "x") -> np.ndarray:
    """Validate and return a 4-D ``(N, C, H, W)`` float tensor.

    A 3-D ``(C, H, W)`` input is *not* silently accepted: ambiguity between
    ``(C, H, W)`` and ``(N, H, W)`` is a classic source of silent bugs, so the
    caller must be explicit.
    """
    arr = as_float_array(tensor, name=name)
    check_ndim(arr, 4, name=name)
    n, c, h, w = arr.shape
    if min(n, c, h, w) <= 0:
        raise ShapeError(
            f"{name!r} has a zero-sized dimension: shape {arr.shape}. "
            "All of (N, C, H, W) must be >= 1."
        )
    return arr


def check_image(image: Any, *, name: str = "image", allow_gray: bool = True) -> np.ndarray:
    """Validate an image in ``HW`` or ``HWC`` layout and return it unchanged.

    The dtype is *preserved* (unlike :func:`as_float_array`) so that callers can
    decide whether to round-trip back to an integer dtype.
    """
    arr = np.asarray(image)
    if arr.dtype == object:
        raise DTypeError(f"{name!r} has dtype=object; expected a numeric array.")
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise DTypeError(f"{name!r} has non-numeric dtype {arr.dtype}.")
    allowed = (2, 3) if allow_gray else (3,)
    check_ndim(arr, allowed, name=name)
    if arr.ndim == 3 and arr.shape[2] == 0:
        raise ShapeError(f"{name!r} has zero channels: shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ShapeError(f"{name!r} has a zero-sized spatial dimension: {arr.shape}.")
    return arr


def to_pair(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    """Normalise ``int`` or ``(a, b)`` into a ``(height, width)`` tuple of ints."""
    if isinstance(value, (bool, np.bool_)):
        raise ParameterError(f"{name!r} must be an int or a pair of ints, got a bool.")
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    elif isinstance(value, Iterable):
        items = list(value)
        if len(items) != 2:
            raise ParameterError(
                f"{name!r} must contain exactly 2 elements, got {len(items)}."
            )
        for item in items:
            if not isinstance(item, (int, np.integer)) or isinstance(item, (bool, np.bool_)):
                raise ParameterError(f"{name!r} must contain integers, got {item!r}.")
        pair = (int(items[0]), int(items[1]))
    else:
        raise ParameterError(
            f"{name!r} must be an int or a sequence of 2 ints, got {type(value).__name__}."
        )
    return pair


def check_positive_int(
    value: int | Sequence[int],
    *,
    name: str,
    minimum: int = 1,
) -> tuple[int, int]:
    """Normalise to a pair and assert both entries are ``>= minimum``."""
    pair = to_pair(value, name=name)
    if pair[0] < minimum or pair[1] < minimum:
        raise ParameterError(f"{name!r} must be >= {minimum}, got {pair}.")
    return pair


def check_choice(value: str, choices: Sequence[str], *, name: str) -> str:
    """Assert ``value`` is one of ``choices`` (case-insensitive) and canonicalise it."""
    if not isinstance(value, str):
        raise ParameterError(
            f"{name!r} must be a string, got {type(value).__name__}."
        )
    lowered = value.lower().strip()
    if lowered not in choices:
        options = ", ".join(repr(c) for c in choices)
        raise ParameterError(f"{name!r} must be one of {options}; got {value!r}.")
    return lowered


def hwc_to_nchw(image: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Convert an ``HW`` / ``HWC`` image into a 1-sample ``NCHW`` tensor."""
    arr = check_image(image, name="image")
    arr = as_float_array(arr, name="image", dtype=dtype)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def nchw_to_hwc(tensor: np.ndarray, *, squeeze_gray: bool = True) -> np.ndarray:
    """Convert an ``NCHW`` tensor with ``N == 1`` back into an ``HWC`` image."""
    arr = check_tensor_nchw(tensor, name="tensor")
    if arr.shape[0] != 1:
        raise ShapeError(
            f"nchw_to_hwc expects a batch of exactly 1, got N={arr.shape[0]}."
        )
    out = np.ascontiguousarray(arr[0].transpose(1, 2, 0))
    if squeeze_gray and out.shape[2] == 1:
        out = out[:, :, 0]
    return out


def dtype_value_range(dtype: np.dtype | type) -> tuple[float, float]:
    """Return the canonical ``(low, high)`` value range for ``dtype``.

    Floating dtypes are assumed to live in ``[0, 1]`` -- the convention used by
    scikit-image and by most deep-learning preprocessing pipelines.
    """
    dt = np.dtype(dtype)
    if dt.name in _INTEGER_RANGES:
        return _INTEGER_RANGES[dt.name]
    if np.issubdtype(dt, np.floating):
        return (0.0, 1.0)
    raise DTypeError(f"No canonical value range is defined for dtype {dt}.")


def restore_dtype(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Cast ``array`` back to ``dtype``, rounding and clipping for integer types.

    Floating results are returned untouched -- clipping a float result would
    silently destroy information the caller may still need.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return array.astype(dt, copy=False)
    info = np.iinfo(dt)
    return np.clip(np.rint(array), info.min, info.max).astype(dt)
