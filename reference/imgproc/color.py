r"""Colour conversion -- principally RGB to grayscale.

Why not just average the channels?
----------------------------------
Averaging assumes the eye is equally sensitive to red, green and blue.  It is
not: the human retina has roughly twice as many green-sensitive cones as
red-sensitive ones and far fewer blue-sensitive ones.  Broadcast standards
encode that with a weighted sum,

.. math::

    Y = w_R R + w_G G + w_B B, \qquad w_R + w_G + w_B = 1

============  =======  =======  =======  =====================================
Standard      w_R      w_G      w_B      Used by
============  =======  =======  =======  =====================================
ITU-R BT.601  0.2990   0.5870   0.1140   SDTV, OpenCV, Pillow, MATLAB
ITU-R BT.709  0.2126   0.7152   0.0722   HDTV, sRGB, scikit-image
ITU-R BT.2020 0.2627   0.6780   0.0593   UHDTV / wide gamut
============  =======  =======  =======  =====================================

Luma vs. luminance (the gamma subtlety)
---------------------------------------
Stored sRGB values are **gamma-encoded**: the number 128 is not "half the
light" of 255, it is about 21% of it.  Applying the weights directly to the
encoded values yields *luma* (:math:`Y'`) -- the fast, conventional answer, and
what OpenCV and Pillow produce.  Linearising first, weighting, then
re-encoding yields true *relative luminance* (:math:`Y`) -- the physically
correct answer, noticeably brighter in saturated regions.

Pass ``gamma_correct=True`` for the photometrically correct version.  Both are
offered because both are "right", for different questions.
"""

from __future__ import annotations

import numpy as np

from .exceptions import ParameterError, ShapeError
from .validation import (
    as_float_array,
    check_choice,
    check_image,
    dtype_value_range,
    restore_dtype,
)

__all__ = [
    "GRAYSCALE_WEIGHTS",
    "GRAYSCALE_METHODS",
    "srgb_to_linear",
    "linear_to_srgb",
    "to_grayscale",
    "to_rgb",
    "drop_alpha",
]

#: Standard luma coefficients, ``(w_R, w_G, w_B)``.
GRAYSCALE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "bt601": (0.299, 0.587, 0.114),
    "bt709": (0.2126, 0.7152, 0.0722),
    "bt2020": (0.2627, 0.6780, 0.0593),
}

#: Every accepted value of ``to_grayscale(method=...)``.
GRAYSCALE_METHODS: tuple[str, ...] = (
    "bt601",
    "bt709",
    "bt2020",
    "luminosity",  # alias of bt601 -- the name used in most textbooks
    "average",
    "lightness",
    "value",
)

_ALIASES = {"luminosity": "bt601", "luma": "bt601", "rec601": "bt601", "rec709": "bt709"}


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    r"""Undo the sRGB transfer function; input and output in ``[0, 1]``.

    .. math::

        C_\text{lin} = \begin{cases}
            C / 12.92 & C \le 0.04045 \\
            \left(\frac{C + 0.055}{1.055}\right)^{2.4} & \text{otherwise}
        \end{cases}

    The linear segment near black avoids the infinite slope a pure power law
    would have at 0.
    """
    v = as_float_array(values, name="values")
    clipped = np.clip(v, 0.0, 1.0)
    return np.where(
        clipped <= 0.04045,
        clipped / 12.92,
        np.power((clipped + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function; input and output in ``[0, 1]``.

    Exact inverse of :func:`srgb_to_linear`.
    """
    v = as_float_array(values, name="values")
    clipped = np.clip(v, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def drop_alpha(image: np.ndarray, *, background: float | None = None) -> np.ndarray:
    """Remove an alpha channel from an ``HWC`` image.

    Parameters
    ----------
    background
        ``None`` (default) simply discards the alpha channel.  A number in the
        image's own value range instead *composites* the image over a constant
        background of that value, which is the correct thing to do whenever the
        alpha is not everywhere opaque.
    """
    arr = check_image(image, name="image")
    if arr.ndim == 2 or arr.shape[2] not in (2, 4):
        return arr
    dtype = arr.dtype
    low, high = dtype_value_range(dtype)
    work = as_float_array(arr, name="image")
    colour, alpha = work[..., :-1], work[..., -1:]
    if background is None:
        return restore_dtype(colour, dtype)
    a = np.clip((alpha - low) / (high - low), 0.0, 1.0)
    composited = colour * a + float(background) * (1.0 - a)
    return restore_dtype(composited, dtype)


def to_grayscale(
    image: np.ndarray,
    method: str = "bt601",
    *,
    gamma_correct: bool = False,
    keep_channel_dim: bool = False,
    alpha_background: float | None = None,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Convert an RGB / RGBA image to a single-channel grayscale image.

    Parameters
    ----------
    image
        ``HW``, ``HWC`` with 1, 3 or 4 channels.  A single-channel image is
        passed through unchanged (already gray).
    method
        One of :data:`GRAYSCALE_METHODS`:

        * ``"bt601"`` / ``"luminosity"`` -- ``0.299R + 0.587G + 0.114B``
          (default; matches OpenCV, Pillow and MATLAB).
        * ``"bt709"``  -- ``0.2126R + 0.7152G + 0.0722B`` (HDTV / sRGB).
        * ``"bt2020"`` -- UHDTV coefficients.
        * ``"average"`` -- ``(R + G + B) / 3``; perceptually wrong but the
          simplest baseline, included for comparison.
        * ``"lightness"`` -- ``(max + min) / 2``, the HSL *L* channel.
        * ``"value"`` -- ``max(R, G, B)``, the HSV *V* channel.
    gamma_correct
        Linearise sRGB before weighting and re-encode afterwards, giving true
        relative luminance instead of luma.  Only meaningful for the weighted
        methods.
    keep_channel_dim
        Return ``(H, W, 1)`` instead of ``(H, W)``.
    alpha_background
        Passed to :func:`drop_alpha` for 4-channel input.
    preserve_dtype
        Cast the result back to the input dtype (rounding and clipping for
        integer images).  With ``False`` a ``float64`` array is returned.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` or ``(H, W, 1)`` grayscale image.

    Examples
    --------
    >>> rgb = np.zeros((1, 1, 3), dtype=np.uint8); rgb[0, 0] = (255, 0, 0)
    >>> int(to_grayscale(rgb, "bt601")[0, 0])
    76
    >>> int(to_grayscale(rgb, "average")[0, 0])
    85
    """
    arr = check_image(image, name="image")
    original_dtype = arr.dtype
    key = check_choice(method, GRAYSCALE_METHODS, name="method")
    key = _ALIASES.get(key, key)

    if arr.ndim == 3 and arr.shape[2] in (2, 4):
        arr = drop_alpha(arr, background=alpha_background)

    if arr.ndim == 2:
        gray = as_float_array(arr, name="image")
    elif arr.shape[2] == 1:
        gray = as_float_array(arr[..., 0], name="image")
    elif arr.shape[2] == 3:
        gray = _rgb_to_gray(arr, key, gamma_correct, original_dtype)
    else:
        raise ShapeError(
            f"Expected 1, 3 or 4 channels for grayscale conversion, got "
            f"{arr.shape[2]} (image shape {arr.shape})."
        )

    if keep_channel_dim:
        gray = gray[:, :, None]
    gray = np.ascontiguousarray(gray)
    return restore_dtype(gray, original_dtype) if preserve_dtype else gray


def _rgb_to_gray(
    rgb: np.ndarray, key: str, gamma_correct: bool, dtype: np.dtype
) -> np.ndarray:
    work = as_float_array(rgb, name="image")

    if key == "average":
        return work.mean(axis=2)
    if key == "lightness":
        return 0.5 * (work.max(axis=2) + work.min(axis=2))
    if key == "value":
        return work.max(axis=2)

    weights = np.asarray(GRAYSCALE_WEIGHTS[key], dtype=np.float64)

    if not gamma_correct:
        return work @ weights

    low, high = dtype_value_range(dtype)
    scale = high - low
    if scale <= 0:  # pragma: no cover - defensive
        raise ParameterError(f"Degenerate value range for dtype {dtype}.")
    unit = (work - low) / scale
    linear = srgb_to_linear(unit)
    luminance = linear @ weights
    return linear_to_srgb(luminance) * scale + low


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Broadcast a grayscale image to 3 identical channels.

    The inverse direction of :func:`to_grayscale`; handy for overlaying a
    colour annotation on a grayscale result, or for feeding a single-channel
    image to a model that expects RGB.
    """
    arr = check_image(image, name="image")
    if arr.ndim == 2:
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    if arr.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(arr, 3, axis=2))
    if arr.shape[2] == 3:
        return np.ascontiguousarray(arr)
    if arr.shape[2] in (2, 4):
        return np.ascontiguousarray(to_rgb(drop_alpha(arr)))
    raise ShapeError(f"Cannot interpret an image with {arr.shape[2]} channels as RGB.")
