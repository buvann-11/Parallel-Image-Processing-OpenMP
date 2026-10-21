r"""Intensity normalisation -- the step that makes training actually converge.

Why normalise at all?
---------------------
Raw pixels live in ``[0, 255]``.  Feed them to a network and the first layer's
pre-activations have a large, input-dependent scale and a large positive mean;
gradients are correspondingly large and badly conditioned, saturating
activations and forcing tiny learning rates.  Normalisation puts every input
feature on a comparable, roughly zero-centred scale, which is why it is the
first transform in essentially every vision pipeline.

The three schemes implemented here
----------------------------------
**Min-max** rescales the observed range onto a target interval:

.. math:: x' = a + (b - a)\,\frac{x - \min x}{\max x - \min x}

Bounded output, but one outlier pixel compresses everything else.

**Standardisation (z-score)** removes the mean and divides by the standard
deviation:

.. math:: x' = \frac{x - \mu}{\sigma}

Unbounded output, but robust to the overall exposure of the image.  This is
what ``torchvision.transforms.Normalize`` does with the fixed ImageNet
statistics :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD`.

**Histogram equalisation** reshapes the intensity *distribution* itself so
that it is approximately uniform, maximising global contrast.

Per-image or per-dataset?
-------------------------
``per_channel=True`` computes the statistics separately for R, G and B, which
also removes colour casts.  For inference you normally want *fixed* dataset
statistics (pass ``mean``/``std`` explicitly) so that the same input always maps
to the same tensor -- computing them per image makes the mapping depend on the
image's own exposure.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import ParameterError, ShapeError
from .validation import as_float_array, check_image, dtype_value_range, restore_dtype

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "to_unit_range",
    "from_unit_range",
    "min_max_normalize",
    "standardize",
    "normalize",
    "denormalize",
    "contrast_stretch",
    "histogram_equalize",
]

#: Channel means of the ImageNet training set, for images already in ``[0, 1]``.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
#: Channel standard deviations of the ImageNet training set.
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_EPS = 1e-12


def _reduction_axes(array: np.ndarray, per_channel: bool, channel_axis: int) -> tuple[int, ...] | None:
    """Axes to reduce over: everything, or everything except the channel axis."""
    if not per_channel:
        return None
    if array.ndim < 2:  # pragma: no cover - guarded by check_image
        raise ShapeError("per_channel normalisation needs at least a 2-D array.")
    axis = channel_axis % array.ndim
    return tuple(i for i in range(array.ndim) if i != axis)


def to_unit_range(image: np.ndarray) -> np.ndarray:
    """Map an image onto ``[0, 1]`` using its dtype's *canonical* range.

    This is a pure change of units -- ``uint8`` 255 becomes 1.0 -- not a
    contrast adjustment: two images normalised this way remain comparable.
    Contrast the behaviour with :func:`min_max_normalize`, which uses the
    *observed* range and therefore alters relative brightness.

    >>> to_unit_range(np.array([[0, 128, 255]], dtype=np.uint8)).round(4).tolist()
    [[0.0, 0.502, 1.0]]
    """
    arr = check_image(image, name="image")
    low, high = dtype_value_range(arr.dtype)
    work = as_float_array(arr, name="image")
    return (work - low) / (high - low)


def from_unit_range(image: np.ndarray, dtype: np.dtype | type = np.uint8) -> np.ndarray:
    """Inverse of :func:`to_unit_range`: map ``[0, 1]`` back onto ``dtype``'s range.

    Values outside ``[0, 1]`` are clipped, because a display-ready image cannot
    represent them.
    """
    work = as_float_array(image, name="image")
    low, high = dtype_value_range(dtype)
    scaled = np.clip(work, 0.0, 1.0) * (high - low) + low
    return restore_dtype(scaled, dtype)


def min_max_normalize(
    image: np.ndarray,
    feature_range: Sequence[float] = (0.0, 1.0),
    *,
    per_channel: bool = False,
    channel_axis: int = -1,
    eps: float = _EPS,
) -> np.ndarray:
    """Rescale the *observed* value range onto ``feature_range``.

    Parameters
    ----------
    feature_range
        Target ``(low, high)``.  ``(0, 1)`` by default; ``(-1, 1)`` is the other
        common choice (it pairs well with ``tanh`` outputs).
    per_channel
        Compute min and max separately for each channel.
    channel_axis
        Which axis holds the channels.  ``-1`` for ``HWC`` images, ``1`` for
        ``NCHW`` tensors.
    eps
        Guard for constant inputs.  When ``max == min`` the result is a
        constant image at ``feature_range[0]`` -- a defined, documented answer
        rather than a ``NaN``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array in ``feature_range``.

    Examples
    --------
    >>> min_max_normalize(np.array([[0., 5., 10.]])).tolist()
    [[0.0, 0.5, 1.0]]
    """
    arr = check_image(image, name="image")
    work = as_float_array(arr, name="image")

    lo_hi = tuple(float(v) for v in feature_range)
    if len(lo_hi) != 2:
        raise ParameterError(f"'feature_range' must have 2 entries, got {feature_range!r}.")
    low, high = lo_hi
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ParameterError(f"'feature_range' must be finite, got {feature_range!r}.")
    if high <= low:
        raise ParameterError(f"'feature_range' must satisfy high > low, got {feature_range!r}.")
    if eps <= 0:
        raise ParameterError(f"'eps' must be positive, got {eps!r}.")

    axes = _reduction_axes(work, per_channel, channel_axis)
    data_min = work.min(axis=axes, keepdims=True)
    data_max = work.max(axis=axes, keepdims=True)
    span = data_max - data_min
    # Where the span is degenerate, map everything to `low` instead of dividing.
    safe_span = np.where(span < eps, 1.0, span)
    unit = np.where(span < eps, 0.0, (work - data_min) / safe_span)
    return np.ascontiguousarray(unit * (high - low) + low)


def standardize(
    image: np.ndarray,
    *,
    mean: float | Sequence[float] | None = None,
    std: float | Sequence[float] | None = None,
    per_channel: bool = False,
    channel_axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """Zero-mean, unit-variance normalisation (z-score).

    Parameters
    ----------
    mean, std
        Fixed statistics to use.  When ``None`` (default) they are estimated
        from the image itself.  Supply them explicitly at inference time so
        that preprocessing is deterministic.
    per_channel
        Estimate (or broadcast) statistics per channel.
    eps
        Added to ``std`` before dividing, so a constant channel yields zeros
        rather than ``inf``.

    Examples
    --------
    >>> out = standardize(np.array([[1., 2., 3., 4.]]))
    >>> bool(abs(out.mean()) < 1e-12), bool(abs(out.std() - 1.0) < 1e-6)
    (True, True)
    """
    arr = check_image(image, name="image")
    work = as_float_array(arr, name="image")
    if eps <= 0:
        raise ParameterError(f"'eps' must be positive, got {eps!r}.")

    axes = _reduction_axes(work, per_channel, channel_axis)

    if mean is None:
        mu = work.mean(axis=axes, keepdims=True)
    else:
        mu = _broadcastable_stat(mean, work, per_channel, channel_axis, name="mean")
    if std is None:
        sigma = work.std(axis=axes, keepdims=True)
    else:
        sigma = _broadcastable_stat(std, work, per_channel, channel_axis, name="std")
        if np.any(sigma <= 0):
            raise ParameterError(f"'std' entries must be positive, got {std!r}.")

    return np.ascontiguousarray((work - mu) / (sigma + eps))


def _broadcastable_stat(
    value: float | Sequence[float],
    array: np.ndarray,
    per_channel: bool,
    channel_axis: int,
    *,
    name: str,
) -> np.ndarray:
    """Reshape a scalar or per-channel statistic so it broadcasts against ``array``."""
    stat = as_float_array(value, name=name).reshape(-1)
    if stat.size == 1:
        return stat.reshape((1,) * array.ndim)
    if not per_channel and array.ndim < 3:
        raise ParameterError(
            f"A vector {name!r} was given but the image has no channel axis."
        )
    axis = channel_axis % array.ndim
    channels = array.shape[axis]
    if stat.size != channels:
        raise ShapeError(
            f"{name!r} has {stat.size} entries but the image has {channels} "
            f"channels along axis {axis}."
        )
    shape = [1] * array.ndim
    shape[axis] = channels
    return stat.reshape(shape)


def normalize(
    image: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    *,
    channel_axis: int = -1,
) -> np.ndarray:
    """``torchvision``-style normalisation with **fixed** per-channel statistics.

    Expects an image already scaled to ``[0, 1]`` (see :func:`to_unit_range`).
    Defaults are the ImageNet statistics every pretrained backbone was trained
    with -- use them whenever you feed a pretrained model.

    >>> x = np.full((1, 1, 3), 0.485)
    >>> float(round(normalize(x)[0, 0, 0], 6))
    0.0
    """
    return standardize(
        image, mean=mean, std=std, per_channel=True, channel_axis=channel_axis, eps=1e-12
    )


def denormalize(
    image: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    *,
    channel_axis: int = -1,
) -> np.ndarray:
    """Invert :func:`normalize` -- essential for *looking at* a normalised tensor."""
    work = as_float_array(image, name="image")
    mu = _broadcastable_stat(mean, work, True, channel_axis, name="mean")
    sigma = _broadcastable_stat(std, work, True, channel_axis, name="std")
    return np.ascontiguousarray(work * sigma + mu)


def contrast_stretch(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    *,
    per_channel: bool = False,
    channel_axis: int = -1,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Percentile-based min-max stretch -- min-max normalisation, outlier-proofed.

    A single hot pixel ruins :func:`min_max_normalize`; clipping at the 2nd and
    98th percentiles first makes the stretch depend on the bulk of the
    distribution instead.

    Returns
    -------
    numpy.ndarray
        Stretched image, cast back to the input dtype when
        ``preserve_dtype=True``.
    """
    arr = check_image(image, name="image")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ParameterError(
            "Percentiles must satisfy 0 <= low < high <= 100, got "
            f"({low_percentile}, {high_percentile})."
        )
    work = as_float_array(arr, name="image")
    axes = _reduction_axes(work, per_channel, channel_axis)

    lo = np.percentile(work, low_percentile, axis=axes, keepdims=True)
    hi = np.percentile(work, high_percentile, axis=axes, keepdims=True)
    span = hi - lo
    safe = np.where(span < _EPS, 1.0, span)
    unit = np.clip(np.where(span < _EPS, 0.0, (work - lo) / safe), 0.0, 1.0)

    if preserve_dtype:
        out_low, out_high = dtype_value_range(arr.dtype)
        return restore_dtype(unit * (out_high - out_low) + out_low, arr.dtype)
    return np.ascontiguousarray(unit)


def histogram_equalize(
    image: np.ndarray,
    *,
    nbins: int = 256,
    per_channel: bool = False,
    preserve_dtype: bool = True,
) -> np.ndarray:
    r"""Global histogram equalisation.

    The cumulative distribution function of the intensities is used as the
    transfer function:

    .. math:: x' = \mathrm{cdf}(x)

    Since the CDF of a random variable is uniformly distributed, the output
    histogram is approximately flat -- every available intensity level is used
    equally often, which maximises global contrast.  This is the standard
    remedy for under-exposed or hazy images.

    Parameters
    ----------
    nbins
        Number of histogram bins used to estimate the CDF.
    per_channel
        Equalise each channel independently.  Beware: on a colour image this
        shifts hues, because the channels are stretched by different amounts.
        Equalising a luminance channel only is usually preferable.

    Returns
    -------
    numpy.ndarray
        Equalised image.
    """
    arr = check_image(image, name="image")
    if nbins < 2:
        raise ParameterError(f"'nbins' must be >= 2, got {nbins}.")
    work = as_float_array(arr, name="image")

    def _equalize_plane(plane: np.ndarray) -> np.ndarray:
        lo, hi = float(plane.min()), float(plane.max())
        if hi - lo < _EPS:
            return np.zeros_like(plane)
        hist, edges = np.histogram(plane, bins=nbins, range=(lo, hi))
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1]
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(plane, centers, cdf)

    if per_channel and work.ndim == 3:
        unit = np.stack(
            [_equalize_plane(work[..., c]) for c in range(work.shape[2])], axis=-1
        )
    else:
        unit = _equalize_plane(work)

    if preserve_dtype:
        out_low, out_high = dtype_value_range(arr.dtype)
        return restore_dtype(unit * (out_high - out_low) + out_low, arr.dtype)
    return np.ascontiguousarray(unit)
