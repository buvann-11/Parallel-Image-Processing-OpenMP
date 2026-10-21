"""Image I/O and synthetic test-pattern generators.

File I/O is delegated to Pillow, which is an *optional* dependency: the
numerical core of this package needs nothing but NumPy, and a missing optional
dependency raises a clear :class:`~imgproc.exceptions.DependencyError` rather
than an obscure ``ImportError`` halfway through a pipeline.

The synthetic generators exist so that the demo and the entire test-suite run
with **no image files at all**.  They are also better test material than a
photograph: a checkerboard has known edge positions, a ramp has a known
gradient, and a circle has known curvature, so the expected output of a filter
can be reasoned about rather than merely eyeballed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DependencyError, ParameterError
from .validation import check_image, dtype_value_range, restore_dtype

__all__ = [
    "load_image",
    "save_image",
    "checkerboard",
    "gradient_ramp",
    "circle",
    "sample_rgb_image",
]


def _require_pillow(feature: str) -> Any:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyError("pillow", feature) from exc
    return Image


def load_image(path: str | Path, *, mode: str | None = None) -> np.ndarray:
    """Read an image file into a ``uint8`` NumPy array.

    Parameters
    ----------
    path
        Path to a PNG/JPEG/BMP/TIFF/... file.
    mode
        Optional Pillow mode to convert to (``"L"`` for grayscale, ``"RGB"``,
        ``"RGBA"``).  ``None`` keeps the file's own mode.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` for grayscale or ``(H, W, C)`` for colour.

    Raises
    ------
    DependencyError
        If Pillow is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    """
    image_module = _require_pillow("reading image files")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No such image file: {file_path}")
    with image_module.open(file_path) as handle:
        if mode is not None:
            handle = handle.convert(mode)
        return np.asarray(handle).copy()


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write an array to an image file, converting to ``uint8`` if needed.

    Floating-point input is assumed to lie in ``[0, 1]`` and is scaled and
    clipped accordingly -- the same convention as
    :func:`imgproc.normalization.to_unit_range`.

    Returns
    -------
    pathlib.Path
        The path written to (parent directories are created as needed).
    """
    image_module = _require_pillow("writing image files")
    arr = check_image(image, name="image")

    if arr.dtype != np.uint8:
        low, high = dtype_value_range(arr.dtype)
        scaled = (np.asarray(arr, dtype=np.float64) - low) / (high - low)
        arr = restore_dtype(np.clip(scaled, 0.0, 1.0) * 255.0, np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    image_module.fromarray(arr).save(file_path)
    return file_path


def checkerboard(
    size: int = 64,
    square: int = 8,
    *,
    low: int = 0,
    high: int = 255,
    dtype: Any = np.uint8,
) -> np.ndarray:
    """A square checkerboard -- the classic aliasing and edge test pattern.

    Downscale one without anti-aliasing and the moiré is immediately visible;
    that contrast is exactly what the demo uses it for.

    >>> checkerboard(4, 2).tolist()
    [[0, 0, 255, 255], [0, 0, 255, 255], [255, 255, 0, 0], [255, 255, 0, 0]]
    """
    if size < 1 or square < 1:
        raise ParameterError(f"'size' and 'square' must be >= 1, got {size}, {square}.")
    ys = np.arange(size) // square
    xs = np.arange(size) // square
    pattern = (ys[:, None] + xs[None, :]) % 2
    return np.where(pattern == 0, low, high).astype(dtype)


def gradient_ramp(
    height: int = 64,
    width: int = 64,
    *,
    direction: str = "horizontal",
    dtype: Any = np.uint8,
) -> np.ndarray:
    """A linear intensity ramp.

    Useful for verifying that a filter preserves linear signals (a Gaussian
    should) and that a resize preserves the gradient (it should).
    """
    if height < 1 or width < 1:
        raise ParameterError(f"'height' and 'width' must be >= 1, got {height}, {width}.")
    low, high = dtype_value_range(dtype)
    if direction == "horizontal":
        ramp = np.linspace(low, high, width)[None, :].repeat(height, axis=0)
    elif direction == "vertical":
        ramp = np.linspace(low, high, height)[:, None].repeat(width, axis=1)
    elif direction == "diagonal":
        ys = np.linspace(0.0, 1.0, height)[:, None]
        xs = np.linspace(0.0, 1.0, width)[None, :]
        ramp = (ys + xs) / 2.0 * (high - low) + low
    else:
        raise ParameterError(
            f"'direction' must be 'horizontal', 'vertical' or 'diagonal', got {direction!r}."
        )
    return restore_dtype(ramp, dtype)


def circle(
    size: int = 64,
    radius: float | None = None,
    *,
    dtype: Any = np.uint8,
    soft: bool = True,
) -> np.ndarray:
    """A centred disc -- curved edges in every orientation at once.

    With ``soft=True`` the boundary is anti-aliased over one pixel, so the
    pattern does not itself contain aliasing artefacts that could be mistaken
    for a bug in the code under test.
    """
    if size < 1:
        raise ParameterError(f"'size' must be >= 1, got {size}.")
    if radius is None:
        radius = size / 3.0
    if radius <= 0:
        raise ParameterError(f"'radius' must be positive, got {radius}.")

    coords = np.arange(size) - (size - 1) / 2.0
    distance = np.hypot(coords[:, None], coords[None, :])
    if soft:
        mask = np.clip(radius + 0.5 - distance, 0.0, 1.0)
    else:
        mask = (distance <= radius).astype(np.float64)
    low, high = dtype_value_range(dtype)
    return restore_dtype(mask * (high - low) + low, dtype)


def sample_rgb_image(size: int = 64, *, dtype: Any = np.uint8) -> np.ndarray:
    """A deterministic RGB test image combining a ramp, a disc and a checkerboard.

    Each channel carries different structure, so channel-mixing bugs (a very
    common class of grayscale/normalisation error) show up immediately.

    >>> sample_rgb_image(8).shape
    (8, 8, 3)
    """
    red = gradient_ramp(size, size, direction="horizontal", dtype=dtype)
    green = circle(size, size / 3.0, dtype=dtype)
    blue = checkerboard(size, max(1, size // 8), dtype=dtype)
    return np.ascontiguousarray(np.stack([red, green, blue], axis=-1))
