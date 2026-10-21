r"""A small library of classical 2-D filter kernels.

These are the *hand-designed* ancestors of the filters a CNN learns.  Showing
them side by side with a learned convolution makes the point that the first
layer of a trained network reliably rediscovers exactly these operators --
oriented edges, blobs and colour opponency.

Every function returns a ``float64`` 2-D array ready for
:func:`imgproc.convolution.convolve_image`.

Normalisation convention
------------------------
* **Smoothing** kernels sum to 1, so flat regions keep their brightness.
* **Derivative** kernels sum to 0, so flat regions map to 0 and only changes
  survive.
"""

from __future__ import annotations

import numpy as np

from .exceptions import ParameterError

__all__ = [
    "identity",
    "box_blur",
    "gaussian_blur",
    "gaussian_kernel_1d",
    "sobel",
    "prewitt",
    "scharr",
    "laplacian",
    "sharpen",
    "emboss",
    "KERNELS",
]


def _check_odd_size(size: int, *, name: str = "size") -> int:
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
        raise ParameterError(f"{name!r} must be an int, got {type(size).__name__}.")
    size = int(size)
    if size < 1:
        raise ParameterError(f"{name!r} must be >= 1, got {size}.")
    if size % 2 == 0:
        raise ParameterError(
            f"{name!r} must be odd so the kernel has a well-defined centre, got {size}."
        )
    return size


def identity(size: int = 3) -> np.ndarray:
    """Kernel that leaves an image unchanged (a single 1 at the centre)."""
    size = _check_odd_size(size)
    k = np.zeros((size, size), dtype=np.float64)
    k[size // 2, size // 2] = 1.0
    return k


def box_blur(size: int = 3) -> np.ndarray:
    """Uniform ``size x size`` averaging filter.

    The cheapest smoother, but its frequency response has ringing side-lobes,
    which is why :func:`gaussian_blur` is preferred when quality matters.

    >>> float(box_blur(3).sum())
    1.0
    """
    size = _check_odd_size(size)
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel_1d(sigma: float = 1.0, size: int | None = None) -> np.ndarray:
    r"""Normalised 1-D Gaussian :math:`g[n] \propto e^{-n^2 / 2\sigma^2}`.

    A 2-D Gaussian is *separable*: filtering with a ``k x k`` Gaussian is the
    same as two 1-D passes, which reduces the cost per pixel from
    :math:`O(k^2)` to :math:`O(2k)`.

    Parameters
    ----------
    sigma
        Standard deviation in pixels.  Must be positive.
    size
        Kernel length.  Defaults to ``2 * ceil(3 * sigma) + 1``, which captures
        99.7% of the Gaussian's mass.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ParameterError(f"'sigma' must be a positive finite number, got {sigma!r}.")
    if size is None:
        size = int(2 * np.ceil(3.0 * sigma) + 1)
    size = _check_odd_size(size)
    radius = size // 2
    n = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(n**2) / (2.0 * sigma**2))
    return g / g.sum()


def gaussian_blur(sigma: float = 1.0, size: int | None = None) -> np.ndarray:
    """Separable 2-D Gaussian kernel, normalised to sum to 1.

    >>> float(np.round(gaussian_blur(1.0, 3).sum(), 12))
    1.0
    """
    g = gaussian_kernel_1d(sigma, size)
    return np.outer(g, g)


def sobel(axis: str = "x") -> np.ndarray:
    """Sobel derivative kernel.

    ``axis="x"`` responds to *vertical* edges (a horizontal intensity change);
    ``axis="y"`` responds to horizontal edges.  Sobel is the outer product of a
    ``[1, 2, 1]`` smoother and a ``[-1, 0, 1]`` difference -- it differentiates
    in one direction while denoising in the other.
    """
    smooth = np.array([1.0, 2.0, 1.0])
    diff = np.array([-1.0, 0.0, 1.0])
    if axis.lower() == "x":
        return np.outer(smooth, diff)
    if axis.lower() == "y":
        return np.outer(diff, smooth)
    raise ParameterError(f"'axis' must be 'x' or 'y', got {axis!r}.")


def prewitt(axis: str = "x") -> np.ndarray:
    """Prewitt derivative kernel -- Sobel with a box smoother instead of ``[1,2,1]``."""
    smooth = np.array([1.0, 1.0, 1.0])
    diff = np.array([-1.0, 0.0, 1.0])
    if axis.lower() == "x":
        return np.outer(smooth, diff)
    if axis.lower() == "y":
        return np.outer(diff, smooth)
    raise ParameterError(f"'axis' must be 'x' or 'y', got {axis!r}.")


def scharr(axis: str = "x") -> np.ndarray:
    """Scharr derivative kernel -- weights optimised for rotational symmetry."""
    smooth = np.array([3.0, 10.0, 3.0])
    diff = np.array([-1.0, 0.0, 1.0])
    if axis.lower() == "x":
        return np.outer(smooth, diff)
    if axis.lower() == "y":
        return np.outer(diff, smooth)
    raise ParameterError(f"'axis' must be 'x' or 'y', got {axis!r}.")


def laplacian(diagonal: bool = False) -> np.ndarray:
    """Discrete Laplacian :math:`\\nabla^2` -- an isotropic second-derivative filter.

    ``diagonal=True`` uses the 8-neighbour variant, which responds to diagonal
    edges as strongly as to axis-aligned ones.
    """
    if diagonal:
        return np.array(
            [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64
        )
    return np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )


def sharpen(amount: float = 1.0) -> np.ndarray:
    """Unsharp-mask kernel: ``identity - amount * laplacian``.

    Boosts local contrast by adding back a scaled copy of the image's own
    second derivative.  ``amount=0`` returns the identity.
    """
    if not np.isfinite(amount):
        raise ParameterError(f"'amount' must be finite, got {amount!r}.")
    return identity(3) - float(amount) * laplacian()


def emboss() -> np.ndarray:
    """Directional emboss filter -- a diagonal derivative plus a DC offset of 1."""
    return np.array(
        [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64
    )


#: Name -> zero-argument factory, for the CLI and the demo script.
KERNELS: dict[str, "object"] = {
    "identity": identity,
    "box_blur": box_blur,
    "gaussian_blur": gaussian_blur,
    "sobel_x": lambda: sobel("x"),
    "sobel_y": lambda: sobel("y"),
    "prewitt_x": lambda: prewitt("x"),
    "prewitt_y": lambda: prewitt("y"),
    "scharr_x": lambda: scharr("x"),
    "scharr_y": lambda: scharr("y"),
    "laplacian": laplacian,
    "sharpen": sharpen,
    "emboss": emboss,
}
