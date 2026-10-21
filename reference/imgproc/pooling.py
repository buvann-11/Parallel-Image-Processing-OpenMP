r"""Pooling layers for CNNs -- max, average and global, with backward passes.

What pooling is for
-------------------
Pooling downsamples a feature map by summarising each local neighbourhood with
a single number.  It does three useful things at once:

* **Reduces resolution** (and therefore compute and memory) without parameters.
* **Widens the receptive field** of later layers -- after one 2x2 pool, each
  subsequent 3x3 kernel effectively sees a 6x6 patch of the original image.
* **Buys local translation tolerance**: shifting a strong activation by one
  pixel inside a window leaves a max-pooled output unchanged.

Max vs. average
---------------
.. math::

    y_{\max}[p, q]  &= \max_{(i,j)\in\mathcal{W}(p,q)} x[i, j] \\
    y_{\mathrm{avg}}[p, q] &= \frac{1}{|\mathcal{W}|}
                              \sum_{(i,j)\in\mathcal{W}(p,q)} x[i, j]

Max pooling keeps the strongest evidence of a feature and routes the gradient
to exactly one input per window -- it behaves like a *selector*.  Average
pooling keeps the mean response and spreads the gradient evenly -- it behaves
like a *smoother*.  Modern architectures use max pooling inside the trunk and a
single **global average pool** before the classifier, which is why all three
are provided here.

Conventions
-----------
Geometry (``stride``, ``padding``, ``dilation``, ``ceil_mode``) follows
PyTorch exactly, including the rule that ``padding`` may not exceed half the
kernel size -- otherwise a window could contain nothing but padding and the
result would be meaningless.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._im2col import col2im, im2col
from .exceptions import ParameterError, ShapeError
from .padding import PaddingSpec, conv_output_size, resolve_padding
from .validation import (
    as_float_array,
    check_ndim,
    check_positive_int,
    check_tensor_nchw,
)

__all__ = [
    "pool_output_shape",
    "max_pool2d",
    "max_pool2d_backward",
    "avg_pool2d",
    "avg_pool2d_backward",
    "global_max_pool2d",
    "global_avg_pool2d",
]


def _pool_geometry(
    x: np.ndarray,
    kernel_size,
    stride,
    padding: PaddingSpec,
    dilation,
    ceil_mode: bool,
):
    """Validate inputs and derive every geometric quantity a pool needs."""
    tensor = check_tensor_nchw(x, name="x")
    kh, kw = check_positive_int(kernel_size, name="kernel_size")
    if stride is None:
        sh, sw = kh, kw  # PyTorch default: non-overlapping windows.
    else:
        sh, sw = check_positive_int(stride, name="stride")
    dh, dw = check_positive_int(dilation, name="dilation")

    _, _, h, w = tensor.shape
    (top, bottom), (left, right) = resolve_padding(
        padding,
        kernel_size=(kh, kw),
        input_size=(h, w),
        stride=(sh, sw),
        dilation=(dh, dw),
    )

    # Guard against windows made entirely of padding.
    if max(top, bottom) * 2 > kh or max(left, right) * 2 > kw:
        raise ParameterError(
            f"Pooling padding must not exceed half the kernel size: got padding "
            f"(({top}, {bottom}), ({left}, {right})) with kernel_size {(kh, kw)}. "
            "Otherwise an output position could see padding only."
        )

    out_h = conv_output_size(
        h, kh, stride=sh, padding_before=top, padding_after=bottom,
        dilation=dh, ceil_mode=ceil_mode,
    )
    out_w = conv_output_size(
        w, kw, stride=sw, padding_before=left, padding_after=right,
        dilation=dw, ceil_mode=ceil_mode,
    )

    # ceil_mode can ask for a final window that runs past the padded tensor;
    # the overhang is treated as ignorable, exactly as PyTorch does.
    eff_kh = dh * (kh - 1) + 1
    eff_kw = dw * (kw - 1) + 1
    extra_h = max(0, (out_h - 1) * sh + eff_kh - (h + top + bottom))
    extra_w = max(0, (out_w - 1) * sw + eff_kw - (w + left + right))

    return (
        tensor,
        (kh, kw),
        (sh, sw),
        (dh, dw),
        (top, bottom, left, right),
        (extra_h, extra_w),
        (out_h, out_w),
    )


def pool_output_shape(
    input_hw: Sequence[int],
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] | None = None,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    ceil_mode: bool = False,
) -> tuple[int, int]:
    """Output spatial size of a pooling operation.

    Examples
    --------
    >>> pool_output_shape((32, 32), 2)
    (16, 16)
    >>> pool_output_shape((7, 7), 2, ceil_mode=True)
    (4, 4)
    >>> pool_output_shape((7, 7), 2)
    (3, 3)
    """
    ih, iw = check_positive_int(input_hw, name="input_hw")
    dummy = np.zeros((1, 1, ih, iw))
    *_, out_hw = _pool_geometry(dummy, kernel_size, stride, padding, dilation, ceil_mode)
    return out_hw


def max_pool2d(
    x: np.ndarray,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] | None = None,
    *,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """2-D max pooling over an ``(N, C, H, W)`` tensor.

    Parameters
    ----------
    x
        Input tensor, ``NCHW``.
    kernel_size
        Window size, ``int`` or ``(kh, kw)``.
    stride
        Window step.  Defaults to ``kernel_size`` (non-overlapping windows).
    padding
        Implicit padding.  Pooling pads with :math:`-\\infty` rather than 0, so
        padding can never win a maximum -- padding with 0 would be a bug for
        feature maps that contain negative activations.
    dilation
        Spacing between window elements.
    ceil_mode
        Round the output size up instead of down.
    return_indices
        Also return the argmax positions, needed by
        :func:`max_pool2d_backward` and by unpooling layers.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Pooled output ``(N, C, OH, OW)``; with ``return_indices=True`` also an
        integer array of the same shape holding, for each output, the flat
        index ``row * W + col`` of the winning element **within its own
        ``(H, W)`` plane** -- the same convention PyTorch uses.

    Examples
    --------
    >>> x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    >>> max_pool2d(x, 2)[0, 0]
    array([[ 5.,  7.],
           [13., 15.]])
    """
    (
        tensor,
        (kh, kw),
        (sh, sw),
        (dh, dw),
        (top, bottom, left, right),
        (extra_h, extra_w),
        (out_h, out_w),
    ) = _pool_geometry(x, kernel_size, stride, padding, dilation, ceil_mode)

    n, c, h, w = tensor.shape
    pad_bottom = bottom + extra_h
    pad_right = right + extra_w

    padded = np.full(
        (n, c, h + top + pad_bottom, w + left + pad_right), -np.inf, dtype=tensor.dtype
    )
    padded[:, :, top : top + h, left : left + w] = tensor

    cols = im2col(padded, (kh, kw), (sh, sw), (dh, dw), (out_h, out_w))
    flat = cols.reshape(n, c, kh * kw, out_h, out_w)

    arg = np.argmax(flat, axis=2)
    out = flat.max(axis=2)

    if not np.all(np.isfinite(out)):  # pragma: no cover - guarded by _pool_geometry
        raise ParameterError(
            "A pooling window contained only padding; reduce the padding or "
            "increase the kernel size."
        )

    if not return_indices:
        return np.ascontiguousarray(out)

    # Decompose the within-window argmax into absolute input coordinates.
    tap_i, tap_j = np.divmod(arg, kw)
    p = np.arange(out_h).reshape(1, 1, out_h, 1)
    q = np.arange(out_w).reshape(1, 1, 1, out_w)
    rows = p * sh + tap_i * dh - top
    cols_idx = q * sw + tap_j * dw - left
    indices = (rows * w + cols_idx).astype(np.intp)
    return np.ascontiguousarray(out), np.ascontiguousarray(indices)


def max_pool2d_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    input_shape: Sequence[int],
) -> np.ndarray:
    """Back-propagate through :func:`max_pool2d`.

    Max pooling routes each output's gradient to the single input that produced
    it -- every other element in the window receives nothing.  When windows
    overlap (``stride < kernel_size``) an input can win several windows, and the
    contributions are summed.

    Parameters
    ----------
    grad_output
        Gradient w.r.t. the pooled output, shape ``(N, C, OH, OW)``.
    indices
        The index array returned by ``max_pool2d(..., return_indices=True)``.
    input_shape
        Shape ``(N, C, H, W)`` of the original input.

    Returns
    -------
    numpy.ndarray
        Gradient w.r.t. the input, shape ``input_shape``.
    """
    dout = as_float_array(grad_output, name="grad_output")
    check_ndim(dout, 4, name="grad_output")
    idx = np.asarray(indices)
    if idx.shape != dout.shape:
        raise ShapeError(
            f"'indices' shape {idx.shape} must match 'grad_output' shape {dout.shape}."
        )
    if not np.issubdtype(idx.dtype, np.integer):
        raise ShapeError(f"'indices' must be an integer array, got dtype {idx.dtype}.")

    shape = tuple(int(s) for s in input_shape)
    if len(shape) != 4:
        raise ShapeError(f"'input_shape' must have 4 entries, got {input_shape!r}.")
    n, c, h, w = shape
    if dout.shape[:2] != (n, c):
        raise ShapeError(
            f"'grad_output' has {dout.shape[:2]} (N, C) but 'input_shape' says {(n, c)}."
        )
    plane = h * w
    if idx.size and (idx.min() < 0 or idx.max() >= plane):
        raise ShapeError(
            f"'indices' contains values outside [0, {plane - 1}] for an input "
            f"plane of size {(h, w)}."
        )

    dx = np.zeros(n * c * plane, dtype=dout.dtype)
    base = (np.arange(n * c, dtype=np.intp) * plane).reshape(n, c, 1, 1)
    np.add.at(dx, (base + idx).ravel(), dout.ravel())
    return dx.reshape(shape)


def _avg_divisors(
    shape_padded: tuple[int, int, int, int],
    input_hw: tuple[int, int],
    pads: tuple[int, int, int, int],
    extra: tuple[int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    out_hw: tuple[int, int],
    count_include_pad: bool,
) -> np.ndarray:
    """Per-output-position window sizes, honouring ``count_include_pad``.

    Built by pooling a 0/1 mask with the *same* geometry, which keeps the rule
    in one place instead of scattering ``if`` statements through the maths.
    """
    top, bottom, left, right = pads
    extra_h, extra_w = extra
    h, w = input_hw
    mask = np.zeros((1, 1) + shape_padded[2:], dtype=np.float64)
    if count_include_pad:
        # Explicit padding counts; only the ceil_mode overhang does not.
        mask[:, :, : mask.shape[2] - extra_h, : mask.shape[3] - extra_w] = 1.0
    else:
        mask[:, :, top : top + h, left : left + w] = 1.0
    mask_cols = im2col(mask, kernel, stride, dilation, out_hw)
    return mask_cols.sum(axis=(2, 3))  # (1, 1, OH, OW)


def avg_pool2d(
    x: np.ndarray,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] | None = None,
    *,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> np.ndarray:
    """2-D average pooling over an ``(N, C, H, W)`` tensor.

    Parameters
    ----------
    count_include_pad
        ``True`` (default, PyTorch's default) divides by the full window area,
        so windows overlapping the border are pulled toward zero.  ``False``
        divides by the number of *real* input elements in the window, which
        keeps border averages unbiased -- usually what you want for image
        processing rather than for a CNN trunk.

    Returns
    -------
    numpy.ndarray
        Pooled output ``(N, C, OH, OW)``.

    Examples
    --------
    >>> x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    >>> avg_pool2d(x, 2)[0, 0]
    array([[ 2.5,  4.5],
           [10.5, 12.5]])
    """
    (
        tensor,
        (kh, kw),
        (sh, sw),
        (dh, dw),
        (top, bottom, left, right),
        (extra_h, extra_w),
        (out_h, out_w),
    ) = _pool_geometry(x, kernel_size, stride, padding, dilation, ceil_mode)

    n, c, h, w = tensor.shape
    padded = np.zeros(
        (n, c, h + top + bottom + extra_h, w + left + right + extra_w), dtype=tensor.dtype
    )
    padded[:, :, top : top + h, left : left + w] = tensor

    cols = im2col(padded, (kh, kw), (sh, sw), (dh, dw), (out_h, out_w))
    totals = cols.sum(axis=(2, 3))

    divisors = _avg_divisors(
        padded.shape, (h, w), (top, bottom, left, right), (extra_h, extra_w),
        (kh, kw), (sh, sw), (dh, dw), (out_h, out_w), count_include_pad,
    )
    return np.ascontiguousarray(totals / divisors)


def avg_pool2d_backward(
    grad_output: np.ndarray,
    input_shape: Sequence[int],
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] | None = None,
    *,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> np.ndarray:
    """Back-propagate through :func:`avg_pool2d`.

    Average pooling is linear, so each output distributes its gradient equally
    over the window that produced it -- scaled by the same divisor the forward
    pass used.  No cached activations are needed, only the geometry.
    """
    dout = as_float_array(grad_output, name="grad_output")
    check_ndim(dout, 4, name="grad_output")

    shape = tuple(int(s) for s in input_shape)
    if len(shape) != 4:
        raise ShapeError(f"'input_shape' must have 4 entries, got {input_shape!r}.")

    (
        _,
        (kh, kw),
        (sh, sw),
        (dh, dw),
        (top, bottom, left, right),
        (extra_h, extra_w),
        (out_h, out_w),
    ) = _pool_geometry(
        np.zeros(shape), kernel_size, stride, padding, dilation, ceil_mode
    )

    n, c, h, w = shape
    expected = (n, c, out_h, out_w)
    if dout.shape != expected:
        raise ShapeError(
            f"'grad_output' must have shape {expected} for this pooling "
            f"configuration, got {dout.shape}."
        )

    padded_shape = (n, c, h + top + bottom + extra_h, w + left + right + extra_w)
    divisors = _avg_divisors(
        padded_shape, (h, w), (top, bottom, left, right), (extra_h, extra_w),
        (kh, kw), (sh, sw), (dh, dw), (out_h, out_w), count_include_pad,
    )

    scaled = dout / divisors
    dcols = np.broadcast_to(
        scaled[:, :, None, None, :, :], (n, c, kh, kw, out_h, out_w)
    )
    dpadded = col2im(np.ascontiguousarray(dcols), padded_shape, (kh, kw), (sh, sw), (dh, dw))
    return np.ascontiguousarray(dpadded[:, :, top : top + h, left : left + w])


def global_max_pool2d(x: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    """Collapse each channel to its maximum: ``(N, C, H, W) -> (N, C)``.

    With ``keepdims=True`` the result is ``(N, C, 1, 1)``, which is the shape a
    ``1x1`` convolution head expects.
    """
    tensor = check_tensor_nchw(x, name="x")
    out = tensor.max(axis=(2, 3), keepdims=keepdims)
    return np.ascontiguousarray(out)


def global_avg_pool2d(x: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    """Collapse each channel to its mean: ``(N, C, H, W) -> (N, C)``.

    Global average pooling replaced the large fully-connected head of early
    CNNs (Network-in-Network, 2013): it has zero parameters, is far less prone
    to overfitting, and accepts any input resolution.
    """
    tensor = check_tensor_nchw(x, name="x")
    out = tensor.mean(axis=(2, 3), keepdims=keepdims)
    return np.ascontiguousarray(out)
