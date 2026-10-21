"""``im2col`` / ``col2im``: the workhorse behind fast CPU convolution.

A direct seven-loop convolution is unusable in Python.  The standard trick --
used by Caffe, cuDNN and every practical CPU backend -- is to *lower* the
convolution to a single matrix multiplication:

1. **im2col** copies every sliding window of the input into its own column.
   An ``(N, C, H, W)`` tensor becomes ``(N, C, kh, kw, OH, OW)``.
2. The convolution is then ``weight_matrix @ column_matrix`` -- one BLAS call,
   which NumPy dispatches to a tuned, multi-threaded GEMM kernel.
3. **col2im** is the adjoint: it scatters gradients from column space back to
   image space, *accumulating* wherever windows overlapped.

Cost of the trick: the column buffer is ``kh * kw`` times larger than the
input.  That memory-for-speed trade is the reason im2col-based convolution is
one to two orders of magnitude faster than nested Python loops.

These helpers are private; they are exercised indirectly through
:mod:`imgproc.convolution` and :mod:`imgproc.pooling`, and directly in the
test-suite.
"""

from __future__ import annotations

import numpy as np

__all__ = ["im2col", "col2im"]


def im2col(
    x_padded: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Extract every sliding window of ``x_padded`` into an explicit buffer.

    Parameters
    ----------
    x_padded
        Already-padded tensor of shape ``(N, C, Hp, Wp)``.
    kernel_size, stride, dilation
        ``(height, width)`` pairs.
    out_hw
        ``(OH, OW)`` -- the output spatial size, pre-computed by the caller.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, C, kh, kw, OH, OW)``.  Element ``[n, c, i, j, p, q]`` is
        the input pixel that the kernel tap ``(i, j)`` sees when the window is
        centred on output position ``(p, q)``.

    Notes
    -----
    The loop runs ``kh * kw`` times -- a *constant* independent of the image
    size -- and each iteration is a single strided NumPy copy.
    """
    n, c = x_padded.shape[0], x_padded.shape[1]
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    out_h, out_w = out_hw

    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x_padded.dtype)
    for i in range(kh):
        i_start = i * dh
        i_stop = i_start + sh * out_h
        for j in range(kw):
            j_start = j * dw
            j_stop = j_start + sw * out_w
            cols[:, :, i, j, :, :] = x_padded[
                :, :, i_start:i_stop:sh, j_start:j_stop:sw
            ]
    return cols


def col2im(
    cols: np.ndarray,
    padded_shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> np.ndarray:
    """Adjoint of :func:`im2col` -- scatter-add columns back to image space.

    Overlapping windows are **summed**, which is precisely what the chain rule
    requires: an input pixel seen by several windows receives the sum of the
    gradients flowing through each of them.

    Parameters
    ----------
    cols
        Buffer of shape ``(N, C, kh, kw, OH, OW)``.
    padded_shape
        Shape ``(N, C, Hp, Wp)`` of the padded tensor the columns came from.

    Returns
    -------
    numpy.ndarray
        Tensor of shape ``padded_shape``.
    """
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    out_h, out_w = cols.shape[4], cols.shape[5]

    out = np.zeros(padded_shape, dtype=cols.dtype)
    for i in range(kh):
        i_start = i * dh
        i_stop = i_start + sh * out_h
        for j in range(kw):
            j_start = j * dw
            j_stop = j_start + sw * out_w
            out[:, :, i_start:i_stop:sh, j_start:j_stop:sw] += cols[:, :, i, j, :, :]
    return out
