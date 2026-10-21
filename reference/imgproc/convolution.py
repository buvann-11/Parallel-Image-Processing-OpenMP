r"""2-D convolution / cross-correlation for CNNs, written from scratch.

The forward pass
----------------
For a batch :math:`x \in \mathbb{R}^{N \times C \times H \times W}` and weights
:math:`w \in \mathbb{R}^{F \times C \times k_h \times k_w}`, the operation every
deep-learning framework calls "convolution" is in fact a **cross-correlation**:

.. math::

    y[n, f, p, q] \;=\; b[f] \;+\; \sum_{c=0}^{C-1}\;\sum_{i=0}^{k_h-1}\;
    \sum_{j=0}^{k_w-1} w[f, c, i, j]\;
    x\bigl[n,\, c,\, p\,s_h + i\,d_h,\; q\,s_w + j\,d_w\bigr]

where :math:`s` is the stride and :math:`d` the dilation.  True mathematical
convolution flips the kernel first; pass ``flip_kernel=True`` to get it.  For a
*learned* kernel the distinction is irrelevant (the network simply learns the
flipped weights), but it matters a great deal when the kernel is a fixed filter
such as a Sobel or Gaussian, which is why both are offered here.

Implementation
--------------
The sum above is lowered to a single matrix product via ``im2col``
(see :mod:`imgproc._im2col`), which is what makes a pure-NumPy implementation
competitive rather than merely educational.

Backward pass
-------------
:func:`conv2d_backward` returns :math:`(\partial L/\partial x,
\partial L/\partial w, \partial L/\partial b)` and is verified against
central-difference numerical gradients in the test-suite.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._im2col import col2im, im2col
from .exceptions import ParameterError, ShapeError
from .padding import (
    PaddingSpec,
    conv_output_size,
    pad2d,
    pad2d_backward,
    resolve_padding,
)
from .validation import (
    as_float_array,
    check_choice,
    check_image,
    check_ndim,
    check_positive_int,
    check_tensor_nchw,
    restore_dtype,
)

__all__ = [
    "conv2d",
    "conv2d_backward",
    "conv_output_shape",
    "convolve_image",
]


def _check_groups(in_channels: int, out_channels: int, groups: int) -> None:
    if not isinstance(groups, (int, np.integer)) or isinstance(groups, (bool, np.bool_)):
        raise ParameterError(f"'groups' must be an int, got {type(groups).__name__}.")
    groups = int(groups)
    if groups < 1:
        raise ParameterError(f"'groups' must be >= 1, got {groups}.")
    if in_channels % groups:
        raise ParameterError(
            f"in_channels ({in_channels}) must be divisible by groups ({groups})."
        )
    if out_channels % groups:
        raise ParameterError(
            f"out_channels ({out_channels}) must be divisible by groups ({groups})."
        )


def conv_output_shape(
    input_hw: Sequence[int],
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
) -> tuple[int, int]:
    """Spatial size of :func:`conv2d`'s output, without running it.

    Useful for sizing a network on paper (and for the assertions in the
    test-suite).

    Examples
    --------
    >>> conv_output_shape((32, 32), 3, padding="same")
    (32, 32)
    >>> conv_output_shape((32, 32), 3, stride=2)
    (15, 15)
    """
    ih, iw = check_positive_int(input_hw, name="input_hw")
    kh, kw = check_positive_int(kernel_size, name="kernel_size")
    sh, sw = check_positive_int(stride, name="stride")
    dh, dw = check_positive_int(dilation, name="dilation")
    (top, bottom), (left, right) = resolve_padding(
        padding,
        kernel_size=(kh, kw),
        input_size=(ih, iw),
        stride=(sh, sw),
        dilation=(dh, dw),
    )
    out_h = conv_output_size(
        ih, kh, stride=sh, padding_before=top, padding_after=bottom, dilation=dh
    )
    out_w = conv_output_size(
        iw, kw, stride=sw, padding_before=left, padding_after=right, dilation=dw
    )
    return out_h, out_w


def _prepare(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride,
    padding,
    dilation,
    groups: int,
):
    """Shared validation + geometry resolution for forward and backward."""
    tensor = check_tensor_nchw(x, name="x")
    w = as_float_array(weight, name="weight")
    check_ndim(w, 4, name="weight")

    n, c, h, wd = tensor.shape
    out_channels, in_per_group, kh, kw = w.shape
    if min(w.shape) <= 0:
        raise ShapeError(f"'weight' has a zero-sized dimension: {w.shape}.")

    _check_groups(c, out_channels, groups)
    groups = int(groups)
    if in_per_group != c // groups:
        raise ShapeError(
            f"weight.shape[1] must equal in_channels // groups = {c // groups}, "
            f"got {in_per_group}. (x has {c} channels, groups={groups}.)"
        )

    sh, sw = check_positive_int(stride, name="stride")
    dh, dw = check_positive_int(dilation, name="dilation")
    pads = resolve_padding(
        padding,
        kernel_size=(kh, kw),
        input_size=(h, wd),
        stride=(sh, sw),
        dilation=(dh, dw),
    )
    (top, bottom), (left, right) = pads

    out_h = conv_output_size(
        h, kh, stride=sh, padding_before=top, padding_after=bottom, dilation=dh
    )
    out_w = conv_output_size(
        wd, kw, stride=sw, padding_before=left, padding_after=right, dilation=dw
    )

    b: np.ndarray | None = None
    if bias is not None:
        b = as_float_array(bias, name="bias").reshape(-1)
        if b.size != out_channels:
            raise ShapeError(
                f"'bias' must have {out_channels} elements (one per output "
                f"channel), got {b.size}."
            )

    return tensor, w, b, (sh, sw), (dh, dw), pads, (out_h, out_w), groups, (kh, kw)


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    pad_value: float = 0.0,
    flip_kernel: bool = False,
) -> np.ndarray:
    """Multi-channel, batched 2-D convolution.

    Parameters
    ----------
    x
        Input batch, shape ``(N, C, H, W)``.
    weight
        Kernel bank, shape ``(F, C // groups, kh, kw)``.
    bias
        Optional per-output-channel bias, shape ``(F,)``.
    stride, dilation
        ``int`` or ``(h, w)`` pair.
    padding
        ``int``, ``(ph, pw)``, ``((top, bottom), (left, right))``, or one of
        ``"valid"`` / ``"same"`` / ``"full"``.
    groups
        Split input and output channels into ``groups`` independent paths.
        ``groups=1`` is ordinary convolution; ``groups=C`` with ``F=C`` gives a
        depthwise convolution (the building block of MobileNet).
    padding_mode
        Border mode -- see :data:`imgproc.padding.PAD_MODES`.
    pad_value
        Fill value when ``padding_mode="constant"``.
    flip_kernel
        ``False`` (default) computes cross-correlation, as PyTorch/TensorFlow
        do.  ``True`` computes true mathematical convolution.

    Returns
    -------
    numpy.ndarray
        Output of shape ``(N, F, OH, OW)``, dtype ``float64``.

    Examples
    --------
    An edge-detecting 3x3 kernel on a single grayscale image::

    >>> x = np.zeros((1, 1, 5, 5)); x[0, 0, :, 2:] = 1.0
    >>> w = np.array([[[[-1., 0., 1.]]]])          # horizontal gradient
    >>> conv2d(x, w, padding="same")[0, 0, 2]
    array([ 0.,  1.,  1.,  0., -1.])
    """
    (
        tensor,
        w,
        b,
        (sh, sw),
        (dh, dw),
        pads,
        (out_h, out_w),
        groups,
        (kh, kw),
    ) = _prepare(x, weight, bias, stride, padding, dilation, groups)

    if flip_kernel:
        w = np.ascontiguousarray(w[:, :, ::-1, ::-1])

    padded = pad2d(tensor, pads, mode=padding_mode, value=pad_value)
    cols = im2col(padded, (kh, kw), (sh, sw), (dh, dw), (out_h, out_w))

    n = tensor.shape[0]
    out_channels = w.shape[0]
    in_per_group = w.shape[1]
    fg = out_channels // groups
    taps = in_per_group * kh * kw
    positions = out_h * out_w

    # (N, C, kh, kw, OH, OW) -> (N, G, Cg*kh*kw, OH*OW)
    cols_g = cols.reshape(n, groups, taps, positions)
    # (F, Cg, kh, kw) -> (G, Fg, Cg*kh*kw)
    w_g = w.reshape(groups, fg, taps)

    out = np.matmul(w_g[None, ...], cols_g)  # (N, G, Fg, OH*OW)
    out = out.reshape(n, out_channels, out_h, out_w)

    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return np.ascontiguousarray(out)


def conv2d_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    flip_kernel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Analytic gradients of :func:`conv2d`.

    Given :math:`\partial L/\partial y`, returns
    :math:`(\partial L/\partial x,\ \partial L/\partial w,\ \partial L/\partial b)`.

    The three gradients follow directly from the forward sum:

    .. math::

        \frac{\partial L}{\partial b[f]} &= \sum_{n,p,q} \delta[n,f,p,q] \\
        \frac{\partial L}{\partial w[f,c,i,j]} &=
            \sum_{n,p,q} \delta[n,f,p,q]\; \mathrm{col}[n,c,i,j,p,q] \\
        \frac{\partial L}{\partial x} &=
            \mathrm{col2im}\Bigl(\sum_f w[f,c,i,j]\,\delta[n,f,p,q]\Bigr)

    ``col2im`` accumulates overlapping windows, and the padding is then
    back-propagated too -- for ``reflect``/``replicate``/``circular`` borders the
    gradient of a duplicated pixel is folded back onto its source.

    Returns
    -------
    tuple
        ``(grad_x, grad_weight, grad_bias)``.  ``grad_bias`` is ``None`` when
        ``bias`` was ``None``.
    """
    (
        tensor,
        w,
        b,
        (sh, sw),
        (dh, dw),
        pads,
        (out_h, out_w),
        groups,
        (kh, kw),
    ) = _prepare(x, weight, bias, stride, padding, dilation, groups)

    dout = as_float_array(grad_output, name="grad_output")
    check_ndim(dout, 4, name="grad_output")
    n = tensor.shape[0]
    out_channels = w.shape[0]
    expected = (n, out_channels, out_h, out_w)
    if dout.shape != expected:
        raise ShapeError(
            f"'grad_output' must have shape {expected} to match the forward "
            f"output, got {dout.shape}."
        )

    w_eff = np.ascontiguousarray(w[:, :, ::-1, ::-1]) if flip_kernel else w

    padded = pad2d(tensor, pads, mode=padding_mode)
    cols = im2col(padded, (kh, kw), (sh, sw), (dh, dw), (out_h, out_w))

    in_per_group = w.shape[1]
    fg = out_channels // groups
    taps = in_per_group * kh * kw
    positions = out_h * out_w

    cols_g = cols.reshape(n, groups, taps, positions)
    dout_g = dout.reshape(n, groups, fg, positions)

    # --- dL/dw ------------------------------------------------------------
    # (N, G, Fg, P) @ (N, G, P, K) -> (N, G, Fg, K), summed over the batch.
    dw_g = np.matmul(dout_g, cols_g.transpose(0, 1, 3, 2)).sum(axis=0)
    dweight = dw_g.reshape(out_channels, in_per_group, kh, kw)
    if flip_kernel:
        dweight = np.ascontiguousarray(dweight[:, :, ::-1, ::-1])

    # --- dL/dx ------------------------------------------------------------
    w_g = w_eff.reshape(groups, fg, taps)
    # (G, K, Fg) @ (N, G, Fg, P) -> (N, G, K, P)
    dcols = np.matmul(w_g.transpose(0, 2, 1)[None, ...], dout_g)
    dcols = dcols.reshape(n, tensor.shape[1], kh, kw, out_h, out_w)
    dpadded = col2im(dcols, padded.shape, (kh, kw), (sh, sw), (dh, dw))
    dx = pad2d_backward(dpadded, tensor.shape, pads, mode=padding_mode)

    # --- dL/db ------------------------------------------------------------
    db = dout.sum(axis=(0, 2, 3)) if b is not None else None

    return np.ascontiguousarray(dx), np.ascontiguousarray(dweight), db


def convolve_image(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    padding: PaddingSpec = "same",
    padding_mode: str = "reflect",
    pad_value: float = 0.0,
    mode: str = "convolve",
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Filter an ``HW`` / ``HWC`` image with a single 2-D kernel.

    The kernel is applied **independently to every channel** (a depthwise
    convolution), which is what "blur this photo" means -- the red channel must
    not leak into the blue one.

    Parameters
    ----------
    image
        ``HW`` or ``HWC`` image, any numeric dtype.
    kernel
        2-D filter, e.g. from :mod:`imgproc.kernels`.
    padding
        Defaults to ``"same"`` so the filtered image keeps its size.
    padding_mode
        Defaults to ``"reflect"``: for photographic filtering, mirroring the
        border avoids the dark halo that zero-padding produces.
    mode
        ``"convolve"`` (flip the kernel -- the signal-processing convention,
        and what SciPy's ``convolve2d`` does) or ``"correlate"``.
    preserve_dtype
        If ``True`` (default) the result is cast back to the input dtype,
        rounding and clipping for integer images.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same layout as the input.

    Examples
    --------
    >>> from imgproc.kernels import box_blur
    >>> img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    >>> convolve_image(img, box_blur(3)).shape
    (5, 5)
    """
    arr = check_image(image, name="image")
    original_dtype = arr.dtype
    gray = arr.ndim == 2

    k = as_float_array(kernel, name="kernel")
    check_ndim(k, 2, name="kernel")
    if k.size == 0:
        raise ShapeError("'kernel' must not be empty.")

    how = check_choice(mode, ("convolve", "correlate"), name="mode")

    work = as_float_array(arr, name="image")
    if gray:
        work = work[:, :, None]
    channels = work.shape[2]
    tensor = np.ascontiguousarray(work.transpose(2, 0, 1)[None, ...])

    # One kernel replicated across channels == depthwise convolution.
    weight = np.broadcast_to(k[None, None, :, :], (channels, 1, k.shape[0], k.shape[1]))
    out = conv2d(
        tensor,
        np.ascontiguousarray(weight),
        stride=1,
        padding=padding,
        groups=channels,
        padding_mode=padding_mode,
        pad_value=pad_value,
        flip_kernel=(how == "convolve"),
    )

    result = out[0].transpose(1, 2, 0)
    if gray:
        result = result[:, :, 0]
    result = np.ascontiguousarray(result)
    if preserve_dtype:
        return restore_dtype(result, original_dtype)
    return result
