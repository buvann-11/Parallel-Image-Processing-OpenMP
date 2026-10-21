"""Padding for convolutional neural networks -- implemented from scratch.

Why padding exists
------------------
A convolution with a :math:`k \\times k` kernel and no padding shrinks a feature
map from :math:`H` to :math:`H - k + 1`.  Stack a few layers and the spatial
extent collapses, while border pixels are visited far fewer times than central
ones.  Padding fixes both problems: it preserves resolution and it equalises
the number of times each input position contributes to the output.

Supported border modes
----------------------
=================  ==========================================================
``"zeros"``        Pad with 0.  The default in virtually every CNN.
``"constant"``     Pad with a user-supplied ``value``.
``"reflect"``      Mirror about the border pixel *without* repeating it
                   (``c b | a b c d | c b``).
``"symmetric"``    Mirror about the border edge, repeating it
                   (``b a | a b c d | d c``).
``"replicate"``    Repeat the border pixel (``a a | a b c d | d d``).
                   Alias: ``"edge"``.
``"circular"``     Wrap around (``c d | a b c d | a b``).  Alias: ``"wrap"``.
=================  ==========================================================

Unlike :func:`numpy.pad`, the non-constant modes here accept a padding width
*larger* than the axis length, because the index mapping is computed with
modular folding rather than by repeated reflection.

Symbolic padding specifications
-------------------------------
``"valid"``  no padding.
``"same"``   output size is ``ceil(input / stride)`` (TensorFlow semantics).
``"full"``   every partial overlap of kernel and input is computed.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

from .exceptions import ParameterError, ShapeError
from .validation import (
    as_float_array,
    check_choice,
    check_image,
    check_tensor_nchw,
    to_pair,
)

__all__ = [
    "PAD_MODES",
    "PaddingSpec",
    "resolve_padding",
    "conv_output_size",
    "pad2d",
    "pad2d_backward",
    "unpad2d",
    "pad_image",
]

#: Canonical border modes.  Aliases are resolved by :func:`_canonical_mode`.
PAD_MODES: tuple[str, ...] = (
    "zeros",
    "constant",
    "reflect",
    "symmetric",
    "replicate",
    "circular",
)

_MODE_ALIASES = {
    "zero": "zeros",
    "edge": "replicate",
    "nearest": "replicate",
    "wrap": "circular",
    "mirror": "reflect",
}

#: Anything accepted where a padding amount is expected.
PaddingSpec = Union[int, Sequence[int], Sequence[Sequence[int]], str]

_SYMBOLIC = ("valid", "same", "full")


def _canonical_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ParameterError(f"'mode' must be a string, got {type(mode).__name__}.")
    key = mode.lower().strip()
    key = _MODE_ALIASES.get(key, key)
    return check_choice(key, PAD_MODES, name="mode")


def _as_pad_pairs(padding: PaddingSpec, *, name: str = "padding") -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalise a numeric padding spec to ``((top, bottom), (left, right))``.

    Accepted forms::

        2                      -> ((2, 2), (2, 2))
        (2, 3)                 -> ((2, 2), (3, 3))     # (height, width)
        ((1, 2), (3, 4))       -> ((1, 2), (3, 4))     # explicit, asymmetric
    """
    if isinstance(padding, str):
        raise ParameterError(
            f"{name!r} is symbolic ({padding!r}); resolve it with resolve_padding() "
            "first -- kernel size and stride are needed to turn it into numbers."
        )
    if isinstance(padding, (int, np.integer)) and not isinstance(padding, (bool, np.bool_)):
        p = int(padding)
        pairs = ((p, p), (p, p))
    else:
        try:
            items = list(padding)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ParameterError(
                f"{name!r} must be an int, a pair, or a pair of pairs; got {padding!r}."
            ) from exc
        if len(items) != 2:
            raise ParameterError(f"{name!r} must have exactly 2 entries, got {len(items)}.")
        if all(isinstance(it, (int, np.integer)) and not isinstance(it, (bool, np.bool_)) for it in items):
            ph, pw = int(items[0]), int(items[1])
            pairs = ((ph, ph), (pw, pw))
        else:
            h_pair = to_pair(items[0], name=f"{name}[0]")
            w_pair = to_pair(items[1], name=f"{name}[1]")
            pairs = (h_pair, w_pair)

    for axis, pair in zip(("height", "width"), pairs):
        for side, amount in zip(("before", "after"), pair):
            if amount < 0:
                raise ParameterError(
                    f"{name!r} must be non-negative; got {amount} for the "
                    f"{axis} {side} side."
                )
    return pairs


def conv_output_size(
    input_size: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding_before: int = 0,
    padding_after: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
) -> int:
    r"""Output length of a 1-D sliding-window operation.

    .. math::

        L_{out} = \left\lfloor
            \frac{L_{in} + p_{before} + p_{after} - d\,(k - 1) - 1}{s}
        \right\rfloor + 1

    With ``ceil_mode=True`` the floor becomes a ceiling and a trailing window
    that would start inside the padding only is dropped -- exactly the rule
    used by PyTorch's pooling layers.

    Raises
    ------
    ParameterError
        If the configuration yields a non-positive output size.
    """
    if kernel_size < 1:
        raise ParameterError(f"kernel_size must be >= 1, got {kernel_size}.")
    if stride < 1:
        raise ParameterError(f"stride must be >= 1, got {stride}.")
    if dilation < 1:
        raise ParameterError(f"dilation must be >= 1, got {dilation}.")

    effective_k = dilation * (kernel_size - 1) + 1
    padded = input_size + padding_before + padding_after
    numerator = padded - effective_k

    if numerator < 0:
        raise ParameterError(
            f"Kernel does not fit: padded input size {padded} is smaller than the "
            f"effective kernel size {effective_k} "
            f"(kernel_size={kernel_size}, dilation={dilation}). "
            "Increase the padding or reduce the kernel/dilation."
        )

    if ceil_mode:
        out = int(math.ceil(numerator / stride)) + 1
        # Never let the last window begin inside the trailing padding.
        if (out - 1) * stride >= input_size + padding_before:
            out -= 1
    else:
        out = numerator // stride + 1

    if out < 1:  # pragma: no cover - unreachable given the checks above
        raise ParameterError(
            "Resulting output size is non-positive; check kernel_size, stride, "
            "dilation and padding."
        )
    return out


def resolve_padding(
    padding: PaddingSpec,
    *,
    kernel_size: int | Sequence[int] = 1,
    input_size: Sequence[int] | None = None,
    stride: int | Sequence[int] = 1,
    dilation: int | Sequence[int] = 1,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Turn any padding specification into ``((top, bottom), (left, right))``.

    Numeric specifications are normalised directly.  The symbolic forms
    ``"valid"``, ``"same"`` and ``"full"`` are resolved using the convolution
    geometry:

    * ``"valid"`` -- no padding at all.
    * ``"same"``  -- TensorFlow semantics: the output size is
      ``ceil(input_size / stride)``.  When the required total padding is odd,
      the extra pixel goes to the *bottom*/*right*, matching TensorFlow,
      PyTorch's ``Conv2d(padding='same')`` (for stride 1) and cuDNN.
    * ``"full"``  -- ``effective_kernel - 1`` on each side, so every position
      where the kernel overlaps the input at all produces an output.

    ``input_size`` is required for ``"same"`` (and only for ``"same"``).

    Examples
    --------
    >>> resolve_padding(1)
    ((1, 1), (1, 1))
    >>> resolve_padding("same", kernel_size=3, input_size=(8, 8))
    ((1, 1), (1, 1))
    >>> resolve_padding("same", kernel_size=2, input_size=(8, 8))
    ((0, 1), (0, 1))
    """
    if not isinstance(padding, str):
        return _as_pad_pairs(padding)

    key = check_choice(padding, _SYMBOLIC, name="padding")
    kh, kw = to_pair(kernel_size, name="kernel_size")
    sh, sw = to_pair(stride, name="stride")
    dh, dw = to_pair(dilation, name="dilation")
    for label, value in (("kernel_size", (kh, kw)), ("stride", (sh, sw)), ("dilation", (dh, dw))):
        if min(value) < 1:
            raise ParameterError(f"{label} entries must be >= 1, got {value}.")

    if key == "valid":
        return ((0, 0), (0, 0))

    if key == "full":
        return ((dh * (kh - 1),) * 2, (dw * (kw - 1),) * 2)

    # --- "same" -----------------------------------------------------------
    if input_size is None:
        raise ParameterError(
            "padding='same' needs 'input_size' to be known "
            "(the required padding depends on the input size whenever stride > 1)."
        )
    ih, iw = to_pair(input_size, name="input_size")
    if ih < 1 or iw < 1:
        raise ParameterError(f"input_size entries must be >= 1, got {(ih, iw)}.")

    def _same(in_size: int, k: int, s: int, d: int) -> tuple[int, int]:
        effective_k = d * (k - 1) + 1
        out = int(math.ceil(in_size / s))
        total = max((out - 1) * s + effective_k - in_size, 0)
        before = total // 2
        return before, total - before

    return (_same(ih, kh, sh, dh), _same(iw, kw, sw, dw))


def _index_map(length: int, before: int, after: int, mode: str) -> np.ndarray:
    """Source index for every position of the padded axis (non-constant modes)."""
    idx = np.arange(-before, length + after, dtype=np.intp)

    if mode == "replicate":
        return np.clip(idx, 0, length - 1)

    if mode == "circular":
        return np.mod(idx, length)

    if mode == "reflect":
        if length == 1:
            return np.zeros_like(idx)
        period = 2 * length - 2
        folded = np.mod(idx, period)
        return np.where(folded >= length, period - folded, folded)

    if mode == "symmetric":
        period = 2 * length
        folded = np.mod(idx, period)
        return np.where(folded >= length, period - 1 - folded, folded)

    raise ParameterError(f"Unsupported non-constant padding mode {mode!r}.")  # pragma: no cover


def pad2d(
    x: np.ndarray,
    padding: PaddingSpec,
    *,
    mode: str = "zeros",
    value: float = 0.0,
) -> np.ndarray:
    """Pad the spatial dimensions of an ``(N, C, H, W)`` tensor.

    Parameters
    ----------
    x
        Input tensor in ``NCHW`` layout.
    padding
        ``int``, ``(pad_h, pad_w)`` or ``((top, bottom), (left, right))``.
        Symbolic strings must be resolved with :func:`resolve_padding` first.
    mode
        One of :data:`PAD_MODES` (or an alias such as ``"edge"``/``"wrap"``).
    value
        Fill value for ``mode="constant"``.  Ignored otherwise.

    Returns
    -------
    numpy.ndarray
        Tensor of shape ``(N, C, H + top + bottom, W + left + right)``.

    Notes
    -----
    Implemented with explicit index arithmetic rather than :func:`numpy.pad`,
    both to show the mechanism and to lift NumPy's restriction that the
    reflection width must be smaller than the axis length.

    Examples
    --------
    >>> x = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    >>> pad2d(x, 1)[0, 0]
    array([[0., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 2., 3., 0.],
           [0., 0., 0., 0.]])
    """
    tensor = check_tensor_nchw(x, name="x")
    canonical = _canonical_mode(mode)
    (top, bottom), (left, right) = _as_pad_pairs(padding)

    if top == bottom == left == right == 0:
        return tensor.copy()

    n, c, h, w = tensor.shape

    if canonical in ("zeros", "constant"):
        fill = 0.0 if canonical == "zeros" else float(value)
        if not np.isfinite(fill) and canonical == "constant":
            raise ParameterError(f"'value' must be finite, got {value!r}.")
        out = np.full((n, c, h + top + bottom, w + left + right), fill, dtype=tensor.dtype)
        out[:, :, top : top + h, left : left + w] = tensor
        return out

    rows = _index_map(h, top, bottom, canonical)
    cols = _index_map(w, left, right, canonical)
    return np.ascontiguousarray(tensor[:, :, rows[:, None], cols[None, :]])


def unpad2d(x: np.ndarray, padding: PaddingSpec) -> np.ndarray:
    """Inverse of :func:`pad2d`: crop the padded border away.

    ``unpad2d(pad2d(x, p), p)`` reproduces ``x`` exactly for every mode.
    """
    tensor = check_tensor_nchw(x, name="x")
    (top, bottom), (left, right) = _as_pad_pairs(padding)
    h, w = tensor.shape[2], tensor.shape[3]
    if top + bottom >= h or left + right >= w:
        raise ShapeError(
            f"Cannot remove padding ((({top}, {bottom}), ({left}, {right}))) from a "
            f"tensor of spatial size {(h, w)}: nothing would be left."
        )
    return np.ascontiguousarray(
        tensor[:, :, top : h - bottom, left : w - right]
    )


def pad2d_backward(
    grad_padded: np.ndarray,
    input_shape: Sequence[int],
    padding: PaddingSpec,
    *,
    mode: str = "zeros",
) -> np.ndarray:
    """Back-propagate a gradient through :func:`pad2d`.

    Padding is a linear operator, so its gradient is simply its adjoint:

    * **zeros / constant** -- padded positions do not depend on any input, so
      the gradient is obtained by *cropping* the border away.
    * **reflect / symmetric / replicate / circular** -- each padded position is
      a *copy* of some interior pixel.  A copy is a gather, and the adjoint of
      a gather is a scatter-add: the gradient of every duplicate is folded back
      onto the pixel it was copied from.

    Forgetting this fold-back is a classic bug; it silently biases the
    gradients of every border pixel.

    Parameters
    ----------
    grad_padded
        Gradient with respect to the *padded* tensor, shape ``(N, C, Hp, Wp)``.
    input_shape
        Shape ``(N, C, H, W)`` of the tensor that was padded.
    padding, mode
        Exactly the values passed to :func:`pad2d`.

    Returns
    -------
    numpy.ndarray
        Gradient with respect to the unpadded input, shape ``input_shape``.
    """
    grad = check_tensor_nchw(grad_padded, name="grad_padded")
    canonical = _canonical_mode(mode)
    (top, bottom), (left, right) = _as_pad_pairs(padding)

    shape = tuple(int(s) for s in input_shape)
    if len(shape) != 4:
        raise ShapeError(f"'input_shape' must have 4 entries, got {input_shape!r}.")
    n, c, h, w = shape
    expected = (n, c, h + top + bottom, w + left + right)
    if grad.shape != expected:
        raise ShapeError(
            f"'grad_padded' has shape {grad.shape} but padding {((top, bottom), (left, right))} "
            f"applied to input shape {shape} gives {expected}."
        )

    if top == bottom == left == right == 0:
        return grad.copy()

    if canonical in ("zeros", "constant"):
        return np.ascontiguousarray(grad[:, :, top : top + h, left : left + w])

    rows = _index_map(h, top, bottom, canonical)
    cols = _index_map(w, left, right, canonical)

    # Scatter matrices: row_map[p, i] == 1 iff padded row p copies input row i.
    row_map = np.zeros((rows.size, h), dtype=grad.dtype)
    row_map[np.arange(rows.size), rows] = 1.0
    col_map = np.zeros((cols.size, w), dtype=grad.dtype)
    col_map[np.arange(cols.size), cols] = 1.0

    folded = np.tensordot(grad, row_map, axes=([2], [0]))   # (N, C, Wp, H)
    folded = np.tensordot(folded, col_map, axes=([2], [0]))  # (N, C, H, W)
    return np.ascontiguousarray(folded)


def pad_image(
    image: np.ndarray,
    padding: PaddingSpec,
    *,
    mode: str = "zeros",
    value: float = 0.0,
) -> np.ndarray:
    """Pad an ``HW`` / ``HWC`` image, preserving layout and dtype.

    A thin, convenient wrapper around :func:`pad2d` for the image-processing
    half of the project.
    """
    arr = check_image(image, name="image")
    original_dtype = arr.dtype
    gray = arr.ndim == 2
    work = as_float_array(arr, name="image")
    if gray:
        work = work[:, :, None]
    tensor = np.ascontiguousarray(work.transpose(2, 0, 1)[None, ...])
    padded = pad2d(tensor, padding, mode=mode, value=value)
    out = padded[0].transpose(1, 2, 0)
    if gray:
        out = out[:, :, 0]
    if np.issubdtype(original_dtype, np.floating):
        return np.ascontiguousarray(out.astype(original_dtype, copy=False))
    info = np.iinfo(original_dtype)
    return np.ascontiguousarray(
        np.clip(np.rint(out), info.min, info.max).astype(original_dtype)
    )
