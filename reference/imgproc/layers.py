"""A minimal layer API on top of the functional ops.

The functions in :mod:`imgproc.convolution` and :mod:`imgproc.pooling` are
stateless.  Real networks need objects that own their parameters, cache what
the backward pass needs, and compose.  This module supplies exactly that --
just enough to build and *train* a small CNN in pure NumPy, without pretending
to be a framework.

Every layer follows the same three-method contract::

    y = layer.forward(x)     # caches whatever backward() will need
    dx = layer.backward(dy)  # accumulates parameter gradients, returns dL/dx
    layer.zero_grad()        # clears the accumulators

Example
-------
>>> import numpy as np
>>> net = Sequential(
...     Conv2D(1, 4, 3, padding="same", seed=0),
...     ReLU(),
...     MaxPool2D(2),
...     Flatten(),
... )
>>> x = np.random.default_rng(0).normal(size=(2, 1, 8, 8))
>>> net.forward(x).shape
(2, 64)
>>> dx = net.backward(np.ones((2, 64)))
>>> dx.shape
(2, 1, 8, 8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

import numpy as np

from .convolution import conv2d, conv2d_backward, conv_output_shape
from .exceptions import ParameterError, ShapeError
from .padding import PaddingSpec
from .pooling import (
    avg_pool2d,
    avg_pool2d_backward,
    max_pool2d,
    max_pool2d_backward,
    pool_output_shape,
)
from .validation import as_float_array, check_positive_int, check_tensor_nchw

__all__ = [
    "Layer",
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "ReLU",
    "Flatten",
    "Sequential",
]


class Layer(ABC):
    """Abstract base class for every layer."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the output and cache what :meth:`backward` will need."""

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Return ``dL/dx``, accumulating parameter gradients on the way."""

    def parameters(self) -> dict[str, np.ndarray]:
        """Trainable arrays, keyed by name.  Empty for parameter-free layers."""
        return {}

    def gradients(self) -> dict[str, np.ndarray]:
        """Gradients matching :meth:`parameters`, same keys."""
        return {}

    def zero_grad(self) -> None:
        """Reset gradient accumulators to zero."""
        for grad in self.gradients().values():
            grad.fill(0.0)

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        """Output shape for a given input shape, without allocating anything."""
        return tuple(int(s) for s in input_shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def _require_cache(self) -> dict[str, Any]:
        if not self._cache:
            raise RuntimeError(
                f"{type(self).__name__}.backward() was called before forward(); "
                "there is no cached activation to differentiate."
            )
        return self._cache


class Conv2D(Layer):
    """Trainable 2-D convolution layer.

    Parameters
    ----------
    in_channels, out_channels
        Channel counts.  ``out_channels`` is the number of filters.
    kernel_size, stride, padding, dilation, groups, padding_mode
        Geometry; identical in meaning to :func:`imgproc.convolution.conv2d`.
    bias
        Add a learnable per-filter bias.
    init
        Weight initialiser:

        * ``"he"`` (default) -- :math:`\\mathcal{N}(0, 2/fan_{in})`, the right
          choice for ReLU networks (He et al., 2015), because ReLU discards
          half the variance of its input.
        * ``"xavier"`` -- :math:`\\mathcal{U}(\\pm\\sqrt{6/(fan_{in}+fan_{out})})`,
          for symmetric activations such as ``tanh`` (Glorot & Bengio, 2010).
        * ``"zeros"`` -- for tests only; a zero-initialised layer can never
          break symmetry and will not learn.
    seed
        Seed for reproducible initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        init: str = "he",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ParameterError(
                f"Channel counts must be >= 1, got in={in_channels}, out={out_channels}."
            )
        if in_channels % groups or out_channels % groups:
            raise ParameterError(
                f"in_channels ({in_channels}) and out_channels ({out_channels}) "
                f"must both be divisible by groups ({groups})."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = check_positive_int(kernel_size, name="kernel_size")
        self.stride = check_positive_int(stride, name="stride")
        self.padding = padding
        self.dilation = check_positive_int(dilation, name="dilation")
        self.groups = int(groups)
        self.padding_mode = padding_mode

        kh, kw = self.kernel_size
        fan_in = (self.in_channels // self.groups) * kh * kw
        fan_out = (self.out_channels // self.groups) * kh * kw
        shape = (self.out_channels, self.in_channels // self.groups, kh, kw)
        rng = np.random.default_rng(seed)

        if init == "he":
            self.weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)
        elif init == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight = rng.uniform(-limit, limit, size=shape)
        elif init == "zeros":
            self.weight = np.zeros(shape, dtype=np.float64)
        else:
            raise ParameterError(
                f"'init' must be 'he', 'xavier' or 'zeros', got {init!r}."
            )

        self.bias = np.zeros(self.out_channels, dtype=np.float64) if bias else None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        tensor = check_tensor_nchw(x, name="x")
        if tensor.shape[1] != self.in_channels:
            raise ShapeError(
                f"{self!r} expects {self.in_channels} input channels, got {tensor.shape[1]}."
            )
        self._cache = {"x": tensor}
        return conv2d(
            tensor,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode,
        )

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        cache = self._require_cache()
        dx, dw, db = conv2d_backward(
            grad_output,
            cache["x"],
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=self.padding_mode,
        )
        self.grad_weight += dw
        if self.grad_bias is not None and db is not None:
            self.grad_bias += db
        return dx

    def parameters(self) -> dict[str, np.ndarray]:
        params = {"weight": self.weight}
        if self.bias is not None:
            params["bias"] = self.bias
        return params

    def gradients(self) -> dict[str, np.ndarray]:
        grads = {"weight": self.grad_weight}
        if self.grad_bias is not None:
            grads["bias"] = self.grad_bias
        return grads

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        n, _, h, w = (int(s) for s in input_shape)
        out_hw = conv_output_shape(
            (h, w), self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
        )
        return (n, self.out_channels, *out_hw)

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding!r}, groups={self.groups})"
        )


class MaxPool2D(Layer):
    """Max-pooling layer.  Caches argmax indices for the backward pass."""

    def __init__(
        self,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
        *,
        padding: PaddingSpec = 0,
        dilation: int | Sequence[int] = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = check_positive_int(kernel_size, name="kernel_size")
        self.stride = None if stride is None else check_positive_int(stride, name="stride")
        self.padding = padding
        self.dilation = check_positive_int(dilation, name="dilation")
        self.ceil_mode = bool(ceil_mode)

    def forward(self, x: np.ndarray) -> np.ndarray:
        tensor = check_tensor_nchw(x, name="x")
        out, indices = max_pool2d(
            tensor, self.kernel_size, self.stride,
            padding=self.padding, dilation=self.dilation,
            ceil_mode=self.ceil_mode, return_indices=True,
        )
        self._cache = {"indices": indices, "shape": tensor.shape}
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        cache = self._require_cache()
        return max_pool2d_backward(grad_output, cache["indices"], cache["shape"])

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        n, c, h, w = (int(s) for s in input_shape)
        out_hw = pool_output_shape(
            (h, w), self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, ceil_mode=self.ceil_mode,
        )
        return (n, c, *out_hw)

    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2D(Layer):
    """Average-pooling layer.  Needs only the input shape for its backward pass."""

    def __init__(
        self,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
        *,
        padding: PaddingSpec = 0,
        dilation: int | Sequence[int] = 1,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = check_positive_int(kernel_size, name="kernel_size")
        self.stride = None if stride is None else check_positive_int(stride, name="stride")
        self.padding = padding
        self.dilation = check_positive_int(dilation, name="dilation")
        self.ceil_mode = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)

    def forward(self, x: np.ndarray) -> np.ndarray:
        tensor = check_tensor_nchw(x, name="x")
        self._cache = {"shape": tensor.shape}
        return avg_pool2d(
            tensor, self.kernel_size, self.stride,
            padding=self.padding, dilation=self.dilation,
            ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
        )

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        cache = self._require_cache()
        return avg_pool2d_backward(
            grad_output, cache["shape"], self.kernel_size, self.stride,
            padding=self.padding, dilation=self.dilation,
            ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
        )

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        n, c, h, w = (int(s) for s in input_shape)
        out_hw = pool_output_shape(
            (h, w), self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, ceil_mode=self.ceil_mode,
        )
        return (n, c, *out_hw)

    def __repr__(self) -> str:
        return f"AvgPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class ReLU(Layer):
    """Rectified linear unit, ``max(0, x)``.

    Included so the demo network is a genuine CNN rather than a stack of linear
    maps: without a nonlinearity, any number of convolutions collapses into a
    single convolution.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = as_float_array(x, name="x")
        self._cache = {"mask": arr > 0}
        return np.maximum(arr, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        cache = self._require_cache()
        dout = as_float_array(grad_output, name="grad_output")
        if dout.shape != cache["mask"].shape:
            raise ShapeError(
                f"'grad_output' shape {dout.shape} does not match the cached "
                f"activation shape {cache['mask'].shape}."
            )
        return dout * cache["mask"]

    def __repr__(self) -> str:
        return "ReLU()"


class Flatten(Layer):
    """Flatten ``(N, C, H, W)`` to ``(N, C*H*W)`` -- the bridge to a dense head."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = as_float_array(x, name="x")
        self._cache = {"shape": arr.shape}
        return arr.reshape(arr.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        cache = self._require_cache()
        dout = as_float_array(grad_output, name="grad_output")
        return dout.reshape(cache["shape"])

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        dims = [int(s) for s in input_shape]
        return (dims[0], int(np.prod(dims[1:])))

    def __repr__(self) -> str:
        return "Flatten()"


class Sequential(Layer):
    """Chain layers end to end.

    ``forward`` runs them in order; ``backward`` runs them in reverse, which is
    exactly the chain rule applied to a composition of functions.
    """

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ParameterError(
                    f"Sequential accepts Layer instances, got {type(layer).__name__}."
                )
        self.layers: list[Layer] = list(layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self._cache = {"ran": True}
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self._require_cache()
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{name}": array
            for i, layer in enumerate(self.layers)
            for name, array in layer.parameters().items()
        }

    def gradients(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{name}": array
            for i, layer in enumerate(self.layers)
            for name, array in layer.gradients().items()
        }

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def output_shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        shape = tuple(int(s) for s in input_shape)
        for layer in self.layers:
            shape = layer.output_shape(shape)
        return shape

    def summary(self, input_shape: Sequence[int]) -> str:
        """A Keras-style table of layers, output shapes and parameter counts."""
        shape = tuple(int(s) for s in input_shape)
        rows = [("Layer", "Output shape", "Params"), ("-" * 44, "-" * 20, "-" * 10)]
        total = 0
        for layer in self.layers:
            shape = layer.output_shape(shape)
            count = sum(p.size for p in layer.parameters().values())
            total += count
            rows.append((repr(layer)[:44], str(shape), f"{count:,}"))
        rows.append(("-" * 44, "-" * 20, "-" * 10))
        rows.append(("Total", "", f"{total:,}"))
        return "\n".join(f"{a:<46}{b:<22}{c:>10}" for a, b, c in rows)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {inner}\n)"
