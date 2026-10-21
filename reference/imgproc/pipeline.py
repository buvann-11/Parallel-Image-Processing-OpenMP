"""Composable preprocessing transforms.

A production pipeline is a *declared* sequence of steps, not a pile of ad-hoc
function calls: written down once, it can be logged, compared between training
and inference, and reproduced exactly.  Mismatched preprocessing between the
two is one of the most common causes of a model that scores well offline and
badly in production.

The API mirrors ``torchvision.transforms``: every transform is a callable
object, and :class:`Compose` chains them.

Example
-------
>>> import numpy as np
>>> pipeline = Compose(
...     Resize((32, 32), method="bilinear"),
...     Grayscale(method="bt709"),
...     ToUnitRange(),
...     Standardize(),
...     ToTensor(),
... )
>>> tensor = pipeline(np.random.default_rng(0).integers(0, 256, (64, 48, 3), dtype=np.uint8))
>>> tensor.shape
(1, 1, 32, 32)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import numpy as np

from .color import to_grayscale, to_rgb
from .exceptions import ParameterError
from .normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    from_unit_range,
    histogram_equalize,
    min_max_normalize,
    normalize,
    standardize,
    to_unit_range,
)
from .padding import PaddingSpec, pad_image
from .resize import center_crop, resize, resize_preserve_aspect
from .validation import check_image, hwc_to_nchw

__all__ = [
    "Transform",
    "Compose",
    "Lambda",
    "Grayscale",
    "ToRGB",
    "Resize",
    "ResizePreserveAspect",
    "CenterCrop",
    "Pad",
    "ToUnitRange",
    "FromUnitRange",
    "MinMaxNormalize",
    "Standardize",
    "Normalize",
    "HistogramEqualize",
    "ToTensor",
]


class Transform(ABC):
    """Base class: a callable, self-describing preprocessing step."""

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class Compose(Transform):
    """Apply transforms left to right.

    >>> Compose(ToUnitRange())(np.array([[0, 255]], dtype=np.uint8)).tolist()
    [[0.0, 1.0]]
    """

    def __init__(self, *transforms: Callable[[np.ndarray], np.ndarray]) -> None:
        flat: list[Callable[[np.ndarray], np.ndarray]] = []
        for item in transforms:
            # Accept both Compose(a, b) and Compose([a, b]).
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        for item in flat:
            if not callable(item):
                raise ParameterError(
                    f"Compose accepts callables, got {type(item).__name__}."
                )
        self.transforms = flat

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = image
        for transform in self.transforms:
            out = transform(out)
        return out

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose(\n  {inner}\n)"


class Lambda(Transform):
    """Wrap an arbitrary callable, with an optional label for ``repr``."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = "lambda") -> None:
        if not callable(func):
            raise ParameterError(f"'func' must be callable, got {type(func).__name__}.")
        self.func = func
        self.name = str(name)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.func(image)

    def __repr__(self) -> str:
        return f"Lambda({self.name})"


class Grayscale(Transform):
    """Convert to single-channel grayscale -- see :func:`imgproc.color.to_grayscale`."""

    def __init__(self, method: str = "bt601", *, gamma_correct: bool = False,
                 keep_channel_dim: bool = False) -> None:
        self.method = method
        self.gamma_correct = bool(gamma_correct)
        self.keep_channel_dim = bool(keep_channel_dim)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return to_grayscale(
            image, self.method, gamma_correct=self.gamma_correct,
            keep_channel_dim=self.keep_channel_dim,
        )

    def __repr__(self) -> str:
        return f"Grayscale(method={self.method!r}, gamma_correct={self.gamma_correct})"


class ToRGB(Transform):
    """Broadcast a single-channel image to 3 channels."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return to_rgb(image)


class Resize(Transform):
    """Resize to a fixed ``(height, width)`` -- see :func:`imgproc.resize.resize`."""

    def __init__(self, size: Sequence[int], *, method: str = "bilinear",
                 antialias: bool = True, preserve_dtype: bool = True) -> None:
        self.size = tuple(int(s) for s in size)
        self.method = method
        self.antialias = bool(antialias)
        self.preserve_dtype = bool(preserve_dtype)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return resize(
            image, self.size, method=self.method, antialias=self.antialias,
            preserve_dtype=self.preserve_dtype,
        )

    def __repr__(self) -> str:
        return f"Resize(size={self.size}, method={self.method!r}, antialias={self.antialias})"


class ResizePreserveAspect(Transform):
    """Letterbox into ``(height, width)`` without distorting the aspect ratio."""

    def __init__(self, size: Sequence[int], *, method: str = "bilinear",
                 antialias: bool = True, pad_value: float = 0.0) -> None:
        self.size = tuple(int(s) for s in size)
        self.method = method
        self.antialias = bool(antialias)
        self.pad_value = float(pad_value)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return resize_preserve_aspect(
            image, self.size, method=self.method, antialias=self.antialias,
            pad_value=self.pad_value,
        )

    def __repr__(self) -> str:
        return f"ResizePreserveAspect(size={self.size}, method={self.method!r})"


class CenterCrop(Transform):
    """Crop the central ``(height, width)`` region."""

    def __init__(self, size: Sequence[int]) -> None:
        self.size = tuple(int(s) for s in size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return center_crop(image, self.size)

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class Pad(Transform):
    """Pad the border -- see :func:`imgproc.padding.pad_image`."""

    def __init__(self, padding: PaddingSpec, *, mode: str = "zeros", value: float = 0.0) -> None:
        self.padding = padding
        self.mode = mode
        self.value = float(value)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return pad_image(image, self.padding, mode=self.mode, value=self.value)

    def __repr__(self) -> str:
        return f"Pad(padding={self.padding!r}, mode={self.mode!r})"


class ToUnitRange(Transform):
    """Map the dtype's canonical range onto ``[0, 1]`` (``uint8`` 255 -> 1.0)."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return to_unit_range(image)


class FromUnitRange(Transform):
    """Map ``[0, 1]`` back onto an integer dtype -- the inverse of :class:`ToUnitRange`."""

    def __init__(self, dtype: Any = np.uint8) -> None:
        self.dtype = dtype

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return from_unit_range(image, self.dtype)

    def __repr__(self) -> str:
        return f"FromUnitRange(dtype={np.dtype(self.dtype).name})"


class MinMaxNormalize(Transform):
    """Rescale the observed range onto ``feature_range``."""

    def __init__(self, feature_range: Sequence[float] = (0.0, 1.0), *,
                 per_channel: bool = False) -> None:
        self.feature_range = tuple(float(v) for v in feature_range)
        self.per_channel = bool(per_channel)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return min_max_normalize(
            image, self.feature_range, per_channel=self.per_channel
        )

    def __repr__(self) -> str:
        return f"MinMaxNormalize(feature_range={self.feature_range}, per_channel={self.per_channel})"


class Standardize(Transform):
    """Zero-mean / unit-variance normalisation, estimated per image."""

    def __init__(self, *, per_channel: bool = False) -> None:
        self.per_channel = bool(per_channel)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return standardize(image, per_channel=self.per_channel)

    def __repr__(self) -> str:
        return f"Standardize(per_channel={self.per_channel})"


class Normalize(Transform):
    """Fixed per-channel normalisation (ImageNet statistics by default)."""

    def __init__(self, mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD) -> None:
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return normalize(image, self.mean, self.std)

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class HistogramEqualize(Transform):
    """Flatten the intensity histogram to maximise global contrast."""

    def __init__(self, *, nbins: int = 256, per_channel: bool = False) -> None:
        self.nbins = int(nbins)
        self.per_channel = bool(per_channel)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return histogram_equalize(image, nbins=self.nbins, per_channel=self.per_channel)

    def __repr__(self) -> str:
        return f"HistogramEqualize(nbins={self.nbins}, per_channel={self.per_channel})"


class ToTensor(Transform):
    """``HWC`` image -> ``NCHW`` float tensor, ready for :mod:`imgproc.layers`.

    This is the final step of a pipeline: it changes the *layout* (channels
    first) and adds the batch axis.  It deliberately does **not** rescale
    values -- unlike ``torchvision``'s ``ToTensor``, whose hidden ``/255`` is a
    recurring source of confusion.  Put :class:`ToUnitRange` in the pipeline
    when you want that.
    """

    def __init__(self, *, dtype: Any = np.float64) -> None:
        self.dtype = dtype

    def __call__(self, image: np.ndarray) -> np.ndarray:
        check_image(image, name="image")
        return hwc_to_nchw(image, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"ToTensor(dtype={np.dtype(self.dtype).name})"
