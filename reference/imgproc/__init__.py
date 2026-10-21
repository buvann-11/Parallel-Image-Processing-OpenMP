"""``imgproc`` -- CNN primitives and classical image processing, from scratch.

The package has two halves, matching the two parts of the project:

**Part 1 -- CNN primitives** (:mod:`~imgproc.padding`, :mod:`~imgproc.convolution`,
:mod:`~imgproc.pooling`, :mod:`~imgproc.layers`)
    Padding with six border modes, im2col-accelerated convolution with stride /
    dilation / groups, max and average pooling, and analytically differentiated
    backward passes for all of them.

**Part 2 -- Image processing** (:mod:`~imgproc.color`, :mod:`~imgproc.normalization`,
:mod:`~imgproc.resize`, :mod:`~imgproc.pipeline`)
    Grayscale conversion with the broadcast luma standards, intensity
    normalisation, and high-quality resampling with proper anti-aliasing --
    composable through a ``torchvision``-style transform pipeline.

Everything is implemented with NumPy alone.  Pillow, Matplotlib and SciPy are
optional: they are used for file I/O, figures and cross-validation
respectively, never by the numerical core.

Array layout conventions
------------------------
* Feature maps and mini-batches: ``(N, C, H, W)`` -- *channels first*.
* Images: ``(H, W)`` or ``(H, W, C)`` -- *channels last*.
* :func:`imgproc.validation.hwc_to_nchw` and
  :func:`~imgproc.validation.nchw_to_hwc` convert between them.

Quick start
-----------
>>> import numpy as np
>>> from imgproc import conv2d, max_pool2d, to_grayscale, resize
>>> x = np.random.default_rng(0).normal(size=(2, 3, 16, 16))
>>> w = np.random.default_rng(1).normal(size=(8, 3, 3, 3))
>>> conv2d(x, w, padding="same").shape
(2, 8, 16, 16)
>>> max_pool2d(conv2d(x, w, padding="same"), 2).shape
(2, 8, 8, 8)
"""

from __future__ import annotations

from .color import (
    GRAYSCALE_METHODS,
    GRAYSCALE_WEIGHTS,
    drop_alpha,
    linear_to_srgb,
    srgb_to_linear,
    to_grayscale,
    to_rgb,
)
from .convolution import conv2d, conv2d_backward, conv_output_shape, convolve_image
from .exceptions import (
    DependencyError,
    DTypeError,
    ImgProcError,
    ParameterError,
    ShapeError,
)
from .io_utils import (
    checkerboard,
    circle,
    gradient_ramp,
    load_image,
    sample_rgb_image,
    save_image,
)
from .kernels import (
    KERNELS,
    box_blur,
    emboss,
    gaussian_blur,
    gaussian_kernel_1d,
    identity,
    laplacian,
    prewitt,
    scharr,
    sharpen,
    sobel,
)
from .layers import AvgPool2D, Conv2D, Flatten, Layer, MaxPool2D, ReLU, Sequential
from .normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    contrast_stretch,
    denormalize,
    from_unit_range,
    histogram_equalize,
    min_max_normalize,
    normalize,
    standardize,
    to_unit_range,
)
from .padding import (
    PAD_MODES,
    conv_output_size,
    pad2d,
    pad2d_backward,
    pad_image,
    resolve_padding,
    unpad2d,
)
from .pipeline import (
    CenterCrop,
    Compose,
    FromUnitRange,
    Grayscale,
    HistogramEqualize,
    Lambda,
    MinMaxNormalize,
    Normalize,
    Pad,
    Resize,
    ResizePreserveAspect,
    Standardize,
    ToRGB,
    ToTensor,
    ToUnitRange,
    Transform,
)
from .pooling import (
    avg_pool2d,
    avg_pool2d_backward,
    global_avg_pool2d,
    global_max_pool2d,
    max_pool2d,
    max_pool2d_backward,
    pool_output_shape,
)
from .resize import (
    RESIZE_METHODS,
    center_crop,
    rescale,
    resize,
    resize_preserve_aspect,
)
from .validation import hwc_to_nchw, nchw_to_hwc

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # -- exceptions ------------------------------------------------------
    "ImgProcError",
    "ShapeError",
    "DTypeError",
    "ParameterError",
    "DependencyError",
    # -- Part 1: padding --------------------------------------------------
    "PAD_MODES",
    "pad2d",
    "pad2d_backward",
    "unpad2d",
    "pad_image",
    "resolve_padding",
    "conv_output_size",
    # -- Part 1: convolution ---------------------------------------------
    "conv2d",
    "conv2d_backward",
    "conv_output_shape",
    "convolve_image",
    # -- Part 1: pooling --------------------------------------------------
    "max_pool2d",
    "max_pool2d_backward",
    "avg_pool2d",
    "avg_pool2d_backward",
    "global_max_pool2d",
    "global_avg_pool2d",
    "pool_output_shape",
    # -- Part 1: layers ---------------------------------------------------
    "Layer",
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "ReLU",
    "Flatten",
    "Sequential",
    # -- Part 1: kernels --------------------------------------------------
    "KERNELS",
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
    # -- Part 2: colour ---------------------------------------------------
    "GRAYSCALE_WEIGHTS",
    "GRAYSCALE_METHODS",
    "to_grayscale",
    "to_rgb",
    "drop_alpha",
    "srgb_to_linear",
    "linear_to_srgb",
    # -- Part 2: normalisation --------------------------------------------
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
    # -- Part 2: resize ---------------------------------------------------
    "RESIZE_METHODS",
    "resize",
    "rescale",
    "resize_preserve_aspect",
    "center_crop",
    # -- Part 2: pipeline --------------------------------------------------
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
    # -- utilities ---------------------------------------------------------
    "hwc_to_nchw",
    "nchw_to_hwc",
    "load_image",
    "save_image",
    "checkerboard",
    "gradient_ramp",
    "circle",
    "sample_rgb_image",
]
