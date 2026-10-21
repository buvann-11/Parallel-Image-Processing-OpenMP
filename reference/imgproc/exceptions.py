"""Exception hierarchy for the :mod:`imgproc` package.

A single, well-defined exception hierarchy lets callers write precise
``except`` clauses instead of catching bare ``ValueError``/``TypeError``
and guessing what went wrong.

Hierarchy
---------
``ImgProcError``
  ├── ``ShapeError``          -- array rank / geometry is wrong
  ├── ``DTypeError``          -- array dtype is unsupported
  ├── ``ParameterError``      -- a keyword argument has an invalid value
  └── ``DependencyError``     -- an optional third-party dependency is missing

All of them inherit from :class:`ValueError` as well, so existing code that
catches ``ValueError`` keeps working.
"""

from __future__ import annotations

__all__ = [
    "ImgProcError",
    "ShapeError",
    "DTypeError",
    "ParameterError",
    "DependencyError",
]


class ImgProcError(ValueError):
    """Base class for every error raised by :mod:`imgproc`."""


class ShapeError(ImgProcError):
    """Raised when an array has an unexpected rank or geometry."""


class DTypeError(ImgProcError, TypeError):
    """Raised when an array has an unsupported dtype."""


class ParameterError(ImgProcError):
    """Raised when an argument is outside its permitted domain."""


class DependencyError(ImgProcError, ImportError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, package: str, feature: str) -> None:
        super().__init__(
            f"The optional dependency {package!r} is required for {feature}. "
            f"Install it with: pip install {package}"
        )
        self.package = package
        self.feature = feature
