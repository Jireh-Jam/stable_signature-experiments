"""
Compatibility wrapper for the transformations pipeline.

Use `common.transformations.ImageTransformations` going forward.
This module re-exports the class for backward compatibility.
"""

from common.transformations import ImageTransformations  # noqa: F401

__all__ = ["ImageTransformations"]