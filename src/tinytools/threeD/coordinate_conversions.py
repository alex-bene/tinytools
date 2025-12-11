"""Coordinate conversion utilities.

This module provides matrices for converting between different 3D coordinate systems,
such as OpenGL and PyTorch3D.
"""

from __future__ import annotations

import numpy as np

from tinytools.metaclasses import FrozenNamespaceMeta


class CoordinateConversions(metaclass=FrozenNamespaceMeta):
    """Coordinate conversion matrices.

    Supports dot access and dict-style access.
    """

    opengl_to_pt3d = np.array(
        [
            [-1.0, 0.0, 0.0],  # Flip X (right to left)
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],  # Flip Z (forward to backward)
        ]
    )
    pt3d_to_opengl = np.array(
        [
            [-1.0, 0.0, 0.0],  # Flip X (left to right)
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],  # Flip Z (backward to forward)
        ]
    )
    cv_to_pt3d = np.array(
        [
            [-1.0, 0.0, 0.0],  # Flip X (right to left)
            [0.0, -1.0, 0.0],  # Flip Y (top to bottom)
            [0.0, 0.0, 1.0],
        ]
    )
    pt3d_to_cv = np.array(
        [
            [-1.0, 0.0, 0.0],  # Flip X (left to right)
            [0.0, -1.0, 0.0],  # Flip Y (bottom to top)
            [0.0, 0.0, 1.0],
        ]
    )
