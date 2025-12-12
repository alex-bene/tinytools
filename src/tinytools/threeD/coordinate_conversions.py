"""Coordinate conversion utilities.

This module provides matrices for converting between different 3D coordinate systems,
such as OpenGL, OpenCV and PyTorch3D.
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
    opengl_to_cv = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],  # Flip Y (top to bottom)
            [0.0, 0.0, -1.0],  # Flip Z (forward to backward)
        ]
    )
    cv_to_pt3d = np.array(
        [
            [-1.0, 0.0, 0.0],  # Flip X (right to left)
            [0.0, -1.0, 0.0],  # Flip Y (top to bottom)
            [0.0, 0.0, 1.0],
        ]
    )
    pt3d_to_cv = cv_to_pt3d
    cv_to_opengl = opengl_to_cv
    pt3d_to_opengl = opengl_to_pt3d
