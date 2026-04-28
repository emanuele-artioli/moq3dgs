"""Unit tests for priority calculation."""

from __future__ import annotations

import numpy as np
import pytest

from moq3dgs.models import LoDLevel
from moq3dgs.viewport.frustum import extract_frustum, projection_matrix_from_fov
from moq3dgs.viewport.priority import compute_priority


def _default_frustum():
    proj = projection_matrix_from_fov(60.0)
    view = np.eye(4, dtype=np.float64)
    return extract_frustum(proj @ view)


class TestPriority:
    """Priority scoring tests."""

    def test_close_centre_base_is_highest(self) -> None:
        """A close, centred, base-LoD cluster should have near-zero priority."""
        frustum = _default_frustum()
        p = compute_priority(
            camera_pos=np.array([0, 0, 0]),
            camera_forward=np.array([0, 0, -1]),
            cluster_bbox_min=np.array([-1, -1, -5]),
            cluster_bbox_max=np.array([1, 1, -3]),
            frustum=frustum,
            lod_level=LoDLevel.BASE,
            max_distance=50.0,
        )
        assert p < 50, f"Expected high priority (<50) but got {p}"

    def test_far_behind_enhancement_is_lowest(self) -> None:
        """A far, behind-camera, enhancement cluster should have high priority."""
        frustum = _default_frustum()
        p = compute_priority(
            camera_pos=np.array([0, 0, 0]),
            camera_forward=np.array([0, 0, -1]),
            cluster_bbox_min=np.array([-1, -1, 40]),
            cluster_bbox_max=np.array([1, 1, 50]),
            frustum=frustum,
            lod_level=LoDLevel.ENHANCEMENT,
            max_distance=50.0,
        )
        assert p > 150, f"Expected low priority (>150) but got {p}"

    def test_base_beats_enhancement(self) -> None:
        """Same position, base should beat enhancement."""
        frustum = _default_frustum()
        kwargs = dict(
            camera_pos=np.array([0, 0, 0]),
            camera_forward=np.array([0, 0, -1]),
            cluster_bbox_min=np.array([-1, -1, -10]),
            cluster_bbox_max=np.array([1, 1, -5]),
            frustum=frustum,
            max_distance=50.0,
        )
        p_base = compute_priority(**kwargs, lod_level=LoDLevel.BASE)
        p_enh = compute_priority(**kwargs, lod_level=LoDLevel.ENHANCEMENT)
        assert p_base < p_enh

    def test_priority_in_range(self) -> None:
        """Priority must be in [0, 255]."""
        frustum = _default_frustum()
        for dist in [0.1, 1, 10, 100]:
            p = compute_priority(
                camera_pos=np.array([0, 0, 0]),
                camera_forward=np.array([0, 0, -1]),
                cluster_bbox_min=np.array([-1, -1, -dist - 1]),
                cluster_bbox_max=np.array([1, 1, -dist]),
                frustum=frustum,
                lod_level=LoDLevel.BASE,
            )
            assert 0 <= p <= 255
