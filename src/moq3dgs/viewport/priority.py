"""Dynamic priority calculation for MoQ subscription scheduling.

Priority ranges from 0 (highest / critical) to 255 (lowest / droppable).
It is computed from three factors:
    1. Distance to camera — closer = higher priority.
    2. Frustum position — centre screen = highest, periphery = medium,
       occluded / behind = lowest.
    3. Layer type — base LoD (Object 0) gets a boost over enhancement
       LoD (Object 1).
"""

from __future__ import annotations

from typing import List

import numpy as np

from moq3dgs.models import ImportanceTier
from moq3dgs.viewport.frustum import (
    Frustum,
    Visibility,
    aabb_center,
    check_aabb_frustum,
)


def _distance_score(
    camera_pos: np.ndarray,
    cluster_center: np.ndarray,
    max_distance: float,
) -> float:
    """Map Euclidean distance to a [0, 1] score where 0 = close, 1 = far.

    Distances beyond ``max_distance`` are clamped to 1.0.
    """
    dist = float(np.linalg.norm(camera_pos - cluster_center))
    return min(dist / max(max_distance, 1e-6), 1.0)


def _frustum_score(
    cluster_center: np.ndarray,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    visibility: Visibility,
) -> float:
    """Map frustum position to a [0, 1] score.

    0.0 = dead centre of the view, 0.5 = periphery, 1.0 = outside / behind.

    The centre-ness is measured by the cosine of the angle between the
    camera forward vector and the direction to the cluster centre.
    """
    if visibility == Visibility.OUTSIDE:
        return 1.0

    direction = cluster_center - camera_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-8:
        return 0.0
    cos_angle = float(np.dot(camera_forward, direction / dist))
    # cos_angle in [-1, 1]; map to [0, 1] where 1 = behind
    score = (1.0 - cos_angle) / 2.0
    return score


def _lod_score(tier: ImportanceTier) -> float:
    """Base geometry (0-2) gets highest priority boost, enhancements (3-4) get less."""
    scores = {
        ImportanceTier.BASE_LARGE: 0.0,
        ImportanceTier.BASE_MEDIUM: 0.05,
        ImportanceTier.BASE_SMALL: 0.10,
        ImportanceTier.ENHANCE_LARGE: 0.20,
        ImportanceTier.ENHANCE_MEDIUM: 0.30,
    }
    return scores.get(tier, 0.5)


def compute_priority(
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    cluster_bbox_min: np.ndarray,
    cluster_bbox_max: np.ndarray,
    frustum: Frustum,
    subgroup_id: ImportanceTier,
    max_distance: float = 50.0,
    *,
    weight_distance: float = 0.4,
    weight_frustum: float = 0.4,
    weight_lod: float = 0.2,
) -> int:
    """Compute the MoQ priority (0-255) for a cluster + LoD combination.

    Lower values mean higher priority.  The three sub-scores are
    independently normalised to [0, 1] and then combined with the given
    weights before scaling to the 0-255 integer range.

    Args:
        camera_pos: (3,) camera world position.
        camera_forward: (3,) unit camera forward vector.
        cluster_bbox_min: (3,) AABB lower corner.
        cluster_bbox_max: (3,) AABB upper corner.
        frustum: Current view frustum.
        lod_level: Which LoD layer this priority is for.
        max_distance: Distance at which a cluster drops to minimum priority.
        weight_distance: Weight for the distance component.
        weight_frustum: Weight for the frustum-position component.
        weight_lod: Weight for the LoD component.

    Returns:
        Integer priority in [0, 255].
    """
    center = aabb_center(cluster_bbox_min, cluster_bbox_max)
    visibility = check_aabb_frustum(cluster_bbox_min, cluster_bbox_max, frustum)

    d_score = _distance_score(camera_pos, center, max_distance)
    f_score = _frustum_score(center, camera_pos, camera_forward, visibility)
    l_score = _lod_score(subgroup_id)

    combined = (
        weight_distance * d_score
        + weight_frustum * f_score
        + weight_lod * l_score
    )
    return int(np.clip(combined * 255.0, 0, 255))


def batch_compute_priorities(
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    frustum: Frustum,
    clusters: List[dict],
    subgroup_id: ImportanceTier,
    max_distance: float = 50.0,
) -> List[int]:
    """Compute priorities for a batch of clusters at once.

    Each entry in *clusters* must have keys ``bbox_min`` and ``bbox_max``
    (both np.ndarray of shape (3,)).

    Args:
        camera_pos: (3,) camera world position.
        camera_forward: (3,) unit camera forward vector.
        frustum: Current view frustum.
        clusters: List of dicts with ``bbox_min`` and ``bbox_max``.
        lod_level: LoD layer to score.
        max_distance: Distance ceiling.

    Returns:
        List of integer priorities, one per cluster.
    """
    return [
        compute_priority(
            camera_pos,
            camera_forward,
            c["bbox_min"],
            c["bbox_max"],
            frustum,
            subgroup_id,
            max_distance,
        )
        for c in clusters
    ]
