"""MoQ manifest generation.

Converts the spatial partitioning results (clusters, octree leaves) into
a :class:`SceneManifest` that the server distributes to clients so they
know which track/group IDs exist and where they sit in world space.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from moq3dgs.models import GroupInfo, SceneManifest, TrackInfo
from moq3dgs.scene.clustering import Cluster


def build_manifest_from_clusters(
    clusters: Dict[int, Cluster],
    broadcast_name: str = "3dgs-scene",
    groups_per_track: int = 1,
) -> SceneManifest:
    """Build a SceneManifest where each cluster becomes a Track with one Group.

    For finer granularity, set ``groups_per_track > 1`` to sub-divide
    each cluster into multiple Groups (not yet implemented — placeholder).

    Args:
        clusters: Mapping of cluster_id → :class:`Cluster`.
        broadcast_name: Human-readable broadcast name.
        groups_per_track: Number of MoQ Groups per Track.

    Returns:
        :class:`SceneManifest`.
    """
    total_gaussians = 0
    tracks: List[TrackInfo] = []

    for cid, cluster in sorted(clusters.items()):
        total_gaussians += cluster.num_gaussians
        track_id = f"track-{cid:04d}"
        group_id = f"group-{cid:04d}-0"

        group = GroupInfo(
            group_id=group_id,
            num_gaussians=cluster.num_gaussians,
            bbox_min=cluster.bbox_min.tolist(),
            bbox_max=cluster.bbox_max.tolist(),
            available_objects=[0, 1],
        )
        track = TrackInfo(
            track_id=track_id,
            bbox_min=cluster.bbox_min.tolist(),
            bbox_max=cluster.bbox_max.tolist(),
            groups=[group],
        )
        tracks.append(track)

    return SceneManifest(
        broadcast_name=broadcast_name,
        total_gaussians=total_gaussians,
        tracks=tracks,
    )


def build_manifest_from_octree_leaves(
    leaves: list,
    broadcast_name: str = "3dgs-scene",
) -> SceneManifest:
    """Build a SceneManifest from octree leaf nodes.

    Each leaf becomes a Group under a Track derived from its parent's
    node_id.

    Args:
        leaves: List of :class:`OctreeNode` leaf nodes.
        broadcast_name: Human-readable broadcast name.

    Returns:
        :class:`SceneManifest`.
    """
    from moq3dgs.scene.clustering import OctreeNode

    # Group leaves by their parent (everything before last '_<octant>')
    track_map: Dict[str, List[OctreeNode]] = {}
    total_gaussians = 0
    for leaf in leaves:
        parts = leaf.node_id.rsplit("_", 1)
        parent_id = parts[0] if len(parts) > 1 else leaf.node_id
        track_map.setdefault(parent_id, []).append(leaf)
        if leaf.indices is not None:
            total_gaussians += len(leaf.indices)

    tracks: List[TrackInfo] = []
    for parent_id, group_leaves in sorted(track_map.items()):
        groups: List[GroupInfo] = []
        track_bmin = np.full(3, np.inf)
        track_bmax = np.full(3, -np.inf)

        for i, leaf in enumerate(group_leaves):
            track_bmin = np.minimum(track_bmin, leaf.bbox_min)
            track_bmax = np.maximum(track_bmax, leaf.bbox_max)
            ng = len(leaf.indices) if leaf.indices is not None else 0
            groups.append(
                GroupInfo(
                    group_id=f"{leaf.node_id}",
                    num_gaussians=ng,
                    bbox_min=leaf.bbox_min.tolist(),
                    bbox_max=leaf.bbox_max.tolist(),
                    available_objects=[0, 1],
                )
            )

        tracks.append(
            TrackInfo(
                track_id=parent_id,
                bbox_min=track_bmin.tolist(),
                bbox_max=track_bmax.tolist(),
                groups=groups,
            )
        )

    return SceneManifest(
        broadcast_name=broadcast_name,
        total_gaussians=total_gaussians,
        tracks=tracks,
    )
