"""Spatial clustering of Gaussians for MoQ Track / Group assignment.

Two strategies are provided:
1. **K-means** — fast, flat partitioning suitable for moderate scene sizes.
2. **Octree** — hierarchical partitioning that naturally maps to the
   MoQ Track (coarse node) → Group (leaf node) hierarchy and allows
   multi-resolution frustum culling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from moq3dgs.scene.loader import GaussianScene


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """A spatial cluster of Gaussians.

    Stores the indices into the original scene arrays and the axis-aligned
    bounding box of the cluster.
    """

    cluster_id: int
    indices: np.ndarray          # int64 indices into the scene
    bbox_min: np.ndarray         # (3,)
    bbox_max: np.ndarray         # (3,)
    centroid: np.ndarray         # (3,)

    @property
    def num_gaussians(self) -> int:
        return len(self.indices)


@dataclass
class OctreeNode:
    """A single node in the spatial octree.

    Leaf nodes hold Gaussian indices; internal nodes hold children.
    """

    node_id: str
    depth: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    children: List["OctreeNode"] = field(default_factory=list)
    indices: Optional[np.ndarray] = None   # only set for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def centroid(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2.0


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------


def cluster_gaussians_kmeans(
    scene: GaussianScene,
    num_clusters: int = 64,
    batch_size: int = 4096,
    random_state: int = 42,
) -> Dict[int, Cluster]:
    """Partition Gaussians into spatial clusters via Mini-Batch K-Means.

    Each cluster maps to a single MoQ Group.  The caller decides how to
    assign clusters to Tracks (e.g., one track = one cluster, or grouping
    nearby clusters into a shared track).

    Args:
        scene: Loaded 3DGS scene.
        num_clusters: Target number of spatial clusters.
        batch_size: Mini-batch size for scalability on large scenes.
        random_state: Seed for reproducibility.

    Returns:
        Dict mapping cluster_id → :class:`Cluster`.
    """
    means_np: np.ndarray = scene.means.numpy()  # (N, 3)

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(means_np)

    clusters: Dict[int, Cluster] = {}
    for cid in range(num_clusters):
        mask = labels == cid
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        pts = means_np[idx]
        clusters[cid] = Cluster(
            cluster_id=cid,
            indices=idx,
            bbox_min=pts.min(axis=0),
            bbox_max=pts.max(axis=0),
            centroid=pts.mean(axis=0),
        )
    return clusters


# ---------------------------------------------------------------------------
# Octree partitioning
# ---------------------------------------------------------------------------


def _build_octree_recursive(
    node: OctreeNode,
    points: np.ndarray,
    indices: np.ndarray,
    max_depth: int,
    min_points: int,
) -> None:
    """Recursively subdivide an octree node into 8 children."""
    if node.depth >= max_depth or len(indices) <= min_points:
        node.indices = indices
        return

    mid = node.centroid
    for octant in range(8):
        # Determine octant bounds
        child_min = node.bbox_min.copy()
        child_max = node.bbox_max.copy()
        for axis in range(3):
            if octant & (1 << axis):
                child_min[axis] = mid[axis]
            else:
                child_max[axis] = mid[axis]

        # Filter points inside this octant.
        # Use inclusive upper bound for the "high" octants (those touching
        # the node's bbox_max) so that points sitting exactly on the
        # boundary are not lost.
        mask = np.ones(len(indices), dtype=bool)
        for axis in range(3):
            mask &= points[indices, axis] >= child_min[axis]
            if octant & (1 << axis):
                # High side — include upper boundary
                mask &= points[indices, axis] <= child_max[axis]
            else:
                # Low side — strict upper bound to avoid double-counting
                mask &= points[indices, axis] < child_max[axis]

        child_indices = indices[mask]
        if len(child_indices) == 0:
            continue

        child = OctreeNode(
            node_id=f"{node.node_id}_{octant}",
            depth=node.depth + 1,
            bbox_min=child_min,
            bbox_max=child_max,
        )
        _build_octree_recursive(child, points, child_indices, max_depth, min_points)
        node.children.append(child)

    # If all points ended up in a single child, collapse
    if len(node.children) == 1:
        only = node.children[0]
        node.indices = only.indices if only.is_leaf else None
        node.children = only.children


def build_octree(
    scene: GaussianScene,
    max_depth: int = 4,
    min_points: int = 256,
) -> OctreeNode:
    """Build a spatial octree over the Gaussian centres.

    The root node covers the scene AABB.  Subdivision continues until
    ``max_depth`` or until a node has fewer than ``min_points`` Gaussians.

    Args:
        scene: Loaded 3DGS scene.
        max_depth: Maximum octree depth (2^depth cells per axis at most).
        min_points: Stop splitting when a node has this few points or less.

    Returns:
        Root :class:`OctreeNode`.
    """
    means_np = scene.means.numpy()
    all_indices = np.arange(len(means_np), dtype=np.int64)

    root = OctreeNode(
        node_id="root",
        depth=0,
        bbox_min=means_np.min(axis=0).copy(),
        bbox_max=means_np.max(axis=0).copy(),
    )
    _build_octree_recursive(root, means_np, all_indices, max_depth, min_points)
    return root


def collect_leaves(node: OctreeNode) -> List[OctreeNode]:
    """Flatten the octree into a list of leaf nodes.

    Each leaf becomes a MoQ Group; its parent chain determines the
    Track assignment.
    """
    if node.is_leaf:
        return [node]
    leaves: List[OctreeNode] = []
    for child in node.children:
        leaves.extend(collect_leaves(child))
    return leaves
