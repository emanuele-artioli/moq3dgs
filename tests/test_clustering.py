"""Unit tests for spatial clustering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moq3dgs.scene.clustering import (
    Cluster,
    OctreeNode,
    build_octree,
    cluster_gaussians_kmeans,
    collect_leaves,
)
from moq3dgs.scene.loader import GaussianScene


def _make_scene(n: int = 1000) -> GaussianScene:
    """Create a synthetic scene with N random Gaussians."""
    rng = np.random.default_rng(42)
    return GaussianScene(
        means=torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32)),
        opacities=torch.zeros(n, 1),
        scales=torch.zeros(n, 3),
        rotations=torch.tensor([[1, 0, 0, 0]] * n, dtype=torch.float32),
        sh_dc=torch.zeros(n, 1, 3),
        sh_rest=torch.zeros(n, 0, 3),
        num_gaussians=n,
    )


class TestKMeansClustering:
    """K-means clustering tests."""

    def test_basic_clustering(self) -> None:
        scene = _make_scene(500)
        clusters = cluster_gaussians_kmeans(scene, num_clusters=8)
        assert len(clusters) > 0
        # All indices should be covered
        all_idx = np.concatenate([c.indices for c in clusters.values()])
        assert len(np.unique(all_idx)) == 500

    def test_cluster_bboxes(self) -> None:
        scene = _make_scene(200)
        clusters = cluster_gaussians_kmeans(scene, num_clusters=4)
        for c in clusters.values():
            pts = scene.means.numpy()[c.indices]
            np.testing.assert_array_less(c.bbox_min - 1e-6, pts.min(axis=0))
            np.testing.assert_array_less(pts.max(axis=0), c.bbox_max + 1e-6)


class TestOctree:
    """Octree partitioning tests."""

    def test_octree_leaves_cover_all_points(self) -> None:
        scene = _make_scene(800)
        root = build_octree(scene, max_depth=3, min_points=50)
        leaves = collect_leaves(root)
        all_idx = np.concatenate([l.indices for l in leaves if l.indices is not None])
        assert len(np.unique(all_idx)) == 800

    def test_octree_depth(self) -> None:
        scene = _make_scene(100)
        root = build_octree(scene, max_depth=2, min_points=10)
        leaves = collect_leaves(root)
        for leaf in leaves:
            assert leaf.depth <= 2
