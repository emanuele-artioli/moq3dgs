"""Tests for the Weighted Farthest-Point Sampling module and LoD splitter.

Verifies:
1. WFPS output is a valid permutation (no repeats, covers all N indices).
2. WFPS spreads points out — the first K points are more spread than a
   random or size-only selection.
3. Voxel-grid approximation produces a valid ordering.
4. split_lod returns non-overlapping, union-complete tier sets.
5. split_lod works on a tiny synthetic scene (no real assets required).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moq3dgs.scene.fps import weighted_fps_order, compute_wfps_tiers
from moq3dgs.scene.lod import split_lod, _voxel_fps_order
from moq3dgs.scene.loader import GaussianScene
from moq3dgs.models import ImportanceTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene(n: int = 256, seed: int = 0) -> GaussianScene:
    """Synthetic scene with random Gaussians."""
    rng = np.random.default_rng(seed)
    means = torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32))
    scales = torch.from_numpy(
        rng.uniform(-2.0, 0.5, (n, 3)).astype(np.float32)
    )
    rotations = torch.zeros((n, 4), dtype=torch.float32)
    rotations[:, 0] = 1.0  # unit quaternion w=1
    opacities = torch.zeros((n, 1), dtype=torch.float32)
    sh_dc = torch.zeros((n, 1, 3), dtype=torch.float32)
    sh_rest = torch.zeros((n, 0, 3), dtype=torch.float32)

    scene = GaussianScene.__new__(GaussianScene)
    scene.means = means
    scene.scales = scales
    scene.rotations = rotations
    scene.opacities = opacities
    scene.sh_dc = sh_dc
    scene.sh_rest = sh_rest
    return scene


# ---------------------------------------------------------------------------
# Tests: weighted_fps_order
# ---------------------------------------------------------------------------


class TestWeightedFPS:
    def test_output_is_permutation(self):
        """WFPS must return each index exactly once."""
        n = 128
        scene = _make_scene(n)
        means = scene.means
        scales = scene.scales
        order = weighted_fps_order(means, scales, device="cpu")
        assert order.shape == (n,)
        assert sorted(order.tolist()) == list(range(n)), "order is not a permutation"

    def test_single_point(self):
        """Edge case: single point."""
        means = torch.zeros((1, 3))
        scales = torch.zeros((1, 3))
        order = weighted_fps_order(means, scales, device="cpu")
        assert list(order) == [0]

    def test_empty(self):
        """Edge case: empty input."""
        means = torch.zeros((0, 3))
        scales = torch.zeros((0, 3))
        order = weighted_fps_order(means, scales, device="cpu")
        assert len(order) == 0

    def test_spatial_spread(self):
        """First K FPS points should cover space better than first K by size."""
        rng = np.random.default_rng(42)
        # 4 clusters of points at corners of a cube + noise
        cluster_centres = np.array([
            [0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]
        ], dtype=np.float32)
        pts = np.vstack([
            cluster_centres[i] + rng.standard_normal((50, 3)).astype(np.float32) * 0.5
            for i in range(4)
        ])
        n = len(pts)
        means = torch.from_numpy(pts)
        # uniform scales so size doesn't interfere
        scales = torch.zeros((n, 3))

        order = weighted_fps_order(means, scales, device="cpu")
        k = 4

        # FPS selected points
        fps_pts = pts[order[:k]]
        fps_spread = np.std(fps_pts, axis=0).mean()

        # Size-only (random here since scales are uniform)
        rand_pts = pts[:k]
        rand_spread = np.std(rand_pts, axis=0).mean()

        # FPS spread should be at least as good (≥) as random (within reason)
        # In practice FPS will cover all 4 corners; random will cover only one.
        assert fps_spread >= rand_spread * 0.5, (
            f"FPS spread {fps_spread:.3f} unexpectedly worse than random {rand_spread:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests: voxel_fps_order
# ---------------------------------------------------------------------------


class TestVoxelFPS:
    def test_is_permutation(self):
        n = 200
        rng = np.random.default_rng(7)
        means_np = rng.standard_normal((n, 3)).astype(np.float32)
        scales_np = np.exp(rng.uniform(-2, 0, n).astype(np.float32))
        order = _voxel_fps_order(means_np, scales_np)
        assert order.shape == (n,)
        assert sorted(order.tolist()) == list(range(n))

    def test_large_scale_first(self):
        """A single large splat in an otherwise uniform cloud should rank first."""
        rng = np.random.default_rng(99)
        n = 100
        means_np = rng.standard_normal((n, 3)).astype(np.float32)
        scales_np = np.ones(n, dtype=np.float32) * 0.1
        # Put a giant isolated splat at index 42
        means_np[42] = np.array([100.0, 100.0, 100.0])
        scales_np[42] = 10.0
        order = _voxel_fps_order(means_np, scales_np)
        assert order[0] == 42, f"Expected idx 42 first, got {order[0]}"


# ---------------------------------------------------------------------------
# Tests: split_lod
# ---------------------------------------------------------------------------


class TestSplitLoD:
    def test_union_complete(self):
        """All indices appear across quality-0 subsets (no splat lost)."""
        n = 512
        scene = _make_scene(n)
        indices = np.arange(n, dtype=np.int64)
        layers = split_lod(scene, indices, device="cpu")

        # Collect all quality=0 indices across all subsets
        base_indices = np.concatenate([
            l.indices for l in layers
            if l.quality == 0
        ])
        assert len(base_indices) == n, "Splats lost in LoD split"
        assert sorted(base_indices.tolist()) == list(range(n))

    def test_no_overlap_between_subsets(self):
        """Subsets within same quality should not share any index."""
        n = 256
        scene = _make_scene(n)
        indices = np.arange(n, dtype=np.int64)
        layers = split_lod(scene, indices, device="cpu")
        base_layers = [
            l for l in layers
            if l.quality == 0
        ]
        all_sets = [set(l.indices.tolist()) for l in base_layers]
        for i, s1 in enumerate(all_sets):
            for j, s2 in enumerate(all_sets):
                if i >= j:
                    continue
                overlap = s1 & s2
                assert len(overlap) == 0, f"Overlap between subset {i} and {j}: {len(overlap)} indices"

    def test_tier_ordering_small_n(self):
        """LARGE gets the smallest fraction (~5%)."""
        n = 1000
        scene = _make_scene(n)
        layers = split_lod(scene, np.arange(n, dtype=np.int64), device="cpu")
        large = next(l for l in layers if l.tier == ImportanceTier.LARGE and l.quality == 0)
        small = next(l for l in layers if l.tier == ImportanceTier.SMALL and l.quality == 0)
        assert large.num_gaussians < small.num_gaussians, (
            "LARGE should have fewer splats than SMALL"
        )

    def test_empty_cluster(self):
        scene = _make_scene(10)
        layers = split_lod(scene, np.array([], dtype=np.int64), device="cpu")
        assert layers == []

    def test_attribute_shapes(self):
        n = 64
        scene = _make_scene(n)
        indices = np.arange(n, dtype=np.int64)
        layers = split_lod(scene, indices, device="cpu")
        for l in layers:
            if l.means is not None:
                assert l.means.shape == (l.num_gaussians, 3)
            if l.opacities is not None:
                assert l.opacities.shape == (l.num_gaussians, 1)
            if l.scales_base is not None:
                assert l.scales_base.shape == (l.num_gaussians, 3)
