"""Unit tests for manifest generation."""

from __future__ import annotations

import numpy as np
import pytest

from moq3dgs.scene.clustering import Cluster
from moq3dgs.transport.manifest import build_manifest_from_clusters


def _make_clusters(n: int = 4) -> dict:
    clusters = {}
    for i in range(n):
        clusters[i] = Cluster(
            cluster_id=i,
            indices=np.arange(100 * i, 100 * (i + 1)),
            bbox_min=np.array([i, 0, 0], dtype=np.float32),
            bbox_max=np.array([i + 1, 1, 1], dtype=np.float32),
            centroid=np.array([i + 0.5, 0.5, 0.5], dtype=np.float32),
        )
    return clusters


class TestManifest:
    def test_manifest_track_count(self) -> None:
        clusters = _make_clusters(8)
        manifest = build_manifest_from_clusters(clusters)
        assert len(manifest.tracks) == 8

    def test_manifest_total_gaussians(self) -> None:
        clusters = _make_clusters(4)
        manifest = build_manifest_from_clusters(clusters)
        assert manifest.total_gaussians == 400

    def test_manifest_json_roundtrip(self) -> None:
        clusters = _make_clusters(2)
        manifest = build_manifest_from_clusters(clusters, broadcast_name="test-scene")
        data = manifest.model_dump_json()
        from moq3dgs.models import SceneManifest
        restored = SceneManifest.model_validate_json(data)
        assert restored.broadcast_name == "test-scene"
        assert len(restored.tracks) == 2
