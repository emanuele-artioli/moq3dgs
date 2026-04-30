"""Integration test: server + client end-to-end over localhost.

Tests the complete pipeline:
  - Server starts with synthetic scene data.
  - Client connects, receives manifest.
  - Client sends viewport update.
  - Server streams matching clusters.
  - Client receives and caches clusters.
  - Client renders a frame via CPU fallback and saves to disk.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from moq3dgs.models import SceneManifest, ViewportUpdate
from moq3dgs.scene.clustering import Cluster, cluster_gaussians_kmeans
from moq3dgs.scene.loader import GaussianScene
from moq3dgs.scene.lod import LoDLayer, split_lod
from moq3dgs.transport.client import MoQClient
from moq3dgs.transport.manifest import build_manifest_from_clusters
from moq3dgs.transport.server import MoQServer


def _make_scene(n: int = 200) -> GaussianScene:
    """Create a synthetic scene for testing."""
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


def _build_server(port: int) -> MoQServer:
    """Build a server with synthetic data."""
    scene = _make_scene()
    clusters = cluster_gaussians_kmeans(scene, num_clusters=4)
    lod_cache: Dict[int, List[LoDLayer]] = {}
    for cid, c in clusters.items():
        lod_cache[cid] = split_lod(scene, c.indices)
    manifest = build_manifest_from_clusters(clusters)
    return MoQServer(scene, clusters, manifest, lod_cache, host="127.0.0.1", port=port)


@pytest.mark.asyncio
async def test_server_client_roundtrip() -> None:
    """Start a server, connect a client, send a viewport update, receive clusters."""
    port = 14433  # use a high port to avoid conflicts
    server = _build_server(port)
    await server.start()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MoQClient(
                host="127.0.0.1",
                port=port,
                output_dir=tmpdir,
                render_enabled=False,  # skip rendering in unit tests
                device="cpu",
            )
            manifest = await client.connect()
            assert len(manifest.tracks) > 0

            await client.start_receiving()

            update = ViewportUpdate(
                client_id="test-client",
                timestamp_ms=0,
                camera_position=[0.0, 0.0, 0.0],
                view_matrix=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                fov=90.0,
            )
            await client.send_viewport_update(update)
            # Give the server time to respond
            await asyncio.sleep(0.5)

            # We should have received at least some clusters
            assert client.cache.num_entries > 0
            assert client._clusters_received > 0

            # Verify cache has valid data
            assembled = client.cache.assemble_base()
            assert assembled is not None
            assert assembled["num_gaussians"] > 0
            assert assembled["means"] is not None

            # Verify manifest was saved
            assert Path(tmpdir, "manifest.json").exists()

            await client.disconnect()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_render_cpu_fallback() -> None:
    """Test that the client can render a frame using CPU fallback."""
    port = 14434
    server = _build_server(port)
    await server.start()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MoQClient(
                host="127.0.0.1",
                port=port,
                output_dir=tmpdir,
                render_enabled=True,
                device="cpu",  # force CPU fallback
                image_width=320,
                image_height=240,
            )
            await client.connect()
            await client.start_receiving()

            update = ViewportUpdate(
                client_id="test-client",
                timestamp_ms=0,
                camera_position=[0.0, 0.0, 5.0],
                view_matrix=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -5],
                    [0, 0, 0, 1],
                ],
                fov=60.0,
            )
            await client.send_viewport_update(update)
            await asyncio.sleep(0.5)

            # Render
            path = client.render_frame(
                frame_idx=0,
                view_matrix=update.view_matrix,
                fov=update.fov,
                camera_position=update.camera_position,
            )

            await client.disconnect()

            # Should have rendered something (even if mostly black)
            assert path is not None
            assert path.exists()
            assert path.suffix == ".mp4"

            # Check metrics file
            metrics_path = Path(tmpdir, "frames", "metrics.jsonl")
            assert metrics_path.exists()
    finally:
        await server.stop()
