"""Integration test: server + client end-to-end over localhost."""

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
        client = MoQClient(host="127.0.0.1", port=port)
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
        await asyncio.sleep(0.3)

        received = []
        while not client.received_queue.empty():
            received.append(client.received_queue.get_nowait())

        # We should have received at least some clusters
        assert len(received) > 0
        # Each cluster should have a track_id and group_id
        for cluster in received:
            assert "track_id" in cluster
            assert "group_id" in cluster
            assert cluster["num_gaussians"] > 0

        await client.disconnect()
    finally:
        await server.stop()
