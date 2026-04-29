"""Asynchronous MoQ server over QUIC.

Loads a 3DGS scene, partitions it, builds a manifest, and streams
Gaussian clusters to subscribed clients over async TCP streams
(placeholder for real QUIC via aioquic).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

from moq3dgs.decorators import network_bound
from moq3dgs.models import (
    ImportanceTier, MoQSubscription, SceneManifest,
    ServerStatus, SubscriptionAck, ViewportUpdate,
)
from moq3dgs.scene.clustering import Cluster, cluster_gaussians_kmeans
from moq3dgs.scene.loader import GaussianScene, find_ply_in_scene_dir, load_ply
from moq3dgs.scene.lod import LoDLayer, split_lod
from moq3dgs.transport.manifest import build_manifest_from_clusters
from moq3dgs.transport.protocol import encode_cluster
from moq3dgs.viewport.frustum import (
    Visibility, extract_frustum,
    projection_matrix_from_fov, check_aabb_frustum,
)
from moq3dgs.viewport.priority import compute_priority
from moq3dgs.viewport.trace import camera_forward_from_view_matrix

logger = structlog.get_logger(__name__)


@dataclass
class ClientSession:
    """Per-client session state."""
    client_id: str
    writer: asyncio.StreamWriter
    subscriptions: Dict[str, MoQSubscription] = field(default_factory=dict)
    last_viewport: Optional[ViewportUpdate] = None
    bytes_sent: int = 0


class MoQServer:
    """Async TCP server simulating MoQ-over-QUIC semantics."""

    def __init__(
        self, scene: GaussianScene, clusters: Dict[int, Cluster],
        manifest: SceneManifest, lod_cache: Dict[int, List[LoDLayer]],
        host: str = "0.0.0.0", port: int = 4433,
    ) -> None:
        self.scene = scene
        self.clusters = clusters
        self.manifest = manifest
        self.lod_cache = lod_cache
        self.host = host
        self.port = port
        self.sessions: Dict[str, ClientSession] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self._start_time = time.monotonic()
        self._total_bytes_sent = 0

    async def start(self) -> None:
        """Start listening for incoming connections."""
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port,
        )
        logger.info("server_started", host=self.host, port=self.port,
                     tracks=len(self.manifest.tracks))

    async def stop(self) -> None:
        """Gracefully shut down."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        """Block until the server is cancelled."""
        if not self._server:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    @network_bound
    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        cid = f"client-{peer[0]}:{peer[1]}" if peer else "unknown"
        session = ClientSession(client_id=cid, writer=writer)
        self.sessions[cid] = session
        logger.info("client_connected", client_id=cid)
        try:
            await self._send_json(writer, {
                "type": "manifest", "data": self.manifest.model_dump(),
            })
            while True:
                raw = await self._read_message(reader)
                if raw is None:
                    break
                await self._dispatch(session, raw)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self.sessions.pop(cid, None)
            writer.close()
            logger.info("client_disconnected", client_id=cid)

    async def _dispatch(self, session: ClientSession, msg: dict) -> None:
        t = msg.get("type")
        if t == "viewport_update":
            await self._on_viewport(session, msg["data"])
        elif t == "subscribe":
            await self._on_subscribe(session, msg["data"])

    async def _on_viewport(self, session: ClientSession, data: dict) -> None:
        update = ViewportUpdate(**data)
        session.last_viewport = update
        # Server can use this for dynamic priority adjustment of in-flight data,
        # but for static scenes, data is sent on subscribe.

    async def _on_subscribe(self, session: ClientSession, data: dict) -> None:
        sub = MoQSubscription(**data)
        session.subscriptions[f"{sub.track_id}/{sub.group_id}"] = sub
        ack = SubscriptionAck(track_id=sub.track_id, group_id=sub.group_id, accepted=True)
        await self._send_json(session.writer, {"type": "subscribe_ack", "data": ack.model_dump()})

        # Find the cluster for this track/group and send it
        # Assuming track_id="track-0000", group_id="group-0000-0"
        try:
            cid = int(sub.group_id.split("-")[1])
        except (IndexError, ValueError):
            return

        cluster = self.clusters.get(cid)
        if not cluster:
            return

        # Send all requested LoD layers
        for lod in self.lod_cache.get(cid, []):
            if lod.tier > sub.max_subgroup_id:
                continue
                
            sh = (lod.sh_dc.reshape(lod.num_gaussians, -1)
                  if lod.sh_dc is not None
                  else (lod.sh_rest.reshape(lod.num_gaussians, -1)
                        if lod.sh_rest is not None else None))
            sc = lod.scales_base if lod.scales_base is not None else lod.scales_delta
            frame = encode_cluster(
                sub.track_id, sub.group_id,
                int(lod.tier), lod.num_gaussians,
                lod.means, lod.opacities, sh, sc, lod.rotations,
            )
            await self._send_binary(session.writer, frame)
            session.bytes_sent += len(frame)
            self._total_bytes_sent += len(frame)

    @staticmethod
    async def _send_json(w: asyncio.StreamWriter, obj: dict) -> None:
        p = json.dumps(obj).encode()
        w.write(len(p).to_bytes(4, "little") + p)
        await w.drain()

    @staticmethod
    async def _send_binary(w: asyncio.StreamWriter, data: bytes) -> None:
        w.write(len(data).to_bytes(4, "little") + data)
        await w.drain()

    @staticmethod
    async def _read_message(r: asyncio.StreamReader) -> Optional[dict]:
        hdr = await r.read(4)
        if len(hdr) < 4:
            return None
        length = int.from_bytes(hdr, "little")
        payload = await r.readexactly(length)
        return json.loads(payload)

    def get_status(self) -> ServerStatus:
        return ServerStatus(
            active_subscriptions=sum(len(s.subscriptions) for s in self.sessions.values()),
            total_tracks=len(self.manifest.tracks),
            total_groups=sum(len(t.groups) for t in self.manifest.tracks),
            bytes_sent=self._total_bytes_sent,
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
        )


def create_server(
    scene_path: str | Path, num_clusters: int = 64,
    host: str = "0.0.0.0", port: int = 4433,
) -> MoQServer:
    """Load scene → cluster → build manifest → return server."""
    ply = find_ply_in_scene_dir(scene_path)
    logger.info("loading_scene", path=str(ply))
    scene = load_ply(ply)
    logger.info("scene_loaded", n=scene.num_gaussians)
    clusters = cluster_gaussians_kmeans(scene, num_clusters=num_clusters)
    lod_cache: Dict[int, List[LoDLayer]] = {}
    for cid, c in clusters.items():
        lod_cache[cid] = split_lod(scene, c.indices)
    manifest = build_manifest_from_clusters(clusters)
    return MoQServer(scene, clusters, manifest, lod_cache, host, port)
