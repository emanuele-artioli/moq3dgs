"""Asynchronous MoQ client over QUIC.

Connects to the server, receives the manifest, replays a camera trace,
sends viewport updates, and receives Gaussian cluster frames which are
deposited into an asyncio.Queue for the rendering pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

from moq3dgs.decorators import network_bound
from moq3dgs.models import (
    LoDLevel, MoQSubscription, SceneManifest, ViewportUpdate,
)
from moq3dgs.transport.protocol import decode_cluster
from moq3dgs.viewport.frustum import (
    Visibility, extract_frustum,
    projection_matrix_from_fov, check_aabb_frustum,
)
from moq3dgs.viewport.priority import compute_priority
from moq3dgs.viewport.trace import (
    camera_forward_from_view_matrix, load_trace, replay_trace,
)

logger = structlog.get_logger(__name__)


class MoQClient:
    """Async TCP client that connects to a MoQServer."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 4433,
        client_id: str = "client-0",
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.manifest: Optional[SceneManifest] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self.received_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False

    async def connect(self) -> SceneManifest:
        """Connect and receive the manifest."""
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port,
        )
        logger.info("connected", host=self.host, port=self.port)
        msg = await self._read_json()
        assert msg and msg["type"] == "manifest"
        self.manifest = SceneManifest(**msg["data"])
        logger.info("manifest_received", tracks=len(self.manifest.tracks))
        return self.manifest

    async def send_viewport_update(self, update: ViewportUpdate) -> None:
        """Send a viewport update to the server."""
        assert self._writer is not None
        await self._send_json({"type": "viewport_update", "data": update.model_dump()})

    async def subscribe(self, sub: MoQSubscription) -> None:
        """Send a subscription request."""
        assert self._writer is not None
        await self._send_json({"type": "subscribe", "data": sub.model_dump()})

    async def start_receiving(self) -> None:
        """Start background task that reads incoming frames."""
        self._running = True
        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Read binary frames from the server and enqueue them."""
        assert self._reader is not None
        try:
            while self._running:
                hdr = await self._reader.read(4)
                if len(hdr) < 4:
                    break
                length = int.from_bytes(hdr, "little")
                data = await self._reader.readexactly(length)
                # Try JSON first (control messages), fall back to binary
                try:
                    msg = json.loads(data)
                    logger.debug("control_msg", type=msg.get("type"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    cluster = decode_cluster(data)
                    await self.received_queue.put(cluster)
        except (asyncio.CancelledError, ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self._running = False

    async def disconnect(self) -> None:
        """Close the connection."""
        self._running = False
        if self._writer:
            self._writer.close()

    async def _send_json(self, obj: dict) -> None:
        assert self._writer is not None
        p = json.dumps(obj).encode()
        self._writer.write(len(p).to_bytes(4, "little") + p)
        await self._writer.drain()

    async def _read_json(self) -> Optional[dict]:
        assert self._reader is not None
        hdr = await self._reader.read(4)
        if len(hdr) < 4:
            return None
        length = int.from_bytes(hdr, "little")
        payload = await self._reader.readexactly(length)
        return json.loads(payload)


async def run_client_trace(
    host: str, port: int, trace_path: str | Path,
    output_dir: str | Path, client_id: str = "client-0",
) -> None:
    """Connect, replay a trace, collect frames, and save results.

    This is the main client entry point for evaluation runs.
    """
    client = MoQClient(host=host, port=port, client_id=client_id)
    manifest = await client.connect()
    await client.start_receiving()

    frames = load_trace(trace_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save manifest
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    for i, update in enumerate(replay_trace(frames, client_id)):
        await client.send_viewport_update(update)
        # Give the server a moment to respond
        await asyncio.sleep(0.05)
        # Drain received clusters
        received = []
        while not client.received_queue.empty():
            received.append(client.received_queue.get_nowait())
        logger.info("frame", idx=i, ts=update.timestamp_ms, clusters=len(received))

        # Save received cluster metadata per frame
        frame_meta = {
            "frame": i,
            "timestamp_ms": update.timestamp_ms,
            "clusters_received": len(received),
            "cluster_ids": [
                f"{c['track_id']}/{c['group_id']}/obj{c['object_id']}"
                for c in received
            ],
        }
        (out / f"frame_{i:04d}.json").write_text(json.dumps(frame_meta, indent=2))

    await client.disconnect()
    logger.info("trace_complete", frames=len(frames))
