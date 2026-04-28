"""Pydantic data contracts for all client ↔ server messaging.

Every message crossing the MoQ transport boundary is defined here so that
serialisation / validation is centralised and enforced.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LoDLevel(IntEnum):
    """Level-of-Detail identifiers mapped to MoQ Object IDs.

    Object 0 carries base geometry + opacity + DC colour (SH0).
    Object 1 carries high-frequency details (SH1-SH3) + scale refinements.
    """

    BASE = 0
    ENHANCEMENT = 1


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------


class ViewportUpdate(BaseModel):
    """Client message requesting new spatial data based on movement trace.

    Sent each time the client camera moves so the server can adjust
    which tracks / groups are published with which priority.
    """

    client_id: str
    timestamp_ms: int
    camera_position: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z]"
    )
    view_matrix: List[List[float]] = Field(
        ..., description="4×4 view matrix (row-major)"
    )
    fov: float = Field(..., gt=0, lt=180, description="Vertical FOV in degrees")


class MoQSubscription(BaseModel):
    """Subscription request from client to server for a specific track/group.

    max_object_id defines the requested quality level: 0 = base only,
    1 = base + enhancement.
    """

    track_id: str
    group_id: str
    max_object_id: int = Field(
        ..., ge=0, le=1, description="Requested LoD (0=base, 1=full)"
    )
    priority: int = Field(
        ..., ge=0, le=255, description="0=highest, 255=lowest"
    )


# ---------------------------------------------------------------------------
# Scene manifest (Server → Client)
# ---------------------------------------------------------------------------


class GroupInfo(BaseModel):
    """Metadata for a single MoQ Group (a chunk of Gaussians within a track).

    Stores the axis-aligned bounding box and the number of Gaussians so
    that the client can do local frustum culling without downloading data.
    """

    group_id: str
    num_gaussians: int
    bbox_min: List[float] = Field(..., min_length=3, max_length=3)
    bbox_max: List[float] = Field(..., min_length=3, max_length=3)
    available_objects: List[int] = Field(
        default=[0, 1], description="Which LoD layers are available"
    )


class TrackInfo(BaseModel):
    """Metadata for a single MoQ Track (a spatial volume / octree node).

    A track contains one or more groups, each covering a sub-region of the
    volume.
    """

    track_id: str
    bbox_min: List[float] = Field(..., min_length=3, max_length=3)
    bbox_max: List[float] = Field(..., min_length=3, max_length=3)
    groups: List[GroupInfo]


class SceneManifest(BaseModel):
    """Top-level manifest for the entire 3DGS broadcast.

    Distributed once at session start; the client uses it to know which
    track/group IDs exist and where they sit in world space.
    """

    broadcast_name: str
    total_gaussians: int
    tracks: List[TrackInfo]


# ---------------------------------------------------------------------------
# Gaussian data payload
# ---------------------------------------------------------------------------


class GaussianCluster(BaseModel):
    """Serialisable payload for a batch of Gaussians sent over the wire.

    Tensor data is flattened to lists for JSON transport; the receiver
    reconstructs torch tensors from these.  For binary transport the
    ``raw_bytes`` field carries the pre-packed buffer instead.
    """

    track_id: str
    group_id: str
    object_id: int = Field(
        ..., ge=0, le=1, description="0=base, 1=enhancement"
    )
    num_gaussians: int

    # Flattened attribute arrays (JSON path)
    means: Optional[List[float]] = None          # [N*3]
    opacities: Optional[List[float]] = None      # [N]
    sh_coeffs: Optional[List[float]] = None      # [N*C]  (C depends on LoD)
    scales: Optional[List[float]] = None         # [N*3]
    rotations: Optional[List[float]] = None      # [N*4]

    # Binary path — mutually exclusive with the lists above
    raw_bytes: Optional[bytes] = None


# ---------------------------------------------------------------------------
# Server → Client control
# ---------------------------------------------------------------------------


class SubscriptionAck(BaseModel):
    """Server acknowledgement of a subscription request."""

    track_id: str
    group_id: str
    accepted: bool
    reason: Optional[str] = None


class ServerStatus(BaseModel):
    """Periodic heartbeat from the server carrying session-level stats."""

    active_subscriptions: int
    total_tracks: int
    total_groups: int
    bytes_sent: int
    uptime_ms: int
