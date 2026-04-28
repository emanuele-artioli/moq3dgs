"""Movement trace replay for camera path evaluation.

Reads a JSON trace file containing a sequence of camera poses
(position + view matrix + FOV) and replays them at a configurable
frame rate.  Each frame yields a :class:`ViewportUpdate` that the
client sends to the server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

import numpy as np

from moq3dgs.models import ViewportUpdate


def load_trace(path: str | Path) -> List[dict]:
    """Load a movement trace JSON file.

    Expected format — a JSON array of frame objects::

        [
            {
                "timestamp_ms": 0,
                "camera_position": [x, y, z],
                "view_matrix": [[4x4 row-major]],
                "fov": 60.0
            },
            ...
        ]

    Args:
        path: Path to the trace JSON file.

    Returns:
        List of raw frame dicts (validated later on iteration).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    return data


def replay_trace(
    frames: List[dict],
    client_id: str = "client-0",
) -> Iterator[ViewportUpdate]:
    """Yield one :class:`ViewportUpdate` per frame in the trace.

    Validation is done via Pydantic so malformed frames raise immediately.

    Args:
        frames: List of frame dicts as returned by :func:`load_trace`.
        client_id: Client identifier to stamp on every update.

    Yields:
        :class:`ViewportUpdate` for each frame.
    """
    for frame in frames:
        yield ViewportUpdate(
            client_id=client_id,
            timestamp_ms=frame["timestamp_ms"],
            camera_position=frame["camera_position"],
            view_matrix=frame["view_matrix"],
            fov=frame["fov"],
        )


def camera_forward_from_view_matrix(view_matrix: List[List[float]]) -> np.ndarray:
    """Extract the world-space forward direction from a 4×4 view matrix.

    The view matrix transforms world → camera.  The camera looks along -Z
    in camera space, so the world-space forward direction is the negated
    third row (or column, depending on convention) of the rotation part.

    We assume **row-major** layout matching the OpenGL convention used in
    the original 3DGS code.

    Args:
        view_matrix: 4×4 row-major view matrix.

    Returns:
        (3,) unit forward vector in world space.
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    # The inverse rotation's third column gives the camera forward.
    # For an orthogonal rotation R, R^-1 = R^T.
    forward = -vm[:3, 2]  # third column of rotation part
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return forward / norm
