#!/usr/bin/env python3
"""Generate a synthetic camera movement trace for testing.

Creates a JSON trace file with camera positions orbiting around the
scene at a fixed elevation, looking towards the origin.  This exercises
different spatial clusters entering/leaving the frustum.

Usage::

    python scripts/generate_test_trace.py [--output assets/eval_trace.json] [--frames 120]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List


def look_at(
    eye: List[float], target: List[float], up: List[float] = [0, 1, 0]
) -> List[List[float]]:
    """Compute a 4×4 view matrix (world-to-camera, row-major).

    Uses right-handed OpenGL convention (camera looks down -Z in camera space).
    """
    import numpy as np

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    new_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = new_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(new_up, eye)
    view[2, 3] = np.dot(forward, eye)

    return view.tolist()


def generate_orbit_trace(
    num_frames: int = 120,
    radius: float = 8.0,
    height: float = 2.0,
    target: List[float] = [0.0, 0.0, -3.0],
    fov: float = 60.0,
    fps: int = 30,
) -> List[dict]:
    """Generate an orbital camera trace around a target point.

    Args:
        num_frames: Number of frames in the trace.
        radius: Orbit radius.
        height: Camera height above the target's Y.
        target: Point the camera always looks at.
        fov: Vertical field of view in degrees.
        fps: Frames per second (determines timestamp_ms spacing).

    Returns:
        List of frame dicts suitable for the MoQ client.
    """
    frames = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        x = radius * math.cos(angle) + target[0]
        z = radius * math.sin(angle) + target[2]
        y = height + target[1]

        eye = [x, y, z]
        view_matrix = look_at(eye, target)

        frames.append({
            "timestamp_ms": int(i * 1000 / fps),
            "camera_position": eye,
            "view_matrix": view_matrix,
            "fov": fov,
        })

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic camera trace")
    parser.add_argument("--output", type=str, default="assets/eval_trace.json")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = generate_orbit_trace(
        num_frames=args.frames,
        radius=args.radius,
        height=args.height,
        fov=args.fov,
        fps=args.fps,
    )

    out.write_text(json.dumps({"frames": frames}, indent=2))
    print(f"Written {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
