#!/usr/bin/env python3
"""Generate a synthetic 3DGS PLY scene for development and testing.

Creates a point_cloud.ply with the same vertex schema as the original
3DGS training code (Kerbl et al.), containing Gaussians arranged in
a recognisable pattern (a floor plane + a few clusters) so that
viewport-dependent streaming behaviour is visually verifiable.

Usage::

    python scripts/generate_test_scene.py [--output assets/point_cloud.ply] [--num-points 50000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def generate_scene(num_points: int = 50000, seed: int = 42) -> dict:
    """Generate synthetic Gaussian attributes.

    Layout:
    - 40% floor plane at y ≈ -1, spread over x ∈ [-10, 10], z ∈ [-10, 10]
    - 20% central object (sphere-ish cluster near origin)
    - 15% left object  (cluster at x ≈ -4, y ≈ 0, z ≈ -3)
    - 15% right object (cluster at x ≈ +4, y ≈ 0, z ≈ -3)
    - 10% background scatter

    Args:
        num_points: Total number of Gaussians to generate.
        seed: Random seed for reproducibility.

    Returns:
        Dict of attribute arrays ready for PLY serialisation.
    """
    rng = np.random.default_rng(seed)

    n_floor = int(num_points * 0.40)
    n_center = int(num_points * 0.20)
    n_left = int(num_points * 0.15)
    n_right = int(num_points * 0.15)
    n_bg = num_points - n_floor - n_center - n_left - n_right

    # Floor plane
    floor_xyz = np.column_stack([
        rng.uniform(-10, 10, n_floor),
        rng.normal(-1.0, 0.05, n_floor),
        rng.uniform(-10, 10, n_floor),
    ])

    # Central object (sphere cluster)
    theta = rng.uniform(0, 2 * np.pi, n_center)
    phi = rng.uniform(0, np.pi, n_center)
    r = rng.normal(1.5, 0.3, n_center)
    center_xyz = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.cos(phi) + 0.5,
        r * np.sin(phi) * np.sin(theta) - 3.0,
    ])

    # Left object
    left_xyz = rng.normal(0, 0.5, (n_left, 3)) + np.array([-4, 0.5, -3])

    # Right object
    right_xyz = rng.normal(0, 0.5, (n_right, 3)) + np.array([4, 0.5, -3])

    # Background scatter
    bg_xyz = rng.uniform(-15, 15, (n_bg, 3))
    bg_xyz[:, 1] = rng.uniform(-2, 5, n_bg)

    means = np.vstack([floor_xyz, center_xyz, left_xyz, right_xyz, bg_xyz]).astype(np.float32)
    n = len(means)

    # Opacities (logit space; higher = more opaque)
    opacities = rng.normal(2.0, 0.5, (n,)).astype(np.float32)

    # Scales (log space)
    scales = rng.normal(-3.5, 0.3, (n, 3)).astype(np.float32)

    # Rotations (unit quaternions; mostly identity with small perturbation)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    quats[:, 1:] = rng.normal(0, 0.05, (n, 3))
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= norms

    # SH coefficients: DC (f_dc_0..2) + rest (f_rest_0..44) for degree 3
    # DC ≈ rough colour, rest ≈ small noise
    sh_dc = np.zeros((n, 3), dtype=np.float32)

    # Colour-code by region for visual debugging
    # Floor = green-ish
    sh_dc[:n_floor] = [0.0, 0.8, 0.0]
    # Center = red-ish
    off = n_floor
    sh_dc[off:off + n_center] = [0.8, 0.1, 0.1]
    off += n_center
    # Left = blue-ish
    sh_dc[off:off + n_left] = [0.1, 0.1, 0.8]
    off += n_left
    # Right = yellow-ish
    sh_dc[off:off + n_right] = [0.8, 0.8, 0.0]
    off += n_right
    # Background = grey
    sh_dc[off:] = [0.3, 0.3, 0.3]

    # Add a bit of noise to DC
    sh_dc += rng.normal(0, 0.05, sh_dc.shape).astype(np.float32)

    # Higher-order SH (45 coefficients = 15 per channel for degree 3)
    num_sh_rest = 45
    sh_rest = rng.normal(0, 0.01, (n, num_sh_rest)).astype(np.float32)

    return {
        "means": means,
        "opacities": opacities,
        "scales": scales,
        "rotations": quats,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }


def write_ply(path: Path, attrs: dict) -> None:
    """Write Gaussian attributes to a 3DGS-compatible PLY file."""
    n = len(attrs["means"])

    # Build dtype list matching the original 3DGS format
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(attrs["sh_rest"].shape[1]):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype.extend([
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    vertex = np.empty(n, dtype=dtype)
    vertex["x"] = attrs["means"][:, 0]
    vertex["y"] = attrs["means"][:, 1]
    vertex["z"] = attrs["means"][:, 2]
    vertex["nx"] = 0.0
    vertex["ny"] = 0.0
    vertex["nz"] = 0.0
    vertex["f_dc_0"] = attrs["sh_dc"][:, 0]
    vertex["f_dc_1"] = attrs["sh_dc"][:, 1]
    vertex["f_dc_2"] = attrs["sh_dc"][:, 2]
    for i in range(attrs["sh_rest"].shape[1]):
        vertex[f"f_rest_{i}"] = attrs["sh_rest"][:, i]
    vertex["opacity"] = attrs["opacities"]
    vertex["scale_0"] = attrs["scales"][:, 0]
    vertex["scale_1"] = attrs["scales"][:, 1]
    vertex["scale_2"] = attrs["scales"][:, 2]
    vertex["rot_0"] = attrs["rotations"][:, 0]
    vertex["rot_1"] = attrs["rotations"][:, 1]
    vertex["rot_2"] = attrs["rotations"][:, 2]
    vertex["rot_3"] = attrs["rotations"][:, 3]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(str(path))
    print(f"Written {n} Gaussians to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic 3DGS PLY scene")
    parser.add_argument("--output", type=str, default="assets/point_cloud.ply")
    parser.add_argument("--num-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    attrs = generate_scene(num_points=args.num_points, seed=args.seed)
    write_ply(out, attrs)


if __name__ == "__main__":
    main()
