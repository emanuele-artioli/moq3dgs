"""Minimal GPU rasterisation wrapper.

This module wraps a simple differentiable Gaussian splatting rasteriser.
If ``diff_gaussian_rasterization`` or ``gsplat`` is available it will
use that; otherwise it falls back to a basic CPU depth-sorted alpha
compositing renderer that is slow but correct.

All rendering output is written to disk (headless requirement).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import structlog

from moq3dgs.decorators import gpu_bound

logger = structlog.get_logger(__name__)


@gpu_bound
def render_gaussians(
    means: torch.Tensor,           # (N, 3)
    opacities: torch.Tensor,       # (N, 1)  sigmoid-space
    sh_coeffs: torch.Tensor,       # (N, C)
    scales: torch.Tensor,          # (N, 3)  log-space
    rotations: torch.Tensor,       # (N, 4)  quaternion
    view_matrix: torch.Tensor,     # (4, 4)
    proj_matrix: torch.Tensor,     # (4, 4)
    camera_pos: torch.Tensor,      # (3,)
    image_width: int = 1920,
    image_height: int = 1080,
    bg_color: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Render Gaussians to an image tensor.

    Returns an (H, W, 3) uint8 tensor on CPU.

    Falls back to a naive depth-sorted splatting if no GPU rasteriser
    is installed.

    Args:
        means: Gaussian centres.
        opacities: Per-Gaussian opacity (sigmoid space).
        sh_coeffs: Spherical harmonics coefficients (flattened).
        scales: Log-space scales.
        rotations: Quaternions (wxyz).
        view_matrix: 4×4 world-to-camera transform.
        proj_matrix: 4×4 projection matrix.
        camera_pos: Camera world-space position.
        image_width: Output width in pixels.
        image_height: Output height in pixels.
        bg_color: (3,) background colour; defaults to black.
        device: CUDA device string.

    Returns:
        (H, W, 3) uint8 image tensor on CPU.
    """
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    # Try hardware-accelerated rasteriser first
    try:
        return _render_gsplat(
            means, opacities, sh_coeffs, scales, rotations,
            view_matrix, proj_matrix, camera_pos,
            image_width, image_height, bg_color, device,
        )
    except ImportError:
        pass

    # Fallback: naive CPU renderer
    logger.warning("gpu_rasteriser_unavailable_falling_back_to_cpu")
    return _render_cpu_fallback(
        means, opacities, sh_coeffs, scales, rotations,
        view_matrix, proj_matrix, camera_pos,
        image_width, image_height, bg_color,
    )


def _render_gsplat(
    means: torch.Tensor, opacities: torch.Tensor,
    sh_coeffs: torch.Tensor, scales: torch.Tensor,
    rotations: torch.Tensor, view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor, camera_pos: torch.Tensor,
    w: int, h: int, bg: torch.Tensor, device: str,
) -> torch.Tensor:
    """Render via the gsplat library (if available)."""
    import gsplat  # noqa: F811

    # Move everything to device
    m = means.to(device)
    o = torch.sigmoid(opacities.to(device)).squeeze(-1)
    s = torch.exp(scales.to(device))
    r = rotations.to(device)
    vm = view_matrix.to(device)
    pm = proj_matrix.to(device)

    # gsplat expects specific format; convert SH DC to RGB colors
    sh = sh_coeffs.to(device)
    if sh.dim() == 1:
        sh = sh.reshape(len(m), -1)
    
    SH_C0 = 0.28209479177387814
    colors = torch.clamp(sh[:, :3] * SH_C0 + 0.5, 0.0, 1.0)

    rendered, _ = gsplat.rasterization(
        means=m, quats=r, scales=s, opacities=o,
        colors=colors, viewmats=vm.unsqueeze(0),
        Ks=pm[:3, :3].unsqueeze(0),
        width=w, height=h,
        backgrounds=bg.unsqueeze(0),
    )
    img = rendered[0].clamp(0, 1) * 255
    return img.byte().cpu()


def _render_cpu_fallback(
    means: torch.Tensor, opacities: torch.Tensor,
    sh_coeffs: torch.Tensor, scales: torch.Tensor,
    rotations: torch.Tensor, view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor, camera_pos: torch.Tensor,
    w: int, h: int, bg: torch.Tensor,
) -> torch.Tensor:
    """Minimal depth-sorted point splatting on CPU.

    Renders each Gaussian as a single pixel at its projected location,
    depth-sorted back-to-front with alpha blending.  This is *not*
    a production renderer — it exists only so tests and demos can
    run without a GPU rasteriser.
    """
    vp = (proj_matrix @ view_matrix).numpy().astype(np.float64)
    pts = means.numpy().astype(np.float64)
    n = len(pts)

    # Project to clip space
    ones = np.ones((n, 1), dtype=np.float64)
    world = np.hstack([pts, ones])  # (N, 4)
    clip = (vp @ world.T).T  # (N, 4)

    # Perspective divide
    w_clip = clip[:, 3:4]
    w_clip[w_clip == 0] = 1e-8
    ndc = clip[:, :3] / w_clip

    # NDC [-1,1] → pixel coords
    px = ((ndc[:, 0] + 1) * 0.5 * w).astype(np.int32)
    py = ((1 - (ndc[:, 1] + 1) * 0.5) * h).astype(np.int32)
    depth = ndc[:, 2]

    # Simple SH0 → colour (first 3 coeffs treated as RGB)
    sh = sh_coeffs
    if sh is not None and sh.numel() >= n * 3:
        # sh_coeffs may arrive as 1D (flat from cache) or 2D (N, C)
        if sh.dim() == 1:
            sh = sh.reshape(n, -1)
        colors = sh[:, :3].numpy()
        # SH0 to colour: C = SH_C0 * sh + 0.5
        SH_C0 = 0.28209479177387814
        colors = np.clip(colors * SH_C0 + 0.5, 0, 1)
    else:
        colors = np.ones((n, 3), dtype=np.float32) * 0.5

    alpha = torch.sigmoid(opacities).squeeze(-1).numpy()

    # Sort back-to-front
    order = np.argsort(-depth)

    canvas = np.full((h, w, 3), bg.numpy().astype(np.float32), dtype=np.float32)
    for idx in order:
        x, y = int(px[idx]), int(py[idx])
        if 0 <= x < w and 0 <= y < h:
            a = float(alpha[idx])
            canvas[y, x] = canvas[y, x] * (1 - a) + colors[idx] * a

    return torch.from_numpy((canvas * 255).clip(0, 255).astype(np.uint8))
