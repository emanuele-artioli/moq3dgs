"""Weighted Farthest-Point Sampling (WFPS) for volume-covering splat selection.

Classic Farthest-Point Sampling greedily picks the point maximally distant
from all already-selected points.  We extend it with a *volume weight*: the
effective "importance" of each Gaussian is::

    importance(i) = min_dist_to_selected(i) * volume_weight(i)

where::

    volume_weight(i) = max(scales_i)  (proxy for Gaussian volume / radius)

This ensures large *and* isolated splats are chosen for the base layer,
giving broad spatial coverage even with a small fraction of splats.

The algorithm is implemented in two modes:

* **GPU (exact)**  — runs on the CUDA device in O(N × K) with batched
  cdist; recommended for clusters of up to ~500K splats.
* **CPU (fallback)** — same logic, numpy-only, slower but always available.

The result is a *permutation index array* of length N (original cluster size)
where position 0 is the most-important splat, position 1 the next-most, etc.
The caller can then split this ordering into importance tiers by percentile.

Reference: Qi et al. (2017), "PointNet++", Appendix A.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from moq3dgs.decorators import gpu_bound


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def weighted_fps_order(
    means: torch.Tensor,
    scales: torch.Tensor,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Order Gaussians by coverage importance using Weighted FPS.

    Args:
        means:  (N, 3) float32 tensor of Gaussian centres.
        scales: (N, 3) float32 tensor of log-scales (raw scene values).
        device: ``"cuda:X"`` or ``"cpu"`` to run on.
        chunk_size: Number of candidate points processed per inner loop
            iteration to cap peak VRAM usage.

    Returns:
        ``order`` — int64 numpy array of shape (N,) with the WFPS permutation.
        order[0] is the most important splat, order[-1] the least.
    """
    N = means.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)

    # Compute per-Gaussian volume weight from log-scale
    # exp(log_scale) → actual scale; use max per splat as radius proxy
    radii = torch.exp(scales).max(dim=1).values.to(device, dtype=torch.float32)  # (N,)
    pts = means.to(device, dtype=torch.float32)

    # Normalise radii to [0, 1] so they don't dominate distance entirely
    r_min, r_max = radii.min(), radii.max()
    if r_max > r_min:
        radii_norm = (radii - r_min) / (r_max - r_min)
    else:
        radii_norm = torch.ones(N, device=device)

    # Distance buffer: dist[i] = current "effective distance" of point i from
    # the selected set.  Initialise to +inf so every point is a candidate.
    dist = torch.full((N,), float("inf"), device=device, dtype=torch.float32)

    order = np.empty(N, dtype=np.int64)

    # Seed: pick the splat with the largest volume
    first = int(radii.argmax().item())
    order[0] = first
    selected_pt = pts[first]  # (3,)

    # Update distances from seed
    d = torch.sum((pts - selected_pt.unsqueeze(0)) ** 2, dim=1).sqrt()
    weighted = d * (1.0 + radii_norm)
    dist = torch.minimum(dist, weighted)
    dist[first] = -1.0  # Mark as selected (will never be max again)

    for k in range(1, N):
        # Pick the point with the maximum weighted distance
        nxt = int(dist.argmax().item())
        order[k] = nxt
        dist[nxt] = -1.0  # Mark as selected

        if k < N - 1:
            # Update distances from the newly selected point (chunked)
            nxt_pt = pts[nxt]
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                d_chunk = torch.sum(
                    (pts[start:end] - nxt_pt.unsqueeze(0)) ** 2, dim=1
                ).sqrt()
                weighted_chunk = d_chunk * (1.0 + radii_norm[start:end])
                dist[start:end] = torch.minimum(dist[start:end], weighted_chunk)

    return order


@gpu_bound
def compute_wfps_tiers(
    means: torch.Tensor,
    scales: torch.Tensor,
    device: str = "cpu",
    tier_fractions: tuple[float, ...] = (0.05, 0.15, 0.80),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run WFPS and split the result into three coverage tiers.

    The first ``tier_fractions[0]`` fraction of the WFPS order forms the
    *large* coverage tier (best volume coverage), the next fraction forms
    the *medium* tier, and the remainder forms the *small / fine detail*
    tier.

    Default fractions (5%, 15%, 80%) mean:
    - BASE_LARGE  → top 5%  → ~4700 splats per cluster out of 94K average
    - BASE_MEDIUM → next 15% → dense skeleton
    - BASE_SMALL  → remaining → fine detail

    Args:
        means:           (N, 3) Gaussian centres.
        scales:          (N, 3) log-scales.
        device:          Compute device.
        tier_fractions:  Must sum to ≤ 1.0. (large, medium, rest).

    Returns:
        Tuple of three int64 numpy arrays: (idx_large, idx_medium, idx_small)
        each containing original scene indices.
    """
    assert abs(sum(tier_fractions) - 1.0) < 1e-6, "tier_fractions must sum to 1.0"

    N = means.shape[0]
    order = weighted_fps_order(means, scales, device=device)

    i1 = max(1, int(N * tier_fractions[0]))
    i2 = max(i1 + 1, int(N * (tier_fractions[0] + tier_fractions[1])))

    idx_large  = order[:i1]
    idx_medium = order[i1:i2]
    idx_small  = order[i2:]

    return idx_large, idx_medium, idx_small
