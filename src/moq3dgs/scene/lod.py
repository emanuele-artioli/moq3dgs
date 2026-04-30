"""Level-of-Detail splitting for MoQ Importance Tiers.

Splats within each Group are ordered by **Weighted Farthest-Point Sampling
(WFPS)** so that the base layer achieves broad *spatial coverage* rather
than simply picking the geometrically largest splats.

The 5-tier hierarchy is:

    Tier 0  BASE_LARGE   — top 5% WFPS (large + isolated splats).
    Tier 1  BASE_MEDIUM  — next 15% WFPS (scene skeleton).
    Tier 2  BASE_SMALL   — remaining 80% WFPS (fine detail).
    Tier 3  ENHANCE_LARGE— higher-order SH for tier-0 splats.
    Tier 4  ENHANCE_MEDIUM— higher-order SH for tiers 1+2.

The WFPS ordering is computed **once at server startup** via
:func:`split_lod` and stored in :class:`LoDLayer`.  During streaming the
server simply reads pre-sorted indices and encodes them.

For very large clusters (>50K splats) a fast voxel-grid approximation is
used: splats are assigned to a 3-D grid and sorted by the product of voxel
occupancy and max-scale, which gives a good approximation of WFPS in O(N)
time without the expensive nearest-neighbour computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from moq3dgs.models import ImportanceTier
from moq3dgs.scene.loader import GaussianScene


# Threshold above which the fast voxel approximation is used instead of
# exact WFPS. The voxel method is O(N) but slightly less accurate.
_WFPS_EXACT_THRESHOLD = 30_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LoDLayer:
    """A single LoD layer for a subset of Gaussians.

    ``indices`` refer to the original scene arrays.  The attribute tensors
    are pre-sliced so the layer can be serialised independently.
    """

    tier: ImportanceTier
    indices: np.ndarray

    # Base attributes (present in BASE_* tiers)
    means: torch.Tensor | None = None
    opacities: torch.Tensor | None = None
    sh_dc: torch.Tensor | None = None
    rotations: torch.Tensor | None = None
    scales_base: torch.Tensor | None = None

    # Enhancement attributes (present in ENHANCE_* tiers)
    sh_rest: torch.Tensor | None = None
    scales_delta: torch.Tensor | None = None

    @property
    def num_gaussians(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _voxel_fps_order(
    means_np: np.ndarray,
    max_scales_np: np.ndarray,
    grid_res: int = 32,
) -> np.ndarray:
    """Fast O(N) voxel-grid approximation of WFPS ordering.

    Strategy:
        1. Quantise splat centres into a ``grid_res³`` voxel grid.
        2. For each splat compute a *coverage score*::

               score(i) = max_scale(i) / (count_in_voxel(i) + 1)

           This penalises splats that share a voxel (crowded = less unique
           coverage) and rewards larger splats.
        3. Sort descending by score → produces a coverage-aware ordering
           without the O(N²) cost of exact FPS.

    Args:
        means_np:      (N, 3) float32 Gaussian centres.
        max_scales_np: (N,)  float32 per-splat maximum scale.
        grid_res:      Voxel grid resolution per axis.

    Returns:
        int64 array of shape (N,) — ordering from most to least important.
    """
    N = len(means_np)
    if N == 0:
        return np.array([], dtype=np.int64)

    # Normalise to [0, grid_res)
    mn = means_np.min(axis=0)
    mx = means_np.max(axis=0)
    span = mx - mn
    span[span == 0] = 1.0  # avoid div-by-zero

    coords = ((means_np - mn) / span * (grid_res - 1)).astype(np.int32)
    coords = np.clip(coords, 0, grid_res - 1)

    # Flat voxel index
    flat = (
        coords[:, 0] * grid_res * grid_res
        + coords[:, 1] * grid_res
        + coords[:, 2]
    )

    # Count occupancy per voxel
    counts = np.bincount(flat, minlength=grid_res ** 3)
    voxel_count = counts[flat].astype(np.float32)  # (N,)

    # Coverage score: large + isolated → high score
    score = max_scales_np / (voxel_count + 1.0)

    return np.argsort(score)[::-1].copy().astype(np.int64)


def _compute_tier_split(
    scene: GaussianScene,
    idx: torch.LongTensor,
    device: str = "cpu",
    tier_fractions: Tuple[float, float, float] = (0.05, 0.15, 0.80),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute WFPS-ordered tier split for a subset of the scene.

    Selects between exact WFPS and the fast voxel approximation based on
    cluster size.

    Args:
        scene:          Full scene (CPU tensors).
        idx:            LongTensor of indices into the scene.
        device:         Compute device for exact WFPS.
        tier_fractions: (large, medium, small) fractions summing to 1.

    Returns:
        (idx_large, idx_medium, idx_small) — three int64 numpy arrays.
    """
    N = len(idx)
    if N == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    scales_sub = scene.scales[idx]                      # (N, 3)
    max_scales = torch.exp(scales_sub).max(dim=1).values  # (N,) real-space

    i1 = max(1, int(N * tier_fractions[0]))
    i2 = max(i1 + 1, int(N * (tier_fractions[0] + tier_fractions[1])))

    if N <= _WFPS_EXACT_THRESHOLD:
        # --- Exact WFPS on GPU / CPU ----------------------------------------
        from moq3dgs.scene.fps import weighted_fps_order
        means_sub = scene.means[idx].to(torch.float32)
        order = weighted_fps_order(means_sub, scales_sub.to(torch.float32), device=device)
    else:
        # --- Fast voxel approximation ----------------------------------------
        means_np = scene.means[idx].numpy().astype(np.float32)
        max_scales_np = max_scales.numpy()
        order = _voxel_fps_order(means_np, max_scales_np)

    # Map relative order back to original scene indices
    idx_np = idx.numpy()
    idx_large  = idx_np[order[:i1]]
    idx_medium = idx_np[order[i1:i2]]
    idx_small  = idx_np[order[i2:]]

    return idx_large, idx_medium, idx_small


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_lod(
    scene: GaussianScene,
    indices: np.ndarray,
    device: str = "cpu",
) -> List[LoDLayer]:
    """Split a Gaussian subset into 5 Importance Tiers using WFPS ordering.

    The base layers (0-2) achieve broad spatial coverage so the client sees
    a spatially uniform coarse scene first, rather than one over-detailed
    cluster with the background missing.

    Enhancement layers (3-4) carry higher-order SH coefficients and are
    sent after all base layers for a given group.

    Args:
        scene:   Full loaded scene (tensors on CPU).
        indices: Integer indices of the Gaussians in this group.
        device:  Compute device for WFPS (``"cpu"`` or ``"cuda:X"``).
                 Falls back silently to CPU if CUDA is unavailable.

    Returns:
        A list of :class:`LoDLayer` objects (one for each active tier).
    """
    idx = torch.from_numpy(indices).long()

    if len(idx) == 0:
        return []

    # Resolve compute device (fall back to CPU if CUDA unavailable)
    try:
        torch.zeros(1, device=device)
        compute_device = device
    except Exception:
        compute_device = "cpu"

    idx_large, idx_medium, idx_small = _compute_tier_split(
        scene, idx, device=compute_device
    )

    layers: List[LoDLayer] = []

    def make_base(tier: ImportanceTier, sub_idx: np.ndarray) -> LoDLayer | None:
        if len(sub_idx) == 0:
            return None
        t = torch.from_numpy(sub_idx).long()
        return LoDLayer(
            tier=tier,
            indices=sub_idx,
            means=scene.means[t],
            opacities=scene.opacities[t],
            sh_dc=scene.sh_dc[t],
            rotations=scene.rotations[t],
            scales_base=scene.scales[t],
        )

    def make_enhance(tier: ImportanceTier, sub_idx: np.ndarray) -> LoDLayer | None:
        if len(sub_idx) == 0 or scene.sh_rest.shape[1] == 0:
            return None
        t = torch.from_numpy(sub_idx).long()
        return LoDLayer(
            tier=tier,
            indices=sub_idx,
            sh_rest=scene.sh_rest[t],
            scales_delta=torch.zeros_like(scene.scales[t]),
        )

    l0 = make_base(ImportanceTier.BASE_LARGE, idx_large)
    l1 = make_base(ImportanceTier.BASE_MEDIUM, idx_medium)
    l2 = make_base(ImportanceTier.BASE_SMALL, idx_small)
    l3 = make_enhance(ImportanceTier.ENHANCE_LARGE, idx_large)

    idx_med_small = (
        np.concatenate([idx_medium, idx_small])
        if len(idx_medium) + len(idx_small) > 0
        else np.array([], dtype=np.int64)
    )
    l4 = make_enhance(ImportanceTier.ENHANCE_MEDIUM, idx_med_small)

    for layer in [l0, l1, l2, l3, l4]:
        if layer is not None:
            layers.append(layer)

    return layers
