"""Level-of-Detail splitting for MoQ Object layering.

Object 0 (base): positions + opacity + DC colour (SH0) + rotations.
Object 1 (enhancement): higher-order SH bands (SH1-SH3) + scale refinements.

This separation lets the transport layer prioritise the base geometry
(which is visually critical) over enhancement details, and lets the
client render a coarse preview immediately while enhancement data arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from moq3dgs.models import ImportanceTier
from moq3dgs.scene.loader import GaussianScene


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


def split_lod(
    scene: GaussianScene,
    indices: np.ndarray,
) -> List[LoDLayer]:
    """Split a Gaussian subset into 5 Importance Tiers.

    Base geometry is split by splat volume (scale).
    Enhancement details are split similarly.

    Args:
        scene: Full loaded scene (tensors on CPU).
        indices: Integer indices of the Gaussians to split.

    Returns:
        A list of :class:`LoDLayer` objects (one for each active tier).
    """
    idx = torch.from_numpy(indices).long()

    if len(idx) == 0:
        return []

    # Sort splats by max scale (proxy for volume)
    scales = scene.scales[idx]
    max_scales = scales.max(dim=1).values
    # sort descending so largest splats are first
    sorted_idx_relative = torch.argsort(max_scales, descending=True)
    sorted_idx = idx[sorted_idx_relative]

    N = len(sorted_idx)
    i1 = int(N * 0.2)
    i2 = int(N * 0.5)

    idx_large = sorted_idx[:i1]
    idx_medium = sorted_idx[i1:i2]
    idx_small = sorted_idx[i2:]

    layers = []

    def make_base(tier, sub_idx):
        if len(sub_idx) == 0: return None
        return LoDLayer(
            tier=tier,
            indices=sub_idx.numpy(),
            means=scene.means[sub_idx],
            opacities=scene.opacities[sub_idx],
            sh_dc=scene.sh_dc[sub_idx],
            rotations=scene.rotations[sub_idx],
            scales_base=scene.scales[sub_idx],
        )

    def make_enhance(tier, sub_idx):
        if len(sub_idx) == 0 or scene.sh_rest.shape[1] == 0: return None
        return LoDLayer(
            tier=tier,
            indices=sub_idx.numpy(),
            sh_rest=scene.sh_rest[sub_idx],
            scales_delta=torch.zeros_like(scene.scales[sub_idx]),
        )

    l0 = make_base(ImportanceTier.BASE_LARGE, idx_large)
    l1 = make_base(ImportanceTier.BASE_MEDIUM, idx_medium)
    l2 = make_base(ImportanceTier.BASE_SMALL, idx_small)
    l3 = make_enhance(ImportanceTier.ENHANCE_LARGE, idx_large)
    # Combine medium and small for enhance_medium
    idx_med_small = torch.cat([idx_medium, idx_small]) if len(idx_medium) or len(idx_small) else torch.tensor([], dtype=torch.long)
    l4 = make_enhance(ImportanceTier.ENHANCE_MEDIUM, idx_med_small)

    for l in [l0, l1, l2, l3, l4]:
        if l is not None:
            layers.append(l)

    return layers
