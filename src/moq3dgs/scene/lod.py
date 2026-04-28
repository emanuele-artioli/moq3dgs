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

from moq3dgs.models import LoDLevel
from moq3dgs.scene.loader import GaussianScene


@dataclass
class LoDLayer:
    """A single LoD layer for a subset of Gaussians.

    ``indices`` refer to the original scene arrays.  The attribute tensors
    are pre-sliced so the layer can be serialised independently.
    """

    level: LoDLevel
    indices: np.ndarray

    # Base attributes (always present in Object 0, None in Object 1)
    means: torch.Tensor | None = None        # (M, 3)
    opacities: torch.Tensor | None = None    # (M, 1)
    sh_dc: torch.Tensor | None = None        # (M, 1, 3)
    rotations: torch.Tensor | None = None    # (M, 4)
    # Coarse scales for base layer
    scales_base: torch.Tensor | None = None  # (M, 3)

    # Enhancement attributes (present in Object 1 only)
    sh_rest: torch.Tensor | None = None      # (M, K, 3)
    scales_delta: torch.Tensor | None = None # (M, 3)  refinement on top of base

    @property
    def num_gaussians(self) -> int:
        return len(self.indices)


def split_lod(
    scene: GaussianScene,
    indices: np.ndarray,
) -> List[LoDLayer]:
    """Split a Gaussian subset into base and enhancement LoD layers.

    The base layer carries everything needed for a coarse but complete
    render; the enhancement layer carries the residuals that improve
    visual fidelity.

    Args:
        scene: Full loaded scene (tensors on CPU).
        indices: Integer indices of the Gaussians to split.

    Returns:
        A list of two :class:`LoDLayer` objects, one per LoD level.
    """
    idx = torch.from_numpy(indices).long()

    # --- Object 0: base geometry -------------------------------------------
    base = LoDLayer(
        level=LoDLevel.BASE,
        indices=indices,
        means=scene.means[idx],
        opacities=scene.opacities[idx],
        sh_dc=scene.sh_dc[idx],
        rotations=scene.rotations[idx],
        scales_base=scene.scales[idx],
    )

    # --- Object 1: enhancement details -------------------------------------
    enhancement = LoDLayer(
        level=LoDLevel.ENHANCEMENT,
        indices=indices,
        sh_rest=scene.sh_rest[idx] if scene.sh_rest.shape[1] > 0 else None,
        # Scale delta is zero for now; a future training pass could learn a
        # coarse-to-fine factorisation.
        scales_delta=torch.zeros_like(scene.scales[idx]),
    )

    return [base, enhancement]
