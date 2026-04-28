"""Frame-to-disk writer.

Saves rendered image tensors as PNG files and optionally logs per-frame
metrics (timing, number of Gaussians, cache size) to a JSON lines file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import structlog

logger = structlog.get_logger(__name__)


class FrameWriter:
    """Writes rendered frames and metrics to disk."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.jsonl"
        self._frame_count = 0

    def save_frame(
        self,
        image: torch.Tensor,
        frame_idx: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> Path:
        """Save an (H, W, 3) uint8 tensor as a PNG file.

        Args:
            image: Rendered image tensor (CPU, uint8).
            frame_idx: Optional explicit frame index; auto-increments
                if not provided.
            metrics: Optional dict of per-frame metrics to log.

        Returns:
            Path to the saved PNG file.
        """
        idx = frame_idx if frame_idx is not None else self._frame_count
        self._frame_count = idx + 1

        path = self.output_dir / f"frame_{idx:06d}.png"
        img = Image.fromarray(image.numpy())
        img.save(str(path))
        logger.debug("frame_saved", path=str(path))

        if metrics is not None:
            metrics["frame_idx"] = idx
            metrics["timestamp"] = time.time()
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return path
