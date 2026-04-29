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
import imageio
import structlog

logger = structlog.get_logger(__name__)


class FrameWriter:
    """Writes rendered frames to a video file and metrics to disk."""

    def __init__(self, output_dir: str | Path, width: int = 640, height: int = 480, fps: int = 10) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.jsonl"
        self._video_path = self.output_dir / "output.mp4"
        self._frame_count = 0
        
        # Initialize imageio writer
        try:
            self._writer = imageio.get_writer(
                str(self._video_path), 
                fps=fps, 
                codec='libx264', 
                format='FFMPEG', 
                pixelformat='yuv420p',
                macro_block_size=None
            )
            logger.debug("video_writer_initialized", path=str(self._video_path))
        except Exception as e:
            logger.error("video_writer_init_failed", error=str(e))
            self._writer = None

    def save_frame(
        self,
        image: torch.Tensor,
        frame_idx: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> Path:
        """Write an (H, W, 3) uint8 tensor to the video writer.

        Args:
            image: Rendered image tensor (CPU, uint8).
            frame_idx: Optional explicit frame index; auto-increments
                if not provided.
            metrics: Optional dict of per-frame metrics to log.

        Returns:
            Path to the saved video file.
        """
        idx = frame_idx if frame_idx is not None else self._frame_count
        self._frame_count = idx + 1

        # Debug: save first frame as PNG
        if idx == 0:
            try:
                from PIL import Image
                Image.fromarray(image.numpy()).save(self.output_dir / "debug_frame_0.png")
            except ImportError:
                pass

        # Write frame to video
        if self._writer is not None:
            try:
                self._writer.append_data(image.numpy())
            except Exception as e:
                logger.error("frame_write_failed", frame_idx=idx, error=str(e))

        if metrics is not None:
            metrics["frame_idx"] = idx
            metrics["timestamp"] = time.time()
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return self._video_path
        
    def close(self):
        """Close the video writer."""
        if hasattr(self, '_writer') and self._writer is not None:
            self._writer.close()
            logger.debug("video_writer_closed")
