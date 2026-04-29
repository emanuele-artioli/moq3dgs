"""Client-side persistent splat cache.

The cache stores received Gaussian clusters indexed by
(track_id, group_id, object_id).  On each render frame the cache
assembles the complete set of Gaussians that the client has received
so far, merging base and enhancement layers.

Per the architecture spec: **do not resend splats**.  Once a cluster
is cached, it stays until explicitly evicted.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import torch
import structlog

logger = structlog.get_logger(__name__)

CacheKey = Tuple[str, str, int, int]  # (track_id, group_id, subgroup_id, object_id)


class SplatCache:
    """Thread-safe, persistent cache for received Gaussian clusters.

    Each entry holds raw attribute tensors (on CPU).  The ``assemble()``
    method concatenates all cached base-layer clusters into a single set
    of tensors ready for GPU rasterisation.
    """

    def __init__(self) -> None:
        self._store: Dict[CacheKey, dict] = {}
        self._lock = threading.Lock()

    def put(self, cluster: dict) -> None:
        """Insert or update a cluster in the cache.

        Args:
            cluster: Dict with keys ``track_id``, ``group_id``,
                ``subgroup_id``, ``object_id``, and tensor attributes.
        """
        key: CacheKey = (
            cluster["track_id"],
            cluster["group_id"],
            cluster["subgroup_id"],
            cluster["object_id"],
        )
        with self._lock:
            self._store[key] = cluster
        logger.debug("cache_put", key=key, n=cluster.get("num_gaussians"))

    def has(self, track_id: str, group_id: str, subgroup_id: int, object_id: int) -> bool:
        """Check whether a cluster is already cached."""
        return (track_id, group_id, subgroup_id, object_id) in self._store

    def get(self, track_id: str, group_id: str, object_id: int) -> Optional[dict]:
        """Retrieve a cached cluster."""
        return self._store.get((track_id, group_id, object_id))

    def evict(self, track_id: str, group_id: str, object_id: int) -> None:
        """Remove a cluster from the cache."""
        key = (track_id, group_id, object_id)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Evict all clusters."""
        with self._lock:
            self._store.clear()

    @property
    def num_entries(self) -> int:
        return len(self._store)

    def assemble_base(self) -> Optional[dict]:
        """Concatenate all base-layer (object_id=0) clusters.

        Returns:
            Dict with concatenated ``means``, ``opacities``, ``sh_coeffs``,
            ``scales``, ``rotations`` tensors, or ``None`` if the cache
            is empty.
        """
        with self._lock:
            base_entries = [
                v for (_, _, oid), v in self._store.items() if oid == 0
            ]
        if not base_entries:
            return None

        def _cat(key: str) -> Optional[torch.Tensor]:
            parts = [e[key] for e in base_entries if e.get(key) is not None]
            if not parts:
                return None
            return torch.cat(parts, dim=0)

        return {
            "means": _cat("means"),
            "opacities": _cat("opacities"),
            "sh_coeffs": _cat("sh_coeffs"),
            "scales": _cat("scales"),
            "rotations": _cat("rotations"),
            "num_gaussians": sum(e["num_gaussians"] for e in base_entries),
        }
