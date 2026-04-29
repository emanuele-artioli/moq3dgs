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
        self._dirty = True
        self._cached_assembled: Optional[dict] = None
        self._cached_device: Optional[str] = None

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
            self._dirty = True
        logger.debug("cache_put", key=key, n=cluster.get("num_gaussians"))

    def has(self, track_id: str, group_id: str, subgroup_id: int, object_id: int) -> bool:
        """Check whether a cluster is already cached."""
        return (track_id, group_id, subgroup_id, object_id) in self._store

    def get(self, track_id: str, group_id: str, subgroup_id: int, object_id: int) -> Optional[dict]:
        """Retrieve a cached cluster."""
        return self._store.get((track_id, group_id, subgroup_id, object_id))

    def evict(self, track_id: str, group_id: str, subgroup_id: int, object_id: int) -> None:
        """Remove a cluster from the cache."""
        key = (track_id, group_id, subgroup_id, object_id)
        with self._lock:
            if key in self._store:
                self._store.pop(key)
                self._dirty = True

    def clear(self) -> None:
        """Evict all clusters."""
        with self._lock:
            self._store.clear()
            self._dirty = True
            self._cached_assembled = None

    @property
    def num_entries(self) -> int:
        return len(self._store)

    def assemble_base(self, device: str = "cpu", visible_keys: Optional[Set[Tuple[str, str]]] = None) -> Optional[dict]:
        """Concatenate all base-layer clusters.
 
        If the cache is not dirty, visible_keys haven't changed, and the 
        requested device matches the cached device, returns the previously 
        assembled result.
 
        Args:
            device: Target device for the concatenated tensors.
            visible_keys: Optional set of (track_id, group_id) to include.
                If None, all base clusters are included.
        """
        with self._lock:
            # For simplicity with culling, we only use the dirty cache for the "all" case.
            # If visible_keys is provided, we re-assemble to ensure correctness,
            # but we still benefit from tensors already being on device.
            if visible_keys is None and not self._dirty and self._cached_assembled is not None and self._cached_device == device:
                return self._cached_assembled
 
            base_entries = []
            for (tid, gid, sub_id, oid), v in self._store.items():
                if sub_id in (0, 1, 2) and oid == 0:
                    if visible_keys is None or (tid, gid) in visible_keys:
                        base_entries.append(v)
 
            if not base_entries:
                return None

            def _cat(key: str) -> Optional[torch.Tensor]:
                parts = [e[key] for e in base_entries if e.get(key) is not None]
                if not parts:
                    return None
                return torch.cat(parts, dim=0).to(device)

            result = {
                "means": _cat("means"),
                "opacities": _cat("opacities"),
                "sh_coeffs": _cat("sh_coeffs"),
                "scales": _cat("scales"),
                "rotations": _cat("rotations"),
                "num_gaussians": sum(e["num_gaussians"] for e in base_entries),
            }
            
            self._cached_assembled = result
            self._cached_device = device
            self._dirty = False
            return result
