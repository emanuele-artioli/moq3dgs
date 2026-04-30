"""Wire protocol for Gaussian cluster serialisation over MoQ/QUIC.

This module provides compact binary encoding / decoding of Gaussian
attribute tensors so that they can be shipped over QUIC streams without
going through JSON.

Frame format
------------

::

    [ header: 32 bytes ][ payload: variable ]

    Header layout (all little-endian):
        0..3    magic          (uint32)  0x47535033  ("GSP0")
        4..7    version        (uint32)  1
        8..11   track_id_len   (uint32)
       12..15   group_id_len   (uint32)
       16..19   object_id      (uint32)
       20..23   num_gaussians  (uint32)
       24..27   payload_len    (uint32)
       28..31   subgroup_id    (uint32)

    Payload:
        track_id bytes, group_id bytes, then attribute arrays in
        a fixed order: means, opacities, sh_coeffs, scales, rotations.
        Each array is prefixed by its byte length (uint32).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np
import torch

from moq3dgs.decorators import network_bound

MAGIC = 0x47535033  # "GSP0"
VERSION = 1
HEADER_SIZE = 32
HEADER_FMT = "<8I"  # 8 × uint32


@network_bound
def encode_cluster(
    track_id: str,
    group_id: str,
    subgroup_id: int,
    object_id: int,
    num_gaussians: int,
    means: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
    sh_coeffs: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
) -> bytes:
    """Encode Gaussian attributes into a compact binary frame.

    ``None`` attributes are encoded as zero-length arrays so the receiver
    knows they were intentionally omitted (e.g., enhancement layer carries
    only sh_coeffs + scales).

    Args:
        track_id: MoQ track identifier.
        group_id: MoQ group identifier.
        object_id: LoD layer (0 or 1).
        num_gaussians: Number of Gaussians in this cluster.
        means: (N, 3) positions.
        opacities: (N, 1) opacities.
        sh_coeffs: (N, C) SH coefficients (flattened).
        scales: (N, 3) log-space scales.
        rotations: (N, 4) quaternions.

    Returns:
        Binary frame bytes.
    """
    tid = track_id.encode("utf-8")
    gid = group_id.encode("utf-8")

    def _tensor_bytes(t: Optional[torch.Tensor]) -> bytes:
        if t is None:
            return b""
        return t.detach().cpu().contiguous().numpy().astype(np.float32).tobytes()

    arrays = [
        _tensor_bytes(means),
        _tensor_bytes(opacities),
        _tensor_bytes(sh_coeffs),
        _tensor_bytes(scales),
        _tensor_bytes(rotations),
    ]

    # Build payload: tid + gid + array data (each prefixed with uint32 length)
    payload = bytearray()
    payload.extend(tid)
    payload.extend(gid)
    for arr in arrays:
        payload.extend(struct.pack("<I", len(arr)))
        payload.extend(arr)

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        len(tid),
        len(gid),
        object_id,
        num_gaussians,
        len(payload),
        subgroup_id,
    )
    return bytes(header) + bytes(payload)


@network_bound
def decode_cluster(data: bytes) -> dict:
    """Decode a binary frame back into attribute tensors.

    Args:
        data: Raw bytes previously produced by :func:`encode_cluster`.

    Returns:
        Dict with keys ``track_id``, ``group_id``, ``object_id``,
        ``num_gaussians``, ``means``, ``opacities``, ``sh_coeffs``,
        ``scales``, ``rotations``.  Tensor values may be ``None`` if the
        corresponding array had zero length.

    Raises:
        ValueError: If the magic number or version is wrong.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(data)} < {HEADER_SIZE}")

    (
        magic,
        version,
        tid_len,
        gid_len,
        object_id,
        num_gaussians,
        payload_len,
        subgroup_id,
    ) = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    payload = data[HEADER_SIZE:]
    if len(payload) < payload_len:
        raise ValueError("Truncated payload")

    offset = 0
    track_id = payload[offset : offset + tid_len].decode("utf-8")
    offset += tid_len
    group_id = payload[offset : offset + gid_len].decode("utf-8")
    offset += gid_len

    def _read_array() -> Optional[torch.Tensor]:
        nonlocal offset
        (arr_len,) = struct.unpack("<I", payload[offset : offset + 4])
        offset += 4
        if arr_len == 0:
            return None
        arr = np.frombuffer(payload[offset : offset + arr_len], dtype=np.float32).copy()
        offset += arr_len
        return torch.from_numpy(arr)

    means = _read_array()
    opacities = _read_array()
    sh_coeffs = _read_array()
    scales = _read_array()
    rotations = _read_array()

    # Reshape based on num_gaussians
    if means is not None:
        means = means.reshape(num_gaussians, 3)
    if opacities is not None:
        opacities = opacities.reshape(num_gaussians, 1)
    if scales is not None:
        scales = scales.reshape(num_gaussians, 3)
    if rotations is not None:
        rotations = rotations.reshape(num_gaussians, 4)
    if sh_coeffs is not None:
        sh_coeffs = sh_coeffs.reshape(num_gaussians, -1)

    return {
        "track_id": track_id,
        "group_id": group_id,
        "subgroup_id": subgroup_id,
        "object_id": object_id,
        "num_gaussians": num_gaussians,
        "means": means,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
        "scales": scales,
        "rotations": rotations,
    }
