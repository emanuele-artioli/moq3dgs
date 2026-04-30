"""Unit tests for the binary wire protocol."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moq3dgs.transport.protocol import decode_cluster, encode_cluster


class TestProtocol:
    """Binary encode/decode roundtrip tests."""

    def test_roundtrip_full(self) -> None:
        """Full cluster with all attributes."""
        n = 10
        means = torch.randn(n, 3)
        opacities = torch.randn(n, 1)
        sh = torch.randn(n, 3)
        scales = torch.randn(n, 3)
        rots = torch.randn(n, 4)

        data = encode_cluster(
            "track-0001", "group-0001-0", 1, 0, n,
            means, opacities, sh, scales, rots,
        )
        result = decode_cluster(data)

        assert result["track_id"] == "track-0001"
        assert result["group_id"] == "group-0001-0"
        assert result["subgroup_id"] == 1
        assert result["object_id"] == 0
        assert result["num_gaussians"] == n
        torch.testing.assert_close(result["means"], means, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["opacities"], opacities, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["scales"], scales, atol=1e-6, rtol=1e-5)

    def test_roundtrip_partial(self) -> None:
        """Enhancement layer with only sh_coeffs."""
        n = 5
        sh = torch.randn(n, 12)

        data = encode_cluster(
            "track-0002", "group-0002-0", 0, 1, n,
            sh_coeffs=sh,
        )
        result = decode_cluster(data)

        assert result["subgroup_id"] == 0
        assert result["object_id"] == 1
        assert result["means"] is None
        assert result["sh_coeffs"] is not None

    def test_bad_magic_raises(self) -> None:
        data = b"\x00" * 32 + b"\x00" * 10
        with pytest.raises(ValueError, match="Bad magic"):
            decode_cluster(data)

    def test_truncated_header(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            decode_cluster(b"\x00" * 10)
