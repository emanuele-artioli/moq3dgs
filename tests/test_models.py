"""Unit tests for Pydantic data contracts."""

from __future__ import annotations

import pytest

from moq3dgs.models import (
    GaussianCluster,
    GroupInfo,
    ImportanceTier,
    MoQSubscription,
    SceneManifest,
    ServerStatus,
    SubscriptionAck,
    TrackInfo,
    ViewportUpdate,
)


class TestViewportUpdate:
    """Validation tests for ViewportUpdate."""

    def test_valid_viewport(self) -> None:
        vu = ViewportUpdate(
            client_id="c1",
            timestamp_ms=100,
            camera_position=[1.0, 2.0, 3.0],
            view_matrix=[[1, 0, 0, 0]] * 4,
            fov=60.0,
        )
        assert vu.client_id == "c1"
        assert vu.camera_position == [1.0, 2.0, 3.0]

    def test_invalid_fov_too_large(self) -> None:
        with pytest.raises(Exception):
            ViewportUpdate(
                client_id="c1",
                timestamp_ms=0,
                camera_position=[0, 0, 0],
                view_matrix=[[1, 0, 0, 0]] * 4,
                fov=200.0,
            )

    def test_invalid_position_length(self) -> None:
        with pytest.raises(Exception):
            ViewportUpdate(
                client_id="c1",
                timestamp_ms=0,
                camera_position=[0, 0],  # only 2 elements
                view_matrix=[[1, 0, 0, 0]] * 4,
                fov=60.0,
            )


class TestMoQSubscription:
    """Validation tests for MoQSubscription."""

    def test_valid_subscription(self) -> None:
        sub = MoQSubscription(
            track_id="track-0001",
            group_id="group-0001-0",
            max_subgroup_id=1,
            priority=128,
        )
        assert sub.max_subgroup_id == 1
        assert sub.priority == 128

    def test_priority_out_of_range(self) -> None:
        with pytest.raises(Exception):
            MoQSubscription(
                track_id="t", group_id="g",
                max_subgroup_id=0, priority=300,
            )


class TestSceneManifest:
    """Test manifest construction."""

    def test_manifest_roundtrip(self) -> None:
        manifest = SceneManifest(
            broadcast_name="test",
            total_gaussians=1000,
            tracks=[
                TrackInfo(
                    track_id="t0",
                    bbox_min=[0, 0, 0],
                    bbox_max=[1, 1, 1],
                    groups=[
                        GroupInfo(
                            group_id="g0",
                            num_gaussians=1000,
                            bbox_min=[0, 0, 0],
                            bbox_max=[1, 1, 1],
                        )
                    ],
                )
            ],
        )
        # Roundtrip through JSON
        data = manifest.model_dump_json()
        restored = SceneManifest.model_validate_json(data)
        assert restored.total_gaussians == 1000
        assert len(restored.tracks) == 1


class TestImportanceTier:
    def test_enum_values(self):
        assert ImportanceTier.BASE_LARGE == 0
        assert ImportanceTier.ENHANCE_MEDIUM == 4
