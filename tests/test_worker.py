"""Tests unitaires worker.main — mocke SupabaseWorkerClient + pipeline.

Pas de réseau, pas de Supabase joignable. Vérifie le flow claim → process →
notify (success + failure) et les invariants du build_redacted_bucket_path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worker.main import (
    GracefulShutdown,
    VideoTooLongError,
    _build_redacted_bucket_path,
    _safe_notify_failed,
    process_one_video,
)
from worker.settings import WorkerSettings
from worker.supabase_client import ClaimedVideo

# ─── _build_redacted_bucket_path ────────────────────────────────────────────


@pytest.mark.parametrize(
    "source,expected",
    [
        ("user-abc/clip.mp4", "user-abc/clip_redacted.mp4"),
        ("user-abc/clip.mov", "user-abc/clip_redacted.mp4"),
        ("user-abc/clip.mkv", "user-abc/clip_redacted.mp4"),
        ("u/sub/clip.mp4", "u/sub/clip_redacted.mp4"),
        ("clip.mp4", "clip_redacted.mp4"),
    ],
)
def test_build_redacted_bucket_path(source: str, expected: str):
    assert _build_redacted_bucket_path(source) == expected


# ─── Fixtures helpers ───────────────────────────────────────────────────────


@pytest.fixture
def fake_settings(tmp_path: Path) -> WorkerSettings:
    return WorkerSettings(
        supabase_url="https://fake.supabase.co",
        supabase_service_role_key="fake-key",
        poll_interval_idle_s=30,
        work_dir=tmp_path,
        face_model_path=tmp_path / "face.pt",
        plate_model_path=tmp_path / "plate.pt",
        max_video_duration_s=600,
        request_timeout_s=10.0,
    )


@pytest.fixture
def fake_claimed() -> ClaimedVideo:
    return ClaimedVideo(
        id="vid-123",
        source_bucket_path="user-abc/clip.mp4",
        gps_log_path=None,
        file_size_bytes=12345,
    )


# ─── process_one_video ──────────────────────────────────────────────────────


def test_process_one_video_success_flow(
    fake_settings: WorkerSettings,
    fake_claimed: ClaimedVideo,
):
    """Flow nominal : download → process → upload → notify_redacted."""
    client = MagicMock()

    # Simule download : crée le fichier local
    def _fake_download(path: str, dest: Path) -> Path:
        dest.write_bytes(b"fake video bytes")
        return dest

    client.download_source.side_effect = _fake_download

    models = MagicMock()

    fake_stats = MagicMock()
    fake_stats.n_frames = 100
    fake_stats.process_s = 6.6
    fake_stats.real_time_factor = 0.66
    fake_stats.n_faces_total_tracks = 12
    fake_stats.n_plates_total_tracks = 5

    with patch("worker.main._probe_duration_s", return_value=10.0):
        with patch("worker.main.process_video", return_value=fake_stats) as mock_process:
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "unlink"):
                    stats = process_one_video(fake_claimed, client, fake_settings, models)

    assert stats is fake_stats
    mock_process.assert_called_once()
    client.upload_redacted.assert_called_once()
    upload_args = client.upload_redacted.call_args
    assert upload_args[0][1] == "user-abc/clip_redacted.mp4"
    client.notify_redacted.assert_called_once_with(
        video_id="vid-123",
        redacted_bucket_path="user-abc/clip_redacted.mp4",
        faces_count=12,
        plates_count=5,
    )


def test_process_one_video_too_long_raises_video_too_long(
    fake_settings: WorkerSettings,
    fake_claimed: ClaimedVideo,
):
    """Vidéo > MAX_VIDEO_DURATION_S → VideoTooLongError, pas de process."""
    client = MagicMock()

    def _fake_download(path: str, dest: Path) -> Path:
        dest.write_bytes(b"x")
        return dest

    client.download_source.side_effect = _fake_download
    models = MagicMock()

    with patch("worker.main._probe_duration_s", return_value=900.0):  # 15 min > 10 min
        with patch("worker.main.process_video") as mock_process:
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "unlink"):
                    with pytest.raises(VideoTooLongError, match="video_too_long_max_600s"):
                        process_one_video(fake_claimed, client, fake_settings, models)

    mock_process.assert_not_called()
    client.upload_redacted.assert_not_called()
    client.notify_redacted.assert_not_called()


def test_process_one_video_cleans_up_local_files_on_error(
    fake_settings: WorkerSettings,
    fake_claimed: ClaimedVideo,
):
    """En cas d'exception, les fichiers locaux sont toujours supprimés (finally)."""
    client = MagicMock()

    def _fake_download(path: str, dest: Path) -> Path:
        dest.write_bytes(b"x")
        return dest

    client.download_source.side_effect = _fake_download
    models = MagicMock()

    unlink_calls: list[Path] = []

    def _record_unlink(self):
        unlink_calls.append(self)

    with patch("worker.main._probe_duration_s", return_value=10.0):
        with patch("worker.main.process_video", side_effect=RuntimeError("boom")):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "unlink", _record_unlink):
                    with pytest.raises(RuntimeError, match="boom"):
                        process_one_video(fake_claimed, client, fake_settings, models)

    # On a tenté de supprimer les 2 fichiers (in + out)
    assert len(unlink_calls) == 2


# ─── _safe_notify_failed ────────────────────────────────────────────────────


def test_safe_notify_failed_swallows_client_exception():
    """Si notify_failed elle-même throw, le worker ne crash pas (best-effort)."""
    client = MagicMock()
    client.notify_failed.side_effect = RuntimeError("notify down")
    # Ne doit pas re-throw
    _safe_notify_failed(client, "vid-123", "boom")
    client.notify_failed.assert_called_once_with("vid-123", "boom")


# ─── GracefulShutdown ───────────────────────────────────────────────────────


def test_graceful_shutdown_initial_state():
    """Au démarrage, should_exit = False."""
    shutdown = GracefulShutdown()
    assert shutdown.should_exit is False


def test_graceful_shutdown_handler_sets_flag():
    shutdown = GracefulShutdown()
    shutdown._handler(15, None)  # SIGTERM
    assert shutdown.should_exit is True
