"""Focused tests for deterministic corpus metadata and frame sampling."""

from pathlib import Path

import pytest

from hand_benchmark.dataset import (
    VideoInfo,
    build_frame_file_name,
    extract_frames,
    iter_sample_points,
    video_provenance,
)
from hand_benchmark.mcap_video import VideoExportResult


def test_one_fps_sampling_uses_source_frame_times() -> None:
    video = VideoInfo(Path("clip.mp4"), 640, 480, 29.97, 3.1, 93)

    assert list(iter_sample_points(video, 1.0)) == [
        (0, 0.0),
        (29, 29 / 29.97),
        (59, 59 / 29.97),
        (89, 89 / 29.97),
    ]


def test_frame_names_are_unique_across_mcap_stems() -> None:
    first = build_frame_file_name("recording-a", 0, 0.0, "jpg")
    second = build_frame_file_name("recording-b", 0, 0.0, "jpg")

    assert first != second
    assert first == "recording-a_frame000000_0000000000ms.jpg"


def test_video_provenance_contains_required_mcap_fields(tmp_path: Path) -> None:
    result = VideoExportResult(
        output_video_path=tmp_path / "videos" / "source.mp4",
        source_mcap_path=tmp_path / "source.mcap",
        video_topic="/head_left/video",
        frame_id="head_left",
        frame_count=12,
        time_range_sec=(1.0, 1.4),
        average_fps=30.0,
        format="h264",
        warnings=[],
    )

    record = video_provenance(result, tmp_path / "videos")

    assert record["source_mcap_stem"] == "source"
    assert record["video_topic"] == "/head_left/video"
    assert record["output_video_path"] == "source.mp4"


def test_extract_frames_rejects_cached_video_without_provenance(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "source.mp4").touch()

    with pytest.raises(ValueError, match="Missing MCAP provenance"):
        extract_frames(video_dir, tmp_path / "missing.jsonl", tmp_path / "frames", tmp_path / "frames.jsonl")


def test_frame_extraction_start_index_selects_a_later_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    for name in ("first.mp4", "second.mp4"):
        (video_dir / name).touch()
    (video_dir / "metadata.jsonl").write_text(
        "\n".join(
            f'{{"output_video_path":"{name}","source_mcap_path":"/{name}.mcap","source_mcap_stem":"{name}","video_topic":"/head_left/video"}}'
            for name in ("first.mp4", "second.mp4")
        ) + "\n"
    )
    monkeypatch.setattr(
        "hand_benchmark.dataset.probe_video",
        lambda path: VideoInfo(path, 10, 10, 1.0, 1.0, 1),
    )
    monkeypatch.setattr("hand_benchmark.dataset.extract_video_frames", lambda *args: None)

    video_count, frame_count = extract_frames(
        video_dir, video_dir / "metadata.jsonl", tmp_path / "frames", tmp_path / "frames.jsonl", start_index=1
    )

    assert (video_count, frame_count) == (1, 1)
    assert '"source_video": "second.mp4"' in (tmp_path / "frames.jsonl").read_text()
