"""Unit tests for MCAP topic validation and H.264 GOP helpers."""

from pathlib import Path

import pytest

from hand_detection_benchmark import mcap_video


def test_export_rejects_missing_topic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = mcap_video.McapSummary(tmp_path / "source.mcap", {}, {}, [])
    monkeypatch.setattr(mcap_video, "read_mcap_summary", lambda _: summary)

    with pytest.raises(ValueError, match="Video topic not found"):
        mcap_video.export_video_topic(
            mcap_path=summary.path,
            output_video_path=tmp_path / "output.mp4",
            video_topic="/head_left/video",
        )


def test_h264_parameter_sets_enable_following_idr() -> None:
    parameter_sets = b"\x00\x00\x01\x67\x01\x00\x00\x01\x68\x02"
    idr = b"\x00\x00\x01\x65\x03"
    output_path = Path("ignored.h264")
    writer = mcap_video._BitstreamWriter(handle=__import__("io").BytesIO())

    mcap_video._begin_gop(writer, parameter_sets)
    mcap_video._begin_gop(writer, idr)

    assert writer.seen_idr is True
    assert writer.packet_count == 2
    assert writer.handle.getvalue() == parameter_sets + idr
