"""Shared defaults for local hand-dataset artifacts."""

from pathlib import Path

HEAD_LEFT_VIDEO_TOPIC = "/head_left/video"
DEFAULT_VIDEO_DIR = Path("data/videos")
DEFAULT_VIDEO_METADATA_PATH = DEFAULT_VIDEO_DIR / "metadata.jsonl"
DEFAULT_FRAMES_DIR = Path("data/frames")
DEFAULT_FRAME_METADATA_PATH = DEFAULT_FRAMES_DIR / "metadata.jsonl"
DEFAULT_FRAME_FPS = 1.0
