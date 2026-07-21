"""Shared defaults for local hand-dataset artifacts."""

from pathlib import Path

HEAD_LEFT_VIDEO_TOPIC = "/head_left/video"
DEFAULT_VIDEO_DIR = Path("data/videos")
DEFAULT_VIDEO_METADATA_PATH = DEFAULT_VIDEO_DIR / "metadata.jsonl"
DEFAULT_FRAMES_DIR = Path("data/frames")
DEFAULT_FRAME_METADATA_PATH = DEFAULT_FRAMES_DIR / "metadata.jsonl"
DEFAULT_FRAME_FPS = 1.0
DEFAULT_MODEL_DIR = Path("models/wilor")
DEFAULT_WILOR_DETECTOR_PATH = DEFAULT_MODEL_DIR / "detector.pt"
DEFAULT_WILOR_DETECTOR_METADATA_PATH = DEFAULT_MODEL_DIR / "detector.metadata.json"
DEFAULT_PREDICTIONS_PATH = Path("data/predictions/wilor-detector.jsonl")
DEFAULT_ROBOFLOW_EXPORT_DIR = Path("data/roboflow-export/wilor-detector-yolo")
DEFAULT_WILOR_DETECTOR_URL = (
    "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
    "pretrained_models/detector.pt"
)
DEFAULT_WILOR_CONFIDENCE = 0.3
