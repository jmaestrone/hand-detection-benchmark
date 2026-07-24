"""Tests for WiLoR detector metadata and raw prediction normalization."""

import pytest

from hand_benchmark.wilor import detections_from_result, validate_wilor_class_names


class _Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    xyxy = _Tensor([[1.0, -2.0, 22.0, 18.0]])
    cls = _Tensor([0.0])
    conf = _Tensor([0.9])

    def __len__(self):
        return 1


class _Result:
    boxes = _Boxes()


def test_wilor_class_mapping_requires_left_then_right() -> None:
    validate_wilor_class_names({0: "left", 1: "right"})

    with pytest.raises(ValueError, match="Unexpected WiLoR class mapping"):
        validate_wilor_class_names({0: "right", 1: "left"})


def test_detection_boxes_are_clipped_and_labeled() -> None:
    detections = detections_from_result(_Result(), width=20, height=16)

    assert detections[0].category == "left_hand"
    assert detections[0].bbox_xyxy == [1.0, 0.0, 20, 16]
