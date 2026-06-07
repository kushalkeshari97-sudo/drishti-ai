import numpy as np
from sabnetra_ai.core.detector import Detection, Detector
from sabnetra_ai.config import DetectorConfig


def test_detection_creation():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = Detection(bbox, 0.85, 0)
    assert det.confidence == 0.85
    assert det.class_id == 0
    assert det.track_id == -1
    assert det.crop is None


def test_detection_properties():
    bbox = np.array([10, 20, 110, 220], dtype=float)
    det = Detection(bbox, 0.9, 0)
    assert det.x1 == 10
    assert det.y1 == 20
    assert det.x2 == 110
    assert det.y2 == 220
    assert det.cx == 60
    assert det.cy == 120
    assert det.width == 100
    assert det.height == 200
    assert det.area == 20000


def test_detection_to_dict():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = Detection(bbox, 0.85, 0)
    d = det.to_dict()
    assert d["confidence"] == 0.85
    assert d["class_id"] == 0
    assert d["track_id"] == -1


def test_detector_preprocess():
    config = DetectorConfig()
    det = Detector(config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = det.preprocess(frame)
    assert result is frame


def test_extract_crops():
    config = DetectorConfig()
    det = Detector(config)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    detections = [
        Detection(np.array([10, 10, 100, 200], dtype=float), 0.9, 0),
        Detection(np.array([200, 100, 300, 300], dtype=float), 0.8, 0),
    ]
    crops = det.extract_crops(frame, detections)
    assert len(crops) == 2
    assert crops[0] is not None
    assert detections[0].crop is not None


def test_extract_crops_out_of_bounds():
    config = DetectorConfig()
    det = Detector(config)
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(np.array([-10, -10, 50, 50], dtype=float), 0.9, 0),
        Detection(np.array([0, 0, 200, 200], dtype=float), 0.8, 0),
    ]
    crops = det.extract_crops(frame, detections)
    assert len(crops) == 2
    assert crops[0] is not None
    assert crops[1] is not None


def test_postprocess_empty_results():
    config = DetectorConfig()
    det = Detector(config)
    assert det.postprocess(None) == []
    assert det.postprocess([]) == []
