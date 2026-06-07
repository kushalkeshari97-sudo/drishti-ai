import numpy as np
from sabnetra_ai.utils.geometry import (
    compute_iou, compute_giou, cosine_similarity, l2_normalize,
    nms, box_center, box_area, euclidean_distance,
)


def test_compute_iou():
    box1 = np.array([0, 0, 10, 10], dtype=float)
    box2 = np.array([5, 0, 15, 10], dtype=float)
    iou = compute_iou(box1, box2)
    assert 0.3 < iou < 0.35


def test_compute_iou_no_overlap():
    box1 = np.array([0, 0, 10, 10], dtype=float)
    box2 = np.array([20, 20, 30, 30], dtype=float)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_identical():
    box = np.array([0, 0, 10, 10], dtype=float)
    assert compute_iou(box, box) == 1.0


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_similarity_zero_norm():
    a = np.zeros(5)
    b = np.ones(5)
    assert cosine_similarity(a, b) == 0.0


def test_l2_normalize():
    v = np.array([3.0, 4.0])
    n = l2_normalize(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-6


def test_l2_normalize_zero():
    v = np.zeros(5)
    n = l2_normalize(v)
    assert np.all(n == 0)


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.45)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep


def test_nms_empty():
    keep = nms(np.array([]).reshape(0, 4), np.array([]), 0.45)
    assert len(keep) == 0


def test_box_center():
    box = np.array([0, 0, 10, 10])
    cx, cy = box_center(box)
    assert cx == 5.0 and cy == 5.0


def test_box_area():
    box = np.array([0, 0, 10, 10])
    assert box_area(box) == 100.0


def test_box_area_zero():
    box = np.array([5, 5, 5, 5])
    assert box_area(box) == 0.0


def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert abs(euclidean_distance(a, b) - 5.0) < 1e-6
