import numpy as np
from typing import Tuple, Optional


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_giou(box1: np.ndarray, box2: np.ndarray) -> float:
    iou = compute_iou(box1, box2)
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    c_area = max(0, x2 - x1) * max(0, y2 - y1)
    return iou - (c_area - ( (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - (c_area * iou) )) / c_area if c_area > 0 else iou


def box_center(box: np.ndarray) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_area(box: np.ndarray) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    ious = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            ious[i, j] = compute_iou(boxes1[i], boxes2[j])
    return ious


def warp_boxes(boxes: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes.shape[0] == 0:
        return boxes
    ones = np.ones((boxes.shape[0], 1))
    corners = np.concatenate([
        boxes[:, 0:1], boxes[:, 1:2], ones,
        boxes[:, 2:3], boxes[:, 1:2], ones,
        boxes[:, 0:1], boxes[:, 3:4], ones,
        boxes[:, 2:3], boxes[:, 3:4], ones,
    ], axis=1).reshape(-1, 3)
    warped = M @ corners.T
    warped = warped.T
    warped = warped[:, :2] / warped[:, 2:3]
    warped = warped.reshape(-1, 4, 2)
    x1 = np.min(warped[:, :, 0], axis=1)
    y1 = np.min(warped[:, :, 1], axis=1)
    x2 = np.max(warped[:, :, 0], axis=1)
    y2 = np.max(warped[:, :, 1], axis=1)
    warped_boxes = np.stack([x1, y1, x2, y2], axis=1)
    warped_boxes[:, 0] = np.clip(warped_boxes[:, 0], 0, w)
    warped_boxes[:, 1] = np.clip(warped_boxes[:, 1], 0, h)
    warped_boxes[:, 2] = np.clip(warped_boxes[:, 2], 0, w)
    warped_boxes[:, 3] = np.clip(warped_boxes[:, 3], 0, h)
    return warped_boxes


def is_occluded(box: np.ndarray, other_boxes: np.ndarray, iou_thresh: float = 0.1) -> bool:
    for other in other_boxes:
        if np.array_equal(box, other):
            continue
        if compute_iou(box, other) > iou_thresh:
            return True
    return False


def compute_optical_flow_boxes(prev_gray: np.ndarray, curr_gray: np.ndarray,
                               boxes: np.ndarray) -> np.ndarray:
    try:
        import cv2
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        warped = boxes.copy().astype(np.float32)
        for i, box in enumerate(boxes):
            cx, cy = box_center(box)
            cx_i, cy_i = int(round(cx)), int(round(cy))
            if 0 <= cy_i < flow.shape[0] and 0 <= cx_i < flow.shape[1]:
                dx, dy = flow[cy_i, cx_i]
                warped[i, 0] += dx
                warped[i, 1] += dy
                warped[i, 2] += dx
                warped[i, 3] += dy
        return warped
    except Exception:
        return boxes


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)
