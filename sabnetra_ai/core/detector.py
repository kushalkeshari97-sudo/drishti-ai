import numpy as np
import torch
import logging
from typing import List, Optional, Tuple

from sabnetra_ai.config import DetectorConfig
from sabnetra_ai.models.model_manager import ModelManager
from sabnetra_ai.utils.helpers import Timer, is_low_light
from sabnetra_ai.utils.geometry import nms

logger = logging.getLogger(__name__)


class Detection:
    """Represents a single detection with bounding box and metadata."""

    __slots__ = ("bbox", "confidence", "class_id", "track_id", "crop")

    def __init__(self, bbox: np.ndarray, confidence: float, class_id: int):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.track_id = -1
        self.crop = None

    @property
    def x1(self) -> float: return self.bbox[0]
    @property
    def y1(self) -> float: return self.bbox[1]
    @property
    def x2(self) -> float: return self.bbox[2]
    @property
    def y2(self) -> float: return self.bbox[3]
    @property
    def cx(self) -> float: return (self.bbox[0] + self.bbox[2]) / 2
    @property
    def cy(self) -> float: return (self.bbox[1] + self.bbox[3]) / 2
    @property
    def width(self) -> float: return self.bbox[2] - self.bbox[0]
    @property
    def height(self) -> float: return self.bbox[3] - self.bbox[1]
    @property
    def area(self) -> float: return self.width * self.height

    def to_dict(self) -> dict:
        """Serialize detection to a dictionary."""
        return {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "track_id": self.track_id,
        }


class Detector:
    """Wraps a YOLO model to detect and filter objects in frames."""

    def __init__(self, config: Optional[DetectorConfig] = None,
                 model_manager: Optional[ModelManager] = None):
        """Initialize detector with config and model manager."""
        self.config = config or DetectorConfig()
        self.model_manager = model_manager or ModelManager()
        self._model = None

    @property
    def model(self):
        """Lazy-loaded YOLO detection model."""
        if self._model is None:
            self._model = self.model_manager.detector
        return self._model

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame before inference."""
        return frame

    def postprocess(self, results) -> List[Detection]:
        """Convert model outputs to filtered Detection list."""
        detections = []
        if results is None or len(results) == 0:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            if confs[i] < self.config.confidence_threshold:
                continue
            if self.config.classes and cls_ids[i] not in self.config.classes:
                continue
            w = xyxy[i][2] - xyxy[i][0]
            h = xyxy[i][3] - xyxy[i][1]
            if w <= 0 or h <= 0:
                continue
            if w > h * 1.1:
                continue
            if w * h < self.config.min_person_area:
                continue
            det = Detection(
                bbox=xyxy[i],
                confidence=float(confs[i]),
                class_id=int(cls_ids[i]),
            )
            detections.append(det)
        if detections:
            boxes_arr = np.array([d.bbox for d in detections])
            scores_arr = np.array([d.confidence for d in detections])
            keep = nms(boxes_arr, scores_arr, self.config.nms_threshold)
            detections = [detections[i] for i in keep]
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame and return filtered detections.
        Args:
            frame: Input image as numpy array.
        Returns:
            List of Detection objects.
        """
        with Timer("detector"):
            if self.config.skip_frame_on_low_light and is_low_light(frame, self.config.low_light_threshold):
                return []
            processed = self.preprocess(frame)
            results = self.model(
                processed,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                max_det=self.config.max_det,
                half=self.config.half_precision,
                device=self.config.device,
                classes=self.config.classes,
                verbose=False,
            )
            detections = self.postprocess(results)
            return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames.
        Args:
            frames: List of input frames.
        Returns:
            List of Detection lists per frame.
        """
        if not frames:
            return []
        with Timer("detector_batch"):
            valid_idxs = []
            valid_frames = []
            for i, f in enumerate(frames):
                if not (self.config.skip_frame_on_low_light and is_low_light(f, 30.0)):
                    valid_idxs.append(i)
                    valid_frames.append(f)
            if not valid_frames:
                return [[] for _ in frames]
            results = self.model(
                valid_frames,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                max_det=self.config.max_det,
                half=self.config.half_precision,
                device=self.config.device,
                classes=self.config.classes,
                verbose=False,
            )
            all_detections = [[] for _ in frames]
            if not isinstance(results, (list, tuple)):
                results = [results]
            for idx, result in zip(valid_idxs, results):
                all_detections[idx] = self.postprocess(result)
            return all_detections

    def extract_crops(self, frame: np.ndarray,
                      detections: List[Detection]) -> List[np.ndarray]:
        """Extract image crops for each detection bounding box.
        Args:
            frame: Source image.
            detections: List of detections to crop.
        Returns:
            List of cropped images.
        """
        crops = []
        for det in detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(frame.shape[1], int(det.x2))
            y2 = min(frame.shape[0], int(det.y2))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                det.crop = crop
                crops.append(crop)
            else:
                crops.append(None)
        return crops
