import cv2
import numpy as np
import torch
import logging
from typing import List, Optional, Tuple

from sabnetra_ai.config import FeatureConfig
from sabnetra_ai.models.model_manager import ModelManager
from sabnetra_ai.utils.helpers import Timer
from sabnetra_ai.utils.geometry import l2_normalize

logger = logging.getLogger(__name__)


class IdentityFeatures:
    __slots__ = ("face_embedding", "body_embedding", "clothing_descriptor",
                 "gait_descriptor", "track_id", "camera_id", "timestamp",
                 "bbox")

    def __init__(self, track_id: int = -1, camera_id: str = "",
                 timestamp: float = 0.0):
        self.face_embedding: Optional[np.ndarray] = None
        self.body_embedding: Optional[np.ndarray] = None
        self.clothing_descriptor: Optional[np.ndarray] = None
        self.gait_descriptor: Optional[np.ndarray] = None
        self.track_id = track_id
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.bbox: Optional[np.ndarray] = None

    def has_any_embedding(self) -> bool:
        return any(x is not None for x in [
            self.face_embedding, self.body_embedding,
            self.clothing_descriptor, self.gait_descriptor])

    def to_dict(self) -> dict:
        return {
            "has_face": self.face_embedding is not None,
            "face_dim": len(self.face_embedding) if self.face_embedding is not None else 0,
            "has_body": self.body_embedding is not None,
            "body_dim": len(self.body_embedding) if self.body_embedding is not None else 0,
            "has_clothing": self.clothing_descriptor is not None,
            "has_gait": self.gait_descriptor is not None,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
        }

    def normalize_all(self):
        if self.face_embedding is not None:
            self.face_embedding = l2_normalize(self.face_embedding)
        if self.body_embedding is not None:
            self.body_embedding = l2_normalize(self.body_embedding)
        if self.clothing_descriptor is not None:
            self.clothing_descriptor = l2_normalize(self.clothing_descriptor)
        if self.gait_descriptor is not None:
            self.gait_descriptor = l2_normalize(self.gait_descriptor)


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None,
                 model_manager: Optional[ModelManager] = None):
        self.config = config or FeatureConfig()
        self.model_manager = model_manager or ModelManager()
        import torch.nn as nn
        self._clothing_conv: Optional[nn.Conv2d] = None

    def extract_face(self, frame: np.ndarray,
                     bbox: np.ndarray) -> Optional[np.ndarray]:
        with Timer("face_extract"):
            try:
                model = self.model_manager.face_model
                if model is None:
                    return None
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    return None
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    return None
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                faces = model.get(rgb)
                if not faces:
                    return None
                best_face = max(faces, key=lambda f: f.det_score)
                if best_face.det_score < self.config.face_det_thresh:
                    return None
                embedding = best_face.normed_embedding
                return l2_normalize(embedding.astype(np.float32))
            except Exception as e:
                logger.debug(f"Face extraction failed: {e}")
                return None

    def extract_body(self, frame: np.ndarray,
                     bbox: np.ndarray) -> Optional[np.ndarray]:
        with Timer("reid_extract"):
            try:
                model = self.model_manager.reid_model
                if model is None:
                    return None
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 < 30 or y2 - y1 < 30:
                    return None
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                emb = model.predict(rgb)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                if emb.ndim > 1:
                    emb = emb.flatten()
                return l2_normalize(emb.astype(np.float32))
            except Exception as e:
                logger.debug(f"ReID extraction failed: {e}")
                return None

    def extract_clothing(self, frame: np.ndarray,
                         bbox: np.ndarray) -> Optional[np.ndarray]:
        with Timer("clothing_extract"):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                h = y2 - y1
                body_top = y1 + int(h * 0.15)
                body_bottom = y1 + int(h * 0.65)
                body_top = max(y1, body_top)
                body_bottom = min(y2, body_bottom)
                if body_bottom <= body_top:
                    return None
                crop = frame[body_top:body_bottom, x1:x2]
                if crop.size == 0:
                    return None
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (128, 256))
                resized = resized.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                resized = (resized - mean) / std
                resized = np.transpose(resized, (2, 0, 1))
                tensor = torch.from_numpy(resized).unsqueeze(0)
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                if self._clothing_conv is None:
                    self._clothing_conv = nn.Conv2d(
                        3, self.config.clothing_feature_dim,
                        kernel_size=1).to(tensor.device)
                with torch.no_grad():
                    feat = self._clothing_conv(tensor).flatten().cpu().numpy()
                return l2_normalize(feat.astype(np.float32))
            except Exception as e:
                logger.debug(f"Clothing extraction failed: {e}")
                return None

    def extract_gait(self, frames: List[np.ndarray],
                     bboxes: List[np.ndarray]) -> Optional[np.ndarray]:
        if not self.config.gait_enabled:
            return None
        with Timer("gait_extract"):
            try:
                model = self.model_manager.gait_model
                if model is None or len(frames) < self.config.gait_sequence_length:
                    return None
                silhs = []
                for frame, bbox in zip(frames, bboxes):
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, silh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    silh = cv2.resize(silh, (64, 64))
                    silhs.append(silh)
                if len(silhs) < self.config.gait_sequence_length:
                    return None
                silh_seq = np.stack(silhs[:self.config.gait_sequence_length])
                silh_seq = silh_seq.astype(np.float32) / 255.0
                silh_seq = torch.from_numpy(silh_seq).unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                    silh_seq = silh_seq.cuda()
                emb = model(silh_seq)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy().flatten()
                return l2_normalize(emb.astype(np.float32))
            except Exception as e:
                logger.debug(f"Gait extraction failed: {e}")
                return None

    def extract_all(self, frame: np.ndarray, bbox: np.ndarray,
                    track_id: int, camera_id: str, timestamp: float,
                    gait_frames: Optional[List[np.ndarray]] = None,
                    gait_bboxes: Optional[List[np.ndarray]] = None
                    ) -> IdentityFeatures:
        features = IdentityFeatures(
            track_id=track_id,
            camera_id=camera_id,
            timestamp=timestamp,
        )
        features.bbox = bbox
        features.face_embedding = self.extract_face(frame, bbox)
        features.body_embedding = self.extract_body(frame, bbox)
        features.clothing_descriptor = self.extract_clothing(frame, bbox)
        if gait_frames and gait_bboxes:
            features.gait_descriptor = self.extract_gait(gait_frames, gait_bboxes)
        features.normalize_all()
        return features
