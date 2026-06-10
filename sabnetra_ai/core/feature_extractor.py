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
    """Container for multi-modal identity embeddings of a single detection."""

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
        """Check if at least one embedding is present."""
        return any(x is not None for x in [
            self.face_embedding, self.body_embedding,
            self.clothing_descriptor, self.gait_descriptor])

    def to_dict(self) -> dict:
        """Serialize feature presence and metadata to dict."""
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
        """L2-normalize all stored embeddings in-place."""
        if self.face_embedding is not None:
            self.face_embedding = l2_normalize(self.face_embedding)
        if self.body_embedding is not None:
            self.body_embedding = l2_normalize(self.body_embedding)
        if self.clothing_descriptor is not None:
            self.clothing_descriptor = l2_normalize(self.clothing_descriptor)
        if self.gait_descriptor is not None:
            self.gait_descriptor = l2_normalize(self.gait_descriptor)


class FeatureExtractor:
    """Extracts multi-modal features (face, body, clothing, gait) from detections."""

    def __init__(self, config: Optional[FeatureConfig] = None,
                 model_manager: Optional[ModelManager] = None,
                 enable_face_recognition: bool = True,
                 enable_reid: bool = True,
                 enable_clothing: bool = True,
                 enable_gait: bool = False):
        """Initialize extractor with config and per-modality toggles."""
        self.config = config or FeatureConfig()
        self.model_manager = model_manager or ModelManager()
        self.enable_face_recognition = enable_face_recognition
        self.enable_reid = enable_reid
        self.enable_clothing = enable_clothing
        self.enable_gait = enable_gait

    def extract_face(self, frame: np.ndarray,
                     bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a bounding box region.
        Args:
            frame: Source image.
            bbox: (x1, y1, x2, y2) bounding box.
        Returns:
            L2-normalized face embedding or None.
        """
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
                logger.warning(f"Face extraction failed: {e}")
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
                rgb_resized = cv2.resize(rgb, (128, 256))
                rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float()
                if torch.cuda.is_available():
                    rgb_tensor = rgb_tensor.cuda()
                with torch.no_grad():
                    emb = model(rgb_tensor)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                if emb.ndim > 1:
                    emb = emb.flatten()
                return l2_normalize(emb.astype(np.float32))
            except Exception as e:
                logger.warning(f"ReID extraction failed: {e}")
                return None

    def extract_clothing(self, frame: np.ndarray,
                          bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract HSV-based clothing color descriptor from the upper body.
        Args:
            frame: Source image.
            bbox: (x1, y1, x2, y2) bounding box.
        Returns:
            L2-normalized clothing descriptor or None.
        """
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
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h_bins, s_bins, v_bins = 32, 16, 16
                h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
                v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
                for hist in (h_hist, s_hist, v_hist):
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                descriptor = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
                return l2_normalize(descriptor.astype(np.float32))
            except Exception as e:
                logger.warning(f"Clothing extraction failed: {e}")
                return None

    def extract_gait(self, frames: List[np.ndarray],
                     bboxes: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extract gait embedding from a sequence of frames.
        Args:
            frames: Sequence of source images.
            bboxes: Corresponding bounding boxes.
        Returns:
            L2-normalized gait embedding or None.
        """
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
                logger.warning(f"Gait extraction failed: {e}")
                return None

    def extract_all(self, frame: np.ndarray, bbox: np.ndarray,
                    track_id: int, camera_id: str, timestamp: float,
                    gait_frames: Optional[List[np.ndarray]] = None,
                    gait_bboxes: Optional[List[np.ndarray]] = None
                    ) -> IdentityFeatures:
        """Extract all enabled feature modalities for a detection.
        Args:
            frame: Source image.
            bbox: Bounding box of the detection.
            track_id: Track identifier.
            camera_id: Camera identifier.
            timestamp: Frame timestamp.
            gait_frames: Optional gait sequence frames.
            gait_bboxes: Optional gait sequence bboxes.
        Returns:
            IdentityFeatures with extracted embeddings.
        """
        features = IdentityFeatures(
            track_id=track_id,
            camera_id=camera_id,
            timestamp=timestamp,
        )
        features.bbox = bbox
        if self.enable_face_recognition:
            features.face_embedding = self.extract_face(frame, bbox)
        if self.enable_reid:
            features.body_embedding = self.extract_body(frame, bbox)
        if self.enable_clothing:
            features.clothing_descriptor = self.extract_clothing(frame, bbox)
        if self.enable_gait and gait_frames and gait_bboxes:
            features.gait_descriptor = self.extract_gait(gait_frames, gait_bboxes)
        features.normalize_all()
        return features
