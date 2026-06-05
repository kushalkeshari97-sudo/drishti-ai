import cv2
import numpy as np
import time
import logging
import os
from typing import Optional, List

from sabnetra_ai.core.matcher import SuspectProfile
from sabnetra_ai.core.feature_extractor import FeatureExtractor
from sabnetra_ai.config import FeatureConfig
from sabnetra_ai.utils.geometry import l2_normalize
from sabnetra_ai.utils.helpers import Timer

logger = logging.getLogger(__name__)


class FIRInputModule:
    def __init__(self, feature_config: Optional[FeatureConfig] = None,
                 feature_extractor: Optional[FeatureExtractor] = None):
        self.config = feature_config or FeatureConfig()
        self.feature_extractor = feature_extractor or FeatureExtractor(self.config)
        self._suspect_counter = 0

    def enroll_from_image(self, image_path: str, case_id: str = "",
                          suspect_id: Optional[str] = None,
                          metadata: Optional[dict] = None
                          ) -> Optional[SuspectProfile]:
        with Timer("enroll_image"):
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            h, w = frame.shape[:2]
            full_body_bbox = np.array([0, 0, w, h])
            features = self.feature_extractor.extract_all(
                frame, full_body_bbox, -1,
                "enrollment", time.time())
            if not features.has_any_embedding():
                logger.warning(f"No features extracted from {image_path}")
                return None
            sid = suspect_id or self._generate_suspect_id()
            profile = SuspectProfile(
                suspect_id=sid,
                case_id=case_id,
                face_emb=features.face_embedding,
                body_emb=features.body_embedding,
                clothing_emb=features.clothing_descriptor,
                gait_emb=features.gait_descriptor,
                metadata={
                    **(metadata or {}),
                    "source": "image",
                    "source_path": image_path,
                    "enrollment_time": time.time(),
                },
            )
            logger.info(f"Enrolled suspect {sid} from image: {image_path}")
            return profile

    def enroll_from_video(self, video_path: str, case_id: str = "",
                          suspect_id: Optional[str] = None,
                          metadata: Optional[dict] = None,
                          sample_rate: int = 15
                          ) -> Optional[SuspectProfile]:
        with Timer("enroll_video"):
            if not os.path.exists(video_path):
                logger.error(f"Video not found: {video_path}")
                return None
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            face_embs = []
            body_embs = []
            clothing_embs = []
            gait_frames = []
            gait_bboxes = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            sample_interval = max(1, int(fps / sample_rate))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % sample_interval != 0:
                    frame_count += 1
                    continue
                h, w = frame.shape[:2]
                bbox = np.array([0, 0, w, h])
                feats = self.feature_extractor.extract_all(
                    frame, bbox, track_id=-1,
                    camera_id="enrollment", timestamp=time.time())
                if feats.face_embedding is not None:
                    face_embs.append(feats.face_embedding)
                if feats.body_embedding is not None:
                    body_embs.append(feats.body_embedding)
                if feats.clothing_descriptor is not None:
                    clothing_embs.append(feats.clothing_descriptor)
                if feats.gait_descriptor is not None or self.config.gait_enabled:
                    gait_frames.append(frame)
                    gait_bboxes.append(bbox)
                frame_count += 1
            cap.release()
            if not face_embs and not body_embs:
                logger.warning(f"No features extracted from video: {video_path}")
                return None
            sid = suspect_id or self._generate_suspect_id()
            profile = SuspectProfile(
                suspect_id=sid,
                case_id=case_id,
                face_emb=self._fuse_embeddings(face_embs) if face_embs else None,
                body_emb=self._fuse_embeddings(body_embs) if body_embs else None,
                clothing_emb=self._fuse_embeddings(clothing_embs) if clothing_embs else None,
                gait_emb=self.feature_extractor.extract_gait(
                    gait_frames, gait_bboxes) if gait_frames else None,
                metadata={
                    **(metadata or {}),
                    "source": "video",
                    "source_path": video_path,
                    "enrollment_time": time.time(),
                    "frames_processed": frame_count // sample_interval,
                },
            )
            logger.info(f"Enrolled suspect {sid} from video: {video_path}")
            return profile

    def enroll_fir(self, images: List[str], videos: Optional[List[str]] = None,
                   case_id: str = "", suspect_id: Optional[str] = None,
                   metadata: Optional[dict] = None
                   ) -> Optional[SuspectProfile]:
        sid = suspect_id or self._generate_suspect_id()
        face_embs = []
        body_embs = []
        clothing_embs = []
        gait_embs = []
        for img_path in images:
            profile = self.enroll_from_image(img_path, case_id, sid, metadata)
            if profile:
                if profile.face_embedding is not None:
                    face_embs.append(profile.face_embedding)
                if profile.body_embedding is not None:
                    body_embs.append(profile.body_embedding)
                if profile.clothing_descriptor is not None:
                    clothing_embs.append(profile.clothing_descriptor)
        if videos:
            for vid_path in videos:
                profile = self.enroll_from_video(vid_path, case_id, sid, metadata)
                if profile:
                    if profile.face_embedding is not None:
                        face_embs.append(profile.face_embedding)
                    if profile.body_embedding is not None:
                        body_embs.append(profile.body_embedding)
                    if profile.clothing_descriptor is not None:
                        clothing_embs.append(profile.clothing_descriptor)
                    if profile.gait_descriptor is not None:
                        gait_embs.append(profile.gait_descriptor)
        if not face_embs and not body_embs:
            logger.error("FIR enrollment failed: no features extracted")
            return None
        final_profile = SuspectProfile(
            suspect_id=sid,
            case_id=case_id,
            face_emb=self._fuse_embeddings(face_embs) if face_embs else None,
            body_emb=self._fuse_embeddings(body_embs) if body_embs else None,
            clothing_emb=self._fuse_embeddings(clothing_embs) if clothing_embs else None,
            gait_emb=self._fuse_embeddings(gait_embs) if gait_embs else None,
            metadata={
                **(metadata or {}),
                "source": "fir",
                "num_images": len(images),
                "num_videos": len(videos) if videos else 0,
                "enrollment_time": time.time(),
            },
        )
        logger.info(f"FIR enrollment complete for suspect {sid}")
        return final_profile

    def _fuse_embeddings(self, embeddings: list) -> np.ndarray:
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return embeddings[0]
        stacked = np.stack(embeddings)
        fused = np.mean(stacked, axis=0)
        return l2_normalize(fused)

    def _generate_suspect_id(self) -> str:
        self._suspect_counter += 1
        return f"S{self._suspect_counter:04d}"
