import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, List, Dict, Callable
from collections import defaultdict

from sabnetra_ai.config import SabNetraConfig
from sabnetra_ai.core.frame_buffer import FrameBufferManager, Frame
from sabnetra_ai.core.detector import Detector, Detection
from sabnetra_ai.core.tracker import BoTSORT, Track
from sabnetra_ai.core.feature_extractor import FeatureExtractor, IdentityFeatures
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem
from sabnetra_ai.core.matcher import MatchingEngine
from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.alert_system import AlertSystem, Alert
from sabnetra_ai.core.suspect_manager import GlobalSuspectManager
from sabnetra_ai.core.cross_camera import CrossCameraTracker
from sabnetra_ai.models.model_manager import ModelManager
from sabnetra_ai.stream.rtsp_manager import RTSPManager
from sabnetra_ai.enrollment.fir_module import FIRInputModule
from sabnetra_ai.utils.helpers import Timer, FrameRateCounter, draw_detection

logger = logging.getLogger(__name__)


class CameraPipeline:
    def __init__(self, camera_id: str, config: SabNetraConfig,
                 detector: Detector, tracker: BoTSORT,
                 feature_extractor: FeatureExtractor):
        self.camera_id = camera_id
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.feature_extractor = feature_extractor
        self.frame_rate_counter = FrameRateCounter()
        self._frame_skip_counter = 0
        self._gait_buffers: Dict[int, dict] = {}

    def process_frame(self, frame: np.ndarray, timestamp: float
                      ) -> List[dict]:
        self.frame_rate_counter.tick()
        self._frame_skip_counter += 1
        if self._frame_skip_counter < self.config.pipeline.process_every_n_frames:
            return []
        self._frame_skip_counter = 0
        detections = self.detector.detect(frame)
        if not detections:
            return []
        detection_dicts = []
        for det in detections:
            detection_dicts.append({
                "bbox": det.bbox.copy(),
                "confidence": det.confidence,
                "class_id": det.class_id,
            })
        frame_embeddings = []
        for det in detections:
            features = self.feature_extractor.extract_all(
                frame, det.bbox, track_id=-1,
                camera_id=self.camera_id, timestamp=timestamp)
            frame_embeddings.append(features.body_embedding)
        tracks = self.tracker.update(detection_dicts, frame_embeddings)
        results = []
        for track in tracks:
            box = track.bbox.copy()
            features = self.feature_extractor.extract_all(
                frame, box, track.track_id,
                self.camera_id, timestamp)
            results.append({
                "track_id": track.track_id,
                "bbox": box,
                "confidence": track.confidence,
                "features": features,
                "timestamp": timestamp,
            })
        return results


class SabNetraPipeline:
    def __init__(self, config: Optional[SabNetraConfig] = None):
        self.config = config or SabNetraConfig()
        self._running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self.model_manager = ModelManager(
            self.config.device, self.config.pipeline.enable_gait,
            self.config.features.face_det_thresh,
            self.config.detector.model_path)
        self.buffer_manager = FrameBufferManager(
            maxsize=self.config.pipeline.frame_buffer_size)
        self.detector = Detector(self.config.detector, self.model_manager)
        self.feature_extractor = FeatureExtractor(
            self.config.features, self.model_manager)
        self.trackers: Dict[str, BoTSORT] = {}
        self.camera_pipelines: Dict[str, CameraPipeline] = {}
        self.embedding_memory = EmbeddingMemorySystem(self.config)
        self.matcher = MatchingEngine(self.config.matcher)
        self.state_engine = StateClassificationEngine(self.config.matcher)
        self.alert_system = AlertSystem(self.config.alert)
        self.suspect_manager = GlobalSuspectManager(
            self.matcher, self.state_engine, self.embedding_memory)
        self.cross_camera = CrossCameraTracker(
            self.matcher, self.embedding_memory,
            self.config.temporal, self.config.matcher)
        self.rtsp_manager = RTSPManager(self.buffer_manager)
        self.fir_module = FIRInputModule(self.config.features,
                                          self.feature_extractor)
        self._on_detection_callbacks: List[Callable] = []
        self._on_alert_callbacks: List[Callable] = []
        self._on_frame_callbacks: List[Callable] = []
        self._pipeline_stats = {
            "frames_processed": 0,
            "detections": 0,
            "matches": 0,
            "alerts": 0,
        }

    def add_camera(self, rtsp_url: str, camera_id: str) -> bool:
        with self._lock:
            if camera_id in self.camera_pipelines:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            stream = self.rtsp_manager.add_stream(rtsp_url, camera_id)
            tracker = BoTSORT(self.config.tracker)
            pipe = CameraPipeline(
                camera_id, self.config, self.detector, tracker,
                self.feature_extractor)
            self.trackers[camera_id] = tracker
            self.camera_pipelines[camera_id] = pipe
            self.cross_camera.register_camera(camera_id)
            logger.info(f"Camera {camera_id} added to pipeline")
            return True

    def remove_camera(self, camera_id: str):
        with self._lock:
            self.rtsp_manager.remove_stream(camera_id)
            self.trackers.pop(camera_id, None)
            self.camera_pipelines.pop(camera_id, None)

    def enroll_suspect(self, images: List[str],
                       videos: Optional[List[str]] = None,
                       case_id: str = "",
                       suspect_id: Optional[str] = None) -> Optional[str]:
        profile = self.fir_module.enroll_fir(
            images, videos, case_id, suspect_id)
        if profile is None:
            return None
        self.suspect_manager.enroll_suspect(profile)
        return profile.suspect_id

    def start(self):
        if self._running:
            return
        self._running = True
        self.model_manager.warmup()
        self.rtsp_manager.start_all()
        self.alert_system.register_callback(self._handle_alert)
        self._threads.append(threading.Thread(
            target=self._pipeline_loop, name="SabNetra-Pipeline",
            daemon=True))
        self._threads.append(threading.Thread(
            target=self._maintenance_loop, name="SabNetra-Maintenance",
            daemon=True))
        for t in self._threads:
            t.start()
        logger.info("SabNetra Pipeline started")

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self.rtsp_manager.stop_all()
        self.model_manager.release()
        self.buffer_manager.clear_all()
        logger.info("SabNetra Pipeline stopped")

    def _pipeline_loop(self):
        while self._running:
            try:
                for camera_id in self.buffer_manager.camera_ids():
                    buffer = self.buffer_manager.get_buffer(camera_id)
                    if buffer is None:
                        continue
                    frame = buffer.pop()
                    if frame is None:
                        continue
                    self._process_single_frame(
                        frame.data, camera_id, frame.timestamp)
                    self._pipeline_stats["frames_processed"] += 1
                    for cb in self._on_frame_callbacks:
                        try:
                            cb(frame.data, camera_id, frame.timestamp)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}", exc_info=True)

    def _process_single_frame(self, frame: np.ndarray,
                               camera_id: str, timestamp: float):
        pipe = self.camera_pipelines.get(camera_id)
        if pipe is None:
            return
        results = pipe.process_frame(frame, timestamp)
        for res in results:
            self._pipeline_stats["detections"] += 1
            features = res["features"]
            track_id = res["track_id"]
            if not features.has_any_embedding():
                continue
            features_dict = {
                "face_embedding": features.face_embedding,
                "body_embedding": features.body_embedding,
                "clothing_descriptor": features.clothing_descriptor,
                "gait_descriptor": features.gait_descriptor,
            }
            state, suspect_id, score = self.suspect_manager.process_detection(
                track_id, camera_id, features_dict, timestamp)
            res["state"] = state
            res["suspect_id"] = suspect_id
            res["score"] = score
            if state == IdentityState.RED:
                self._pipeline_stats["matches"] += 1
                alert = self.alert_system.trigger(
                    suspect_id, track_id, camera_id, state, score)
                if alert:
                    self._pipeline_stats["alerts"] += 1
                self.cross_camera.find_potential_reappearances(
                    suspect_id, camera_id, features_dict, timestamp)
            for cb in self._on_detection_callbacks:
                try:
                    cb(res, frame)
                except Exception:
                    pass
            if self.config.pipeline.visualize:
                draw_detection(frame, res["bbox"], track_id,
                               state, suspect_id, score)

    def _maintenance_loop(self):
        while self._running:
            time.sleep(60.0)
            try:
                self.embedding_memory.cleanup_stale(timeout=300.0)
                logger.debug("Maintenance: cleaned stale tracks")
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    def _handle_alert(self, alert: Alert):
        for cb in self._on_alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass

    def register_detection_callback(self, callback: Callable):
        self._on_detection_callbacks.append(callback)

    def register_alert_callback(self, callback: Callable):
        self._on_alert_callbacks.append(callback)

    def register_frame_callback(self, callback: Callable):
        self._on_frame_callbacks.append(callback)

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        memory_stats = self.embedding_memory.stats()
        alert_stats = self.alert_system.stats()
        suspect_stats = self.suspect_manager.stats()
        cross_stats = self.cross_camera.stats()
        pipeline_stats = dict(self._pipeline_stats)
        stream_stats = self.rtsp_manager.stats()
        return {
            "pipeline": pipeline_stats,
            "memory": memory_stats,
            "alerts": alert_stats,
            "suspects": suspect_stats,
            "cross_camera": cross_stats,
            "streams": stream_stats,
        }
