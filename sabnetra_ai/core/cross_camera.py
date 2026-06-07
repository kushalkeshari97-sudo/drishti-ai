import time
import logging
from typing import Optional, List, Tuple
from collections import defaultdict

from sabnetra_ai.config import TemporalConfig, MatcherConfig
from sabnetra_ai.core.matcher import MatchingEngine
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem, TrackMemory
from sabnetra_ai.utils.helpers import Timer

logger = logging.getLogger(__name__)


class CameraNode:
    def __init__(self, camera_id: str, position: Optional[Tuple[float, float]] = None):
        self.camera_id = camera_id
        self.position = position
        self.overlapping_cameras: list = []
        self.adjacent_cameras: list = []


class TemporalIntelligence:
    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self._last_seen: dict = {}

    def validate_transition(self, suspect_id: str, from_camera: str,
                            to_camera: str, timestamp: float) -> bool:
        key = f"{suspect_id}_{from_camera}_{to_camera}"
        if key in self._last_seen:
            elapsed = timestamp - self._last_seen[key]
            if elapsed < self.config.min_travel_time:
                return False
            if elapsed > self.config.temporal_window:
                return False
        self._last_seen[key] = timestamp
        return True

    def get_last_seen(self, suspect_id: str, camera_id: str) -> Optional[float]:
        return self._last_seen.get(f"{suspect_id}_{camera_id}")


class CrossCameraTracker:
    def __init__(self, matcher: MatchingEngine,
                 embedding_memory: EmbeddingMemorySystem,
                 temporal_config: Optional[TemporalConfig] = None,
                 matcher_config: Optional[MatcherConfig] = None):
        self.matcher = matcher
        self.embedding_memory = embedding_memory
        self.temporal = TemporalIntelligence(temporal_config or TemporalConfig())
        self.matcher_config = matcher_config or MatcherConfig()
        self._cameras: dict = {}
        self._suspect_camera_map: defaultdict = defaultdict(set)

    def register_camera(self, camera_id: str,
                        position: Optional[Tuple[float, float]] = None):
        self._cameras[camera_id] = CameraNode(camera_id, position)

    def set_camera_overlap(self, cam_a: str, cam_b: str):
        if cam_a in self._cameras and cam_b in self._cameras:
            self._cameras[cam_a].overlapping_cameras.append(cam_b)
            self._cameras[cam_b].overlapping_cameras.append(cam_a)

    def set_camera_adjacent(self, cam_a: str, cam_b: str):
        if cam_a in self._cameras and cam_b in self._cameras:
            self._cameras[cam_a].adjacent_cameras.append(cam_b)
            self._cameras[cam_b].adjacent_cameras.append(cam_a)

    def track_across_cameras(self, suspect_id: str, from_camera: str,
                             to_camera: str, features: dict,
                             timestamp: float) -> Tuple[bool, float]:
        with Timer("cross_camera"):
            if not self.temporal.validate_transition(
                    suspect_id, from_camera, to_camera, timestamp):
                logger.debug(f"Temporal constraint failed for {suspect_id} "
                             f"{from_camera} -> {to_camera}")
                return False, 0.0
            suspect_profile = self.matcher.get_suspect(suspect_id)
            if suspect_profile is None:
                return False, 0.0
            score = self.matcher.match_with_profile(features, suspect_profile)
            is_match = score >= self.matcher_config.yellow_threshold
            if is_match:
                self._suspect_camera_map[suspect_id].add(to_camera)
                logger.info(f"Cross-camera match: {suspect_id} "
                            f"{from_camera} -> {to_camera} score={score:.3f}")
            return is_match, score

    def find_potential_reappearances(self, suspect_id: str,
                                      camera_id: str,
                                      features: dict,
                                      timestamp: float) -> List[dict]:
        results = []
        for cid, cam_node in self._cameras.items():
            if cid == camera_id:
                continue
            is_match, score = self.track_across_cameras(
                suspect_id, camera_id, cid, features, timestamp)
            if is_match:
                results.append({
                    "camera_id": cid,
                    "suspect_id": suspect_id,
                    "score": score,
                    "timestamp": timestamp,
                })
        return results

    def get_suspect_camera_path(self, suspect_id: str) -> List[str]:
        return list(self._suspect_camera_map.get(suspect_id, set()))

    def get_overlapping_tracks(self, camera_id: str,
                                tracks: List[TrackMemory]) -> List[TrackMemory]:
        cam_node = self._cameras.get(camera_id)
        if not cam_node or not cam_node.overlapping_cameras:
            return []
        overlapping = []
        for track in tracks:
            if track.camera_id in cam_node.overlapping_cameras:
                overlapping.append(track)
        return overlapping

    def stats(self) -> dict:
        return {
            "registered_cameras": len(self._cameras),
            "cross_camera_suspects": len(self._suspect_camera_map),
        }
