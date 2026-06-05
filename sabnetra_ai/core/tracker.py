import numpy as np
from collections import OrderedDict, deque
from typing import List, Optional, Tuple
import logging
from scipy.optimize import linear_sum_assignment

from sabnetra_ai.config import TrackerConfig
from sabnetra_ai.utils.geometry import compute_iou, cosine_similarity, l2_normalize
from sabnetra_ai.utils.helpers import Timer

logger = logging.getLogger(__name__)


class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.A = np.eye(4, dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 0.1
        self.P = np.eye(4, dtype=np.float32) * 10.0

    def predict(self, mean: np.ndarray, covariance: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.A @ mean
        covariance = self.A @ covariance @ self.A.T + self.Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (measurement - self.H @ mean)
        covariance = (np.eye(4) - K @ self.H) @ covariance
        return mean, covariance


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    OCCLUDED = 4
    MERGED = 5


class Track:
    def __init__(self, track_id: int, bbox: np.ndarray, detection: dict,
                 appearance_embedding: Optional[np.ndarray] = None,
                 confidence: float = 0.0):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.no_losses = 0
        self.time_since_update = 0
        self.age = 0
        self.confidence = confidence
        self.mean, self.covariance = self._init_kf(bbox)
        self.history = deque(maxlen=120)
        self.history.append(bbox.copy())
        self.bbox = bbox.copy()
        self.appearance_embedding = appearance_embedding
        self.embeddings_history = deque(maxlen=30)
        if appearance_embedding is not None:
            self.embeddings_history.append(appearance_embedding)
        self.smooth_bbox = bbox.copy()
        self.alpha = 0.3
        self.kf = KalmanFilter()
        self.detection = detection

    def _init_kf(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        mean = np.array([cx, cy, w, h], dtype=np.float32)
        covariance = np.eye(4, dtype=np.float32) * 10.0
        return mean, covariance

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        cx, cy, w, h = self.mean
        self.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        return self.bbox

    def update(self, bbox: np.ndarray, embedding: Optional[np.ndarray] = None,
               confidence: float = 0.0):
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        measurement = np.array([cx, cy], dtype=np.float32)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.mean[2] = w
        self.mean[3] = h
        self.bbox = bbox.copy()
        self.history.append(bbox.copy())
        self.smooth_bbox = self.alpha * bbox + (1 - self.alpha) * self.smooth_bbox
        if embedding is not None:
            self.appearance_embedding = embedding
            self.embeddings_history.append(embedding)
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED

    def get_appearance_embedding(self) -> Optional[np.ndarray]:
        if self.appearance_embedding is not None:
            return self.appearance_embedding
        if self.embeddings_history:
            embs = np.array(self.embeddings_history)
            return np.mean(embs, axis=0)
        return None

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_occluded(self):
        self.state = TrackState.OCCLUDED

    def mark_merged(self):
        self.state = TrackState.MERGED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_occluded(self) -> bool:
        return self.state == TrackState.OCCLUDED


class BoTSORT:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._next_id = 1
        self.active_tracks: OrderedDict[int, Track] = OrderedDict()
        self.lost_tracks: OrderedDict[int, Track] = OrderedDict()
        self.fragmented_tracks: list = []

    def _gaussian_heatmap_distance(self, bbox1: np.ndarray,
                                   bbox2: np.ndarray) -> float:
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        w = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
        h = max(bbox1[3] - bbox1[1], bbox2[3] - bbox2[1])
        sigma = max(w, h) * 0.5
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return np.exp(-dist**2 / (2 * sigma**2))

    def _compute_cost_matrix(self, tracks: List[Track],
                             detections: List[dict]) -> np.ndarray:
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_cost = 1 - compute_iou(track.bbox, det["bbox"])
                motion_cost = 1 - self._gaussian_heatmap_distance(
                    track.bbox, det["bbox"])
                has_appearance = track.appearance_embedding is not None and \
                    det.get("embedding") is not None
                appearance_cost = 0.0
                if has_appearance:
                    cos_sim = cosine_similarity(
                        track.appearance_embedding, det["embedding"])
                    appearance_cost = max(0, 1 - cos_sim)
                mw = self.config.motion_weight
                aw = self.config.appearance_weight if has_appearance else 0.0
                iw = 1.0 - mw - aw
                cost = (mw * motion_cost + aw * appearance_cost + iw * iou_cost) / (mw + aw + iw)
                cost_matrix[i, j] = cost
        return cost_matrix

    def _hungarian_match(self, cost_matrix: np.ndarray,
                         thresh: float) -> List[Tuple[int, int]]:
        if cost_matrix.size == 0:
            return []
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
        return matches

    def _merge_fragmented_tracks(self, track: Track):
        for ft in self.fragmented_tracks:
            iou = compute_iou(track.bbox, ft["bbox"])
            if iou > self.config.fragmented_track_merge_iou:
                emb_sim = cosine_similarity(
                    track.get_appearance_embedding(),
                    ft.get("embedding", np.zeros(512))
                ) if track.get_appearance_embedding() is not None else 0
                if emb_sim > self.config.fragmented_track_embedding_thresh:
                    logger.info(f"Merging fragmented track {ft['track_id']} "
                                f"into {track.track_id}")
                    self.fragmented_tracks.remove(ft)
                    break

    def update(self, detections: List[dict],
               frame_embeddings: Optional[List[np.ndarray]] = None
               ) -> List[Track]:
        with Timer("botsort_update"):
            for i, det in enumerate(detections):
                if frame_embeddings is not None and i < len(frame_embeddings):
                    det["embedding"] = frame_embeddings[i]
                else:
                    det["embedding"] = None
            for track in self.active_tracks.values():
                track.predict()

            confirmed = [t for t in self.active_tracks.values()
                         if t.is_confirmed]
            tentative = [t for t in self.active_tracks.values()
                         if t.is_tentative]
            unmatched_dets = list(range(len(detections)))
            matches = []
            tracks_to_update = []

            if confirmed and detections:
                cost_matrix = self._compute_cost_matrix(confirmed, detections)
                matches = self._hungarian_match(
                    cost_matrix, self.config.match_thresh)
                matched_dets = set()
                for r, c in matches:
                    confirmed[r].update(
                        detections[c]["bbox"],
                        detections[c].get("embedding"),
                        detections[c].get("confidence", 0.0),
                    )
                    tracks_to_update.append(confirmed[r].track_id)
                    matched_dets.add(c)
                unmatched_dets = [j for j in range(len(detections))
                                  if j not in matched_dets]
                unmatched_tracks = [i for i, t in enumerate(confirmed)
                                    if t.track_id not in tracks_to_update]
                for ui in unmatched_tracks:
                    confirmed[ui].mark_lost()
                    self.lost_tracks[confirmed[ui].track_id] = confirmed[ui]

            if tentative and detections:
                tentative_cost = self._compute_cost_matrix(tentative, detections)
                tentative_matches = self._hungarian_match(
                    tentative_cost, self.config.match_thresh * 1.2)
                for r, c in tentative_matches:
                    tentative[r].update(
                        detections[c]["bbox"],
                        detections[c].get("embedding"),
                        detections[c].get("confidence", 0.0),
                    )
                    if tentative[r].is_confirmed:
                        tracks_to_update.append(tentative[r].track_id)
                        self.active_tracks[tentative[r].track_id] = tentative[r]
                    unmatched_dets = [j for j in unmatched_dets
                                      if j != c]

            to_remove = set()
            for j in unmatched_dets:
                det = detections[j]
                if det.get("confidence", 0.0) < 0.5:
                    w = det["bbox"][2] - det["bbox"][0]
                    h = det["bbox"][3] - det["bbox"][1]
                    if w * h < 6000:
                        to_remove.add(j)
                        continue
                for existing in self.active_tracks.values():
                    if compute_iou(det["bbox"], existing.bbox) > 0.15:
                        to_remove.add(j)
                        break
            unmatched_dets = [j for j in unmatched_dets if j not in to_remove]
            for j in unmatched_dets:
                self._create_track(detections[j])
            stale_ids = []
            for tid, track in list(self.active_tracks.items()):
                if track.time_since_update > self.config.max_time_lost:
                    stale_ids.append(tid)
                    self.lost_tracks[tid] = track
            for sid in stale_ids:
                del self.active_tracks[sid]

            self._cleanup_lost()
            self._merge_fragmented_tracks_logic()
            return list(self.active_tracks.values())

    def _create_track(self, detection: dict):
        track = Track(
            track_id=self._next_id,
            bbox=detection["bbox"],
            detection=detection,
            appearance_embedding=detection.get("embedding"),
            confidence=detection.get("confidence", 0.0),
        )
        self.active_tracks[self._next_id] = track
        self._next_id += 1

    def _cleanup_lost(self):
        stale = []
        for tid, track in self.lost_tracks.items():
            if track.time_since_update > self.config.track_buffer * 2:
                stale.append(tid)
        for sid in stale:
            ft = {
                "track_id": sid,
                "bbox": self.lost_tracks[sid].bbox.copy(),
                "embedding": self.lost_tracks[sid].get_appearance_embedding(),
            }
            self.fragmented_tracks.append(ft)
            del self.lost_tracks[sid]

    def _merge_fragmented_tracks_logic(self):
        if len(self.fragmented_tracks) > 50:
            self.fragmented_tracks = self.fragmented_tracks[-50:]

    def get_all_active_tracks(self) -> List[Track]:
        return list(self.active_tracks.values())

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        return self.active_tracks.get(track_id)

    def reset(self):
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.fragmented_tracks.clear()
