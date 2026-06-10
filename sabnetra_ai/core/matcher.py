import numpy as np
import faiss
import time
import logging
from typing import List, Optional, Tuple, Dict
from collections import deque

from sabnetra_ai.config import MatcherConfig
from sabnetra_ai.utils.geometry import cosine_similarity, l2_normalize
from sabnetra_ai.utils.helpers import Timer

logger = logging.getLogger(__name__)


class MatchResult:
    """Result of a matching operation against the suspect database."""

    __slots__ = ("suspect_id", "score", "confidence", "similarities",
                 "is_match")

    def __init__(self, suspect_id: str, score: float, confidence: float,
                 similarities: Optional[Dict[str, float]] = None):
        self.suspect_id = suspect_id
        self.score = score
        self.confidence = confidence
        self.similarities = similarities or {}
        self.is_match = score > 0.5


class SuspectProfile:
    """Profile for a known suspect containing multi-modal embeddings."""

    def __init__(self, suspect_id: str, case_id: str = "",
                 face_emb: Optional[np.ndarray] = None,
                 body_emb: Optional[np.ndarray] = None,
                 clothing_emb: Optional[np.ndarray] = None,
                 gait_emb: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None):
        self.suspect_id = suspect_id
        self.case_id = case_id
        self.face_embedding = l2_normalize(face_emb) if face_emb is not None else None
        self.body_embedding = l2_normalize(body_emb) if body_emb is not None else None
        self.clothing_descriptor = l2_normalize(clothing_emb) if clothing_emb is not None else None
        self.gait_descriptor = l2_normalize(gait_emb) if gait_emb is not None else None
        self.metadata = metadata or {}
        self.enrollment_time = time.time()

    def has_modality(self, modality: str) -> bool:
        """Check if this profile has a given modality embedding."""
        return getattr(self, f"{modality}_embedding" if modality != "clothing"
                       else "clothing_descriptor", None) is not None


class SuspectDatabase:
    """FAISS-backed database of suspect profiles for fast similarity search."""

    def __init__(self):
        self.suspects: Dict[str, SuspectProfile] = {}
        self._face_index: Optional[faiss.Index] = None
        self._body_index: Optional[faiss.Index] = None
        self._face_id_map: Dict[int, str] = {}
        self._body_id_map: Dict[int, str] = {}
        self._next_face_id: int = 0
        self._next_body_id: int = 0

    def add_suspect(self, profile: SuspectProfile):
        """Add a suspect profile to the database and indices."""
        self.suspects[profile.suspect_id] = profile
        self._add_to_indices(profile)

    def _add_to_indices(self, profile: SuspectProfile):
        if profile.face_embedding is not None:
            dim = profile.face_embedding.shape[0]
            if self._face_index is None:
                self._face_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            emb = l2_normalize(profile.face_embedding).reshape(1, -1).astype(np.float32)
            fid = self._next_face_id
            self._face_index.add_with_ids(emb, np.array([fid]))
            self._face_id_map[fid] = profile.suspect_id
            self._next_face_id += 1
        if profile.body_embedding is not None:
            dim = profile.body_embedding.shape[0]
            if self._body_index is None:
                self._body_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            emb = l2_normalize(profile.body_embedding).reshape(1, -1).astype(np.float32)
            bid = self._next_body_id
            self._body_index.add_with_ids(emb, np.array([bid]))
            self._body_id_map[bid] = profile.suspect_id
            self._next_body_id += 1

    def remove_suspect(self, suspect_id: str):
        """Remove a suspect profile by ID and rebuild indices."""
        if suspect_id in self.suspects:
            del self.suspects[suspect_id]
            self.rebuild_indices()

    def get(self, suspect_id: str) -> Optional[SuspectProfile]:
        """Retrieve a suspect profile by ID."""
        return self.suspects.get(suspect_id)

    def rebuild_indices(self):
        """Rebuild FAISS indices from all stored profiles."""
        self._face_index = None
        self._body_index = None
        self._face_id_map.clear()
        self._body_id_map.clear()
        self._next_face_id = 0
        self._next_body_id = 0
        for prof in self.suspects.values():
            self._add_to_indices(prof)

    def search_face(self, query: np.ndarray, k: int = 5
                    ) -> List[Tuple[str, float]]:
        """Search face index for top-k matches. Returns list of (suspect_id, score)."""
        if self._face_index is None or self._face_index.ntotal == 0:
            return []
        if np.linalg.norm(query) == 0:
            return []
        query = l2_normalize(query).reshape(1, -1).astype(np.float32)
        scores, indices = self._face_index.search(query, min(k, self._face_index.ntotal))
        return [(self._face_id_map[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0]) if idx in self._face_id_map]

    def search_body(self, query: np.ndarray, k: int = 5
                    ) -> List[Tuple[str, float]]:
        """Search body index for top-k matches. Returns list of (suspect_id, score)."""
        if self._body_index is None or self._body_index.ntotal == 0:
            return []
        if np.linalg.norm(query) == 0:
            return []
        query = l2_normalize(query).reshape(1, -1).astype(np.float32)
        scores, indices = self._body_index.search(query, min(k, self._body_index.ntotal))
        return [(self._body_id_map[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0]) if idx in self._body_id_map]

    @property
    def size(self) -> int:
        """Return the number of suspects in the database."""
        return len(self.suspects)


class MatchingEngine:
    """Orchestrates multi-modal matching with temporal consistency checks."""

    def __init__(self, config: Optional[MatcherConfig] = None):
        """Initialize engine with matcher config and suspect database."""
        self.config = config or MatcherConfig()
        self.suspect_db = SuspectDatabase()
        self._temporal_buffer: Dict[str, deque] = {}

    def match(self, features: dict, camera_id: str = "",
              track_id: int = -1) -> MatchResult:
        with Timer("matching_engine"):
            candidates = []
            weights = []
            if features.get("face_embedding") is not None:
                face_results = self.suspect_db.search_face(
                    features["face_embedding"], self.config.top_k)
                for sid, score in face_results:
                    candidates.append((sid, score, "face", self.config.face_weight))
            if features.get("body_embedding") is not None:
                body_results = self.suspect_db.search_body(
                    features["body_embedding"], self.config.top_k)
                for sid, score in body_results:
                    candidates.append((sid, score, "body", self.config.reid_weight))
            if not candidates:
                return MatchResult("UNKNOWN", 0.0, 0.0)
            scores = {}
            for sid, score, modality, weight in candidates:
                if sid not in scores:
                    scores[sid] = {"total": 0.0, "weight_sum": 0.0,
                                   "modalities": {}}
                scores[sid]["total"] += score * weight
                scores[sid]["weight_sum"] += weight
                scores[sid]["modalities"][modality] = score
            best_sid = max(scores, key=lambda s: scores[s]["total"])
            best = scores[best_sid]
            weighted_score = best["total"] / best["weight_sum"] if best["weight_sum"] > 0 else 0.0
            weighted_score = np.clip(weighted_score, 0.0, 1.0)
            temp_consistent = self._check_temporal_consistency(
                best_sid, weighted_score, track_id, camera_id)
            if not temp_consistent:
                weighted_score *= 0.7
            result = MatchResult(
                suspect_id=best_sid,
                score=weighted_score,
                confidence=min(1.0, best["weight_sum"]),
                similarities=best["modalities"],
            )
            return result

    def match_with_profile(self, features: dict,
                           profile: SuspectProfile) -> float:
        """Compute weighted similarity score against a single suspect profile.
        Args:
            features: Dictionary of query embeddings.
            profile: The suspect profile to compare against.
        Returns:
            Weighted similarity score between 0 and 1.
        """
        scores = []
        weights = []
        if features.get("face_embedding") is not None and \
           profile.face_embedding is not None:
            sim = cosine_similarity(features["face_embedding"],
                                    profile.face_embedding)
            scores.append(sim)
            weights.append(self.config.face_weight)
        if features.get("body_embedding") is not None and \
           profile.body_embedding is not None:
            sim = cosine_similarity(features["body_embedding"],
                                    profile.body_embedding)
            scores.append(sim)
            weights.append(self.config.reid_weight)
        if features.get("clothing_descriptor") is not None and \
           profile.clothing_descriptor is not None:
            sim = cosine_similarity(features["clothing_descriptor"],
                                    profile.clothing_descriptor)
            scores.append(sim)
            weights.append(self.config.clothing_weight)
        if features.get("gait_descriptor") is not None and \
           profile.gait_descriptor is not None:
            sim = cosine_similarity(features["gait_descriptor"],
                                    profile.gait_descriptor)
            scores.append(sim)
            weights.append(self.config.gait_weight)
        if not scores:
            return 0.0
        return np.average(scores, weights=weights)

    def _check_temporal_consistency(self, suspect_id: str, score: float,
                                      track_id: int, camera_id: str) -> bool:
        key = f"{suspect_id}_{track_id}"
        if key not in self._temporal_buffer:
            self._temporal_buffer[key] = deque(maxlen=self.config.temporal_consistency_frames)
        self._temporal_buffer[key].append(score)
        if len(self._temporal_buffer[key]) < self.config.temporal_consistency_frames:
            return True
        scores_arr = np.array(self._temporal_buffer[key])
        above_thresh = np.sum(scores_arr > self.config.yellow_threshold)
        ratio = above_thresh / len(scores_arr)
        return ratio >= self.config.temporal_consistency_ratio

    def cross_camera_match(self, source_features: dict,
                            target_features: dict) -> float:
        """Compute cross-camera similarity between two feature sets.
        Args:
            source_features: Features from source camera.
            target_features: Features from target camera.
        Returns:
            Weighted similarity score.
        """
        sims = []
        if source_features.get("face_embedding") is not None and \
           target_features.get("face_embedding") is not None:
            sim = cosine_similarity(source_features["face_embedding"],
                                    target_features["face_embedding"])
            sims.append((sim, self.config.cross_camera_face_weight))
        if source_features.get("body_embedding") is not None and \
           target_features.get("body_embedding") is not None:
            sim = cosine_similarity(source_features["body_embedding"],
                                    target_features["body_embedding"])
            sims.append((sim, self.config.cross_camera_reid_weight))
        if source_features.get("clothing_descriptor") is not None and \
           target_features.get("clothing_descriptor") is not None:
            sim = cosine_similarity(source_features["clothing_descriptor"],
                                    target_features["clothing_descriptor"])
            sims.append((sim, self.config.cross_camera_clothing_weight))
        if not sims:
            return 0.0
        total_weight = sum(w for _, w in sims)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in sims) / total_weight

    def add_suspect(self, profile: SuspectProfile):
        """Add a suspect profile to the database."""
        self.suspect_db.add_suspect(profile)

    def get_suspect(self, suspect_id: str) -> Optional[SuspectProfile]:
        """Retrieve a suspect profile by ID."""
        return self.suspect_db.get(suspect_id)
