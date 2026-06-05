import numpy as np
import time
from collections import OrderedDict, defaultdict
from typing import Optional, List, Tuple
import logging
import pickle
import os

from sabnetra_ai.utils.geometry import l2_normalize, cosine_similarity
from sabnetra_ai.config import SabNetraConfig

logger = logging.getLogger(__name__)


class TrackMemory:
    def __init__(self, track_id: int, camera_id: str):
        self.track_id = track_id
        self.camera_id = camera_id
        self.face_embeddings: list = []
        self.body_embeddings: list = []
        self.clothing_descriptors: list = []
        self.gait_descriptors: list = []
        self.timestamps: list = []
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()
        self.best_face: Optional[np.ndarray] = None
        self.best_body: Optional[np.ndarray] = None
        self.best_clothing: Optional[np.ndarray] = None
        self.best_gait: Optional[np.ndarray] = None
        self.confidence: float = 0.0
        self.state: str = "GREEN"
        self.suspect_id: Optional[str] = None
        self.match_score: float = 0.0

    def update(self, features: dict, timestamp: float):
        self.last_seen = timestamp
        self.timestamps.append(timestamp)
        if features.get("face_embedding") is not None:
            self.face_embeddings.append(features["face_embedding"])
            self.best_face = self._select_best(self.face_embeddings)
        if features.get("body_embedding") is not None:
            self.body_embeddings.append(features["body_embedding"])
            self.best_body = self._select_best(self.body_embeddings)
        if features.get("clothing_descriptor") is not None:
            self.clothing_descriptors.append(features["clothing_descriptor"])
            self.best_clothing = self._select_best(self.clothing_descriptors)
        if features.get("gait_descriptor") is not None:
            self.gait_descriptors.append(features["gait_descriptor"])
            self.best_gait = self._select_best(self.gait_descriptors)

    def _select_best(self, embeddings: list) -> np.ndarray:
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return embeddings[0]
        stacked = np.stack(embeddings)
        mean_emb = np.mean(stacked, axis=0)
        mean_emb = l2_normalize(mean_emb)
        return mean_emb

    def get_consolidated(self) -> dict:
        return {
            "face": self.best_face,
            "body": self.best_body,
            "clothing": self.best_clothing,
            "gait": self.best_gait,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
            "state": self.state,
            "suspect_id": self.suspect_id,
            "match_score": self.match_score,
        }

    @property
    def age(self) -> float:
        return time.time() - self.first_seen

    def is_stale(self, timeout: float = 300.0) -> bool:
        return (time.time() - self.last_seen) > timeout


class EmbeddingMemorySystem:
    def __init__(self, config: Optional[SabNetraConfig] = None):
        self.config = config or SabNetraConfig()
        self._track_memories: OrderedDict[int, TrackMemory] = OrderedDict()
        self._camera_tracks: defaultdict = defaultdict(list)

    def get_or_create(self, track_id: int, camera_id: str) -> TrackMemory:
        if track_id not in self._track_memories:
            mem = TrackMemory(track_id, camera_id)
            self._track_memories[track_id] = mem
            self._camera_tracks[camera_id].append(track_id)
        return self._track_memories[track_id]

    def update(self, track_id: int, camera_id: str, features: dict,
               timestamp: float):
        mem = self.get_or_create(track_id, camera_id)
        mem.update(features, timestamp)

    def get(self, track_id: int) -> Optional[TrackMemory]:
        return self._track_memories.get(track_id)

    def get_consolidated(self, track_id: int) -> Optional[dict]:
        mem = self.get(track_id)
        if mem is None:
            return None
        return mem.get_consolidated()

    def get_all_active(self, max_age: float = 300.0) -> List[TrackMemory]:
        now = time.time()
        return [
            m for m in self._track_memories.values()
            if (now - m.last_seen) < max_age
        ]

    def cleanup_stale(self, timeout: float = 300.0):
        stale = [
            tid for tid, mem in self._track_memories.items()
            if mem.is_stale(timeout)
        ]
        for tid in stale:
            mem = self._track_memories.pop(tid, None)
            if mem:
                cam_tracks = self._camera_tracks.get(mem.camera_id, [])
                if tid in cam_tracks:
                    cam_tracks.remove(tid)

    def search_by_embedding(self, embedding: np.ndarray,
                            embedding_type: str = "body",
                            threshold: float = 0.5,
                            top_k: int = 10
                            ) -> List[Tuple[int, float, str]]:
        results = []
        for tid, mem in self._track_memories.items():
            target = None
            if embedding_type == "face":
                target = mem.best_face
            elif embedding_type == "body":
                target = mem.best_body
            elif embedding_type == "clothing":
                target = mem.best_clothing
            if target is None:
                continue
            sim = cosine_similarity(embedding, target)
            if sim >= threshold:
                results.append((tid, sim, mem.state))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path: str):
        data = {
            "track_memories": self._track_memories,
            "camera_tracks": dict(self._camera_tracks),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {len(self._track_memories)} track memories to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._track_memories = data.get("track_memories", OrderedDict())
        self._camera_tracks = defaultdict(
            list, data.get("camera_tracks", {}))
        logger.info(f"Loaded {len(self._track_memories)} track memories from {path}")

    @property
    def size(self) -> int:
        return len(self._track_memories)

    def stats(self) -> dict:
        states = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        for mem in self._track_memories.values():
            states[mem.state] = states.get(mem.state, 0) + 1
        return {
            "total_tracks": len(self._track_memories),
            "states": states,
            "cameras": len(self._camera_tracks),
        }
