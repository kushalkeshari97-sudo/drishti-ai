import logging
import time
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np

from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile
from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem
from sabnetra_ai.utils.helpers import Timer
from sabnetra_ai.utils.geometry import cosine_similarity

logger = logging.getLogger(__name__)

RED_REID_THRESHOLD = 0.55
RED_REID_MEMORY_TTL = 600.0


class GlobalSuspectManager:
    """Manages global suspect enrollment, track-to-suspect mapping, and state transitions."""

    def __init__(self, matcher: MatchingEngine,
                 state_engine: StateClassificationEngine,
                 embedding_memory: EmbeddingMemorySystem):
        """Initialize manager with core subsystems."""
        self.matcher = matcher
        self.state_engine = state_engine
        self.embedding_memory = embedding_memory
        self._global_track_to_suspect: Dict[int, str] = {}
        self._suspect_to_tracks: Dict[str, List[int]] = defaultdict(list)
        self._suspect_counter: int = 0
        self._global_suspects: Dict[str, SuspectProfile] = {}
        self._red_suspect_memory: Dict[str, dict] = {}

    def process_detection(self, track_id: int, camera_id: str,
                           features: dict, timestamp: float
                           ) -> tuple:
        """Process a detection: update memory, match, classify, and return state.
        Args:
            track_id: Track identifier.
            camera_id: Camera identifier.
            features: Dictionary of extracted embeddings.
            timestamp: Frame timestamp.
        Returns:
            Tuple of (state, suspect_id, score).
        """
        with Timer("suspect_manager"):
            track_mem = self.embedding_memory.get_or_create(track_id, camera_id)
            track_mem.update(features, timestamp)
            match_result = self.matcher.match(
                features, camera_id=camera_id, track_id=track_id)
            state, suspect_id, score = self.state_engine.classify(
                match_result, track_id, camera_id, track_mem.state)
            track_mem.state = state
            track_mem.match_score = score
            if state == IdentityState.RED:
                if suspect_id and suspect_id != "UNKNOWN":
                    track_mem.suspect_id = suspect_id
                    self._global_track_to_suspect[track_id] = suspect_id
                    if track_id not in self._suspect_to_tracks[suspect_id]:
                        self._suspect_to_tracks[suspect_id].append(track_id)
                elif suspect_id == "UNKNOWN" and track_mem.suspect_id is None:
                    new_id = self._register_new_suspect(features)
                    track_mem.suspect_id = new_id
                    match_result.suspect_id = new_id
                    self._global_track_to_suspect[track_id] = new_id
                    self._suspect_to_tracks[new_id].append(track_id)
                self._store_red_suspect(suspect_id, features, track_id)
            elif state in (IdentityState.GREEN, IdentityState.YELLOW):
                red_reid = self._check_red_reid(features, track_id)
                if red_reid:
                    state, suspect_id, score = red_reid
                    track_mem.state = state
                    track_mem.suspect_id = suspect_id
                    track_mem.match_score = score
                    self._global_track_to_suspect[track_id] = suspect_id
                    if track_id not in self._suspect_to_tracks[suspect_id]:
                        self._suspect_to_tracks[suspect_id].append(track_id)
                    self._store_red_suspect(suspect_id, features, track_id)
            return state, suspect_id, score

    def _store_red_suspect(self, suspect_id: str, features: dict, track_id: int):
        now = time.time()
        entry = self._red_suspect_memory.get(suspect_id)
        if entry is None:
            self._red_suspect_memory[suspect_id] = {
                "best_features": {k: v.copy() for k, v in features.items() if v is not None},
                "last_seen": now,
                "track_ids": {track_id},
            }
        else:
            entry["last_seen"] = now
            entry["track_ids"].add(track_id)
            for k, v in features.items():
                if v is not None:
                    entry["best_features"][k] = v.copy()

    def _check_red_reid(self, features: dict, new_track_id: int) -> Optional[tuple]:
        now = time.time()
        self._red_suspect_memory = {
            sid: e for sid, e in self._red_suspect_memory.items()
            if now - e["last_seen"] < RED_REID_MEMORY_TTL
        }
        best_match = None
        best_score = RED_REID_THRESHOLD
        for suspect_id, entry in self._red_suspect_memory.items():
            if new_track_id in entry["track_ids"]:
                continue
            total_sim = 0.0
            total_weight = 0.0
            for modality, emb in entry["best_features"].items():
                query_emb = features.get(modality)
                if query_emb is None:
                    continue
                sim = cosine_similarity(query_emb, emb)
                w = {"face_embedding": 0.5, "body_embedding": 0.3, "clothing_descriptor": 0.2}.get(modality, 0.2)
                total_sim += sim * w
                total_weight += w
            if total_weight == 0:
                continue
            avg_sim = total_sim / total_weight
            if avg_sim > best_score:
                best_score = avg_sim
                best_match = suspect_id
        if best_match:
            logger.info(f"RED re-ID: new track {new_track_id} matches "
                        f"previous RED suspect {best_match} (score={best_score:.3f})")
            return IdentityState.RED, best_match, best_score
        return None

    def _register_new_suspect(self, features: dict) -> str:
        self._suspect_counter += 1
        suspect_id = f"S{self._suspect_counter:04d}"
        profile = SuspectProfile(
            suspect_id=suspect_id,
            face_emb=features.get("face_embedding"),
            body_emb=features.get("body_embedding"),
            clothing_emb=features.get("clothing_descriptor"),
            gait_emb=features.get("gait_descriptor"),
            metadata={"auto_enrolled": True, "timestamp": time.time()},
        )
        self.matcher.add_suspect(profile)
        self._global_suspects[suspect_id] = profile
        logger.info(f"Auto-registered new suspect: {suspect_id}")
        return suspect_id

    def get_suspect_id_for_track(self, track_id: int) -> Optional[str]:
        """Return the suspect ID mapped to a given track, if any."""
        return self._global_track_to_suspect.get(track_id)

    def get_tracks_for_suspect(self, suspect_id: str) -> List[int]:
        """Return all track IDs associated with a suspect."""
        return self._suspect_to_tracks.get(suspect_id, [])

    def get_suspect_profile(self, suspect_id: str) -> Optional[SuspectProfile]:
        """Retrieve a suspect profile by ID from global or matcher storage."""
        return self._global_suspects.get(suspect_id) or \
               self.matcher.get_suspect(suspect_id)

    def enroll_suspect(self, profile: SuspectProfile):
        """Enroll a new suspect profile into the system."""
        self.matcher.add_suspect(profile)
        self._global_suspects[profile.suspect_id] = profile
        logger.info(f"Enrolled suspect: {profile.suspect_id}")

    def reset_track(self, track_id: int):
        """Reset state engine and remove track-to-suspect mapping."""
        self.state_engine.reset_track(track_id)
        self._global_track_to_suspect.pop(track_id, None)

    def get_all_active_suspects(self) -> List[dict]:
        """Return all tracks in YELLOW or RED state as a list of dicts."""
        suspects = []
        for tid, mem in self.embedding_memory._track_memories.items():
            if mem.state in (IdentityState.YELLOW, IdentityState.RED):
                suspects.append({
                    "track_id": tid,
                    "suspect_id": mem.suspect_id,
                    "state": mem.state,
                    "score": mem.match_score,
                    "camera_id": mem.camera_id,
                })
        return suspects

    def _cleanup_red_memory(self):
        now = time.time()
        stale = [sid for sid, e in self._red_suspect_memory.items()
                 if now - e["last_seen"] >= RED_REID_MEMORY_TTL]
        for sid in stale:
            del self._red_suspect_memory[sid]
        if stale:
            logger.debug(f"Cleaned {len(stale)} stale RED suspect memories")

    def stats(self) -> dict:
        """Return summary statistics about the suspect manager."""
        return {
            "enrolled_suspects": len(self._global_suspects),
            "auto_enrolled": self._suspect_counter,
            "active_red_tracks": sum(
                1 for m in self.embedding_memory._track_memories.values()
                if m.state == IdentityState.RED),
            "active_yellow_tracks": sum(
                1 for m in self.embedding_memory._track_memories.values()
                if m.state == IdentityState.YELLOW),
            "global_track_to_suspect_mappings": len(self._global_track_to_suspect),
            "red_reid_memory_size": len(self._red_suspect_memory),
        }
