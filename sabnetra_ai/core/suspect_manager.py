import logging
import time
from typing import Optional, List, Dict
from collections import defaultdict

from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile
from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem
from sabnetra_ai.utils.helpers import Timer

logger = logging.getLogger(__name__)


class GlobalSuspectManager:
    def __init__(self, matcher: MatchingEngine,
                 state_engine: StateClassificationEngine,
                 embedding_memory: EmbeddingMemorySystem):
        self.matcher = matcher
        self.state_engine = state_engine
        self.embedding_memory = embedding_memory
        self._global_track_to_suspect: Dict[int, str] = {}
        self._suspect_to_tracks: Dict[str, List[int]] = defaultdict(list)
        self._suspect_counter: int = 0
        self._global_suspects: Dict[str, SuspectProfile] = {}

    def process_detection(self, track_id: int, camera_id: str,
                           features: dict, timestamp: float
                           ) -> tuple:
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
            return state, suspect_id, score

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
        return self._global_track_to_suspect.get(track_id)

    def get_tracks_for_suspect(self, suspect_id: str) -> List[int]:
        return self._suspect_to_tracks.get(suspect_id, [])

    def get_suspect_profile(self, suspect_id: str) -> Optional[SuspectProfile]:
        return self._global_suspects.get(suspect_id) or \
               self.matcher.get_suspect(suspect_id)

    def enroll_suspect(self, profile: SuspectProfile):
        self.matcher.add_suspect(profile)
        self._global_suspects[profile.suspect_id] = profile
        logger.info(f"Enrolled suspect: {profile.suspect_id}")

    def reset_track(self, track_id: int):
        self.state_engine.reset_track(track_id)
        self._global_track_to_suspect.pop(track_id, None)

    def get_all_active_suspects(self) -> List[dict]:
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

    def stats(self) -> dict:
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
        }
