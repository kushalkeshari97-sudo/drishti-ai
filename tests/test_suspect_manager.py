import numpy as np
from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile
from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem
from sabnetra_ai.core.suspect_manager import GlobalSuspectManager


def test_suspect_manager_enroll():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    profile = SuspectProfile("S1", face_emb=np.array([1.0, 0.0, 0.0]))
    sm.enroll_suspect(profile)
    prof = sm.get_suspect_profile("S1")
    assert prof is not None
    assert prof.suspect_id == "S1"


def test_suspect_manager_process_green():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    emb = np.array([1.0, 0.0, 0.0])
    state, sid, score = sm.process_detection(
        1, "cam0", {"face_embedding": emb}, 100.0)
    assert state == IdentityState.GREEN


def test_suspect_manager_process_red():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    emb = np.array([1.0, 0.0, 0.0])
    profile = SuspectProfile("S1", face_emb=emb)
    sm.enroll_suspect(profile)
    state, sid, score = sm.process_detection(
        1, "cam0", {"face_embedding": emb}, 100.0)
    assert state == IdentityState.RED
    assert sid == "S1"


def test_suspect_manager_track_mapping():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    emb = np.array([1.0, 0.0, 0.0])
    profile = SuspectProfile("S1", face_emb=emb)
    sm.enroll_suspect(profile)
    sm.process_detection(1, "cam0", {"face_embedding": emb}, 100.0)
    assert sm.get_suspect_id_for_track(1) == "S1"
    assert sm.get_tracks_for_suspect("S1") == [1]


def test_suspect_manager_reset_track():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    emb = np.array([1.0, 0.0, 0.0])
    profile = SuspectProfile("S1", face_emb=emb)
    sm.enroll_suspect(profile)
    sm.process_detection(1, "cam0", {"face_embedding": emb}, 100.0)
    sm.reset_track(1)
    assert sm.get_suspect_id_for_track(1) is None


def test_suspect_manager_stats():
    matcher = MatchingEngine()
    se = StateClassificationEngine()
    ems = EmbeddingMemorySystem()
    sm = GlobalSuspectManager(matcher, se, ems)
    stats = sm.stats()
    assert stats["enrolled_suspects"] == 0
    assert stats["active_red_tracks"] == 0
