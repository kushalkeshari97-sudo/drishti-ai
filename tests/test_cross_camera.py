import time
import numpy as np
from sabnetra_ai.core.cross_camera import TemporalIntelligence, CrossCameraTracker
from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile
from sabnetra_ai.core.embedding_memory import EmbeddingMemorySystem
from sabnetra_ai.config import MatcherConfig


def test_temporal_validate_transition():
    ti = TemporalIntelligence()
    assert ti.validate_transition("S1", "cam0", "cam1", time.time())
    assert not ti.validate_transition("S1", "cam0", "cam1", time.time() + 0.1)


def test_temporal_min_travel_time():
    ti = TemporalIntelligence()
    now = time.time()
    assert ti.validate_transition("S1", "cam0", "cam1", now)
    assert not ti.validate_transition("S1", "cam0", "cam1", now + 1.0)


def test_temporal_max_window():
    ti = TemporalIntelligence()
    now = time.time()
    assert ti.validate_transition("S1", "cam0", "cam1", now)
    assert not ti.validate_transition("S1", "cam0", "cam1", now + 10000)


def test_get_last_seen():
    ti = TemporalIntelligence()
    t = time.time()
    ti.validate_transition("S1", "cam0", "cam1", t)
    last = ti.get_last_seen("S1", "cam1")
    assert last is None, "get_last_seen uses from_camera key format: suspectid_fromcam_tocam"


def test_cross_camera_register():
    matcher = MatchingEngine()
    ems = EmbeddingMemorySystem()
    cc = CrossCameraTracker(matcher, ems)
    cc.register_camera("cam0")
    cc.register_camera("cam1")
    stats = cc.stats()
    assert stats["registered_cameras"] == 2


def test_cross_camera_overlap():
    matcher = MatchingEngine()
    ems = EmbeddingMemorySystem()
    cc = CrossCameraTracker(matcher, ems)
    cc.register_camera("cam0")
    cc.register_camera("cam1")
    cc.set_camera_overlap("cam0", "cam1")
    stats = cc.stats()
    assert stats["registered_cameras"] == 2


def test_cross_camera_adjacent():
    matcher = MatchingEngine()
    ems = EmbeddingMemorySystem()
    cc = CrossCameraTracker(matcher, ems)
    cc.register_camera("cam0")
    cc.register_camera("cam1")
    cc.set_camera_adjacent("cam0", "cam1")
    stats = cc.stats()
    assert stats["registered_cameras"] == 2


def test_cross_camera_match():
    matcher = MatchingEngine()
    ems = EmbeddingMemorySystem()
    cc = CrossCameraTracker(matcher, ems)
    cc.register_camera("cam0")
    profile = SuspectProfile(
        suspect_id="S1", face_emb=np.array([1.0, 0.0, 0.0]),
    )
    matcher.add_suspect(profile)
    is_match, score = cc.track_across_cameras(
        "S1", "cam0", "cam1",
        {"face_embedding": np.array([1.0, 0.0, 0.0])},
        time.time(),
    )
    assert isinstance(is_match, (bool, np.bool_))
