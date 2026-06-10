from sabnetra_ai.core.state_engine import StateClassificationEngine, IdentityState
from sabnetra_ai.core.matcher import MatchResult


def test_green_below_threshold():
    engine = StateClassificationEngine()
    result = MatchResult("UNKNOWN", 0.1, 0.5)
    state, sid, score = engine.classify(result, 1, "cam0")
    assert state == IdentityState.GREEN


def test_yellow_above_yellow_threshold():
    engine = StateClassificationEngine()
    thresh = engine.config.yellow_threshold
    result = MatchResult("SUSPECT", thresh + 0.05, 0.5)
    state, sid, score = engine.classify(result, 1, "cam0")
    assert state == IdentityState.YELLOW


def test_red_above_red_threshold():
    engine = StateClassificationEngine()
    thresh = engine.config.red_threshold
    result = MatchResult("SUSPECT", thresh + 0.05, 0.5)
    state, sid, score = engine.classify(result, 1, "cam0")
    assert state == IdentityState.RED


def test_red_lock_keeps_red_on_subsequent_calls():
    engine = StateClassificationEngine()
    thresh = engine.config.red_threshold
    result1 = MatchResult("S1", thresh + 0.1, 0.5)
    state1, _, _ = engine.classify(result1, 99, "cam0")
    assert state1 == IdentityState.RED
    result2 = MatchResult("S1", 0.0, 0.0)
    state2, _, _ = engine.classify(result2, 99, "cam0")
    assert state2 == IdentityState.RED


def test_red_lock_expires_after_ttl():
    import time
    engine = StateClassificationEngine()
    engine._red_locked[99] = time.time() - 400.0
    result = MatchResult("S1", 0.0, 0.0)
    state, _, _ = engine.classify(result, 99, "cam0")
    assert state != IdentityState.RED


def test_cleanup_stale_removes_expired_locks():
    import time
    engine = StateClassificationEngine()
    engine._red_locked[1] = time.time() - 400.0
    engine._red_locked[2] = time.time() - 10.0
    engine.cleanup_stale(timeout=300.0)
    assert 1 not in engine._red_locked
    assert 2 in engine._red_locked


def test_reset_track_removes_red_lock():
    engine = StateClassificationEngine()
    thresh = engine.config.red_threshold
    result = MatchResult("S1", thresh + 0.1, 0.5)
    engine.classify(result, 42, "cam0")
    assert 42 in engine._red_locked
    engine.reset_track(42)
    assert 42 not in engine._red_locked


def test_reset_all_clears_all_locks():
    engine = StateClassificationEngine()
    engine._red_locked[1] = 100.0
    engine._red_locked[2] = 200.0
    engine.reset_all()
    assert len(engine._red_locked) == 0
    assert engine.red_count == 0
