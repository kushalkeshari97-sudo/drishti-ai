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
