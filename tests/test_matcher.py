import numpy as np
from sabnetra_ai.core.matcher import MatchingEngine, SuspectProfile, MatchResult
from sabnetra_ai.config import MatcherConfig


def test_suspect_enroll_and_match():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    profile = SuspectProfile("TEST", "CASE", face_emb=emb)
    engine.add_suspect(profile)
    result = engine.match({"face_embedding": emb})
    assert result.suspect_id == "TEST"
    assert result.score > 0.5


def test_no_match_for_unknown():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    profile = SuspectProfile("KNOWN", "CASE", face_emb=emb1)
    engine.add_suspect(profile)
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0
    result = engine.match({"face_embedding": emb2})
    assert result.suspect_id == "KNOWN"
    assert result.score < 0.1
