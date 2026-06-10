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


def test_incremental_add_multiple_suspects():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    rng = np.random.RandomState(0)
    for i in range(5):
        emb = rng.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        engine.add_suspect(SuspectProfile(f"S{i}", "C", face_emb=emb))
    assert engine.suspect_db.size == 5
    assert engine.suspect_db._face_index is not None
    assert engine.suspect_db._face_index.ntotal == 5


def test_incremental_add_preserves_existing_matches():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    engine.add_suspect(SuspectProfile("FIRST", "C", face_emb=emb1.copy()))
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0
    engine.add_suspect(SuspectProfile("SECOND", "C", face_emb=emb2.copy()))
    result1 = engine.match({"face_embedding": emb1})
    assert result1.suspect_id == "FIRST"
    result2 = engine.match({"face_embedding": emb2})
    assert result2.suspect_id == "SECOND"


def test_remove_suspect():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    engine.add_suspect(SuspectProfile("RM", "C", face_emb=emb.copy()))
    assert engine.suspect_db.size == 1
    engine.suspect_db.remove_suspect("RM")
    assert engine.suspect_db.size == 0


def test_match_zero_norm_query():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    engine.add_suspect(SuspectProfile("S1", "C", face_emb=emb))
    result = engine.match({"face_embedding": np.zeros(512, dtype=np.float32)})
    assert result.suspect_id == "UNKNOWN"
    assert result.score == 0.0


def test_match_with_body_only():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    engine.add_suspect(SuspectProfile("BODY", "C", body_emb=emb.copy()))
    result = engine.match({"body_embedding": emb.copy()})
    assert result.suspect_id == "BODY"
    assert result.score > 0.5


def test_match_no_features_returns_unknown():
    config = MatcherConfig()
    engine = MatchingEngine(config)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    engine.add_suspect(SuspectProfile("S1", "C", face_emb=emb))
    result = engine.match({})
    assert result.suspect_id == "UNKNOWN"
    assert result.score == 0.0
