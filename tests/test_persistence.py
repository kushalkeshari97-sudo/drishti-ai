import os
import tempfile
import numpy as np
from sabnetra_ai.utils.persistence import save_suspects, load_suspects
from sabnetra_ai.core.matcher import SuspectProfile


def test_save_load_suspects():
    profiles = [
        SuspectProfile(
            suspect_id="S1", case_id="CASE001",
            face_emb=np.array([1.0, 0.0, 0.0]),
            body_emb=np.array([0.0, 1.0, 0.0]),
            metadata={"source": "test"},
        ),
        SuspectProfile(
            suspect_id="S2", case_id="CASE002",
            body_emb=np.array([0.0, 0.0, 1.0]),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_suspects(profiles, tmpdir)
        loaded = load_suspects(tmpdir)
        assert len(loaded) == 2
        ids = {p.suspect_id for p in loaded}
        assert "S1" in ids
        assert "S2" in ids


def test_load_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = load_suspects(tmpdir)
        assert loaded == []


def test_load_nonexistent():
    loaded = load_suspects("nonexistent_dir_12345")
    assert loaded == []
