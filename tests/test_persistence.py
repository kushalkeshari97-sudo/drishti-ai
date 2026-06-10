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


def test_save_load_encrypted():
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        import pytest
        pytest.skip("cryptography not installed")
    profiles = [
        SuspectProfile(
            suspect_id="E1", case_id="ENC",
            face_emb=np.array([0.5, 0.5, 0.0]),
            metadata={"encrypted": True},
        ),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_suspects(profiles, tmpdir)
        loaded = load_suspects(tmpdir)
        assert len(loaded) == 1
        assert loaded[0].suspect_id == "E1"


def test_save_load_roundtrip_with_all_modalities():
    rng = np.random.RandomState(42)
    emb = rng.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    profile = SuspectProfile(
        suspect_id="FULL", case_id="C",
        face_emb=emb.copy(),
        body_emb=emb.copy(),
        clothing_emb=emb.copy(),
        gait_emb=emb.copy(),
        metadata={"all": True},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_suspects([profile], tmpdir)
        loaded = load_suspects(tmpdir)
        assert len(loaded) == 1
        p = loaded[0]
        assert p.suspect_id == "FULL"
        assert p.face_embedding is not None
        assert p.body_embedding is not None
        assert p.clothing_descriptor is not None


def test_save_load_corrupted_file_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        bad_path = os.path.join(tmpdir, "corrupt.pkl")
        with open(bad_path, "wb") as f:
            f.write(b"not a pickle file")
        loaded = load_suspects(tmpdir)
        assert loaded == []
