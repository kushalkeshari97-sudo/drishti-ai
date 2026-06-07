import numpy as np
import time
from sabnetra_ai.core.embedding_memory import TrackMemory, EmbeddingMemorySystem
from sabnetra_ai.config import SabNetraConfig


def test_track_memory_creation():
    mem = TrackMemory(1, "cam0")
    assert mem.track_id == 1
    assert mem.camera_id == "cam0"
    assert mem.first_seen > 0


def test_track_memory_update():
    mem = TrackMemory(1, "cam0")
    emb = np.array([1.0, 0.0, 0.0])
    mem.update({"face_embedding": emb}, time.time() + 1.0)
    assert mem.best_face is not None
    assert len(mem.face_embeddings) == 1


def test_track_memory_consolidated():
    mem = TrackMemory(1, "cam0")
    emb = np.array([1.0, 0.0, 0.0])
    mem.update({"body_embedding": emb}, time.time())
    consolidated = mem.get_consolidated()
    assert consolidated["track_id"] == 1
    assert consolidated["body"] is not None
    assert consolidated["state"] == "GREEN"


def test_track_memory_age():
    mem = TrackMemory(1, "cam0")
    assert mem.age >= 0


def test_track_memory_is_stale():
    mem = TrackMemory(1, "cam0")
    assert not mem.is_stale(timeout=300.0)
    mem.last_seen = 0
    assert mem.is_stale(timeout=1.0)


def test_embedding_memory_get_or_create():
    ems = EmbeddingMemorySystem()
    mem = ems.get_or_create(1, "cam0")
    assert mem.track_id == 1
    same_mem = ems.get_or_create(1, "cam0")
    assert same_mem is mem


def test_embedding_memory_update():
    ems = EmbeddingMemorySystem()
    emb = np.array([1.0, 0.0, 0.0])
    ems.update(1, "cam0", {"face_embedding": emb}, time.time())
    mem = ems.get(1)
    assert mem is not None
    assert mem.best_face is not None


def test_embedding_memory_get_all_active():
    ems = EmbeddingMemorySystem()
    ems.get_or_create(1, "cam0")
    ems.get_or_create(2, "cam1")
    active = ems.get_all_active(max_age=300.0)
    assert len(active) == 2


def test_embedding_memory_cleanup_stale():
    ems = EmbeddingMemorySystem()
    mem = ems.get_or_create(1, "cam0")
    mem.last_seen = 0
    ems.cleanup_stale(timeout=0.0)
    assert ems.get(1) is None


def test_embedding_memory_search():
    ems = EmbeddingMemorySystem()
    emb = np.array([1.0, 0.0, 0.0])
    mem = ems.get_or_create(1, "cam0")
    mem.update({"body_embedding": emb}, time.time())
    results = ems.search_by_embedding(emb, embedding_type="body", threshold=0.0)
    assert len(results) == 1
    assert results[0][0] == 1


def test_embedding_memory_save_load(tmp_path):
    ems = EmbeddingMemorySystem()
    emb = np.array([1.0, 0.0, 0.0])
    ems.get_or_create(1, "cam0").update({"face_embedding": emb}, time.time())
    path = str(tmp_path / "test_memory.pkl")
    ems.save(path)
    ems2 = EmbeddingMemorySystem()
    ems2.load(path)
    assert ems2.get(1) is not None
