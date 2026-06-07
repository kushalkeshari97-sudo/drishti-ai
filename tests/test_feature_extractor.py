import numpy as np
from sabnetra_ai.core.feature_extractor import IdentityFeatures


def test_identity_features_defaults():
    f = IdentityFeatures()
    assert f.face_embedding is None
    assert f.body_embedding is None
    assert f.clothing_descriptor is None
    assert f.gait_descriptor is None
    assert f.track_id == -1


def test_identity_features_with_params():
    f = IdentityFeatures(track_id=42, camera_id="cam1", timestamp=100.0)
    assert f.track_id == 42
    assert f.camera_id == "cam1"
    assert f.timestamp == 100.0


def test_has_any_embedding_none():
    f = IdentityFeatures()
    assert not f.has_any_embedding()


def test_has_any_embedding_face():
    f = IdentityFeatures()
    f.face_embedding = np.array([1.0, 0.0])
    assert f.has_any_embedding()


def test_has_any_embedding_body():
    f = IdentityFeatures()
    f.body_embedding = np.array([0.0, 1.0])
    assert f.has_any_embedding()


def test_to_dict():
    f = IdentityFeatures(track_id=1, camera_id="cam0", timestamp=50.0)
    f.face_embedding = np.array([1.0, 2.0])
    d = f.to_dict()
    assert d["has_face"]
    assert d["face_dim"] == 2
    assert not d["has_body"]
    assert d["track_id"] == 1
    assert d["camera_id"] == "cam0"


def test_normalize_all():
    f = IdentityFeatures()
    f.face_embedding = np.array([3.0, 4.0])
    f.body_embedding = np.array([0.0, 5.0])
    f.normalize_all()
    assert abs(np.linalg.norm(f.face_embedding) - 1.0) < 1e-6
    assert abs(np.linalg.norm(f.body_embedding) - 1.0) < 1e-6
