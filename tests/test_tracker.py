import numpy as np
from sabnetra_ai.core.tracker import Track, TrackState, KalmanFilter, BoTSORT
from sabnetra_ai.config import TrackerConfig


def test_kalman_filter_predict():
    kf = KalmanFilter()
    mean = np.array([50.0, 50.0, 20.0, 40.0], dtype=np.float32)
    cov = np.eye(4, dtype=np.float32) * 10.0
    new_mean, new_cov = kf.predict(mean, cov)
    assert new_mean.shape == (4,)
    assert new_cov.shape == (4, 4)


def test_kalman_filter_update():
    kf = KalmanFilter()
    mean = np.array([50.0, 50.0, 20.0, 40.0], dtype=np.float32)
    cov = np.eye(4, dtype=np.float32) * 10.0
    measurement = np.array([55.0, 52.0], dtype=np.float32)
    new_mean, new_cov = kf.update(mean, cov, measurement)
    assert new_mean.shape == (4,)


def test_track_creation():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det, confidence=0.9)
    assert track.track_id == 1
    assert track.state == TrackState.TENTATIVE
    assert track.hits == 1
    assert not track.is_confirmed
    assert track.is_tentative


def test_track_update_confirmed():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det, confidence=0.9)
    assert not track.is_confirmed
    track.update(np.array([5, 5, 105, 205]), None, 0.85)
    assert track.hits == 2
    assert not track.is_confirmed
    track.update(np.array([10, 10, 110, 210]), None, 0.85)
    assert track.hits == 3
    assert track.is_confirmed


def test_track_predict():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det, confidence=0.9)
    predicted = track.predict()
    assert predicted.shape == (4,)
    assert track.age == 1
    assert track.time_since_update == 1


def test_track_mark_lost():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det, confidence=0.9)
    track.mark_lost()
    assert track.is_lost


def test_track_embedding_history():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    emb = np.array([1.0, 0.0, 0.0])
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det,
                  appearance_embedding=emb, confidence=0.9)
    retrieved = track.get_appearance_embedding()
    assert retrieved is not None
    assert np.allclose(retrieved, emb)


def test_track_smooth_bbox():
    bbox = np.array([0, 0, 100, 200], dtype=float)
    det = {"bbox": bbox, "confidence": 0.9}
    track = Track(track_id=1, bbox=bbox, detection=det, confidence=0.9)
    new_bbox = np.array([10, 10, 110, 210], dtype=float)
    track.update(new_bbox, None, 0.85)
    assert not np.allclose(track.smooth_bbox, new_bbox)
    assert track.alpha == 0.3


def test_botsort_update_empty():
    tracker = BoTSORT()
    tracks = tracker.update([], [])
    assert len(tracks) == 0


def test_botsort_update_new_detections():
    tracker = BoTSORT()
    dets = [
        {"bbox": np.array([0, 0, 100, 200], dtype=float), "confidence": 0.9},
        {"bbox": np.array([150, 50, 250, 250], dtype=float), "confidence": 0.8},
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2


def test_botsort_reset():
    tracker = BoTSORT()
    dets = [{"bbox": np.array([0, 0, 100, 200], dtype=float), "confidence": 0.9}]
    tracker.update(dets)
    assert len(tracker.get_all_active_tracks()) == 1
    tracker.reset()
    assert len(tracker.get_all_active_tracks()) == 0
