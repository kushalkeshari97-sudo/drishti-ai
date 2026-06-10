import os
import sys
from unittest.mock import patch, MagicMock

os.environ["SABNETRA_API_KEY"] = ""

from fastapi.testclient import TestClient
from scripts.run_api import app, pipeline


client = TestClient(app)


def test_health():
    pipeline._running = True
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["running"]
    pipeline._running = False


def test_health_not_running():
    pipeline._running = False
    resp = client.get("/health")
    assert resp.status_code == 200
    assert not resp.json()["running"]


def test_stats():
    with patch.object(pipeline, "stats") as mock_stats:
        mock_stats.return_value = {
            "pipeline": {"frames_processed": 10, "detections": 5, "matches": 1, "alerts": 1},
            "memory": {"total_tracks": 3, "states": {}, "cameras": 1},
            "alerts": {"total_alerts": 1, "cameras": {}, "active_cooldowns": 0},
            "suspects": {"enrolled_suspects": 2, "auto_enrolled": 0, "active_red_tracks": 0, "active_yellow_tracks": 0, "global_track_to_suspect_mappings": 0},
            "cross_camera": {},
            "streams": {},
        }
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["pipeline"]["frames_processed"] == 10


def test_alerts_default():
    resp = client.get("/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_alerts_with_limit():
    resp = client.get("/alerts?limit=5")
    assert resp.status_code == 200


def test_suspects_active():
    with patch.object(pipeline.suspect_manager, "get_all_active_suspects") as mock_get:
        mock_get.return_value = []
        resp = client.get("/suspects/active")
        assert resp.status_code == 200


def test_suspects_stats():
    with patch.object(pipeline.suspect_manager, "stats") as mock_stats:
        mock_stats.return_value = {
            "enrolled_suspects": 0,
            "auto_enrolled": 0,
            "active_red_tracks": 0,
            "active_yellow_tracks": 0,
            "global_track_to_suspect_mappings": 0,
        }
        resp = client.get("/suspects")
        assert resp.status_code == 200


def test_enroll_suspect_file_not_found():
    resp = client.post("/suspects/enroll", json={
        "image_path": "/nonexistent/path.jpg",
    })
    assert resp.status_code == 422


def test_enroll_suspect_success():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(b"fake image data")
        tmp.flush()
        with patch.object(pipeline, "enroll_suspect") as mock_enroll, \
             patch("scripts.run_api.save_suspects"):
            mock_enroll.return_value = "S0001"
            with patch.object(pipeline.suspect_manager, "get_suspect_profile") as mock_get:
                mock_get.return_value = MagicMock()
                resp = client.post("/suspects/enroll", json={
                    "image_path": tmp.name,
                    "case_id": "C1",
                    "suspect_id": "S1",
                })
                assert resp.status_code == 200
                assert resp.json()["suspect_id"] == "S0001"


def test_enroll_suspect_validation_empty_path():
    resp = client.post("/suspects/enroll", json={"image_path": ""})
    assert resp.status_code == 422


def test_add_camera():
    with patch.object(pipeline, "add_camera") as mock_add:
        mock_add.return_value = True
        resp = client.post("/cameras", json={"url": "rtsp://host/stream", "name": "cam0"})
        assert resp.status_code == 200
        assert resp.json()["ok"]


def test_add_camera_invalid_url():
    resp = client.post("/cameras", json={"url": "invalid"})
    assert resp.status_code == 422


def test_add_camera_empty_name():
    resp = client.post("/cameras", json={"url": "rtsp://host/stream", "name": ""})
    assert resp.status_code == 422


def test_remove_camera():
    with patch.object(pipeline, "remove_camera") as mock_rm:
        resp = client.delete("/cameras/cam0")
        assert resp.status_code == 200
        assert resp.json()["ok"]


def test_list_cameras():
    with patch.object(pipeline.rtsp_manager, "stats") as mock_stats:
        mock_stats.return_value = {}
        resp = client.get("/cameras")
        assert resp.status_code == 200


def test_health_with_api_key():
    os.environ["SABNETRA_API_KEY"] = "test-key-123"
    import importlib
    import scripts.run_api
    importlib.reload(scripts.run_api)
    from scripts.run_api import app as app2
    c2 = TestClient(app2)
    resp = c2.get("/health")
    assert resp.status_code == 401
    resp = c2.get("/health", headers={"Authorization": "Bearer test-key-123"})
    assert resp.status_code == 200
    os.environ.pop("SABNETRA_API_KEY", None)
    importlib.reload(scripts.run_api)
