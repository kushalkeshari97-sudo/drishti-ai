from sabnetra_ai.utils.serializers import alert_to_json, stats_to_json, active_suspects_to_json
from sabnetra_ai.core.alert_system import Alert


def test_alert_to_json():
    alert = Alert(alert_id="ALERT_1", suspect_id="SUSPECT", track_id=1,
                  camera_id="cam0", state="RED", score=0.78, timestamp=1000.0)
    j = alert_to_json(alert)
    assert j["alert_id"] == "ALERT_1"
    assert j["score"] == 0.78
    assert "timestamp" in j


def test_stats_to_json():
    stats = {
        "pipeline": {"frames_processed": 100, "detections": 50, "matches": 5, "alerts": 2},
        "suspects": {"enrolled_suspects": 3, "active_red_tracks": 1, "active_yellow_tracks": 0},
        "alerts": {"total_alerts": 10, "active_cooldowns": 2},
        "streams": {},
    }
    j = stats_to_json(stats)
    assert j["pipeline"]["frames_processed"] == 100


def test_active_suspects_to_json():
    suspects = [
        {"track_id": 1, "suspect_id": "SUSPECT", "state": "RED", "score": 0.78, "camera_id": "cam0"},
    ]
    j = active_suspects_to_json(suspects)
    assert len(j) == 1
    assert j[0]["state"] == "RED"
