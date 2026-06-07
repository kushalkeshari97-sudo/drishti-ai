import time
from sabnetra_ai.core.alert_system import AlertSystem, Alert
from sabnetra_ai.config import AlertConfig


def test_alert_creation():
    alert = Alert(
        alert_id="ALERT_1", suspect_id="S1", track_id=1,
        camera_id="cam0", state="RED", score=0.85, timestamp=time.time(),
    )
    assert alert.alert_id == "ALERT_1"
    assert alert.suspect_id == "S1"
    assert alert.details == {}


def test_alert_trigger_red():
    system = AlertSystem()
    alert = system.trigger("S1", 1, "cam0", "RED", 0.85)
    assert alert is not None
    assert alert.suspect_id == "S1"
    assert alert.state == "RED"


def test_alert_cooldown():
    config = AlertConfig(cooldown_seconds=60.0)
    system = AlertSystem(config)
    a1 = system.trigger("S1", 1, "cam0", "RED", 0.85)
    assert a1 is not None
    a2 = system.trigger("S1", 1, "cam0", "RED", 0.85)
    assert a2 is None


def test_alert_rate_limit():
    config = AlertConfig(max_alerts_per_minute=2, cooldown_seconds=0.0)
    system = AlertSystem(config)
    assert system.trigger("S1", 1, "cam0", "RED", 0.85) is not None
    assert system.trigger("S2", 2, "cam0", "RED", 0.85) is not None
    assert system.trigger("S3", 3, "cam0", "RED", 0.85) is None


def test_alert_green_not_triggered():
    system = AlertSystem()
    assert system.trigger("S1", 1, "cam0", "GREEN", 0.1) is None


def test_alert_callback():
    calls = []
    system = AlertSystem()
    system.register_callback(lambda a: calls.append(a.alert_id))
    system.trigger("S1", 1, "cam0", "RED", 0.85)
    assert len(calls) == 1
    assert calls[0].startswith("ALERT_S1_1")


def test_get_recent_alerts():
    system = AlertSystem()
    system.trigger("S1", 1, "cam0", "RED", 0.85)
    system.trigger("S2", 2, "cam1", "RED", 0.90)
    recent = system.get_recent_alerts(1)
    assert len(recent) == 1
    assert recent[0].suspect_id == "S2"


def test_clear():
    system = AlertSystem()
    system.trigger("S1", 1, "cam0", "RED", 0.85)
    system.clear()
    assert len(system.get_recent_alerts()) == 0


def test_stats():
    system = AlertSystem()
    system.trigger("S1", 1, "cam0", "RED", 0.85)
    s = system.stats()
    assert s["total_alerts"] == 1
    assert "cam0" in s["cameras"]
    assert s["active_cooldowns"] == 1
