import numpy as np
import time
from sabnetra_ai.utils.helpers import Timer, FrameRateCounter, is_low_light
from sabnetra_ai.core.matcher import MatchResult
from sabnetra_ai.core.state_engine import IdentityState


def test_timer():
    with Timer("test") as t:
        pass
    assert t.elapsed >= 0
    assert t.name == "test"


def test_timer_start_stop():
    t = Timer()
    t.start()
    time.sleep(0.01)
    elapsed = t.stop()
    assert elapsed >= 0.01


def test_framerate_counter():
    frc = FrameRateCounter(window=5)
    assert frc.fps == 0.0
    frc.tick()
    assert frc.tick() >= 0


def test_is_low_light_bright():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
    assert not is_low_light(frame, threshold=100.0)


def test_is_low_light_dark():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 10
    assert is_low_light(frame, threshold=50.0)


def test_identity_state():
    assert IdentityState.GREEN == "GREEN"
    assert IdentityState.YELLOW == "YELLOW"
    assert IdentityState.RED == "RED"


def test_color_mapping():
    assert IdentityState.color("GREEN") == 1
    assert IdentityState.color("YELLOW") == 2
    assert IdentityState.color("RED") == 3
    assert IdentityState.color("UNKNOWN") == 0
