import time
import logging
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger("SabNetra")


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed*1000:.1f}ms")

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed


class FrameRateCounter:
    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []

    def tick(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) < 2:
            return 0.0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])

    @property
    def fps(self) -> float:
        return self.tick()


def is_low_light(frame: np.ndarray, threshold: float = 30.0) -> bool:
    gray = frame if len(frame.shape) == 2 else cv2_to_gray(frame)
    mean_brightness = np.mean(gray)
    return mean_brightness < threshold


def cv2_to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def draw_detection(frame: np.ndarray, box: np.ndarray, track_id: int,
                   state: str, suspect_id: Optional[str] = None,
                   score: float = 0.0):
    import cv2
    colors = {
        "GREEN": (0, 255, 0),
        "YELLOW": (0, 255, 255),
        "RED": (0, 0, 255),
    }
    color = colors.get(state, (255, 255, 255))
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_parts = [f"ID:{track_id}"]
    if state:
        label_parts.append(state)
    if suspect_id:
        label_parts.append(f"S:{suspect_id}")
    if score > 0:
        label_parts.append(f"{score:.2f}")
    label = " ".join(label_parts)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window
