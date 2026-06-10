import time
import logging
from collections import defaultdict, deque
from typing import Optional, Callable, List
from dataclasses import dataclass

from sabnetra_ai.config import AlertConfig

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Data class representing a single alert event."""

    alert_id: str
    suspect_id: str
    track_id: int
    camera_id: str
    state: str
    score: float
    timestamp: float
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class AlertSystem:
    """Manages alert triggering with cooldown, rate limiting, and callbacks."""

    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize alert system with config."""
        self.config = config or AlertConfig()
        self._alert_history: deque = deque(maxlen=1000)
        self._cooldown_tracker: dict = {}
        self._alert_counters: defaultdict = defaultdict(int)
        self._reset_time = time.time()
        self._callbacks: List[Callable] = []

    def trigger(self, suspect_id: str, track_id: int, camera_id: str,
                state: str, score: float, details: Optional[dict] = None
                ) -> Optional[Alert]:
        """Trigger an alert if cooldown and rate-limit checks pass.
        Args:
            suspect_id: The suspect identifier.
            track_id: The track identifier.
            camera_id: The camera identifier.
            state: Alert state (GREEN/YELLOW/RED).
            score: Match score.
            details: Optional additional details.
        Returns:
            Alert object if triggered, None otherwise.
        """
        if state == "GREEN" and not self.config.alert_on_yellow:
            return None
        if state == "YELLOW" and not self.config.alert_on_yellow:
            return None
        if state == "RED" and not self.config.alert_on_red:
            return None
        if self._is_on_cooldown(suspect_id):
            return None
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded, suppressing alert")
            return None
        alert_id = f"ALERT_{suspect_id}_{track_id}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            suspect_id=suspect_id,
            track_id=track_id,
            camera_id=camera_id,
            state=state,
            score=score,
            timestamp=time.time(),
            details=details or {},
        )
        self._alert_history.append(alert)
        self._cooldown_tracker[suspect_id] = time.time()
        self._alert_counters[camera_id] += 1
        self._notify(alert)
        logger.warning(f"ALERT [{state}] {suspect_id} on {camera_id} "
                       f"track {track_id} score={score:.3f}")
        return alert

    def _is_on_cooldown(self, suspect_id: str) -> bool:
        if suspect_id not in self._cooldown_tracker:
            return False
        elapsed = time.time() - self._cooldown_tracker[suspect_id]
        return elapsed < self.config.cooldown_seconds

    def _check_rate_limit(self) -> bool:
        window_start = time.time() - 60
        recent = sum(1 for a in self._alert_history
                     if a.timestamp > window_start)
        return recent < self.config.max_alerts_per_minute

    def _notify(self, alert: Alert):
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_callback(self, callback: Callable):
        """Register a callback function invoked on every alert trigger."""
        self._callbacks.append(callback)

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Return the most recent alerts, up to count."""
        return list(self._alert_history)[-count:]

    def get_alerts_since(self, since: float) -> List[Alert]:
        """Return all alerts with timestamp >= since."""
        return [a for a in self._alert_history if a.timestamp >= since]

    def clear(self):
        """Clear all alert history, cooldowns, and counters."""
        self._alert_history.clear()
        self._cooldown_tracker.clear()
        self._alert_counters.clear()

    def stats(self) -> dict:
        """Return summary statistics about the alert system."""
        return {
            "total_alerts": len(self._alert_history),
            "cameras": dict(self._alert_counters),
            "active_cooldowns": len(self._cooldown_tracker),
        }
