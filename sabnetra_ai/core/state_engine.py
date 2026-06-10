import logging
import time
from typing import Optional, Tuple, Dict

from sabnetra_ai.config import MatcherConfig
from sabnetra_ai.core.matcher import MatchResult

logger = logging.getLogger(__name__)

RED_LOCKED_TTL = 300.0


class IdentityState:
    """State constants for identity classification levels."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @staticmethod
    def color(state: str) -> str:
        """Return numeric value for a given state string."""
        colors = {"GREEN": 1, "YELLOW": 2, "RED": 3}
        return colors.get(state, 0)


class StateClassificationEngine:
    """Classifies match results into GREEN/YELLOW/RED states with red-lock TTL."""

    def __init__(self, config: Optional[MatcherConfig] = None):
        """Initialize engine with matcher config."""
        self.config = config or MatcherConfig()
        self._red_locked: Dict[int, float] = {}

    def classify(self, match_result: MatchResult,
                 track_id: int,
                 camera_id: str,
                 current_state: str = IdentityState.GREEN
                 ) -> Tuple[str, str, float]:
        """Classify a match result into a (state, suspect_id, score) tuple.
        Args:
            match_result: The match result to classify.
            track_id: The track identifier.
            camera_id: The camera identifier.
            current_state: The current state of the track.
        Returns:
            Tuple of (state, suspect_id, score).
        """
        if track_id in self._red_locked:
            if time.time() - self._red_locked[track_id] < RED_LOCKED_TTL:
                return (IdentityState.RED, match_result.suspect_id,
                        max(match_result.score, 0.78))
            self._red_locked.pop(track_id, None)

        score = match_result.score
        suspect_id = match_result.suspect_id

        if score >= self.config.red_threshold:
            self._red_locked[track_id] = time.time()
            logger.warning(f"RED ALERT: Track {track_id} matched suspect "
                           f"{suspect_id} with score {score:.3f}")
            return (IdentityState.RED, suspect_id, score)

        if score >= self.config.yellow_threshold:
            return (IdentityState.YELLOW, suspect_id, score)

        return (IdentityState.GREEN, suspect_id, score)

    def reset_track(self, track_id: int):
        """Remove red-lock for a specific track."""
        self._red_locked.pop(track_id, None)

    def reset_all(self):
        """Clear all red-locked tracks."""
        self._red_locked.clear()

    def cleanup_stale(self, timeout: float = RED_LOCKED_TTL):
        """Remove red-locked tracks older than timeout."""
        now = time.time()
        stale = [tid for tid, ts in self._red_locked.items()
                 if now - ts > timeout]
        for tid in stale:
            self._red_locked.pop(tid, None)
        if stale:
            logger.debug(f"Cleaned {len(stale)} stale red-locked tracks")

    @property
    def red_count(self) -> int:
        """Return the number of currently red-locked tracks."""
        return len(self._red_locked)
