import logging
from typing import Optional, Tuple

from sabnetra_ai.config import MatcherConfig
from sabnetra_ai.core.matcher import MatchResult

logger = logging.getLogger(__name__)


class IdentityState:
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @staticmethod
    def color(state: str) -> str:
        colors = {"GREEN": 1, "YELLOW": 2, "RED": 3}
        return colors.get(state, 0)


class StateClassificationEngine:
    def __init__(self, config: Optional[MatcherConfig] = None):
        self.config = config or MatcherConfig()
        self._red_locked: set = set()

    def classify(self, match_result: MatchResult,
                 track_id: int,
                 camera_id: str,
                 current_state: str = IdentityState.GREEN
                 ) -> Tuple[str, str, float]:
        if track_id in self._red_locked:
            return (IdentityState.RED, match_result.suspect_id,
                    max(match_result.score, 0.78))

        score = match_result.score
        suspect_id = match_result.suspect_id

        if score >= self.config.red_threshold:
            self._red_locked.add(track_id)
            logger.warning(f"RED ALERT: Track {track_id} matched suspect "
                           f"{suspect_id} with score {score:.3f}")
            return (IdentityState.RED, suspect_id, score)

        if score >= self.config.yellow_threshold:
            return (IdentityState.YELLOW, suspect_id, score)

        return (IdentityState.GREEN, suspect_id, score)

    def reset_track(self, track_id: int):
        self._red_locked.discard(track_id)

    def reset_all(self):
        self._red_locked.clear()

    @property
    def red_count(self) -> int:
        return len(self._red_locked)
