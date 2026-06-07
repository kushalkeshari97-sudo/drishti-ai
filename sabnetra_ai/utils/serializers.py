import numpy as np
from datetime import datetime
from typing import List, Optional


def alert_to_json(alert):
    return {
        "alert_id": alert.alert_id,
        "suspect_id": alert.suspect_id,
        "track_id": alert.track_id,
        "camera_id": alert.camera_id,
        "state": alert.state,
        "score": round(alert.score, 4),
        "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
        "details": alert.details or {},
    }


def track_to_json(track, state: str = "GREEN", suspect_id: str = "UNKNOWN",
                  score: float = 0.0, modalities: Optional[list] = None):
    return {
        "track_id": track.track_id,
        "bbox": [round(float(x), 1) for x in track.bbox],
        "confidence": round(track.confidence, 4),
        "state": state,
        "suspect_id": suspect_id,
        "score": round(score, 4),
        "modalities": modalities or [],
        "is_confirmed": track.is_confirmed,
    }


def detection_to_json(det):
    return {
        "bbox": det.bbox.tolist() if isinstance(det.bbox, np.ndarray) else det.bbox,
        "confidence": round(det.confidence, 4),
        "class_id": det.class_id,
    }


def stats_to_json(stats: dict) -> dict:
    return {
        "pipeline": {
            "frames_processed": stats["pipeline"]["frames_processed"],
            "detections": stats["pipeline"]["detections"],
            "matches": stats["pipeline"]["matches"],
            "alerts": stats["pipeline"]["alerts"],
        },
        "suspects": {
            "enrolled": stats["suspects"]["enrolled_suspects"],
            "active_red": stats["suspects"]["active_red_tracks"],
            "active_yellow": stats["suspects"]["active_yellow_tracks"],
        },
        "alerts": {
            "total": stats["alerts"]["total_alerts"],
            "active_cooldowns": stats["alerts"]["active_cooldowns"],
        },
        "streams": stats.get("streams", {}),
    }


def active_suspects_to_json(suspects: List[dict]) -> list:
    return [
        {
            "track_id": s["track_id"],
            "suspect_id": s["suspect_id"],
            "state": s["state"],
            "score": round(s["score"], 4),
            "camera_id": s["camera_id"],
        }
        for s in suspects
    ]
