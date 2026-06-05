# SabNetra AI

CCTV-first real-time surveillance intelligence system for suspect detection, tracking, and cross-camera identity recognition.

## Architecture

```
sabnetra_ai/
├── __init__.py              Package entry, exports create_pipeline()
├── config.py                All configuration dataclasses (detector, tracker, matcher, etc.)
├── cli.py                   CLI: enroll, add-camera, start, stats
├── core/
│   ├── detector.py          YOLOv8 person detector with NMS + aspect-ratio / area filters
│   ├── tracker.py           BoT-SORT tracker (Kalman filter, Hungarian matching, IoU gating)
│   ├── feature_extractor.py Multi-modal feature extraction: face, ReID, clothing, gait
│   ├── matcher.py           FAISS-based matching engine + SuspectProfile database
│   ├── embedding_memory.py  Per-track memory: accumulates and consolidates embeddings
│   ├── state_engine.py      Classifies match scores into GREEN / YELLOW / RED
│   ├── suspect_manager.py   Global suspect registry, track-to-suspect mapping
│   ├── alert_system.py      Alert generation with cooldowns and rate limiting
│   ├── cross_camera.py      Cross-camera tracking with temporal validation
│   └── frame_buffer.py      Thread-safe frame queue per camera
├── models/
│   └── model_manager.py     Singleton lazy-loader for YOLO, InsightFace, OSNet, gait
├── enrollment/
│   └── fir_module.py        FIR (First Information Report) enrollment from images/videos
├── pipeline/
│   └── orchestrator.py      SabNetraPipeline: capture -> detect -> track -> match -> alert
├── stream/
│   └── rtsp_manager.py      RTSP stream reader with auto-reconnect
└── utils/
    ├── geometry.py          IoU, NMS, cosine similarity, optical flow, box utilities
    └── helpers.py           Timer, FrameRateCounter, draw_detection, low-light check
```

## Pipeline Flow

```
RTSP Camera -> FrameBuffer -> Detector (YOLO) -> Tracker (BoT-SORT)
                                              -> FeatureExtractor (face/ReID/clothing)
                                                     |
                                                     v
                                              Matcher (FAISS)
                                                     |
                                                     v
                                              StateEngine (GREEN/YELLOW/RED)
                                                     |
                                              +------+------+
                                              |             |
                                          AlertSystem   CrossCameraTracker
```

## Quick Start

### 1. Setup

```bash
conda create -n sabnetra python=3.10
conda activate sabnetra
pip install -r requirements.txt
pip install -e .
```

### 2. Enroll a suspect

```bash
sabnetra enroll suspect.jpg --case CASE001 --id S0001
```

Enroll from multiple images or videos:

```bash
sabnetra enroll front.jpg side.jpg --videos walk.mp4 --case CASE001 --id S0002
```

### 3. Start monitoring a camera

```bash
sabnetra start rtsp://192.168.1.100:554/stream --name entrance
```

Press Ctrl+C to stop.

### 4. Webcam test

```bash
python test_webcam.py
```

Places `sabnetra_ai/suspect.jpg` in the FAISS gallery, then runs live webcam tracking. Box colors:

- **GREEN** -- no match
- **YELLOW** -- possible match (score >= yellow_threshold)
- **RED** -- confirmed match (score >= red_threshold)

## CLI Commands

| Command | Description |
|---|---|
| `sabnetra enroll <images> [--videos ...] [--case CASE] [--id ID]` | Enroll suspect from image(s) and/or video(s) |
| `sabnetra add-camera <url> [--name NAME]` | Register an RTSP camera |
| `sabnetra start <url> [--name NAME]` | Add camera and begin monitoring |
| `sabnetra stats` | Show live pipeline statistics |

## Configuration

All parameters live in `config.py` under dataclass groups:

| Config | Key Parameters |
|---|---|
| `DetectorConfig` | `confidence_threshold` (0.35), `iou_threshold` (0.45), `img_size` (640) |
| `TrackerConfig` | `match_thresh` (0.28), `max_time_lost` (60), `appearance_weight` (0.75) |
| `MatcherConfig` | `yellow_threshold` (0.55), `red_threshold` (0.78), per-modality weights |
| `AlertConfig` | `cooldown_seconds` (30), `max_alerts_per_minute` (10) |
| `FeatureConfig` | `face_model` (buffalo_l), `reid_model_name` (osnet_x1_0) |
| `PipelineConfig` | `process_every_n_frames` (2), enable/disable face/ReID/clothing/gait |

Example override:

```python
from sabnetra_ai import create_pipeline, SabNetraConfig

config = SabNetraConfig()
config.detector.confidence_threshold = 0.55
config.matcher.red_threshold = 0.45
pipeline = create_pipeline(config)
```

## Modality Weights

Match score is a weighted combination:

```
score = face_weight * face_sim
      + reid_weight * body_sim
      + clothing_weight * clothing_sim
      + gait_weight * gait_sim
```

Defaults: face 0.40, ReID 0.30, clothing 0.15, gait 0.15.

## State Classification

| State | Condition |
|---|---|
| GREEN | score < yellow_threshold (0.55) |
| YELLOW | score >= yellow_threshold |
| RED | score >= red_threshold (0.78) and temporally consistent for N frames |

RED lock persists per track to prevent flickering.

## Cross-Camera Tracking

When a suspect is matched (RED) on one camera, the system searches for reappearances on other registered cameras. Temporal gating prevents impossible transitions (too fast / too slow).

## Dependencies

| Library | Purpose |
|---|---|
| ultralytics (YOLOv8) | Person detection |
| insightface | Face detection and recognition |
| torchreid (OSNet) | Person re-identification |
| faiss-cpu | Fast similarity search |
| opencv-python | Frame capture and visualization |
| scipy | Hungarian algorithm (linear sum assignment) |

Optional: `torchreid` (ReID), `opengait` (gait recognition).

## Notes

- All embeddings are L2-normalized before storage and matching. FAISS uses inner product (cosine similarity).
- Track memory consolidates multiple observations via mean-pooling.
- The gait module is placeholder-only (`opengaitext` is not a real package) and disabled by default.
- Detected persons with width > height or area < 4000 px are filtered out as noise.
