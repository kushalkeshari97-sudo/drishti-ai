# SabNetra AI

Real-time surveillance intelligence system for multi-modal suspect detection, tracking, and cross-camera identity recognition in live CCTV streams.

---

## Overview

SabNetra AI is a modular computer vision pipeline that ingests video streams, detects persons, tracks them across frames, and matches them against a database of known suspects using face recognition, body re-identification (ReID), and clothing analysis. When a match meets confidence thresholds, the system flags the track and generates alerts.

```
Camera → Detection → Tracking → Feature Extraction → Matching → Alert
```

---

## Architecture

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Detection** | `core/detector.py` | YOLOv8-based person detection with noise filtering (size, aspect ratio, confidence) and NMS |
| **Tracking** | `core/tracker.py` | BoT-SORT tracker assigning unique IDs via Kalman filters and Hungarian algorithm with IoU + appearance + motion cost |
| **Feature Extraction** | `core/feature_extractor.py` | Multi-modal embedding extraction — face (InsightFace ArcFace, 512-d), body (OSNet ReID, 512-d), clothing (HSV histogram, 64-d) |
| **Matching** | `core/matcher.py` | FAISS IndexFlatIP cosine similarity search with weighted multi-modal scoring and temporal consistency checks |
| **State Classification** | `core/state_engine.py` | Three-state classification: GREEN (unknown), YELLOW (possible), RED (confirmed) with TTL-based lock |
| **Alerting** | `core/alert_system.py` | Configurable alert dispatch with per-suspect cooldowns, global rate limiting, and WebSocket callbacks |

### Detection States

| Color | Meaning | Condition |
|-------|---------|-----------|
| Green | Unknown person | Score < `yellow_threshold` |
| Yellow | Possible match | Score >= `yellow_threshold` |
| Red | Confirmed match | Score >= `red_threshold` (locked) |

---

## Project Structure

```
sabnetra_ai/
├── __init__.py                 # Package exports and logging configuration
├── __main__.py                 # python -m sabnetra_ai entry point
├── config.py                   # Dataclass-based configuration
├── cli.py                      # Command-line interface
├── core/
│   ├── detector.py             # YOLOv8 person detection
│   ├── tracker.py              # BoT-SORT multi-object tracking
│   ├── feature_extractor.py    # Face, body, and clothing feature extraction
│   ├── matcher.py              # FAISS similarity search and suspect database
│   ├── embedding_memory.py     # Per-track temporal embedding storage
│   ├── state_engine.py         # GREEN/YELLOW/RED state classification
│   ├── suspect_manager.py      # Global suspect registry with RED re-ID
│   ├── alert_system.py         # Alert generation and dispatch
│   ├── cross_camera.py         # Cross-camera tracking with temporal gating
│   └── frame_buffer.py         # Thread-safe camera frame queues
├── models/
│   └── model_manager.py        # Singleton model loader (YOLO, InsightFace, OSNet)
├── enrollment/
│   └── fir_module.py           # Suspect enrollment from images and videos
├── pipeline/
│   └── orchestrator.py         # Pipeline orchestration and lifecycle management
├── stream/
│   └── rtsp_manager.py         # RTSP stream handling with auto-reconnect
└── utils/
    ├── config_loader.py        # YAML and .env configuration loading
    ├── geometry.py             # IoU, cosine similarity, NMS utilities
    ├── helpers.py              # Timer, FPS counter, annotation drawing
    ├── persistence.py          # Suspect profile serialization with optional encryption
    └── serializers.py          # JSON serializers for API responses

scripts/
├── run_video.py                # Offline video file processing
├── run_camera.py               # Webcam demo
└── run_api.py                  # FastAPI REST + WebSocket server

tests/                          # Test suite
├── test_detector.py
├── test_tracker.py
├── test_matcher.py
└── ...

data/
└── suspects/                   # Suspect photos (gitignored)

config.yaml                     # Runtime configuration overrides
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported)

### Setup

```bash
conda create -n sabnetra python=3.10
conda activate sabnetra
pip install -r requirements.txt
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `ultralytics` | YOLOv8 detection model |
| `insightface` | Face detection and recognition (ArcFace) |
| `faiss-cpu` | Vector similarity search |
| `opencv-python` | Image processing and camera I/O |
| `scipy` | Hungarian algorithm for tracking |
| `fastapi` / `uvicorn` / `websockets` | REST API and real-time alerts |
| `pyyaml` / `python-dotenv` | Configuration file parsing |
| `cryptography` | Optional disk encryption for suspect data |

---

## Usage

### Suspect Enrollment

```bash
# From a single image
sabnetra enroll suspect.jpg --case CASE001 --id S0001

# From multiple images and videos (embeddings are fused)
sabnetra enroll front.jpg side.jpg --videos walk.mp4 --case CASE001 --id S0002
```

### Video File Processing

```bash
python scripts/run_video.py path/to/video.mp4 --show
```

Options:
- `--suspect path/to/image.jpg` — suspect photo for matching
- `--suspect-id SUSPECT_001` — custom suspect identifier
- `--output result.mp4` — save annotated output video
- `--process-every 3` — process every Nth frame
- `--img-size 256` — detection input resolution
- `--disable-face`, `--disable-reid`, `--disable-clothing` — selectively disable modalities

### Live Camera Monitoring

```bash
# RTSP camera
sabnetra start rtsp://admin:password@192.168.1.100:554/stream --name entrance

# Webcam demo
python scripts/run_camera.py
```

### API Server

```bash
python scripts/run_api.py
```

Produces a FastAPI server at `http://localhost:8000`. Interactive documentation available at `/docs`.

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Pipeline status check |
| `GET` | `/stats` | Runtime statistics (frames, detections, alerts) |
| `GET` | `/alerts?limit=50` | Recent alert history |
| `GET` | `/suspects` | Enrolled suspect statistics |
| `GET` | `/suspects/active` | Currently active (YELLOW/RED) suspects |
| `POST` | `/suspects/enroll` | Enroll a new suspect |
| `POST` | `/cameras` | Register an RTSP camera |
| `GET` | `/cameras` | List registered cameras |
| `DELETE` | `/cameras/{id}` | Remove a camera |
| `WS` | `/ws` | Real-time alert WebSocket |

### CLI Commands

| Command | Description |
|---------|-------------|
| `sabnetra enroll <images> [--videos ...] [--case CASE] [--id ID]` | Enroll suspect from images/videos |
| `sabnetra add-camera <url> [--name NAME]` | Register an RTSP camera |
| `sabnetra start <url> [--name NAME]` | Add camera and start monitoring |
| `sabnetra stats` | Display pipeline statistics |
| `sabnetra list-suspects` | List active YELLOW/RED suspects |
| `sabnetra remove-suspect <id>` | Remove a suspect profile |
| `sabnetra alerts [--count N]` | Show recent alert history |
| `sabnetra reset` | Reset all track states |

---

## Configuration

Configuration is managed through `sabnetra_ai/config.py` dataclasses with overrides from `config.yaml`, `config.local.yaml`, and environment variables (precedence order).

### YAML Configuration

```yaml
detector:
  confidence_threshold: 0.55
  img_size: 256

tracker:
  match_thresh: 0.85
  max_time_lost: 30

matcher:
  face_weight: 0.5
  reid_weight: 0.3
  red_threshold: 0.35
  yellow_threshold: 0.20
```

### Environment Variables

```
SABNETRA_RTSP_URL=rtsp://admin:pass@192.168.1.217:554/stream
SABNETRA_MODEL_PATH=yolov8n.pt
SABNETRA_DEVICE=cuda:0
SABNETRA_API_KEY=your-secret-key
```

### Programmatic Configuration

```python
from sabnetra_ai import create_pipeline, SabNetraConfig

config = SabNetraConfig()
config.detector.confidence_threshold = 0.55
config.matcher.red_threshold = 0.45
pipeline = create_pipeline(config)
```

### Configuration Reference

| Config | Key Parameters | Defaults |
|--------|---------------|----------|
| `DetectorConfig` | `confidence_threshold`, `iou_threshold`, `img_size`, `min_person_area` | 0.35, 0.45, 640, 4000 |
| `TrackerConfig` | `match_thresh`, `max_time_lost`, `appearance_weight`, `motion_weight` | 0.28, 60, 0.75, 0.25 |
| `MatcherConfig` | `face_weight`, `reid_weight`, `red_threshold`, `yellow_threshold` | 0.40, 0.30, 0.78, 0.55 |
| `FeatureConfig` | `face_model`, `reid_model_name`, `face_det_thresh` | buffalo_l, osnet_x1_0, 0.5 |
| `AlertConfig` | `cooldown_seconds`, `max_alerts_per_minute` | 30, 10 |
| `TemporalConfig` | `min_travel_time`, `max_travel_speed`, `temporal_window` | 5s, 15 m/s, 300s |
| `PipelineConfig` | `process_every_n_frames`, `enable_face`, `enable_reid` | 2, True, True |

---

## Module Reference

### Detector
- YOLOv8 object detection with configurable confidence and IoU thresholds
- Filters: aspect ratio (height > width), minimum area (`min_person_area`), NMS

### Tracker
- BoT-SORT algorithm integrating Kalman filter motion prediction with appearance-based re-identification
- Cost matrix: weighted combination of IoU, motion (Gaussian heatmap distance), and appearance (cosine similarity)
- Track lifecycle: TENTATIVE (3 hits) → CONFIRMED → LOST (cleanup after `max_time_lost`)

### Feature Extractor
- Face: InsightFace ArcFace — 512-d embedding from detected face regions
- Body: OSNet ReID — 512-d global appearance embedding from full-body crops
- Clothing: HSV histogram (32+16+16 bins) — lightweight color/texture descriptor from upper torso
- All embeddings are L2-normalized before storage and matching

### Matcher
- Suspect profiles stored in FAISS `IndexIDMap` wrapping `IndexFlatIP` for inner product search
- Multi-modal score: `0.40 * face + 0.30 * body + 0.15 * clothing + 0.15 * gait`
- Temporal consistency: score must remain above threshold across N consecutive frames
- Incremental index updates — no full rebuild on suspect addition

### State Engine
- GREEN: score < `yellow_threshold`
- YELLOW: `yellow_threshold` <= score < `red_threshold`
- RED: score >= `red_threshold` — lock with configurable TTL (default 300s) prevents flickering

### Suspect Manager
- Auto-enrollment for unmatched tracks with persistent suspect IDs
- RED re-ID: new tracks are compared against previously matched RED suspects using stored embeddings
- Track-to-suspect mapping with cross-referencing

### Alert System
- Per-suspect cooldown (30s default) and global rate limiting (10/min default)
- Callback-based dispatch compatible with WebSocket push
- Configurable alerting on YELLOW state

### Cross-Camera Tracking
- Temporal gating: validated transitions require travel time between `min_travel_time` and `max_travel_speed` constraints
- Camera adjacency and overlap configuration
- Suspect path reconstruction across camera network

---

## Testing

```bash
python -m pytest tests/
```

The test suite covers all modules with 200+ tests, including edge cases for matching, state transitions, persistence I/O, and pipeline integration.

---

## Development Notes

- All embeddings are L2-normalized. FAISS `IndexFlatIP` with normalized vectors yields cosine similarity.
- The `ModelManager` follows the singleton pattern — all cameras share model instances.
- Models are loaded lazily on first access; warmup via `model_manager.warmup()` pre-initializes CUDA kernels.
- The gait module (`opengaitext`) is a placeholder and disabled by default.
- Track memory consolidates embeddings via mean-pooling for robust matching.
- Suspect profile persistence supports optional Fernet encryption via the `cryptography` package.
- Evidence frames (snapshots on RED match) are saved to `data/evidence/`.
