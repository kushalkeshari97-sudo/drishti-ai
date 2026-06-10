# SabNetra AI

Real-time surveillance intelligence system for multi-modal suspect detection, tracking, and cross-camera identity recognition in live CCTV streams.

---

## Overview

**What it does:** SabNetra AI ingests live or recorded video streams, detects people in each frame, tracks them across frames using unique identifiers, and matches them against a database of known suspects using face recognition, body appearance (ReID), and clothing analysis. When a match meets configurable confidence thresholds, the system flags the bounding box and generates alerts.

**Why it exists:** Traditional surveillance requires operators to manually monitor dozens of camera feeds simultaneously — a task prone to fatigue and error. SabNetra AI automates the process of detecting persons of interest across multiple cameras, reducing response time and enabling security teams to focus on verified threats.

---

## Features

- **Multi-modal matching** — Combines face embeddings (InsightFace ArcFace, 512-d), body appearance (OSNet ReID, 512-d), and clothing descriptors (HSV histogram) for robust identification under varying conditions.
- **Real-time tracking** — BoT-SORT tracker with Kalman filter motion prediction, Hungarian algorithm association, and appearance-based re-identification maintains consistent track IDs across frames.
- **State-based alerting** — Three-tier classification (GREEN / YELLOW / RED) with configurable thresholds and TTL-based RED lock to prevent flickering.
- **Cross-camera tracking** — Temporal gating validates suspect transitions between cameras with configurable min/max travel time constraints.
- **RED re-identification** — Previously matched RED suspects are re-identified on new tracks using stored embeddings, enabling persistent tracking after track loss.
- **Scalable architecture** — Singleton model manager shares loaded models across all camera pipelines; per-camera frame buffers and dedicated tracker instances support multi-stream processing.
- **REST API and WebSocket server** — FastAPI-based server exposes endpoints for suspect enrollment, camera management, alert history, and real-time alert push over WebSockets.
- **Configurable pipeline** — Per-modality toggles (face, ReID, clothing), frame skipping, and configurable detection/tracking thresholds all exposed via dataclass-based configuration with YAML and environment variable overrides.
- **Evidence capture** — Frames triggering a RED alert are automatically saved to disk with suspect ID, camera, and score metadata.
- **Encrypted persistence** — Suspect profiles can be saved to disk with optional Fernet encryption via the `cryptography` package.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI / ML** | YOLOv8 (person detection), InsightFace ArcFace (face recognition), OSNet (ReID), FAISS (vector similarity search) |
| **Framework** | PyTorch, torchvision |
| **Backend** | Python 3.10, FastAPI, Uvicorn, WebSockets |
| **Computer Vision** | OpenCV, Ultralytics |
| **Tracking** | BoT-SORT, Kalman filter, Hungarian algorithm (SciPy) |
| **Configuration** | PyYAML, python-dotenv |
| **Security** | Cryptography (Fernet), HTTPBearer API key auth |
| **Infrastructure** | Conda, pip, setuptools |

---

## System Architecture

SabNetra AI processes video through a five-stage pipeline:

```
Video Stream → Detection → Tracking → Feature Extraction → Matching → Alert
```

1. **Detection** — YOLOv8 identifies persons in each frame. Detections are filtered by confidence, aspect ratio (height > width), and minimum area.
2. **Tracking** — BoT-SORT assigns each detection a unique track ID. A Kalman filter predicts the next position; the Hungarian algorithm matches predictions to new detections using a weighted cost of IoU, motion distance, and appearance cosine similarity.
3. **Feature Extraction** — For each tracked person, three embeddings are extracted: face (InsightFace), body (OSNet), and clothing (HSV histogram). All embeddings are L2-normalized.
4. **Matching** — FAISS `IndexFlatIP` searches the suspect database for each modality. A weighted ensemble score is computed (`0.40 * face + 0.30 * body + 0.15 * clothing + 0.15 * gait`).
5. **Alert** — Scores above `red_threshold` trigger a RED state lock, dispatch alerts via callbacks, and save evidence frames to disk.

The `ModelManager` is a singleton — all camera pipelines share the same loaded model instances. Each camera stream has its own frame buffer, tracker, and `CameraPipeline` instance, allowing parallel processing of multiple streams.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported)

### Setup

```bash
# Create and activate environment
conda create -n sabnetra python=3.10
conda activate sabnetra

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch / torchvision | >=2.0.0 | Deep learning runtime |
| ultralytics | >=8.0.0 | YOLOv8 detection model |
| insightface | >=0.7.0 | Face detection and recognition |
| faiss-cpu | >=1.7.0 | Vector similarity search |
| opencv-python | >=4.8.0 | Image processing and camera I/O |
| scipy | >=1.10.0 | Hungarian algorithm |
| fastapi / uvicorn / websockets | >=0.100.0 | REST API and real-time alerts |
| pyyaml / python-dotenv | >=6.0 | Configuration parsing |
| cryptography | >=41.0.0 | Optional disk encryption |

---

## Usage

### Suspect Enrollment

```bash
# From a single image
sabnetra enroll suspect.jpg --case CASE001 --id S0001

# From multiple images and videos (embeddings are fused via mean-pooling)
sabnetra enroll front.jpg side.jpg --videos walk.mp4 --case CASE001 --id S0002
```

### Video File Processing

```bash
python scripts/run_video.py path/to/video.mp4 --show

# With options
python scripts/run_video.py video.mp4 --show --output result.mp4 --img-size 192 --disable-reid
```

| Flag | Default | Description |
|------|---------|-------------|
| `--suspect path` | `data/suspects/suspect.jpg` | Suspect photo |
| `--suspect-id` | `SUSPECT_001` | Custom suspect ID |
| `--output path` | None | Save annotated video |
| `--process-every N` | 3 | Process every Nth frame |
| `--img-size N` | 256 | Detection input resolution |
| `--disable-face` | off | Skip face recognition |
| `--disable-reid` | off | Skip ReID body recognition |
| `--disable-clothing` | off | Skip clothing descriptor |

### Live Camera

```bash
# Local webcam (default)
python scripts/run_camera.py

# RTSP camera
python scripts/run_camera.py --rtsp rtsp://admin:pass@192.168.1.100:554/stream

# CLI-based monitoring
sabnetra start rtsp://admin:pass@192.168.1.100:554/stream --name entrance
```

### API Server

```bash
python scripts/run_api.py
```

Server starts at `http://localhost:8000`. Interactive docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Pipeline status |
| GET | `/stats` | Runtime statistics |
| GET | `/alerts?limit=50` | Recent alerts |
| GET | `/suspects` | Suspect statistics |
| GET | `/suspects/active` | Active YELLOW/RED suspects |
| POST | `/suspects/enroll` | Enroll a suspect |
| POST | `/cameras` | Register an RTSP camera |
| GET | `/cameras` | List cameras |
| DELETE | `/cameras/{id}` | Remove a camera |
| WS | `/ws` | Real-time alert WebSocket |

### CLI Reference

| Command | Description |
|---------|-------------|
| `sabnetra enroll <images> [--videos] [--case] [--id]` | Enroll suspect |
| `sabnetra add-camera <url> [--name]` | Register RTSP camera |
| `sabnetra start <url> [--name]` | Add camera and start monitoring |
| `sabnetra stats` | Show pipeline statistics |
| `sabnetra list-suspects` | List active YELLOW/RED suspects |
| `sabnetra remove-suspect <id>` | Delete a suspect profile |
| `sabnetra alerts [--count N]` | Show recent alerts |
| `sabnetra reset` | Reset all track states |

---

## Project Structure

```
sabnetra_ai/
├── __init__.py              # Package exports and logging
├── __main__.py              # python -m entry point
├── config.py                # Dataclass configuration definitions
├── cli.py                   # Command-line interface
├── core/                    # Core processing modules
│   ├── detector.py          # YOLOv8 person detection
│   ├── tracker.py           # BoT-SORT tracking
│   ├── feature_extractor.py # Face, body, clothing feature extraction
│   ├── matcher.py           # FAISS search + suspect database
│   ├── embedding_memory.py  # Per-track temporal embedding storage
│   ├── state_engine.py      # GREEN/YELLOW/RED classification
│   ├── suspect_manager.py   # Global suspect registry + RED re-ID
│   ├── alert_system.py      # Alert dispatch with cooldowns
│   ├── cross_camera.py      # Cross-camera temporal matching
│   └── frame_buffer.py      # Thread-safe frame queue
├── models/
│   └── model_manager.py     # Singleton model loader
├── enrollment/
│   └── fir_module.py        # Suspect enrollment from images/videos
├── pipeline/
│   └── orchestrator.py      # Pipeline lifecycle management
├── stream/
│   └── rtsp_manager.py      # RTSP stream handler with reconnect
└── utils/
    ├── config_loader.py     # YAML + .env config loading
    ├── geometry.py          # IoU, cosine similarity, NMS
    ├── helpers.py           # Timer, FPS counter, drawing
    ├── persistence.py       # Profile save/load with encryption
    └── serializers.py       # JSON API serializers

scripts/                     # Runnable entry points
├── run_video.py             # Offline video processing
├── run_camera.py            # Webcam / RTSP live demo
└── run_api.py               # FastAPI server

tests/                       # 200+ tests
├── test_detector.py
├── test_tracker.py
├── test_matcher.py
└── ...

data/
└── suspects/                # Suspect photos (gitignored)

config.yaml                  # Runtime configuration
```

---

## Future Improvements

- **Gait recognition** — Integrate the gait module (currently a placeholder) for walk-pattern-based identification when face and body are occluded.
- **Distributed processing** — Add support for distributing camera pipelines across multiple GPUs or nodes via message queuing (e.g., RabbitMQ, Redis).
- **Plugin system for feature extractors** — Allow third-party feature extraction models to be registered at runtime without modifying core code.
- **Historical analytics** — Store matched events in a time-series database (e.g., InfluxDB) for forensic search across time and cameras.
- **Automatic camera calibration** — Estimate camera adjacency and overlap automatically using person re-appearance patterns instead of manual configuration.
- **Edge deployment** — Quantize models (ONNX, TensorRT) and optimize for ARM-based edge devices (Jetson, Raspberry Pi).

---

## Contributors

- [Kushal Keshari](https://github.com/kushalkeshari97-sudo) — Initial work and maintenance

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
