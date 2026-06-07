# SabNetra AI

CCTV-first real-time surveillance intelligence system for suspect detection, tracking, and cross-camera identity recognition.

---

## What is this?

SabNetra AI watches live CCTV feeds and automatically detects people, tracks them across frames, and checks if they match a database of known suspects using face recognition, body appearance (ReID), and clothing analysis.

**Think of it as:** You upload a suspect's photo. The system connects to your cameras. When that person appears on any camera, the system turns the bounding box RED and sends an alert.

---

## How It Works (Mental Model)

Every video frame goes through a pipeline of 5 stages:

```
Camera → [1] Detect → [2] Track → [3] Extract Features → [4] Match → [5] Alert
```

| Stage | Module | What happens |
|-------|--------|-------------|
| **1. Detect** | `detector.py` | YOLOv8 finds all people in the frame. Filters out noise (too small, wrong shape). |
| **2. Track** | `tracker.py` | BoT-SORT assigns each person a unique ID and follows them frame-to-frame using Kalman filters and appearance matching. |
| **3. Extract** | `feature_extractor.py` | For each tracked person, extracts: face embedding, body appearance vector, and clothing descriptor. |
| **4. Match** | `matcher.py` | FAISS searches the suspect database. Compares face, body, and clothing to find the closest match. |
| **5. Alert** | `alert_system.py` | If match score is high enough, marks the track RED, locks it, and fires an alert. |

**Result colors on screen:**
- 🟢 **GREEN** — unknown person (no match)
- 🟡 **YELLOW** — possible match (score >= `yellow_threshold`)
- 🔴 **RED** — confirmed match (score >= `red_threshold`)

---

## Project Structure (Learner's Map)

```
sabnetra_ai/                          # Main package
├── __init__.py                       # Exports create_pipeline() + logging
├── __main__.py                       # Allows `python -m sabnetra_ai`
├── config.py                         # All settings as Python dataclasses
├── cli.py                            # Command-line interface (enroll, start, stats)
├── core/                             # The brain of the system
│   ├── detector.py                   # YOLOv8 person detection
│   ├── tracker.py                    # BoT-SORT tracking with Kalman filter
│   ├── feature_extractor.py          # Face, body, clothing feature extraction
│   ├── matcher.py                    # FAISS similarity search + suspect database
│   ├── embedding_memory.py           # Per-track memory (accumulates embeddings over time)
│   ├── state_engine.py               # GREEN/YELLOW/RED classification
│   ├── suspect_manager.py            # Global suspect registry
│   ├── alert_system.py               # Alert generation with cooldowns
│   ├── cross_camera.py               # Cross-camera tracking with temporal logic
│   └── frame_buffer.py               # Thread-safe frame queue (one per camera)
├── models/                           # Model loading (singleton pattern)
│   └── model_manager.py              # Lazy-loads YOLO, InsightFace, OSNet
├── enrollment/                       # Suspect enrollment (FIR)
│   └── fir_module.py                 # Extract features from images/videos
├── pipeline/                         # Orchestrator ties everything together
│   └── orchestrator.py               # SabNetraPipeline + CameraPipeline
├── stream/                           # RTSP camera handling
│   └── rtsp_manager.py               # Stream reader with auto-reconnect
└── utils/                            # Shared utilities
    ├── config_loader.py              # Loads config from YAML + .env
    ├── geometry.py                   # IoU, cosine similarity, NMS, box math
    ├── helpers.py                    # Timer, FPS counter, drawing, low-light check
    ├── persistence.py                # Save/load suspect profiles to disk
    └── serializers.py                # JSON converters for API responses

scripts/                              # Runnable entry points
├── run_camera.py                     # Webcam demo (enrolls suspect.jpg, tracks live)
└── run_api.py                        # FastAPI REST + WebSocket server

tests/                                # 115+ tests covering all modules
├── test_detector.py, test_tracker.py, test_matcher.py, ...

data/
└── suspects/
    └── suspect.jpg                   # Sample suspect photo for webcam demo

config.yaml                           # Production config overrides
.env.example                          # Environment variable template
pyproject.toml                        # Modern Python packaging (replaces setup.py)
```

---

## Quick Start for Learners

### 1. Setup the environment

```bash
conda create -n sabnetra python=3.10
conda activate sabnetra
pip install -r requirements.txt
pip install -e .
```

All dependencies in `requirements.txt`:
- **torch / torchvision** — deep learning framework
- **ultralytics** — YOLOv8 person detection model
- **insightface** — face detection + recognition (ArcFace)
- **faiss-cpu** — Facebook's fast vector similarity search
- **opencv-python** — camera capture, image processing, drawing
- **scipy** — Hungarian algorithm for optimal matching
- **fastapi / uvicorn / websockets** — REST API + real-time alerts
- **pyyaml / python-dotenv** — config file loading

### 2. Enroll a suspect

Take a photo of a person and enroll them:

```bash
sabnetra enroll suspect.jpg --case CASE001 --id S0001
```

This extracts face/body/clothing features from the image and stores them in the FAISS index.

> Place your suspect photos in `data/suspects/` (gitignored — never committed to the repo).
> Or use `sabnetra enroll <path>` to enroll from any location.

From multiple images and videos:

```bash
sabnetra enroll front.jpg side.jpg --videos walk.mp4 --case CASE001 --id S0002
```

The system fuses (averages) all extracted embeddings for a more robust match.

### 3. Try the webcam demo (no RTSP needed)

```bash
python scripts/run_camera.py
```

This:
1. Loads `data/suspects/suspect.jpg` (place your suspect photo there) and enrolls it as suspect "SUSPECT"
2. Opens your webcam
3. For each frame: detects people → tracks them → compares against the enrolled suspect
4. Draws colored boxes: 🟢 GREEN (unknown), 🟡 YELLOW (possible match), 🔴 RED (confirmed match)

### 4. Start monitoring an RTSP camera

```bash
sabnetra start rtsp://admin:password@192.168.1.100:554/stream --name entrance
```

Press `Ctrl+C` to stop.

### 5. Start the API server

```bash
python scripts/run_api.py
```

Opens a FastAPI server at `http://localhost:8000`. Browse to `http://localhost:8000/docs` for the interactive Swagger UI.

---

## How Each Module Works (Deep Dive for Learners)

### Detector (`core/detector.py`)
- Uses YOLOv8 (a pretrained object detection model) to find people in each frame
- Filters out:
  - Detections below `confidence_threshold` (too uncertain)
  - Detections wider than tall (people stand upright)
  - Detections smaller than 4000 pixels (too far / too small)
- Applies Non-Maximum Suppression (NMS) to remove duplicate boxes on the same person

### Tracker (`core/tracker.py`)
- BoT-SORT (ByteTrack + Simple Online and Realtime Tracking)
- Each person gets a **Kalman filter** that predicts their next position based on velocity
- **Hungarian algorithm** optimally matches predicted positions to new detections
- Uses **IoU (Intersection over Union)** + appearance cosine similarity + motion distance for matching
- Tracks have states: TENTATIVE → CONFIRMED → LOST
- A track becomes CONFIRMED after 3 consistent detections
- Lost tracks are cleaned up after `max_time_lost` frames

### Feature Extractor (`core/feature_extractor.py`)
Extracts 3 types of features from each detected person:

| Feature | Source Model | What it captures | Dimension |
|---------|-------------|-----------------|-----------|
| **Face** | InsightFace (ArcFace) | Facial identity | 512 |
| **Body (ReID)** | OSNet | Full-body appearance | 512 |
| **Clothing** | Random CNN projection | Torso color/texture | 256 |

All features are L2-normalized before storage/matching.

### Matcher (`core/matcher.py`)
- **SuspectDatabase** stores enrolled suspect profiles with their face/body/clothing embeddings
- **FAISS IndexFlatIP** performs fast cosine similarity search (inner product on normalized vectors)
- Match score = weighted combination:
  ```
  score = 0.40 * face_similarity + 0.30 * body_similarity + 0.15 * clothing_similarity + 0.15 * gait_similarity
  ```
- Temporal consistency check: score must be stable across N frames before trusting it

### State Engine (`core/state_engine.py`)
- **GREEN** — score < `yellow_threshold` (0.55) → unknown person
- **YELLOW** — score >= `yellow_threshold` → possible match, worth watching
- **RED** — score >= `red_threshold` (0.78) → confirmed match, LOCKED (no re-evaluation)
- RED lock prevents flickering: once a track is RED, it stays RED until reset

### Alert System (`core/alert_system.py`)
- Triggers alert when a track is classified RED
- **Cooldown**: one alert per suspect every 30 seconds (configurable)
- **Rate limit**: max 10 alerts per minute globally
- Fires callbacks (e.g., API WebSocket push, logging)

### Embedding Memory (`core/embedding_memory.py`)
- Each track gets a `TrackMemory` that accumulates embeddings over time
- Best embedding = mean of all observations (mean-pooling)
- Stale tracks (>300s old) are cleaned up automatically
- Supports cross-camera search by embedding similarity

### Cross-Camera Tracking (`core/cross_camera.py`)
- When a suspect is matched on one camera (RED), the system checks other cameras
- **Temporal gating**: travel between cameras must take between 5s and 300s (configurable)
- Uses `TemporalIntelligence` to validate camera-to-camera transitions

### Model Manager (`models/model_manager.py`)
- **Singleton** pattern — only one instance of each model loaded in memory
- Lazy loading — models are loaded on first access, not at import time
- Handles graceful fallback if a model fails to load (returns `None` instead of crashing)
- Warmup pass runs a dummy tensor through YOLO to initialize CUDA kernels

### FIR Module (`enrollment/fir_module.py`)
- FIR = First Information Report (suspect intake)
- Enroll from images or videos
- For videos, samples every `sample_rate` frames (default 15 FPS)
- Fuses multiple embeddings by mean-pooling + L2 normalization

---

## Configuration

All settings are Python dataclasses in `config.py`. You can override them:

### Via `config.yaml` (loaded automatically)

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

### Via `config.local.yaml` (for secrets, gitignored)

```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.217:554/stream"
```

### Via `.env` file

```
SABNETRA_RTSP_URL=rtsp://admin:pass@192.168.1.217:554/stream
SABNETRA_MODEL_PATH=yolov8n.pt
SABNETRA_DEVICE=cuda:0
```

### Via Python code

```python
from sabnetra_ai import create_pipeline, SabNetraConfig

config = SabNetraConfig()
config.detector.confidence_threshold = 0.55
config.matcher.red_threshold = 0.45
pipeline = create_pipeline(config)
```

### Config reference

| Config Dataclass | Key Parameters | Default |
|-----------------|---------------|---------|
| `DetectorConfig` | `confidence_threshold`, `iou_threshold`, `img_size`, `device` | 0.35, 0.45, 640, cuda:0 |
| `TrackerConfig` | `match_thresh`, `max_time_lost`, `appearance_weight`, `motion_weight` | 0.28, 60, 0.75, 0.25 |
| `MatcherConfig` | `face_weight`, `reid_weight`, `red_threshold`, `yellow_threshold` | 0.40, 0.30, 0.78, 0.55 |
| `FeatureConfig` | `face_model`, `reid_model_name`, `face_det_thresh` | buffalo_l, osnet_x1_0, 0.5 |
| `AlertConfig` | `cooldown_seconds`, `max_alerts_per_minute` | 30, 10 |
| `TemporalConfig` | `min_travel_time`, `max_travel_speed`, `temporal_window` | 5s, 15 m/s, 300s |
| `PipelineConfig` | `process_every_n_frames`, `enable_face`, `enable_reid` | 2, True, True |

---

## API Endpoints (for the FastAPI server)

| Method | Path | What it does |
|--------|------|-------------|
| `GET` | `/health` | Is the pipeline running? |
| `GET` | `/stats` | Frame count, detections, alerts, etc. |
| `GET` | `/alerts?limit=50` | Recent alert history |
| `GET` | `/suspects` | Enrolled suspect count |
| `GET` | `/suspects/active` | Currently RED/YELLOW tracked suspects |
| `POST` | `/suspects/enroll` | Enroll a suspect (send `{"image_path": "..."}`) |
| `POST` | `/cameras` | Add an RTSP camera |
| `GET` | `/cameras` | List all cameras |
| `DELETE` | `/cameras/{id}` | Remove a camera |
| `WS` | `/ws` | WebSocket — receives real-time alert JSON |

---

## Learning Path

If you want to understand the code:

| Step | Read these files | What you'll learn |
|------|-----------------|-------------------|
| 1 | `config.py`, `config.yaml` | How settings work |
| 2 | `core/detector.py` | How YOLO detects people |
| 3 | `core/tracker.py` | How tracking assigns IDs |
| 4 | `core/feature_extractor.py` | How face/body features work |
| 5 | `core/matcher.py` | How FAISS searches for matches |
| 6 | `core/state_engine.py` | How states are classified |
| 7 | `pipeline/orchestrator.py` | How everything connects |
| 8 | `tests/` | How each module is tested |
| 9 | `scripts/run_camera.py` | End-to-end flow in action |

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `sabnetra enroll <images> [--videos ...] [--case CASE] [--id ID]` | Enroll suspect from image(s)/video(s) |
| `sabnetra add-camera <url> [--name NAME]` | Register an RTSP camera |
| `sabnetra start <url> [--name NAME]` | Add camera and begin monitoring |
| `sabnetra stats` | Show live pipeline statistics |
| `python -m sabnetra_ai --help` | Show full CLI help |

---

## Notes for Developers

- All embeddings are L2-normalized before storage and matching. FAISS uses inner product = cosine similarity.
- Track memory consolidates multiple observations via mean-pooling for more robust matching.
- The gait module is disabled by default (`opengaitext` is a placeholder).
- Detections with width > height or area < 4000 px are filtered as noise.
- RED state is locked per track once set — no re-evaluation to prevent flickering.
- Cross-camera matching uses temporal gating (min/max travel time).
- The ModelManager is a singleton — all cameras share the same loaded models.
- 115+ tests cover all modules. Run with `python -m pytest tests/`.
