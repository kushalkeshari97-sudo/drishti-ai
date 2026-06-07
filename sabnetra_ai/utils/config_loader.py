import os
from typing import Optional
from sabnetra_ai.config import SabNetraConfig


def _load_yaml(path: str, config: SabNetraConfig):
    try:
        import yaml
    except ImportError:
        return
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        return

    det = data.get("detector", {})
    for key in ("confidence_threshold", "img_size", "iou_threshold",
                 "half_precision", "max_det", "batch_size"):
        if key in det:
            setattr(config.detector, key, det[key])
    if "model_path" in det:
        config.detector.model_path = det["model_path"]
    if "device" in det:
        config.detector.device = det["device"]

    trk = data.get("tracker", {})
    for key in ("match_thresh", "new_track_thresh", "max_time_lost",
                 "appearance_weight", "motion_weight", "min_hits",
                 "track_buffer", "proximity_thresh"):
        if key in trk:
            setattr(config.tracker, key, trk[key])

    mat = data.get("matcher", {})
    for key in ("face_weight", "reid_weight", "clothing_weight",
                 "red_threshold", "yellow_threshold", "temporal_consistency_frames",
                 "top_k", "faiss_nprobe"):
        if key in mat:
            setattr(config.matcher, key, mat[key])

    feat = data.get("features", {})
    if "face_det_thresh" in feat:
        config.features.face_det_thresh = feat["face_det_thresh"]

    pip = data.get("pipeline", {})
    for key in ("enable_reid", "enable_gait", "enable_face_recognition",
                 "process_every_n_frames", "max_cameras", "visualize"):
        if key in pip:
            setattr(config.pipeline, key, pip[key])


def _load_env(config: SabNetraConfig):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    rtsp = os.environ.get("SABNETRA_RTSP_URL")
    if rtsp:
        config._rtsp_url = rtsp
    model = os.environ.get("SABNETRA_MODEL_PATH")
    if model:
        config.detector.model_path = model
    device = os.environ.get("SABNETRA_DEVICE")
    if device:
        config.detector.device = device
        config.device = device


def load_config(path: Optional[str] = None) -> SabNetraConfig:
    config = SabNetraConfig()
    if path is None:
        path = os.environ.get("SABNETRA_CONFIG", "config.yaml")
    _load_yaml(path, config)
    _load_yaml("config.local.yaml", config)
    _load_env(config)
    return config
