import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.55
    track_low_thresh: float = 0.12
    new_track_thresh: float = 0.65
    track_buffer: int = 120
    match_thresh: float = 0.28
    min_box_area: int = 220
    frame_rate: int = 15
    appearance_weight: float = 0.75
    motion_weight: float = 0.25
    min_hits: int = 3
    max_time_lost: int = 60
    proximity_thresh: float = 0.35
    occlusion_angle_thresh: float = 30.0
    reid_match_thresh: float = 0.45
    fragmented_track_merge_iou: float = 0.15
    fragmented_track_embedding_thresh: float = 0.60


@dataclass
class DetectorConfig:
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    img_size: int = 640
    half_precision: bool = True
    max_det: int = 300
    device: str = "cuda:0"
    classes: list = field(default_factory=lambda: [0])
    batch_size: int = 1
    skip_frame_on_low_light: bool = True
    low_light_threshold: float = 30.0


@dataclass
class FeatureConfig:
    face_model: str = "buffalo_l"
    face_det_thresh: float = 0.5
    reid_model_name: str = "osnet_x1_0"
    reid_input_size: tuple = (256, 128)
    gait_enabled: bool = False
    gait_sequence_length: int = 30
    clothing_feature_dim: int = 256
    embedding_cache_ttl: int = 30
    face_embedding_dim: int = 512
    reid_embedding_dim: int = 512


@dataclass
class MatcherConfig:
    face_weight: float = 0.40
    reid_weight: float = 0.30
    clothing_weight: float = 0.15
    gait_weight: float = 0.15
    yellow_threshold: float = 0.55
    red_threshold: float = 0.78
    temporal_consistency_frames: int = 5
    temporal_consistency_ratio: float = 0.6
    cross_camera_face_weight: float = 0.50
    cross_camera_reid_weight: float = 0.35
    cross_camera_clothing_weight: float = 0.15
    faiss_nprobe: int = 10
    faiss_nlist: int = 100
    top_k: int = 5


@dataclass
class TemporalConfig:
    min_travel_time: float = 5.0
    max_travel_speed: float = 15.0
    location_gate_threshold: float = 0.35
    temporal_window: float = 300.0
    max_camera_distance_m: float = 500.0


@dataclass
class AlertConfig:
    cooldown_seconds: float = 30.0
    max_alerts_per_minute: int = 10
    alert_on_yellow: bool = False
    alert_on_red: bool = True


@dataclass
class PipelineConfig:
    max_cameras: int = 16
    frame_buffer_size: int = 4
    process_every_n_frames: int = 2
    skip_frame_on_low_light: bool = True
    low_light_threshold: float = 30.0
    enable_face_recognition: bool = True
    enable_reid: bool = True
    enable_gait: bool = False
    enable_clothing: bool = True
    visualize: bool = False
    log_detections: bool = True


@dataclass
class SabNetraConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    suspect_db_path: str = "suspect_embeddings"
    device: str = "cuda:0"
    verbose: bool = False
