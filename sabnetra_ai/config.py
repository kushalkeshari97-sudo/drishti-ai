import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackerConfig:
    """Configuration for the BoTSORT tracker."""

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
    """Configuration for the YOLO-based object detector."""

    model_path: str = "yolov8n.pt"
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
    min_person_area: int = 4000
    nms_threshold: float = 0.3
    min_face_size: int = 20
    min_body_size: int = 30

    def __post_init__(self):
        assert 0 < self.confidence_threshold <= 1
        assert 0 < self.iou_threshold <= 1
        assert self.img_size > 0
        assert self.max_det > 0
        assert self.min_person_area > 0


@dataclass
class FeatureConfig:
    """Configuration for feature extraction models and parameters."""

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

    def __post_init__(self):
        assert 0 < self.face_det_thresh <= 1
        assert self.gait_sequence_length > 0
        assert self.clothing_feature_dim > 0
        assert self.embedding_cache_ttl > 0


@dataclass
class MatcherConfig:
    """Configuration for the matching engine weights and thresholds."""

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
    """Configuration for temporal constraints in cross-camera tracking."""

    min_travel_time: float = 5.0
    max_travel_speed: float = 15.0
    location_gate_threshold: float = 0.35
    temporal_window: float = 300.0
    max_camera_distance_m: float = 500.0


@dataclass
class AlertConfig:
    """Configuration for alert triggering behavior."""

    cooldown_seconds: float = 30.0
    max_alerts_per_minute: int = 10
    alert_on_yellow: bool = False
    alert_on_red: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the main processing pipeline."""

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
    """Top-level configuration aggregating all sub-configs and global settings."""

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
