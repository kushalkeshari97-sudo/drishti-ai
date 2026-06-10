import numpy as np
from sabnetra_ai.config import (
    SabNetraConfig, DetectorConfig, TrackerConfig,
    MatcherConfig, FeatureConfig, AlertConfig,
    PipelineConfig, TemporalConfig,
)


def test_sabnetra_config_defaults():
    c = SabNetraConfig()
    assert isinstance(c.detector, DetectorConfig)
    assert isinstance(c.tracker, TrackerConfig)
    assert isinstance(c.matcher, MatcherConfig)
    assert isinstance(c.features, FeatureConfig)
    assert isinstance(c.alert, AlertConfig)
    assert isinstance(c.pipeline, PipelineConfig)
    assert isinstance(c.temporal, TemporalConfig)


def test_detector_config_defaults():
    c = DetectorConfig()
    assert c.confidence_threshold == 0.35
    assert c.iou_threshold == 0.45
    assert c.img_size == 640
    assert c.device == "cuda:0"
    assert c.classes == [0]
    assert c.half_precision


def test_tracker_config_defaults():
    c = TrackerConfig()
    assert c.match_thresh == 0.28
    assert c.max_time_lost == 60
    assert c.appearance_weight == 0.75
    assert c.motion_weight == 0.25
    assert c.min_hits == 3


def test_matcher_config_defaults():
    c = MatcherConfig()
    assert c.face_weight == 0.40
    assert c.reid_weight == 0.30
    assert c.clothing_weight == 0.15
    assert c.gait_weight == 0.15
    assert c.yellow_threshold == 0.55
    assert c.red_threshold == 0.78


def test_feature_config_defaults():
    c = FeatureConfig()
    assert c.face_model == "buffalo_l"
    assert c.reid_model_name == "osnet_x1_0"
    assert c.face_det_thresh == 0.5
    assert not c.gait_enabled


def test_alert_config_defaults():
    c = AlertConfig()
    assert c.cooldown_seconds == 30.0
    assert c.max_alerts_per_minute == 10
    assert not c.alert_on_yellow
    assert c.alert_on_red


def test_pipeline_config_defaults():
    c = PipelineConfig()
    assert c.max_cameras == 16
    assert c.frame_buffer_size == 4
    assert c.process_every_n_frames == 2
    assert c.enable_face_recognition
    assert c.enable_reid


def test_temporal_config_defaults():
    c = TemporalConfig()
    assert c.min_travel_time == 5.0
    assert c.max_travel_speed == 15.0
    assert c.temporal_window == 300.0


def test_config_override():
    c = SabNetraConfig()
    c.detector.confidence_threshold = 0.55
    c.matcher.red_threshold = 0.45
    c.tracker.match_thresh = 0.85
    assert c.detector.confidence_threshold == 0.55
    assert c.matcher.red_threshold == 0.45
    assert c.tracker.match_thresh == 0.85


def test_detector_config_validation():
    DetectorConfig(confidence_threshold=0.5)
    import pytest
    with pytest.raises(AssertionError):
        DetectorConfig(confidence_threshold=0.0)
    with pytest.raises(AssertionError):
        DetectorConfig(img_size=-1)
    with pytest.raises(AssertionError):
        DetectorConfig(max_det=0)


def test_feature_config_validation():
    FeatureConfig(face_det_thresh=0.5)
    import pytest
    with pytest.raises(AssertionError):
        FeatureConfig(face_det_thresh=0.0)
    with pytest.raises(AssertionError):
        FeatureConfig(gait_sequence_length=0)
    with pytest.raises(AssertionError):
        FeatureConfig(embedding_cache_ttl=-1)
