import numpy as np
from unittest.mock import MagicMock, patch
from sabnetra_ai.config import SabNetraConfig
from sabnetra_ai.pipeline.orchestrator import SabNetraPipeline, CameraPipeline


def test_pipeline_init():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    assert not p.is_running
    assert p.config is config
    assert len(p.camera_pipelines) == 0


def test_pipeline_add_camera():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "add_stream") as mock_add:
        mock_add.return_value = MagicMock()
        ok = p.add_camera("rtsp://localhost/stream", "cam0")
        assert ok
        assert "cam0" in p.camera_pipelines


def test_pipeline_add_camera_duplicate():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "add_stream") as mock_add:
        mock_add.return_value = MagicMock()
        p.add_camera("rtsp://first/stream", "cam0")
        p.add_camera("rtsp://second/stream", "cam0")
        assert len(p.camera_pipelines) == 1


def test_pipeline_max_cameras_enforced():
    config = SabNetraConfig()
    config.pipeline.max_cameras = 2
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "add_stream") as mock_add:
        mock_add.return_value = MagicMock()
        assert p.add_camera("rtsp://a/stream", "cam0")
        assert p.add_camera("rtsp://b/stream", "cam1")
        assert not p.add_camera("rtsp://c/stream", "cam2")


def test_pipeline_remove_camera():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "add_stream") as mock_add, \
         patch.object(p.rtsp_manager, "remove_stream") as mock_rm:
        mock_add.return_value = MagicMock()
        p.add_camera("rtsp://host/stream", "cam0")
        p.remove_camera("cam0")
        assert "cam0" not in p.camera_pipelines
        mock_rm.assert_called_once_with("cam0")


def test_pipeline_start_stop():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "start_all"), \
         patch.object(p.model_manager, "warmup"), \
         patch.object(p.alert_system, "register_callback"):
        p.start()
        assert p.is_running
        p.stop()
        assert not p.is_running


def test_pipeline_double_start():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    with patch.object(p.rtsp_manager, "start_all"), \
         patch.object(p.model_manager, "warmup"), \
         patch.object(p.alert_system, "register_callback"):
        p.start()
        p.start()
        assert p.is_running
        p.stop()


def test_stats_structure():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    s = p.stats()
    assert "pipeline" in s
    assert "memory" in s
    assert "alerts" in s
    assert "suspects" in s
    assert "cross_camera" in s
    assert "streams" in s


def test_stats_contains_counters():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    s = p.stats()
    assert "frames_processed" in s["pipeline"]
    assert "detections" in s["pipeline"]
    assert "matches" in s["pipeline"]
    assert "alerts" in s["pipeline"]


def test_register_callbacks():
    config = SabNetraConfig()
    p = SabNetraPipeline(config)
    cb = lambda x: None
    p.register_detection_callback(cb)
    p.register_alert_callback(cb)
    p.register_frame_callback(cb)
    assert len(p._on_detection_callbacks) == 1
    assert len(p._on_alert_callbacks) == 1
    assert len(p._on_frame_callbacks) == 1


def test_camera_pipeline_process_frame_skip():
    config = SabNetraConfig()
    config.pipeline.process_every_n_frames = 5
    pipe = CameraPipeline("cam0", config, MagicMock(), MagicMock(), MagicMock())
    from sabnetra_ai.core.detector import Detection
    det = Detection(np.array([0, 0, 50, 100], dtype=float), 0.9, 0)
    pipe.detector.detect.return_value = [det]
    track_mock = MagicMock()
    track_mock.bbox = np.array([0, 0, 50, 100], dtype=float)
    track_mock.track_id = 1
    track_mock.confidence = 0.9
    pipe.tracker.update.return_value = [track_mock]
    pipe.feature_extractor.extract_all.return_value = MagicMock()
    pipe.feature_extractor.extract_all.return_value.body_embedding = np.zeros(512)
    pipe.feature_extractor.extract_all.return_value.has_any_embedding.return_value = True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipe.process_frame(frame, 1.0)
    assert result == []
    result = pipe.process_frame(frame, 2.0)
    assert result == []
    result = pipe.process_frame(frame, 3.0)
    assert result == []
    result = pipe.process_frame(frame, 4.0)
    assert result == []
    result = pipe.process_frame(frame, 5.0)
    assert result != []


def test_camera_pipeline_frame_rate_counter():
    config = SabNetraConfig()
    config.pipeline.process_every_n_frames = 1
    pipe = CameraPipeline("cam0", config, MagicMock(), MagicMock(), MagicMock())
    pipe.detector.detect.return_value = []
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipe.process_frame(frame, 1.0)
    assert pipe.frame_rate_counter.fps >= 0


def test_camera_pipeline_detect_and_track():
    config = SabNetraConfig()
    config.pipeline.process_every_n_frames = 1
    pipe = CameraPipeline("cam0", config, MagicMock(), MagicMock(), MagicMock())
    from sabnetra_ai.core.detector import Detection
    det = Detection(np.array([0, 0, 50, 100], dtype=float), 0.9, 0)
    pipe.detector.detect.return_value = [det]
    track_mock = MagicMock()
    track_mock.bbox = np.array([0, 0, 50, 100], dtype=float)
    track_mock.track_id = 1
    track_mock.confidence = 0.9
    pipe.tracker.update.return_value = [track_mock]
    pipe.feature_extractor.extract_all.return_value = MagicMock()
    pipe.feature_extractor.extract_all.return_value.body_embedding = np.zeros(512)
    pipe.feature_extractor.extract_all.return_value.has_any_embedding.return_value = True
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    results = pipe.process_frame(frame, 1.0)
    assert len(results) == 1
    assert results[0]["track_id"] == 1
