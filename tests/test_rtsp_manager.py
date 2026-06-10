import threading
import time
from unittest.mock import MagicMock, patch
from sabnetra_ai.stream.rtsp_manager import RTSPStream, RTSPManager
from sabnetra_ai.core.frame_buffer import FrameBufferManager


def test_rtsp_stream_init():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    assert stream.camera_id == "cam0"
    assert stream.url == "rtsp://host/stream"
    assert not stream._running
    assert stream._cap is None
    assert stream.max_reconnect_attempts == 10


def test_rtsp_stream_start_stop():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    with patch.object(stream, "_capture_loop"):
        stream.start()
        assert stream._running
        stream.stop()
        assert not stream._running


def test_rtsp_stream_double_start():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    with patch.object(stream, "_capture_loop"):
        stream.start()
        stream.start()
        stream.stop()


def test_rtsp_stream_not_connected_initially():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    assert not stream.is_connected


def test_rtsp_stream_fps_zero_initially():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    assert stream.fps == 0.0


def test_rtsp_stream_stats():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    s = stream.stats()
    assert s["camera_id"] == "cam0"
    assert not s["connected"]
    assert s["frames_captured"] == 0


def test_rtsp_stream_register_frame_callback():
    buf_mgr = FrameBufferManager()
    stream = RTSPStream("rtsp://host/stream", "cam0", buf_mgr)
    cb = lambda f, c: None
    stream.register_frame_callback(cb)
    assert len(stream._on_frame_callbacks) == 1


def test_rtsp_manager_init():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    assert mgr.active_count == 0
    assert mgr.stats() == {}


def test_rtsp_manager_add_stream():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        mock_instance = MagicMock()
        mock_stream.return_value = mock_instance
        stream = mgr.add_stream("rtsp://host/stream", "cam0")
        assert "cam0" in mgr._streams


def test_rtsp_manager_add_stream_replaces_duplicate():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        mock_instance = MagicMock()
        mock_stream.return_value = mock_instance
        mgr.add_stream("rtsp://first/stream", "cam0")
        mgr.add_stream("rtsp://second/stream", "cam0")
        assert len(mgr._streams) == 1


def test_rtsp_manager_remove_stream():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        mock_instance = MagicMock()
        mock_stream.return_value = mock_instance
        mgr.add_stream("rtsp://host/stream", "cam0")
        mgr.remove_stream("cam0")
        assert "cam0" not in mgr._streams
        mock_instance.stop.assert_called_once()


def test_rtsp_manager_start_all():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        inst1 = MagicMock()
        inst2 = MagicMock()
        mock_stream.side_effect = [inst1, inst2]
        mgr.add_stream("rtsp://a/stream", "cam0")
        mgr.add_stream("rtsp://b/stream", "cam1")
        mgr.start_all()
        inst1.start.assert_called_once()
        inst2.start.assert_called_once()


def test_rtsp_manager_stop_all():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        inst1 = MagicMock()
        mock_stream.return_value = inst1
        mgr.add_stream("rtsp://host/stream", "cam0")
        mgr.stop_all()
        inst1.stop.assert_called_once()


def test_rtsp_manager_get_stream():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        mock_instance = MagicMock()
        mock_stream.return_value = mock_instance
        mgr.add_stream("rtsp://host/stream", "cam0")
        assert mgr.get_stream("cam0") is mock_instance
        assert mgr.get_stream("nonexistent") is None


def test_rtsp_manager_active_count():
    buf_mgr = FrameBufferManager()
    mgr = RTSPManager(buf_mgr)
    with patch("sabnetra_ai.stream.rtsp_manager.RTSPStream") as mock_stream:
        inst1 = MagicMock()
        inst1.is_connected = True
        inst2 = MagicMock()
        inst2.is_connected = False
        mock_stream.side_effect = [inst1, inst2]
        mgr.add_stream("rtsp://a/stream", "cam0")
        mgr.add_stream("rtsp://b/stream", "cam1")
        assert mgr.active_count == 1
