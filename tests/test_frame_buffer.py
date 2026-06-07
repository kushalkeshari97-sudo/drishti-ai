import numpy as np
import time
from sabnetra_ai.core.frame_buffer import Frame, FrameBuffer, FrameBufferManager


def test_frame_creation():
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    f = Frame(data, 0, "cam0")
    assert f.frame_id == 0
    assert f.camera_id == "cam0"
    assert f.metadata == {}


def test_frame_timestamp():
    data = np.zeros((10, 10, 3), dtype=np.uint8)
    t = 12345.0
    f = Frame(data, 1, "cam0", timestamp=t)
    assert f.timestamp == t


def test_frame_release():
    data = np.zeros((10, 10, 3), dtype=np.uint8)
    f = Frame(data, 0, "cam0")
    f.release()
    assert f.data is None


def test_buffer_push_pop():
    buf = FrameBuffer(maxsize=4, camera_id="cam0")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    f1 = buf.push(frame)
    assert f1 is not None
    assert f1.frame_id == 0
    f2 = buf.pop()
    assert f2 is not None
    assert f2.frame_id == 0
    assert buf.pop() is None


def test_buffer_maxsize():
    buf = FrameBuffer(maxsize=2, camera_id="cam0")
    for i in range(5):
        buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    assert buf.size == 2
    assert buf.drop_rate > 0


def test_buffer_is_full():
    buf = FrameBuffer(maxsize=2, camera_id="cam0")
    assert not buf.is_full
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    assert not buf.is_full
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    assert buf.is_full


def test_buffer_clear():
    buf = FrameBuffer(maxsize=4, camera_id="cam0")
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    buf.clear()
    assert buf.size == 0
    assert buf.pop() is None


def test_buffer_drop_rate():
    buf = FrameBuffer(maxsize=1, camera_id="cam0")
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    buf.push(np.zeros((10, 10, 3), dtype=np.uint8))
    assert buf.drop_rate > 0


def test_buffer_manager_register():
    mgr = FrameBufferManager(maxsize=4)
    buf = mgr.register_camera("cam0")
    assert buf is not None
    assert buf.camera_id == "cam0"
    assert mgr.get_buffer("cam0") is not None


def test_buffer_manager_push_pop():
    mgr = FrameBufferManager(maxsize=4)
    mgr.register_camera("cam0")
    f = mgr.push_frame("cam0", np.zeros((10, 10, 3), dtype=np.uint8))
    assert f is not None
    popped = mgr.pop_frame("cam0")
    assert popped is not None


def test_buffer_manager_unregister():
    mgr = FrameBufferManager()
    mgr.register_camera("cam0")
    mgr.unregister_camera("cam0")
    assert mgr.get_buffer("cam0") is None
    assert len(mgr.camera_ids()) == 0


def test_buffer_manager_clear_all():
    mgr = FrameBufferManager()
    mgr.register_camera("cam0")
    mgr.register_camera("cam1")
    mgr.clear_all()
    assert len(mgr.camera_ids()) == 0
