import cv2
import numpy as np
import threading
import time
import logging
from collections import OrderedDict, deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Frame:
    __slots__ = ("data", "timestamp", "frame_id", "camera_id", "metadata")

    def __init__(self, data: np.ndarray, frame_id: int, camera_id: str,
                 timestamp: Optional[float] = None):
        self.data = data
        self.timestamp = timestamp or time.time()
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.metadata = {}

    def release(self):
        self.data = None


class FrameBuffer:
    def __init__(self, maxsize: int = 4, camera_id: str = "cam_0"):
        self.maxsize = maxsize
        self.camera_id = camera_id
        self._buffer = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._latest_frame_id = -1
        self._dropped_frames = 0
        self._processed_frames = 0

    def push(self, frame: np.ndarray) -> Optional[Frame]:
        wrapped = Frame(
            data=frame,
            frame_id=self._frame_counter,
            camera_id=self.camera_id,
        )
        self._frame_counter += 1
        with self._lock:
            if len(self._buffer) >= self.maxsize:
                self._dropped_frames += 1
            self._buffer.append(wrapped)
            self._latest_frame_id = wrapped.frame_id
        return wrapped

    def pop(self) -> Optional[Frame]:
        with self._lock:
            if not self._buffer:
                return None
            frame = self._buffer.popleft()
            self._processed_frames += 1
            return frame

    def peek(self) -> Optional[Frame]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return self.size >= self.maxsize

    def clear(self):
        with self._lock:
            for f in self._buffer:
                f.release()
            self._buffer.clear()

    @property
    def drop_rate(self) -> float:
        total = self._frame_counter
        if total == 0:
            return 0.0
        return self._dropped_frames / total

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "buffer_size": self.size,
            "maxsize": self.maxsize,
            "total_frames": self._frame_counter,
            "dropped": self._dropped_frames,
            "processed": self._processed_frames,
            "drop_rate": self.drop_rate,
        }


class FrameBufferManager:
    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._buffers: OrderedDict[str, FrameBuffer] = OrderedDict()
        self._lock = threading.Lock()

    def register_camera(self, camera_id: str) -> FrameBuffer:
        with self._lock:
            if camera_id not in self._buffers:
                buffer = FrameBuffer(maxsize=self.maxsize, camera_id=camera_id)
                self._buffers[camera_id] = buffer
                logger.info(f"Registered camera buffer: {camera_id}")
            return self._buffers[camera_id]

    def unregister_camera(self, camera_id: str):
        with self._lock:
            if camera_id in self._buffers:
                self._buffers[camera_id].clear()
                del self._buffers[camera_id]
                logger.info(f"Unregistered camera buffer: {camera_id}")

    def get_buffer(self, camera_id: str) -> Optional[FrameBuffer]:
        with self._lock:
            return self._buffers.get(camera_id)

    def push_frame(self, camera_id: str, frame: np.ndarray) -> Optional[Frame]:
        buffer = self.get_buffer(camera_id)
        if buffer is None:
            buffer = self.register_camera(camera_id)
        return buffer.push(frame)

    def pop_frame(self, camera_id: str) -> Optional[Frame]:
        buffer = self.get_buffer(camera_id)
        if buffer is None:
            return None
        return buffer.pop()

    def all_buffers(self) -> list:
        with self._lock:
            return list(self._buffers.values())

    def camera_ids(self) -> list:
        with self._lock:
            return list(self._buffers.keys())

    def clear_all(self):
        with self._lock:
            for buf in self._buffers.values():
                buf.clear()
            self._buffers.clear()

    def global_stats(self) -> dict:
        stats = {}
        with self._lock:
            for cid, buf in self._buffers.items():
                stats[cid] = buf.stats()
        return stats
