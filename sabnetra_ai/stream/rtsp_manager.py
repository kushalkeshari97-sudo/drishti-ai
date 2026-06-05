import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, List
from collections import OrderedDict

from sabnetra_ai.core.frame_buffer import FrameBufferManager, Frame

logger = logging.getLogger(__name__)


class RTSPStream:
    def __init__(self, url: str, camera_id: str,
                 buffer_manager: FrameBufferManager,
                 reconnect_delay: float = 3.0,
                 max_reconnect_attempts: int = 10):
        self.url = url
        self.camera_id = camera_id
        self.buffer_manager = buffer_manager
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0
        self._fps = 0.0
        self._last_frame_time = 0.0
        self._lock = threading.Lock()
        self._on_frame_callbacks: List[Callable] = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"RTSP-{self.camera_id}",
            daemon=True)
        self._thread.start()
        logger.info(f"Started RTSP stream: {self.camera_id}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        logger.info(f"Stopped RTSP stream: {self.camera_id}")

    def _capture_loop(self):
        attempts = 0
        while self._running and attempts < self.max_reconnect_attempts:
            try:
                self._cap = cv2.VideoCapture(self.url)
                if not self._cap.isOpened():
                    raise ConnectionError(f"Failed to open RTSP: {self.url}")
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self._cap.set(cv2.CAP_PROP_FPS, 15)
                attempts = 0
                logger.info(f"Connected to {self.camera_id}")
                self._read_loop()
            except Exception as e:
                logger.error(f"Stream {self.camera_id} error: {e}")
                attempts += 1
                if attempts < self.max_reconnect_attempts:
                    time.sleep(self.reconnect_delay)
        logger.warning(f"Stream {self.camera_id} terminated after "
                       f"{attempts} reconnection attempts")

    def _read_loop(self):
        frame_interval = 1.0 / 15.0
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"Frame read failed for {self.camera_id}")
                break
            now = time.time()
            if now - self._last_frame_time < frame_interval * 0.8:
                continue
            self._last_frame_time = now
            self._frame_count += 1
            self.buffer_manager.push_frame(self.camera_id, frame)
            for cb in self._on_frame_callbacks:
                try:
                    cb(frame, self.camera_id)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

    def register_frame_callback(self, callback: Callable):
        self._on_frame_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return self._frame_count / max(time.time() - self._last_frame_time, 0.001)

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "connected": self.is_connected,
            "frames_captured": self._frame_count,
            "fps": self.fps,
        }


class RTSPManager:
    def __init__(self, buffer_manager: FrameBufferManager):
        self.buffer_manager = buffer_manager
        self._streams: OrderedDict[str, RTSPStream] = OrderedDict()

    def add_stream(self, url: str, camera_id: str,
                   reconnect_delay: float = 3.0,
                   max_reconnect_attempts: int = 10) -> RTSPStream:
        if camera_id in self._streams:
            logger.warning(f"Stream {camera_id} already exists, replacing")
            self.remove_stream(camera_id)
        self.buffer_manager.register_camera(camera_id)
        stream = RTSPStream(
            url=url, camera_id=camera_id,
            buffer_manager=self.buffer_manager,
            reconnect_delay=reconnect_delay,
            max_reconnect_attempts=max_reconnect_attempts,
        )
        self._streams[camera_id] = stream
        logger.info(f"Added stream: {camera_id} -> {url}")
        return stream

    def remove_stream(self, camera_id: str):
        if camera_id in self._streams:
            self._streams[camera_id].stop()
            del self._streams[camera_id]
            self.buffer_manager.unregister_camera(camera_id)

    def start_all(self):
        for stream in self._streams.values():
            stream.start()

    def stop_all(self):
        for stream in list(self._streams.values()):
            stream.stop()

    def get_stream(self, camera_id: str) -> Optional[RTSPStream]:
        return self._streams.get(camera_id)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._streams.values() if s.is_connected)

    def stats(self) -> dict:
        return {cid: s.stats() for cid, s in self._streams.items()}
