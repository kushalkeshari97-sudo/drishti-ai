import torch
import logging

logger = logging.getLogger(__name__)

class _Sentinel: pass
_FAILED = _Sentinel()


class ModelManager:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self, device: str = "cuda:0", gait_enabled: bool = False,
                 face_det_thresh: float = 0.5, model_path: str = "yolov8n.pt"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.device = device if torch.cuda.is_available() else "cpu"
        self._detector = None
        self._face_model = None
        self._reid_model = None
        self._gait_model = None
        self._gait_enabled = gait_enabled
        self._face_det_thresh = face_det_thresh
        self._model_path = model_path
        self._initialized = True
        logger.info(f"ModelManager initialized on device: {self.device}")

    @property
    def detector(self):
        if self._detector is None:
            self._load_detector()
        return self._detector if self._detector is not _FAILED else None

    @property
    def face_model(self):
        if self._face_model is None:
            self._load_face_model()
        return self._face_model if self._face_model is not _FAILED else None

    @property
    def reid_model(self):
        if self._reid_model is None:
            self._load_reid_model()
        return self._reid_model if self._reid_model is not _FAILED else None

    @property
    def gait_model(self):
        if not self._gait_enabled:
            return None
        if self._gait_model is None:
            self._load_gait_model()
        return self._gait_model if self._gait_model is not _FAILED else None

    def _load_detector(self):
        try:
            from ultralytics import YOLO
            self._detector = YOLO(self._model_path)
            if self.device != "cpu":
                self._detector.to(self.device)
            logger.info(f"Detector loaded: {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8s: {e}")
            self._detector = _FAILED
            raise

    def _load_face_model(self):
        try:
            from insightface.app import FaceAnalysis
            self._face_model = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device != "cpu"
                else ["CPUExecutionProvider"],
            )
            self._face_model.prepare(ctx_id=0 if self.device != "cpu" else -1, det_thresh=self._face_det_thresh)
            logger.info("InsightFace buffalo_l model loaded")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}")
            self._face_model = _FAILED

    def _load_reid_model(self):
        try:
            from torchreid.reid.utils import FeatureExtractor
            self._reid_model = FeatureExtractor(
                model_name="osnet_x1_0",
                model_path=None,
                device=self.device,
            )
            logger.info("OSNet ReID model loaded")
        except Exception as e:
            logger.warning(f"Failed to load OSNet: {e}")
            self._reid_model = _FAILED

    def _load_gait_model(self):
        logger.warning("Gait model not available (opengaitext package missing)")
        self._gait_model = _FAILED

    def warmup(self):
        dummy = torch.zeros((1, 3, 320, 320), device=self.device)
        try:
            self.detector(dummy, verbose=False)
            logger.info("Detector warmed up")
        except Exception as e:
            logger.warning(f"Detector warmup failed: {e}")

    def release(self):
        self._detector = None
        self._face_model = None
        self._reid_model = None
        self._gait_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model resources released")
