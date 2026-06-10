from unittest.mock import patch, MagicMock
from sabnetra_ai.models.model_manager import ModelManager


def _reset_model_manager():
    ModelManager._instances.pop(ModelManager, None)


def test_model_manager_singleton():
    _reset_model_manager()
    m1 = ModelManager(device="cpu", face_det_thresh=0.5)
    m2 = ModelManager(device="cpu", face_det_thresh=0.5)
    assert m1 is m2


def test_model_manager_default_device():
    _reset_model_manager()
    m = ModelManager(device="cuda:0", face_det_thresh=0.5)
    import torch
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert m.device == expected


def test_model_manager_init_once():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5, model_path="yolo.pt")
    assert m._model_path == "yolo.pt"
    assert not m._gait_enabled


def test_model_manager_gait_enabled():
    _reset_model_manager()
    m = ModelManager(device="cpu", gait_enabled=True, face_det_thresh=0.5)
    assert m._gait_enabled


def test_detector_returns_none_after_failed_load():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5)
    from sabnetra_ai.models.model_manager import _FAILED
    m._detector = _FAILED
    assert m.detector is None


def test_face_model_returns_none_after_failed_load():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5)
    from sabnetra_ai.models.model_manager import _FAILED
    m._face_model = _FAILED
    assert m.face_model is None


def test_reid_model_returns_none_after_failed_load():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5)
    from sabnetra_ai.models.model_manager import _FAILED
    m._reid_model = _FAILED
    assert m.reid_model is None


def test_gait_model_returns_none_after_failed_load():
    _reset_model_manager()
    m = ModelManager(device="cpu", gait_enabled=True, face_det_thresh=0.5)
    from sabnetra_ai.models.model_manager import _FAILED
    m._gait_model = _FAILED
    assert m.gait_model is None


def test_gait_model_none_when_disabled():
    _reset_model_manager()
    m = ModelManager(device="cpu", gait_enabled=False, face_det_thresh=0.5)
    assert m.gait_model is None


def test_gait_model_none_when_enabled_but_unavailable():
    _reset_model_manager()
    m = ModelManager(device="cpu", gait_enabled=True, face_det_thresh=0.5)
    result = m.gait_model
    assert result is None


def test_release_clears_models():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5)
    m._detector = MagicMock()
    m._face_model = MagicMock()
    m._reid_model = MagicMock()
    m.release()
    assert m._detector is None
    assert m._face_model is None
    assert m._reid_model is None


def test_release_with_cuda():
    _reset_model_manager()
    m = ModelManager(device="cpu", face_det_thresh=0.5)
    with patch("torch.cuda.empty_cache") as mock_clear:
        m.release()
    import torch
    if torch.cuda.is_available():
        mock_clear.assert_called_once()
