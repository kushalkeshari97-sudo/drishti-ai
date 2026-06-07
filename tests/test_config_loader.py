import os
import tempfile
import yaml
from sabnetra_ai.utils.config_loader import load_config
from sabnetra_ai.config import SabNetraConfig


def test_load_config_defaults():
    config = SabNetraConfig()
    assert config.detector.confidence_threshold == 0.35
    assert config.matcher.face_weight == 0.40


def test_load_config_from_yaml():
    data = {
        "detector": {"confidence_threshold": 0.50, "img_size": 320},
        "matcher": {"face_weight": 0.6, "red_threshold": 0.70},
        "tracker": {"match_thresh": 0.35},
        "pipeline": {"enable_reid": False},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    config = load_config(path)
    assert config.detector.confidence_threshold == 0.50
    assert config.detector.img_size == 320
    assert config.matcher.face_weight == 0.6
    assert config.matcher.red_threshold == 0.70
    assert config.tracker.match_thresh == 0.35
    assert not config.pipeline.enable_reid
    os.unlink(path)


def test_load_config_from_env():
    os.environ["SABNETRA_MODEL_PATH"] = "custom_model.pt"
    os.environ["SABNETRA_DEVICE"] = "cpu"
    config = load_config()
    assert config.detector.model_path == "custom_model.pt"
    assert config.detector.device == "cpu"
    del os.environ["SABNETRA_MODEL_PATH"]
    del os.environ["SABNETRA_DEVICE"]
