import numpy as np
from unittest.mock import MagicMock, patch
from sabnetra_ai.enrollment.fir_module import FIRInputModule
from sabnetra_ai.config import FeatureConfig


def test_enroll_from_image_missing_file():
    module = FIRInputModule()
    result = module.enroll_from_image("/nonexistent/path.jpg", "C1")
    assert result is None


def test_enroll_from_image_no_features():
    module = FIRInputModule()
    module.feature_extractor = MagicMock()
    module.feature_extractor.extract_all.return_value = MagicMock()
    module.feature_extractor.extract_all.return_value.has_any_embedding.return_value = False
    with patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = module.enroll_from_image("img.jpg", "C1")
        assert result is None


def test_enroll_from_image_success():
    module = FIRInputModule()
    module.feature_extractor = MagicMock()
    fake_feats = MagicMock()
    fake_feats.has_any_embedding.return_value = True
    fake_feats.face_embedding = np.array([1.0, 0.0, 0.0])
    fake_feats.body_embedding = None
    fake_feats.clothing_descriptor = None
    fake_feats.gait_descriptor = None
    module.feature_extractor.extract_all.return_value = fake_feats
    with patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = module.enroll_from_image("img.jpg", "C1", suspect_id="S1")
        assert result is not None
        assert result.suspect_id == "S1"
        assert result.case_id == "C1"
        assert result.face_embedding is not None


def test_enroll_from_image_generates_id():
    module = FIRInputModule()
    module.feature_extractor = MagicMock()
    fake_feats = MagicMock()
    fake_feats.has_any_embedding.return_value = True
    fake_feats.face_embedding = np.array([1.0, 0.0, 0.0])
    fake_feats.body_embedding = None
    fake_feats.clothing_descriptor = None
    fake_feats.gait_descriptor = None
    module.feature_extractor.extract_all.return_value = fake_feats
    with patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = module.enroll_from_image("img.jpg")
        assert result is not None
        assert result.suspect_id.startswith("S")


def test_enroll_from_video_missing_file():
    module = FIRInputModule()
    result = module.enroll_from_video("/nonexistent/video.mp4", "C1")
    assert result is None


def test_enroll_from_video_no_features():
    module = FIRInputModule()
    module.feature_extractor = MagicMock()
    module.feature_extractor.extract_all.return_value = MagicMock()
    module.feature_extractor.extract_all.return_value.has_any_embedding.return_value = True
    module.feature_extractor.extract_all.return_value.face_embedding = None
    module.feature_extractor.extract_all.return_value.body_embedding = None
    cap_mock = MagicMock()
    cap_mock.read.side_effect = [(False, None)]
    cap_mock.isOpened.return_value = True
    with patch("os.path.exists", return_value=True), \
         patch("cv2.VideoCapture", return_value=cap_mock):
        result = module.enroll_from_video("video.mp4", "C1")
        assert result is None


def test_enroll_fir_fuses_multiple_images():
    module = FIRInputModule()
    module.feature_extractor = MagicMock()
    fake_feats = MagicMock()
    fake_feats.has_any_embedding.return_value = True
    fake_feats.face_embedding = np.array([1.0, 0.0, 0.0])
    fake_feats.body_embedding = None
    fake_feats.clothing_descriptor = None
    fake_feats.gait_descriptor = None
    module.feature_extractor.extract_all.return_value = fake_feats
    with patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = module.enroll_fir(images=["img1.jpg", "img2.jpg"], case_id="C1")
        assert result is not None
        assert result.case_id == "C1"
        assert result.face_embedding is not None


def test_enroll_fir_no_sources():
    module = FIRInputModule()
    result = module.enroll_fir(images=[])
    assert result is None


def test_generate_suspect_id():
    module = FIRInputModule()
    id1 = module._generate_suspect_id()
    id2 = module._generate_suspect_id()
    assert id1 != id2
    assert id1.startswith("S")
    assert id2.startswith("S")


def test_fuse_embeddings_single():
    module = FIRInputModule()
    emb = np.array([1.0, 2.0, 3.0])
    result = module._fuse_embeddings([emb])
    assert np.array_equal(result, emb)


def test_fuse_embeddings_multiple():
    module = FIRInputModule()
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0])
    result = module._fuse_embeddings([emb1, emb2])
    expected = np.array([0.5, 0.5])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(result, expected)


def test_fuse_embeddings_empty():
    module = FIRInputModule()
    assert module._fuse_embeddings([]) is None
