import cv2
import numpy as np
import os
from sabnetra_ai import create_pipeline, SabNetraConfig
from sabnetra_ai.core.tracker import BoTSORT
from sabnetra_ai.core.matcher import SuspectProfile

config = SabNetraConfig()
config.detector.confidence_threshold = 0.55
config.detector.img_size = 256
config.detector.iou_threshold = 0.3
config.detector.half_precision = True
config.detector.skip_frame_on_low_light = False
config.matcher.red_threshold = 0.35
config.matcher.yellow_threshold = 0.20
config.matcher.face_weight = 1.0
config.matcher.reid_weight = 0.0
config.matcher.temporal_consistency_frames = 3
config.tracker.match_thresh = 0.85
config.tracker.new_track_thresh = 0.9
config.tracker.min_hits = 1
config.tracker.max_time_lost = 20
config.tracker.appearance_weight = 0.0
config.tracker.motion_weight = 0.2
config.features.face_det_thresh = 0.1
config.pipeline.enable_reid = False

pipeline = create_pipeline(config)
pipeline.model_manager.warmup()

suspect_path = os.path.join(os.path.dirname(__file__), "sabnetra_ai", "suspect.jpg")
suspect_img = cv2.imread(suspect_path)
if suspect_img is None:
    print(f"ERROR: Could not load {suspect_path}")
    exit(1)

h, w = suspect_img.shape[:2]
full_bbox = np.array([0, 0, w, h])
feats_suspect = pipeline.feature_extractor.extract_all(
    suspect_img, full_bbox, -1, "suspect_file", 0)

if feats_suspect.face_embedding is not None:
    profile = SuspectProfile(
        suspect_id="SUSPECT",
        case_id="FILE",
        face_emb=feats_suspect.face_embedding,
        metadata={"source": "suspect.jpg"},
    )
    pipeline.suspect_manager.enroll_suspect(profile)
    print(f"Enrolled suspect from {suspect_path}")
    print(f"Face embedding dim: {len(feats_suspect.face_embedding)}")
else:
    print("ERROR: No face detected in suspect.jpg")
    exit(1)

RTSP_URL = "rtsp://admin:Xzone%40321@192.168.1.217:554/Streaming/Channels/101"
print(f"Connecting to RTSP...")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
print(f"Camera opened: {cap.isOpened()}")
if not cap.isOpened():
    print("ERROR: Cannot connect to camera. Check RTSP URL.")
    exit(1)

tracker = BoTSORT(config.tracker)
frame_count = 0
process_every = 3
do_feat_every = 6
fps_history = []

cv2.namedWindow("SabNetra AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SabNetra AI", 640, 360)

print()
print("=== SabNetra Webcam Test ===")
print("Tracking you against suspect.jpg")
print("RED box = match with suspect")
print("Press 'q' to quit")
print()

cached_state = {}
last_dets = []
last_tracks = []
ticks = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Frame read failed, reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        import time
        time.sleep(1)
        continue

    frame = cv2.resize(frame, (640, 360))
    frame_count += 1

    if frame_count % process_every == 0:
        last_dets = pipeline.detector.detect(frame)
        det_dicts = [{"bbox": d.bbox.copy(), "confidence": d.confidence, "class_id": d.class_id} for d in last_dets]
        last_tracks = tracker.update(det_dicts)
        for _ in range(3):
            cap.grab()

    new_ticks = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (new_ticks - ticks)
    ticks = new_ticks
    fps_history.append(fps)
    if len(fps_history) > 30:
        fps_history.pop(0)

    if frame_count % process_every == 0:
        do_feat = (frame_count % do_feat_every == 0)
    else:
        do_feat = False

    for t in last_tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        c = (0, 255, 0)
        label = f"ID:{t.track_id}"

        if do_feat and t.is_confirmed:
            try:
                feats = pipeline.feature_extractor.extract_all(
                    frame, t.bbox, t.track_id, "webcam", frame_count)
                fd = {"face_embedding": feats.face_embedding, "body_embedding": None,
                      "clothing_descriptor": None, "gait_descriptor": None}
                feat_names = []
                if feats.face_embedding is not None:
                    feat_names.append(f"face({len(feats.face_embedding)})")

                if feats.has_any_embedding():
                    state, sid, score = pipeline.suspect_manager.process_detection(
                        t.track_id, "webcam", fd, frame_count)
                    cached_state[t.track_id] = (state, sid, score, feat_names)
                else:
                    cached_state[t.track_id] = ("GREEN", "", 0.0, ["no_features"])
            except Exception as e:
                print(f"  !! Track {t.track_id} error: {e}")
                import traceback
                traceback.print_exc()

        if t.track_id in cached_state:
            state, sid, score, feat_names = cached_state[t.track_id]
            if state == "RED":
                c = (0, 0, 255)
                label = f"RED {sid} {score:.2f}"
            elif state == "YELLOW":
                c = (0, 255, 255)
                label = f"YELLOW {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    avg_fps = sum(fps_history) / len(fps_history)
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("SabNetra AI", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done")
