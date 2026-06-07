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
config.matcher.face_weight = 0.5
config.matcher.reid_weight = 0.3
config.matcher.clothing_weight = 0.2
config.matcher.temporal_consistency_frames = 3
config.tracker.match_thresh = 0.85
config.tracker.new_track_thresh = 0.9
config.tracker.min_hits = 1
config.tracker.max_time_lost = 30
config.tracker.appearance_weight = 0.4
config.tracker.motion_weight = 0.3
config.features.face_det_thresh = 0.1
config.pipeline.enable_reid = True

pipeline = create_pipeline(config)
pipeline.model_manager.warmup()

suspect_path = os.path.join(os.path.dirname(__file__), "..", "data", "suspects", "suspect.jpg")
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
        body_emb=feats_suspect.body_embedding,
        clothing_emb=feats_suspect.clothing_descriptor,
        metadata={"source": "suspect.jpg"},
    )
    pipeline.suspect_manager.enroll_suspect(profile)
    print(f"Enrolled suspect from {suspect_path}")
    print(f"Face embedding dim: {len(feats_suspect.face_embedding)}")
    if feats_suspect.body_embedding is not None:
        print(f"Body embedding dim: {len(feats_suspect.body_embedding)}")
    if feats_suspect.clothing_descriptor is not None:
        print(f"Clothing descriptor dim: {len(feats_suspect.clothing_descriptor)}")
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
process_every = 5
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

    do_detect = (frame_count % process_every == 0)

    if do_detect:
        last_dets = pipeline.detector.detect(frame)
        det_dicts = []
        app_embs = []
        det_feats = {}  # detection index -> IdentityFeatures
        for i, d in enumerate(last_dets):
            det_dicts.append({"bbox": d.bbox.copy(), "confidence": d.confidence, "class_id": d.class_id})
            all_feats = pipeline.feature_extractor.extract_all(frame, d.bbox, -1, "webcam", frame_count)
            det_feats[i] = all_feats
            if all_feats.face_embedding is not None:
                app_embs.append(all_feats.face_embedding)
            elif all_feats.body_embedding is not None:
                app_embs.append(all_feats.body_embedding)
            else:
                x1, y1, x2, y2 = map(int, d.bbox)
                crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                if crop is not None and crop.size > 0:
                    h = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(h, h)
                    app_embs.append(h.flatten().astype(np.float32))
                else:
                    app_embs.append(None)
        for _ in range(3):
            cap.grab()
    else:
        det_dicts = []
        app_embs = []
        last_dets = []
        det_feats = {}

    tracks = tracker.update(det_dicts, app_embs)

    new_ticks = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (new_ticks - ticks)
    ticks = new_ticks
    fps_history.append(fps)
    if len(fps_history) > 30:
        fps_history.pop(0)

    active_ids = {t.track_id for t in tracks}
    cached_state = {k: v for k, v in cached_state.items() if k in active_ids}

    for t_idx, t in enumerate(tracks):
        x1, y1, x2, y2 = map(int, t.bbox)
        if x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0 or x1 > 640 or y1 > 360:
            continue
        c = (0, 255, 0)
        label = f"ID:{t.track_id}"

        if do_detect and t.is_confirmed:
            current = cached_state.get(t.track_id)
            current_state = current[0] if current else "GREEN"

            # RED demotion: re-verify every 30 detection frames (~6s)
            if current_state == "RED":
                rd = (current[4] if len(current) > 4 else 0) + 1
                cached_state[t.track_id] = (*current[:4], rd)
                if rd % 30 == 0:
                    current_state = "CHECK"

            if current_state != "RED":
                try:
                    best_i = -1
                    best_iou = 0.3
                    for di, d in enumerate(last_dets):
                        b = d.bbox
                        xi1, yi1 = max(t.bbox[0], b[0]), max(t.bbox[1], b[1])
                        xi2, yi2 = min(t.bbox[2], b[2]), min(t.bbox[3], b[3])
                        if xi2 <= xi1 or yi2 <= yi1:
                            continue
                        inter = (xi2 - xi1) * (yi2 - yi1)
                        area_t = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
                        area_d = (b[2] - b[0]) * (b[3] - b[1])
                        iou = inter / (area_t + area_d - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_i = di
                    all_feats = det_feats.get(best_i) if best_i >= 0 else None

                    if all_feats is not None and all_feats.has_any_embedding():
                        fd = {
                            "face_embedding": all_feats.face_embedding,
                            "body_embedding": all_feats.body_embedding,
                            "clothing_descriptor": all_feats.clothing_descriptor,
                            "gait_descriptor": None,
                        }
                        state_str, sid, score = pipeline.suspect_manager.process_detection(
                            t.track_id, "webcam", fd, frame_count)
                        feat_names = [k for k, v in fd.items() if v is not None]
                        cached_state[t.track_id] = (state_str, sid, score, feat_names, 0)
                    else:
                        cached_state[t.track_id] = ("GREEN", "", 0.0, ["no_feats"], 0)
                except Exception as e:
                    print(f"  !! Track {t.track_id} error: {e}")
                    import traceback
                    traceback.print_exc()

        if t.track_id in cached_state:
            state, sid, score, feat_names = cached_state[t.track_id][:4]
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
