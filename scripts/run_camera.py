import cv2
import numpy as np
import os
import argparse
import time
from collections import deque

from sabnetra_ai import create_pipeline, SabNetraConfig
from sabnetra_ai.core.tracker import BoTSORT
from sabnetra_ai.core.matcher import SuspectProfile
from sabnetra_ai.utils.helpers import draw_detection


def main():
    parser = argparse.ArgumentParser(description="SabNetra AI live camera demo")
    parser.add_argument("--rtsp", default=None,
                        help="RTSP stream URL (default: use local webcam)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Local camera device index (default: 0)")
    parser.add_argument("--suspect", default=None,
                        help="Suspect photo path (default: data/suspects/suspect.jpg)")
    parser.add_argument("--suspect-id", default="SUSPECT")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Detection input size")
    parser.add_argument("--disable-face", action="store_true")
    parser.add_argument("--disable-reid", action="store_true")
    parser.add_argument("--disable-clothing", action="store_true")
    args = parser.parse_args()

    config = SabNetraConfig()
    config.detector.confidence_threshold = 0.55
    config.detector.img_size = args.img_size
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
    config.pipeline.enable_reid = not args.disable_reid
    config.pipeline.enable_face_recognition = not args.disable_face
    config.pipeline.enable_clothing = not args.disable_clothing

    pipeline = create_pipeline(config)
    pipeline.model_manager.warmup()

    suspect_path = args.suspect or os.path.join(
        os.path.dirname(__file__), "..", "data", "suspects", "suspect.jpg")
    suspect_img = cv2.imread(suspect_path)
    if suspect_img is None:
        print(f"ERROR: Could not load {suspect_path}")
        exit(1)

    h, w = suspect_img.shape[:2]
    full_bbox = np.array([0, 0, w, h])
    feats = pipeline.feature_extractor.extract_all(
        suspect_img, full_bbox, -1, "suspect_file", 0)

    if feats.face_embedding is None and feats.body_embedding is None:
        print("ERROR: No features extracted from suspect image")
        exit(1)

    profile = SuspectProfile(
        suspect_id=args.suspect_id,
        case_id="FILE",
        face_emb=feats.face_embedding,
        body_emb=feats.body_embedding,
        clothing_emb=feats.clothing_descriptor,
        metadata={"source": os.path.basename(suspect_path)},
    )
    pipeline.suspect_manager.enroll_suspect(profile)
    print(f"Enrolled suspect from {suspect_path}")

    url = args.rtsp or os.environ.get("SABNETRA_RTSP_URL")
    if url:
        print(f"Connecting to RTSP: {url}")
        cap = cv2.VideoCapture(url)
    else:
        print(f"Opening webcam (device {args.camera_id})...")
        cap = cv2.VideoCapture(args.camera_id)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        exit(1)

    print(f"Camera opened: {cap.isOpened()}")
    print()

    tracker = BoTSORT(config.tracker)
    frame_count = 0
    process_every = 3
    fps_history = deque(maxlen=30)
    track_states = {}

    cv2.namedWindow("SabNetra AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SabNetra AI", 960, 540)

    last_tick = cv2.getTickCount()

    print("=== SabNetra AI Live Demo ===")
    if not url:
        print("Source: Webcam")
    else:
        print("Source: RTSP Stream")
    print(f"Suspect: {os.path.basename(suspect_path)}")
    print("Press 'q' to quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, reconnecting...")
            cap.release()
            time.sleep(1)
            if url:
                cap = cv2.VideoCapture(url)
            else:
                cap = cv2.VideoCapture(args.camera_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            continue

        frame_count += 1
        do_detect = (frame_count % process_every == 0)

        if do_detect:
            dets = pipeline.detector.detect(frame)
            det_dicts = []
            app_embs = []
            det_features = {}
            for i, d in enumerate(dets):
                det_dicts.append({"bbox": d.bbox.copy(), "confidence": d.confidence, "class_id": d.class_id})
                feats = pipeline.feature_extractor.extract_all(frame, d.bbox, -1, "webcam", frame_count)
                det_features[i] = feats
                if feats.face_embedding is not None:
                    app_embs.append(feats.face_embedding)
                elif feats.body_embedding is not None:
                    app_embs.append(feats.body_embedding)
                else:
                    app_embs.append(None)
            tracks = tracker.update(det_dicts, app_embs)

            for t in tracks:
                if not t.is_confirmed or t.time_since_update != 0:
                    continue
                best_iou = 0.15
                best_idx = -1
                for i, d in enumerate(dets):
                    b = d.bbox
                    xi1 = max(t.bbox[0], b[0]); yi1 = max(t.bbox[1], b[1])
                    xi2 = min(t.bbox[2], b[2]); yi2 = min(t.bbox[3], b[3])
                    if xi2 <= xi1 or yi2 <= yi1:
                        continue
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    area_t = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
                    area_d = (b[2] - b[0]) * (b[3] - b[1])
                    iou = inter / (area_t + area_d - inter)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx < 0:
                    continue
                feats = det_features.get(best_idx)
                if feats is None or not feats.has_any_embedding():
                    continue
                fd = {
                    "face_embedding": feats.face_embedding,
                    "body_embedding": feats.body_embedding,
                    "clothing_descriptor": feats.clothing_descriptor,
                    "gait_descriptor": None,
                }
                state, sid, score = pipeline.suspect_manager.process_detection(
                    t.track_id, "webcam", fd, frame_count)
                track_states[t.track_id] = (state, sid, score)
        else:
            tracks = list(tracker.active_tracks.values())
            for t in tracks:
                t.predict()

        for t in tracks:
            if t.is_confirmed:
                state, sid, score = track_states.get(t.track_id, ("GREEN", None, 0.0))
                draw_detection(frame, t.bbox, t.track_id, state, sid, score)

        tick = cv2.getTickCount()
        dt = tick - last_tick
        if dt > 0:
            fps_history.append(cv2.getTickFrequency() / dt)
        last_tick = tick
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}  Tracks: {len(tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("SabNetra AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
