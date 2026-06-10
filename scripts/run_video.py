import cv2
import numpy as np
import os
import sys
import time
import argparse
from collections import deque

from sabnetra_ai import create_pipeline, SabNetraConfig
from sabnetra_ai.core.matcher import SuspectProfile
from sabnetra_ai.core.tracker import BoTSORT
from sabnetra_ai.utils.helpers import draw_detection


def main():
    parser = argparse.ArgumentParser(description="Run SabNetra AI on a video file")
    parser.add_argument("video", help="Path to the test video file")
    parser.add_argument("--suspect", default=None,
                        help="Path to suspect image (default: data/suspects/suspect.jpg)")
    parser.add_argument("--suspect-id", default="SUSPECT_001")
    parser.add_argument("--show", action="store_true", help="Display video window")
    parser.add_argument("--output", default=None, help="Save output video to path")
    parser.add_argument("--process-every", type=int, default=3,
                        help="Process detection every N frames")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Detection input size (smaller=faster)")
    parser.add_argument("--disable-face", action="store_true",
                        help="Disable face recognition")
    parser.add_argument("--disable-reid", action="store_true",
                        help="Disable ReID body recognition")
    parser.add_argument("--disable-clothing", action="store_true",
                        help="Disable clothing descriptor")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    suspect_path = args.suspect or os.path.join(
        os.path.dirname(__file__), "..", "data", "suspects", "suspect.jpg")

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
    config.pipeline.process_every_n_frames = 1

    pipeline = create_pipeline(config)
    pipeline.model_manager.warmup()
    print("Models loaded")

    suspect_img = cv2.imread(suspect_path)
    if suspect_img is None:
        print(f"ERROR: Could not load suspect image: {suspect_path}")
        sys.exit(1)

    h, w = suspect_img.shape[:2]
    full_bbox = np.array([0, 0, w, h])
    feats = pipeline.feature_extractor.extract_all(
        suspect_img, full_bbox, -1, "enrollment", 0)

    if feats.face_embedding is None and feats.body_embedding is None:
        print("ERROR: No features extracted from suspect image")
        sys.exit(1)

    profile = SuspectProfile(
        suspect_id=args.suspect_id,
        case_id="TEST",
        face_emb=feats.face_embedding,
        body_emb=feats.body_embedding,
        clothing_emb=feats.clothing_descriptor,
        metadata={"source": suspect_path},
    )
    pipeline.suspect_manager.enroll_suspect(profile)
    print(f"Enrolled suspect: {args.suspect_id}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = 1.0 / src_fps
    print(f"\nVideo: {os.path.basename(args.video)}")
    print(f"Frames: {total_frames} @ {src_fps:.1f} FPS, {width}x{height}")

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))

    tracker = BoTSORT(config.tracker)
    frame_count = 0
    results = deque(maxlen=200)
    match_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    start_time = time.time()

    track_states = {}
    next_frame_time = time.time()

    print("\nProcessing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        now = time.time()
        sleep_time = next_frame_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_frame_time = now
        next_frame_time += frame_interval

        do_detect = (frame_count % args.process_every == 0)

        if do_detect:
            dets = pipeline.detector.detect(frame)
            det_dicts = []
            app_embs = []
            det_features = {}
            for i, d in enumerate(dets):
                det_dicts.append({"bbox": d.bbox.copy(), "confidence": d.confidence, "class_id": d.class_id})
                feats = pipeline.feature_extractor.extract_all(frame, d.bbox, -1, "video", frame_count)
                det_features[i] = feats
                if feats.face_embedding is not None:
                    app_embs.append(feats.face_embedding)
                elif feats.body_embedding is not None:
                    app_embs.append(feats.body_embedding)
                else:
                    app_embs.append(None)
            tracks = tracker.update(det_dicts, app_embs)

            for t in tracks:
                if not t.is_confirmed:
                    continue
                if t.time_since_update != 0:
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
                    iou_val = inter / (area_t + area_d - inter)
                    if iou_val > best_iou:
                        best_iou = iou_val
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
                    t.track_id, "video", fd, frame_count)
                match_counts[state] = match_counts.get(state, 0) + 1
                track_states[t.track_id] = (state, sid, score)
                results.append({
                    "frame": frame_count,
                    "track_id": t.track_id,
                    "state": state,
                    "suspect_id": sid,
                    "score": score,
                })
        else:
            tracks = list(tracker.active_tracks.values())
            for t in tracks:
                t.predict()

        for t in tracks:
            if t.is_confirmed:
                state, sid, score = track_states.get(t.track_id, ("GREEN", None, 0.0))
                draw_detection(frame, t.bbox, t.track_id, state, sid, score)

        info = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.show:
            cv2.imshow("SabNetra AI - Video Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out is not None:
            out.write(frame)

    elapsed = time.time() - start_time
    cap.release()
    if out:
        out.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"RESULTS: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"{'='*50}")
    reds = sum(1 for r in results if r["state"] == "RED")
    yellows = sum(1 for r in results if r["state"] == "YELLOW")
    print(f"RED matches:    {reds}")
    print(f"YELLOW matches: {yellows}")
    print(f"Total matches:  {reds + yellows}")
    print()

    if reds > 0:
        print("RED ALERT frames:")
        for r in results:
            if r["state"] == "RED":
                print(f"  Frame {r['frame']}: track={r['track_id']} "
                      f"suspect={r['suspect_id']} score={r['score']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
