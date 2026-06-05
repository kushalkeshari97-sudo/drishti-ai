import argparse
import sys
import time
from sabnetra_ai import create_pipeline

pipeline = None


def cmd_enroll(args):
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline()
    sid = pipeline.enroll_suspect(
        images=args.images,
        videos=args.videos or None,
        case_id=args.case or "",
        suspect_id=args.id,
    )
    print(sid if sid else "FAILED")


def cmd_add_camera(args):
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline()
    pipeline.add_camera(args.url, args.name)
    print(f"Camera {args.name} added")


def cmd_start(args):
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline()
    pipeline.add_camera(args.url, args.name)
    pipeline.start()
    print(f"SabNetra started on {args.name}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
        print("Stopped")


def cmd_stats(args):
    global pipeline
    if pipeline is None:
        print("Pipeline not running")
        return
    s = pipeline.stats()
    print(f"Frames: {s['pipeline']['frames_processed']}")
    print(f"Detections: {s['pipeline']['detections']}")
    print(f"Alerts: {s['pipeline']['alerts']}")
    print(f"RED tracks: {s['suspects']['active_red_tracks']}")
    print(f"YELLOW tracks: {s['suspects']['active_yellow_tracks']}")
    print(f"Enrolled suspects: {s['suspects']['enrolled_suspects']}")


def main():
    parser = argparse.ArgumentParser(description="SabNetra AI 2.0")
    sub = parser.add_subparsers(dest="command")

    p_enroll = sub.add_parser("enroll", help="Enroll suspect from image(s)")
    p_enroll.add_argument("images", nargs="+", help="Suspect image paths")
    p_enroll.add_argument("--videos", nargs="*", default=[])
    p_enroll.add_argument("--case", default="")
    p_enroll.add_argument("--id")

    p_cam = sub.add_parser("add-camera", help="Add RTSP camera")
    p_cam.add_argument("url", help="RTSP URL")
    p_cam.add_argument("--name", default="cam_0")

    p_start = sub.add_parser("start", help="Add camera & start monitoring")
    p_start.add_argument("url", help="RTSP URL")
    p_start.add_argument("--name", default="cam_0")

    sub.add_parser("stats", help="Show pipeline stats")

    args = parser.parse_args()
    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "add-camera":
        cmd_add_camera(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
