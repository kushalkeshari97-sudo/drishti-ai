import argparse
import sys
import time
from sabnetra_ai import create_pipeline

pipeline = None


def _get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline()
    return pipeline


def cmd_enroll(args):
    """Enroll a suspect from image files."""
    p = _get_pipeline()
    sid = p.enroll_suspect(
        images=args.images,
        videos=args.videos or None,
        case_id=args.case or "",
        suspect_id=args.id,
    )
    print(sid if sid else "FAILED")


def cmd_add_camera(args):
    """Add an RTSP camera stream to the pipeline."""
    p = _get_pipeline()
    ok = p.add_camera(args.url, args.name)
    print(f"Camera {args.name} added" if ok else "FAILED")


def cmd_start(args):
    """Add a camera and start the monitoring pipeline."""
    p = _get_pipeline()
    ok = p.add_camera(args.url, args.name)
    if not ok:
        print("Failed to add camera")
        return
    p.start()
    print(f"SabNetra started on {args.name}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.stop()
        print("Stopped")


def cmd_stats(args):
    """Display current pipeline statistics."""
    p = _get_pipeline()
    s = p.stats()
    print(f"Frames: {s['pipeline']['frames_processed']}")
    print(f"Detections: {s['pipeline']['detections']}")
    print(f"Alerts: {s['pipeline']['alerts']}")
    print(f"RED tracks: {s['suspects']['active_red_tracks']}")
    print(f"YELLOW tracks: {s['suspects']['active_yellow_tracks']}")
    print(f"Enrolled suspects: {s['suspects']['enrolled_suspects']}")


def cmd_list_suspects(args):
    """List all active suspects in YELLOW/RED state."""
    p = _get_pipeline()
    suspects = p.suspect_manager.get_all_active_suspects()
    if not suspects:
        print("No active suspects")
        return
    for s in suspects:
        print(f"  {s['suspect_id'] or 'N/A':<16} state={s['state']:<6} "
              f"score={s['score']:.3f}  camera={s['camera_id']}  track={s['track_id']}")


def cmd_remove_suspect(args):
    """Remove a suspect profile from the database."""
    p = _get_pipeline()
    profile = p.suspect_manager.get_suspect_profile(args.suspect_id)
    if profile is None:
        print(f"Suspect {args.suspect_id} not found")
        return
    p.suspect_manager.matcher.suspect_db.remove_suspect(args.suspect_id)
    p.suspect_manager._global_suspects.pop(args.suspect_id, None)
    print(f"Suspect {args.suspect_id} removed")


def cmd_alerts(args):
    """Display recent alert history."""
    p = _get_pipeline()
    alerts = p.alert_system.get_recent_alerts(count=args.count)
    if not alerts:
        print("No alerts")
        return
    for a in reversed(alerts):
        print(f"  {a.alert_id:<30} state={a.state:<6} score={a.score:.3f}  {a.camera_id}")


def cmd_reset(args):
    """Reset all track states and track-to-suspect mappings."""
    p = _get_pipeline()
    p.state_engine.reset_all()
    p.suspect_manager._global_track_to_suspect.clear()
    print("Track state reset")


def main():
    """SabNetra AI 2.0 CLI entry point."""
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

    sub.add_parser("list-suspects", help="List active suspects")

    p_rm = sub.add_parser("remove-suspect", help="Remove a suspect by ID")
    p_rm.add_argument("suspect_id", help="Suspect ID to remove")

    p_alerts = sub.add_parser("alerts", help="Show recent alerts")
    p_alerts.add_argument("--count", type=int, default=10)

    sub.add_parser("reset", help="Reset track states")

    args = parser.parse_args()
    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "add-camera":
        cmd_add_camera(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "list-suspects":
        cmd_list_suspects(args)
    elif args.command == "remove-suspect":
        cmd_remove_suspect(args)
    elif args.command == "alerts":
        cmd_alerts(args)
    elif args.command == "reset":
        cmd_reset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
