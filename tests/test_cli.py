import sys
import argparse
from unittest.mock import patch, MagicMock
from sabnetra_ai.cli import main


def test_cli_enroll_parser():
    with patch.object(sys, "argv", ["sabnetra", "enroll", "img.jpg", "--case", "C1", "--id", "S1"]):
        with patch("sabnetra_ai.cli.cmd_enroll") as mock:
            main()
            mock.assert_called_once()


def test_cli_enroll_no_id():
    with patch.object(sys, "argv", ["sabnetra", "enroll", "img.jpg"]):
        with patch("sabnetra_ai.cli.cmd_enroll") as mock:
            main()
            mock.assert_called_once()


def test_cli_add_camera():
    with patch.object(sys, "argv", ["sabnetra", "add-camera", "rtsp://localhost/stream", "--name", "cam1"]):
        with patch("sabnetra_ai.cli.cmd_add_camera") as mock:
            main()
            mock.assert_called_once()


def test_cli_add_camera_default_name():
    with patch.object(sys, "argv", ["sabnetra", "add-camera", "rtsp://localhost/stream"]):
        with patch("sabnetra_ai.cli.cmd_add_camera") as mock:
            main()
            mock.assert_called_once()


def test_cli_start():
    with patch.object(sys, "argv", ["sabnetra", "start", "rtsp://localhost/stream"]):
        with patch("sabnetra_ai.cli.cmd_start") as mock:
            main()
            mock.assert_called_once()


def test_cli_stats():
    with patch.object(sys, "argv", ["sabnetra", "stats"]):
        with patch("sabnetra_ai.cli.cmd_stats") as mock:
            main()
            mock.assert_called_once()


def test_cli_list_suspects():
    with patch.object(sys, "argv", ["sabnetra", "list-suspects"]):
        with patch("sabnetra_ai.cli.cmd_list_suspects") as mock:
            main()
            mock.assert_called_once()


def test_cli_remove_suspect():
    with patch.object(sys, "argv", ["sabnetra", "remove-suspect", "S0001"]):
        with patch("sabnetra_ai.cli.cmd_remove_suspect") as mock:
            main()
            mock.assert_called_once()


def test_cli_alerts():
    with patch.object(sys, "argv", ["sabnetra", "alerts", "--count", "5"]):
        with patch("sabnetra_ai.cli.cmd_alerts") as mock:
            main()
            mock.assert_called_once()


def test_cli_alerts_default():
    with patch.object(sys, "argv", ["sabnetra", "alerts"]):
        with patch("sabnetra_ai.cli.cmd_alerts") as mock:
            main()
            mock.assert_called_once()


def test_cli_reset():
    with patch.object(sys, "argv", ["sabnetra", "reset"]):
        with patch("sabnetra_ai.cli.cmd_reset") as mock:
            main()
            mock.assert_called_once()


def test_cli_no_command_shows_help():
    with patch.object(sys, "argv", ["sabnetra"]):
        with patch("argparse.ArgumentParser.print_help") as mock:
            main()
            mock.assert_called_once()


def test_cmd_enroll_success():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        pipe.enroll_suspect.return_value = "S0001"
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_enroll
        args = argparse.Namespace(images=["img.jpg"], videos=None, case="C1", id="S1")
        cmd_enroll(args)
        pipe.enroll_suspect.assert_called_once_with(
            images=["img.jpg"], videos=None, case_id="C1", suspect_id="S1")


def test_cmd_enroll_failure():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        pipe.enroll_suspect.return_value = None
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_enroll
        args = argparse.Namespace(images=["img.jpg"], videos=[], case="", id=None)
        cmd_enroll(args)
        pipe.enroll_suspect.assert_called_once()


def test_cmd_add_camera_success():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        pipe.add_camera.return_value = True
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_add_camera
        args = argparse.Namespace(url="rtsp://host/stream", name="cam1")
        cmd_add_camera(args)
        pipe.add_camera.assert_called_once_with("rtsp://host/stream", "cam1")


def test_cmd_remove_suspect_found():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        pipe.suspect_manager.get_suspect_profile.return_value = MagicMock()
        pipe.suspect_manager.matcher.suspect_db = MagicMock()
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_remove_suspect
        args = argparse.Namespace(suspect_id="S1")
        cmd_remove_suspect(args)
        pipe.suspect_manager.get_suspect_profile.assert_called_once_with("S1")


def test_cmd_remove_suspect_not_found():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        pipe.suspect_manager.get_suspect_profile.return_value = None
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_remove_suspect
        args = argparse.Namespace(suspect_id="S1")
        cmd_remove_suspect(args)
        pipe.suspect_manager.matcher.suspect_db.remove_suspect.assert_not_called()


def test_cmd_reset():
    with patch("sabnetra_ai.cli._get_pipeline") as mock_get:
        pipe = MagicMock()
        mock_get.return_value = pipe
        from sabnetra_ai.cli import cmd_reset
        args = argparse.Namespace()
        cmd_reset(args)
        pipe.state_engine.reset_all.assert_called_once()
