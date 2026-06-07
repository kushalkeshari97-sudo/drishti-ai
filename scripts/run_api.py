import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import threading
import time
import json
import os

from sabnetra_ai import create_pipeline, SabNetraConfig
from sabnetra_ai.utils.serializers import (
    alert_to_json, stats_to_json, active_suspects_to_json, track_to_json,
)
from sabnetra_ai.utils.config_loader import load_config
from sabnetra_ai.utils.persistence import save_suspects

app = FastAPI(title="SabNetra AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

config = load_config()
pipeline = create_pipeline(config)
pipeline.model_manager.warmup()

alert_history = []
alert_history_lock = threading.Lock()
websockets = []


class EnrollRequest(BaseModel):
    image_path: str
    case_id: str = ""
    suspect_id: Optional[str] = None


class CameraRequest(BaseModel):
    url: str
    name: str = "cam_0"


@app.on_event("startup")
def startup():
    pipeline.register_alert_callback(_on_alert)
    pipeline.start()


@app.on_event("shutdown")
def shutdown():
    pipeline.stop()


def _on_alert(alert):
    with alert_history_lock:
        alert_history.append(alert_to_json(alert))
        if len(alert_history) > 1000:
            alert_history[:] = alert_history[-1000:]
    for ws in list(websockets):
        try:
            ws.send_json(alert_to_json(alert))
        except Exception:
            websockets.remove(ws)


@app.get("/health")
def health():
    return {"status": "ok", "running": pipeline.is_running}


@app.get("/stats")
def stats():
    return stats_to_json(pipeline.stats())


@app.get("/alerts")
def get_alerts(limit: int = 50):
    with alert_history_lock:
        return list(reversed(alert_history[-limit:]))


@app.get("/suspects/active")
def get_active_suspects():
    suspects = pipeline.suspect_manager.get_all_active_suspects()
    return active_suspects_to_json(suspects)


@app.get("/suspects")
def get_suspects():
    return pipeline.suspect_manager.stats()


@app.post("/suspects/enroll")
def enroll_suspect(req: EnrollRequest):
    if not os.path.exists(req.image_path):
        return {"error": "file not found"}
    sid = pipeline.enroll_suspect(
        images=[req.image_path],
        case_id=req.case_id or "",
        suspect_id=req.suspect_id,
    )
    if sid:
        save_suspects(
            [pipeline.suspect_manager.get_suspect_profile(sid)],
            config.suspect_db_path,
        )
        return {"suspect_id": sid}
    return {"error": "enrollment failed"}


@app.post("/cameras")
def add_camera(req: CameraRequest):
    ok = pipeline.add_camera(req.url, req.name)
    return {"ok": ok, "camera_id": req.name}


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    pipeline.remove_camera(camera_id)
    return {"ok": True}


@app.get("/cameras")
def list_cameras():
    streams = pipeline.rtsp_manager.stats()
    return streams


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websockets.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        websockets.remove(ws)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
