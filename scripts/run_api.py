import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, AsyncGenerator
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

API_KEY = os.environ.get("SABNETRA_API_KEY", "")
security = HTTPBearer(auto_error=False)

config = load_config()
pipeline = create_pipeline(config)

alert_history = []
alert_history_lock = threading.Lock()
websockets = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    pipeline.model_manager.warmup()
    pipeline.register_alert_callback(_on_alert)
    pipeline.start()
    yield
    pipeline.stop()


app = FastAPI(title="SabNetra AI API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY:
        return True
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True


class EnrollRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    case_id: str = ""
    suspect_id: Optional[str] = None

    @field_validator("image_path")
    @classmethod
    def path_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"file not found: {v}")
        return v


class CameraRequest(BaseModel):
    url: str = Field(..., min_length=1)
    name: str = Field(default="cam_0", min_length=1)

    @field_validator("url")
    @classmethod
    def valid_rtsp(cls, v):
        if not v.startswith("rtsp://") and not v.startswith("rtmp://") and not v.startswith("http"):
            raise ValueError("Invalid stream URL")
        return v


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
def health(auth=Depends(verify_api_key)):
    return {"status": "ok", "running": pipeline.is_running}


@app.get("/stats")
def stats(auth=Depends(verify_api_key)):
    return stats_to_json(pipeline.stats())


@app.get("/alerts")
def get_alerts(limit: int = 50, auth=Depends(verify_api_key)):
    with alert_history_lock:
        return list(reversed(alert_history[-limit:]))


@app.get("/suspects/active")
def get_active_suspects(auth=Depends(verify_api_key)):
    suspects = pipeline.suspect_manager.get_all_active_suspects()
    return active_suspects_to_json(suspects)


@app.get("/suspects")
def get_suspects(auth=Depends(verify_api_key)):
    return pipeline.suspect_manager.stats()


@app.post("/suspects/enroll")
def enroll_suspect(req: EnrollRequest, auth=Depends(verify_api_key)):
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
def add_camera(req: CameraRequest, auth=Depends(verify_api_key)):
    ok = pipeline.add_camera(req.url, req.name)
    return {"ok": ok, "camera_id": req.name}


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str, auth=Depends(verify_api_key)):
    pipeline.remove_camera(camera_id)
    return {"ok": True}


@app.get("/cameras")
def list_cameras(auth=Depends(verify_api_key)):
    streams = pipeline.rtsp_manager.stats()
    return streams


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    api_key = ws.headers.get("Authorization", "").replace("Bearer ", "")
    if API_KEY and api_key != API_KEY:
        await ws.close(code=4001)
        return
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
