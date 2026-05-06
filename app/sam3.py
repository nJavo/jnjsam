from __future__ import annotations

import base64
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import Settings
from app.media import COLORS, extract_frames, ffprobe_video, frame_to_b64, h264_encode

settings = Settings()
sys.path.insert(0, str(settings.sam3_repo))
from sam3.model_builder import build_sam3_video_predictor

sys.path.insert(0, str(settings.sam2_repo))
from sam2.build_sam import build_sam2_video_predictor


@contextmanager
def autocast():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield


sam3 = build_sam3_video_predictor(
    checkpoint_path=str(settings.sam3_checkpoint),
    bpe_path=str(settings.sam3_bpe),
)
sam2 = build_sam2_video_predictor(settings.sam2_config, str(settings.sam2_checkpoint))

sessions: dict = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
app.mount("/videos-raw", StaticFiles(directory=str(settings.videos_dir)), name="videos-raw")


def masks_from_sam3_out(out: dict) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    masks: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if out and "out_binary_masks" in out:
        for i, m in enumerate(out["out_binary_masks"]):
            if m.any():
                masks.append((m.astype(bool), COLORS[i % len(COLORS)]))
    return masks


def close_sam3(sid: str | None) -> None:
    if not sid:
        return
    try:
        sam3.handle_request({"type": "close_session", "session_id": sid})
    except Exception:
        pass


@app.get("/meta")
def meta():
    return {
        "variant": "sam3",
        "title": "SAM3 + SAM2",
        "subtitle": "SAM3 text · SAM2 clicks · propagate matches last prompt type",
        "text_mode_label": "SAM3 text",
        "run_text_status": "Running SAM3…",
    }


@app.get("/")
def index():
    return FileResponse(settings.static_dir / "index.html")


@app.get("/videos")
def list_videos():
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    names = sorted(p.name for p in settings.videos_dir.iterdir() if p.suffix.lower() in exts)
    return {"videos": names, "dir": str(settings.videos_dir)}


@app.get("/video-info")
def video_info(name: str):
    path = settings.videos_dir / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return ffprobe_video(path)


class LoadReq(BaseModel):
    video_path: str
    fps: float = 5.0


@app.post("/load")
def load_video(req: LoadReq):
    sid = str(uuid.uuid4())[:8]
    out_dir = settings.frames_sam3 / sid
    frames = extract_frames(req.video_path, out_dir, fps=req.fps)
    if not frames:
        raise HTTPException(400, "No frames extracted")
    sessions[sid] = {
        "frames_dir": str(out_dir),
        "frame_paths": frames,
        "fps": req.fps,
        "sam3_sid": None,
        "sam2_state": None,
        "last_mode": None,
    }
    return {"session_id": sid, "total_frames": len(frames), "frame": frame_to_b64(frames[0])}


class FrameReq(BaseModel):
    session_id: str
    frame_idx: int


@app.post("/frame")
def get_frame(req: FrameReq):
    s = sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, "Not found")
    return {"frame": frame_to_b64(s["frame_paths"][req.frame_idx])}


class PromptReq(BaseModel):
    session_id: str
    frame_idx: int
    text: str = ""
    points: list = []


@app.post("/prompt")
def add_prompt(req: PromptReq):
    s = sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, "Not found")
    frames = s["frame_paths"]
    masks: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    if req.text:
        close_sam3(s["sam3_sid"])
        s["sam2_state"] = None
        with autocast():
            r = sam3.handle_request({"type": "start_session", "resource_path": s["frames_dir"]})
        s["sam3_sid"] = r["session_id"]
        s["last_mode"] = "text"
        with autocast():
            out = sam3.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": s["sam3_sid"],
                    "frame_index": req.frame_idx,
                    "text": req.text,
                }
            ).get("outputs", {})
        masks = masks_from_sam3_out(out)

    elif req.points:
        if s["sam2_state"] is None:
            with autocast():
                state = sam2.init_state(video_path=s["frames_dir"])
            s["sam2_state"] = state
        else:
            state = s["sam2_state"]
        pts = np.array([[p[0], p[1]] for p in req.points], dtype=np.float32)
        labels = np.array([p[2] for p in req.points], dtype=np.int32)
        with autocast():
            _, _obj_ids, mask_logits = sam2.add_new_points_or_box(
                state, frame_idx=req.frame_idx, obj_id=1, points=pts, labels=labels
            )
        s["last_mode"] = "clicks"
        for i, logit in enumerate(mask_logits):
            m = (logit[0] > 0).cpu().numpy()
            if m.any():
                masks.append((m, COLORS[i % len(COLORS)]))
    else:
        raise HTTPException(400, "Provide text or points")

    pts_render = [[p[0], p[1], p[2] == 1] for p in req.points] if req.points else None
    return {"frame": frame_to_b64(frames[req.frame_idx], masks, pts_render), "n_masks": len(masks)}


class PropReq(BaseModel):
    session_id: str


@app.post("/propagate")
def propagate(req: PropReq):
    s = sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, "Not found")
    frames = s["frame_paths"]
    out_path = settings.frames_sam3 / req.session_id / "annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), s["fps"], (w, h))
    frames_with_masks = 0

    if s["last_mode"] == "text" and s["sam3_sid"]:
        with autocast():
            stream = list(
                sam3.handle_stream_request({"type": "propagate_in_video", "session_id": s["sam3_sid"]})
            )
        for resp in stream:
            fidx = resp["frame_index"]
            img = cv2.imread(str(frames[fidx]))
            for mask, color in masks_from_sam3_out(resp.get("outputs", {})):
                ov = img.copy().astype(np.float32)
                ov[mask] = color
                img = cv2.addWeighted(ov.astype(np.uint8), 0.5, img, 0.5, 0)
                frames_with_masks += 1
            writer.write(img)
    elif s["last_mode"] == "clicks" and s["sam2_state"] is not None:
        frame_imgs = {i: cv2.imread(str(p)) for i, p in enumerate(frames)}
        with autocast():
            for fidx, _obj_ids, mask_logits in sam2.propagate_in_video(s["sam2_state"]):
                img = frame_imgs[fidx]
                for i, logit in enumerate(mask_logits):
                    m = (logit[0] > 0).cpu().numpy()
                    if m.any():
                        ov = img.copy().astype(np.float32)
                        ov[m] = COLORS[i % len(COLORS)]
                        img = cv2.addWeighted(ov.astype(np.uint8), 0.5, img, 0.5, 0)
                        frames_with_masks += 1
                frame_imgs[fidx] = img
        for i in range(len(frames)):
            writer.write(frame_imgs[i])
    else:
        writer.release()
        raise HTTPException(400, "Add a prompt first")

    writer.release()
    h264 = h264_encode(str(out_path))
    video_b64 = base64.b64encode(h264.read_bytes()).decode()
    return {"frames_with_masks": frames_with_masks, "video_b64": video_b64}
