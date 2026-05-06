from __future__ import annotations

import base64
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel
from torchvision.ops import nms as tv_nms

from app.config import Settings
from app.media import COLORS, extract_frames, ffprobe_video, frame_to_b64, h264_encode

settings = Settings()
sys.path.insert(0, str(settings.medsam3_repo))
sys.path.insert(0, str(settings.sam3_repo))

from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3.model_builder import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

sys.path.insert(0, str(settings.sam2_repo))
from sam2.build_sam import build_sam2_video_predictor


@contextmanager
def autocast():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield


with open(settings.medsam3_lora_config) as f:
    lora_cfg = yaml.safe_load(f)["lora"]

medsam3 = build_sam3_image_model(
    device="cuda",
    compile=False,
    load_from_HF=False,
    checkpoint_path=str(settings.sam3_checkpoint),
    bpe_path=str(settings.sam3_bpe),
    eval_mode=True,
)
lora_config = LoRAConfig(
    rank=lora_cfg["rank"],
    alpha=lora_cfg["alpha"],
    dropout=0.0,
    target_modules=lora_cfg["target_modules"],
    apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
    apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
    apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
    apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
    apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
    apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
)
medsam3 = apply_lora_to_model(medsam3, lora_config)
load_lora_weights(medsam3, str(settings.medsam3_lora_weights))
medsam3.cuda().eval()

medsam3_transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

sam2 = build_sam2_video_predictor(settings.sam2_config, str(settings.sam2_checkpoint))

sessions: dict = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
app.mount("/videos-raw", StaticFiles(directory=str(settings.videos_dir)), name="videos-raw")


def medsam3_predict(frame_path: str, text: str, threshold: float, nms_iou: float) -> list[np.ndarray]:
    pil = PILImage.open(frame_path).convert("RGB")
    orig_w, orig_h = pil.size
    dp = Datapoint(
        find_queries=[
            FindQueryLoaded(
                query_text=text,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=0,
                    original_image_id=0,
                    original_category_id=1,
                    original_size=[orig_w, orig_h],
                    object_id=0,
                    frame_index=0,
                ),
            )
        ],
        images=[SAMImage(data=pil, objects=[], size=[orig_h, orig_w])],
    )
    dp = medsam3_transform(dp)
    batch = collate_fn_api([dp], dict_key="input")["input"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    with torch.no_grad(), autocast():
        outputs = medsam3(batch)
    last = outputs[-1]
    pred_logits = last["pred_logits"]
    pred_boxes = last["pred_boxes"]
    pred_masks = last.get("pred_masks", None)
    scores = pred_logits.sigmoid()[0].max(dim=-1)[0]
    keep = scores > threshold
    if not keep.any() or pred_masks is None:
        return []
    boxes_cxcywh = pred_boxes[0, keep]
    kept_scores = scores[keep]
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    keep_nms = tv_nms(boxes_xyxy, kept_scores, nms_iou)
    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5
    masks_full = (
        F.interpolate(
            masks_small.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        > 0.5
    )
    return [m.cpu().numpy() for m in masks_full]


@app.get("/meta")
def meta():
    return {
        "variant": "medsam3",
        "title": "MedSAM3 + SAM2",
        "subtitle": "MedSAM3 (LoRA) text on one frame · SAM2 mask init + propagation",
        "text_mode_label": "MedSAM3 text",
        "run_text_status": "Running MedSAM3…",
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
    out_dir = settings.frames_medsam3 / sid
    frames = extract_frames(req.video_path, out_dir, fps=req.fps)
    if not frames:
        raise HTTPException(400, "No frames extracted")
    sessions[sid] = {"frames_dir": str(out_dir), "frame_paths": frames, "fps": req.fps, "sam2_state": None}
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
    threshold: float = 0.3


@app.post("/prompt")
def add_prompt(req: PromptReq):
    s = sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, "Not found")
    frames = s["frame_paths"]
    masks: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    with autocast():
        state = sam2.init_state(video_path=s["frames_dir"])
    s["sam2_state"] = state

    if req.text:
        raw_masks = medsam3_predict(str(frames[req.frame_idx]), req.text, req.threshold, 0.5)
        if not raw_masks:
            return {"frame": frame_to_b64(frames[req.frame_idx]), "n_masks": 0}
        for obj_id, m in enumerate(raw_masks, start=1):
            with autocast():
                sam2.add_new_mask(state, frame_idx=req.frame_idx, obj_id=obj_id, mask=m)
            masks.append((m.astype(bool), COLORS[(obj_id - 1) % len(COLORS)]))
    elif req.points:
        pts = np.array([[p[0], p[1]] for p in req.points], dtype=np.float32)
        labels = np.array([p[2] for p in req.points], dtype=np.int32)
        with autocast():
            _, _obj_ids, mask_logits = sam2.add_new_points_or_box(
                state, frame_idx=req.frame_idx, obj_id=1, points=pts, labels=labels
            )
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
    if not s or s["sam2_state"] is None:
        raise HTTPException(400, "Add a prompt first")
    frames = s["frame_paths"]
    out_path = settings.frames_medsam3 / req.session_id / "annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), s["fps"], (w, h))
    frames_with_masks = 0
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
    writer.release()
    h264 = h264_encode(str(out_path))
    video_b64 = base64.b64encode(h264.read_bytes()).decode()
    return {"frames_with_masks": frames_with_masks, "video_b64": video_b64}
