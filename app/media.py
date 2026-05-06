from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COLORS = [
    (0, 200, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
]


def frame_to_b64(
    path: Path | str,
    masks: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
    points: list[tuple[float, float, bool]] | None = None,
) -> str:
    img = cv2.imread(str(path))
    if masks:
        for mask, color in masks:
            overlay = img.copy().astype(np.float32)
            overlay[mask] = np.array(color, dtype=np.float32)
            img = cv2.addWeighted(overlay.astype(np.uint8), 0.5, img, 0.5, 0)
    if points:
        for x, y, pos in points:
            col = (0, 255, 0) if pos else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 8, col, -1)
            cv2.circle(img, (int(x), int(y)), 8, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def extract_frames(video_path: str, out_dir: Path, fps: float = 5.0, max_height: int = 720) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-q:v",
            "2",
            "-vf",
            f"fps={fps},scale=-2:min({max_height}\\,ih)",
            str(out_dir / "%07d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def h264_encode(mp4_path: str) -> Path:
    out = Path(mp4_path.replace(".mp4", "_h264.mp4"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-c:v", "libx264", "-crf", "23", str(out)],
        capture_output=True,
        check=False,
    )
    return out


def ffprobe_video(path: Path) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(r.stdout)
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})
    dur = float(fmt.get("duration", 0))
    fps_raw = vs.get("r_frame_rate", "0/1")
    num, den = map(int, fps_raw.split("/"))
    fps = round(num / den, 2) if den else 0.0
    return {
        "duration_s": round(dur, 1),
        "duration_str": f"{int(dur // 60)}m {int(dur % 60)}s",
        "fps": fps,
        "width": vs.get("width"),
        "height": vs.get("height"),
        "codec": vs.get("codec_name"),
        "size_mb": round(int(fmt.get("size", 0)) / 1e6, 1),
    }
