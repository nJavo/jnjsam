from __future__ import annotations

import os
from pathlib import Path


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _app_root().parent


def _load_dotenv() -> None:
    path = _app_root() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _default_workspace() -> Path:
    return Path(os.environ.get("WORKSPACE_ROOT", _repo_root().parent)).resolve()


class Settings:
    def __init__(self) -> None:
        w = _default_workspace()
        self.workspace_root = w
        self.sam3_repo = Path(os.environ.get("SAM3_REPO", w / "sam3"))
        self.sam3_checkpoint = Path(
            os.environ.get("SAM3_CHECKPOINT", self.sam3_repo / "checkpoints" / "sam3.pt")
        )
        self.sam3_bpe = Path(
            os.environ.get(
                "SAM3_BPE",
                self.sam3_repo / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
            )
        )
        self.sam2_repo = Path(os.environ.get("SAM2_REPO", w / "sam2"))
        self.sam2_checkpoint = Path(
            os.environ.get(
                "SAM2_CHECKPOINT",
                self.sam2_repo / "checkpoints" / "sam2.1_hiera_tiny.pt",
            )
        )
        self.sam2_config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.videos_dir = Path(os.environ.get("VIDEOS_DIR", w / "videos"))
        self.frames_sam3 = Path(os.environ.get("FRAMES_SAM3", "/tmp/sam3_annotator"))
        self.frames_medsam3 = Path(os.environ.get("FRAMES_MEDSAM3", "/tmp/medsam3_annotator"))
        self.medsam3_repo = Path(os.environ.get("MEDSAM3_REPO", w / "MedSAM3"))
        self.medsam3_lora_config = Path(
            os.environ.get(
                "MEDSAM3_LORA_CONFIG",
                self.medsam3_repo / "configs" / "full_lora_config.yaml",
            )
        )
        self.medsam3_lora_weights = Path(
            os.environ.get(
                "MEDSAM3_LORA_WEIGHTS",
                self.medsam3_repo / "weights" / "best_lora_weights.pt",
            )
        )
        self.static_dir = Path(os.environ.get("ANNOTATOR_STATIC", _app_root() / "static"))
