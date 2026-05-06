# Surgical video annotator

Two FastAPI apps share one static UI (`static/index.html`). Configure paths via `.env` (see `.env.example`). Default `WORKSPACE_ROOT` is the parent of this directory (expects `sam3/`, `sam2/`, `MedSAM3/`, `videos/` next to `annotator/`).

Install PyTorch for your platform (see [pytorch.org](https://pytorch.org)) before `pip install -r requirements.txt` if it is not already in the environment.

## Run

From this directory, with PyTorch and SAM repos installed as in `.env.example`:

```bash
source /path/to/venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"   # so `app` imports
export PYTHONPATH="/path/to/sam3:/path/to/sam2:${PYTHONPATH}"
uvicorn app.sam3:app --host 127.0.0.1 --port 8000
```

MedSAM3 server (add MedSAM3 repo for `lora_layers`):

```bash
export PYTHONPATH="$(pwd):/path/to/sam3:/path/to/sam2:/path/to/MedSAM3:${PYTHONPATH}"
uvicorn app.medsam3:app --host 127.0.0.1 --port 8001
```

Place checkpoints under paths in `.env`; large `*.pt` files should stay untracked (see `.gitignore`).
