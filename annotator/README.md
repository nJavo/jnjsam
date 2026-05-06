Install PyTorch separately. Then `pip install -r annotator/requirements.txt`.

From the repository root (the directory that contains `annotator/`):

```bash
export PYTHONPATH="$(pwd)/annotator"
export PYTHONPATH="/path/to/sam3:/path/to/sam2:${PYTHONPATH}"
uvicorn app.sam3:app --host 127.0.0.1 --port 8000
```

MedSAM3 server:

```bash
export PYTHONPATH="$(pwd)/annotator:/path/to/sam3:/path/to/sam2:/path/to/MedSAM3"
uvicorn app.medsam3:app --host 127.0.0.1 --port 8001
```

Copy `annotator/.env.example` to `annotator/.env` and adjust paths if defaults are wrong.
