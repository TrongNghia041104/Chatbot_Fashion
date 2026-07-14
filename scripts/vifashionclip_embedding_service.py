"""Remote ViFashionCLIP embedding service.

Run this on the GPU machine, usually Vast.ai:

    python scripts/vifashionclip_embedding_service.py --host 0.0.0.0 --port 18080 --preload

Then on local machine open an SSH tunnel:

    ssh -p <SSH_PORT> root@<VAST_IP> -L 18080:localhost:18080 -N

The local chatbot will call http://localhost:18080/embed for Layer A text
embeddings instead of loading the ViFashionCLIP checkpoint on local CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import PRODUCT_EMBEDDING_BATCH_SIZE, VIFASHIONCLIP_CHECKPOINT  # noqa: E402
from app.core.embeddings import ViFashionCLIPTextEmbeddings  # noqa: E402


app = FastAPI(title="ViFashionCLIP Embedding Service", version="1.0.0")

_model: ViFashionCLIPTextEmbeddings | None = None
_model_lock = Lock()
_started_at = time.time()


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    device: str
    count: int
    elapsed_sec: float


def get_model() -> ViFashionCLIPTextEmbeddings:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ViFashionCLIPTextEmbeddings(
                    checkpoint_path=VIFASHIONCLIP_CHECKPOINT,
                    batch_size=PRODUCT_EMBEDDING_BATCH_SIZE,
                )
    return _model


@app.get("/health")
def health():
    model_loaded = _model is not None
    device = str(_model.device) if _model is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": VIFASHIONCLIP_CHECKPOINT,
        "uptime_sec": round(time.time() - _started_at, 2),
    }


@app.post("/warmup")
def warmup():
    start = time.time()
    model = get_model()
    vector = model.embed_query("áo thun trắng nữ")
    return {
        "status": "ok",
        "device": str(model.device),
        "dim": len(vector),
        "elapsed_sec": round(time.time() - start, 3),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    start = time.time()
    texts = [text if isinstance(text, str) else str(text) for text in request.texts]
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    try:
        model = get_model()
        embeddings = model.embed_documents(texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    dim = len(embeddings[0]) if embeddings else 0
    return EmbedResponse(
        embeddings=embeddings,
        dim=dim,
        device=str(model.device),
        count=len(embeddings),
        elapsed_sec=round(time.time() - start, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ViFashionCLIP text embeddings over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--preload", action="store_true", help="Load model before accepting requests.")
    args = parser.parse_args()

    if args.preload:
        print("[INFO] Preloading ViFashionCLIP model...")
        get_model()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
