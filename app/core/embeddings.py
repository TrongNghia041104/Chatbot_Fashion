"""Embedding backends for research-demo v3.

Layer A product retrieval uses ViFashionCLIP text embeddings (512 dim).
Layer B outfit-rule retrieval keeps BGE-M3 through Ollama (1024 dim).
Both are lazy-loaded so importing the API stays lightweight.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from app.config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PRODUCT_EMBEDDING_BATCH_SIZE,
    PRODUCT_EMBEDDING_BACKEND,
    PROJECTION_DROPOUT,
    PROJECTION_HIDDEN_DIM,
    PROJECTION_NUM_LAYERS,
    REMOTE_EMBEDDING_FALLBACK_LOCAL,
    STUDENT_MAX_LENGTH,
    STUDENT_MODEL_NAME,
    VIFASHIONCLIP_SERVICE_TIMEOUT,
    VIFASHIONCLIP_SERVICE_URL,
    VIFASHIONCLIP_CHECKPOINT,
)


class ResidualMLPBlock(nn.Module):
    """Residual MLP block used by the trained ViFashionCLIP projection head."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ProjectionHead(nn.Module):
    """Map student encoder embeddings into FashionCLIP's 512-dim space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, num_layers - 2)):
            layers.append(ResidualMLPBlock(hidden_dim, dropout=dropout))
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


class Stage2StudentProjection(nn.Module):
    """Student encoder + projection head saved by the ViFashionCLIP training run."""

    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        return self.projection_head(pooled)


class BGEM3Embeddings(Embeddings):
    """BGE-M3 wrapper for Layer B outfit-rule embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.ollama_embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.ollama_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.ollama_embeddings.embed_query(text)


class ViFashionCLIPTextEmbeddings(Embeddings):
    """LangChain wrapper for the fine-tuned Vietnamese FashionCLIP text model."""

    def __init__(
        self,
        checkpoint_path: str | Path = VIFASHIONCLIP_CHECKPOINT,
        batch_size: int = PRODUCT_EMBEDDING_BATCH_SIZE,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ViFashionCLIP checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        student_name = checkpoint.get("student_model_name", STUDENT_MODEL_NAME)
        teacher_dim = int(checkpoint.get("teacher_dim", 512))

        self.tokenizer = AutoTokenizer.from_pretrained(student_name)
        encoder = AutoModel.from_pretrained(student_name)
        projection = ProjectionHead(
            input_dim=encoder.config.hidden_size,
            output_dim=teacher_dim,
            hidden_dim=PROJECTION_HIDDEN_DIM,
            num_layers=PROJECTION_NUM_LAYERS,
            dropout=PROJECTION_DROPOUT,
        )

        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        projection.load_state_dict(checkpoint["projection_state_dict"])
        self.model = Stage2StudentProjection(encoder, projection).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = teacher_dim
        # Use .name to avoid UnicodeEncodeError on Windows consoles when the path has Vietnamese characters
        print(f"[OK] ViFashionCLIP loaded: {self.checkpoint_path.name}")
        print(f"     Device: {self.device} | Dim: {self.embedding_dim}")

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> list[list[float]]:
        all_embeds = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="ViFashionCLIP embed", leave=False):
            batch = [str(text) for text in texts[start : start + self.batch_size]]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=STUDENT_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            embeds = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.detach().cpu())
        return torch.cat(all_embeds, dim=0).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Return cached embedding if available, otherwise encode and cache."""
        if text in _embed_query_cache:
            return _embed_query_cache[text]
        result = self._encode([text])[0]
        with _cache_lock:
            if len(_embed_query_cache) >= _EMBED_CACHE_MAX:
                _embed_query_cache.pop(next(iter(_embed_query_cache)))
            _embed_query_cache[text] = result
        return result


class RemoteViFashionCLIPTextEmbeddings(Embeddings):
    """LangChain embedding client that calls a remote ViFashionCLIP HTTP service."""

    def __init__(
        self,
        service_url: str = VIFASHIONCLIP_SERVICE_URL,
        batch_size: int = PRODUCT_EMBEDDING_BATCH_SIZE,
        timeout: float = VIFASHIONCLIP_SERVICE_TIMEOUT,
    ):
        self.service_url = service_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.embedding_dim = 512
        self._session = requests.Session()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._session.post(
                f"{self.service_url}/embed",
                json={"texts": [str(text) for text in texts]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Remote ViFashionCLIP embedding service is unavailable. "
                f"Expected service at {self.service_url}. "
                "Start it on Vast.ai and open the SSH tunnel, or set "
                "PRODUCT_EMBEDDING_BACKEND=local to use local CPU/GPU."
            ) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Invalid embedding response from {self.service_url}: {data}")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            all_vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """Return cached embedding if available, otherwise call remote service and cache."""
        if text in _embed_query_cache:
            return _embed_query_cache[text]
        result = self._embed_batch([text])[0]
        with _cache_lock:
            if len(_embed_query_cache) >= _EMBED_CACHE_MAX:
                _embed_query_cache.pop(next(iter(_embed_query_cache)))
            _embed_query_cache[text] = result
        return result

    def health(self) -> dict:
        response = self._session.get(f"{self.service_url}/health", timeout=min(self.timeout, 10))
        response.raise_for_status()
        return response.json()


_rule_embeddings: BGEM3Embeddings | None = None
_product_embeddings: Embeddings | None = None
_local_product_embeddings: ViFashionCLIPTextEmbeddings | None = None
_embedding_lock = Lock()

# Simple in-memory cache for embed_query — avoids redundant HTTP calls / GPU runs
# for the same (rewritten) query within a session or across concurrent sessions.
_embed_query_cache: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 256
_cache_lock = Lock()


def _cached_embed_query(embedder: Embeddings, text: str) -> list[float]:
    """Return cached embedding or compute and store it."""
    if text in _embed_query_cache:
        return _embed_query_cache[text]
    result = embedder.embed_query.__wrapped__(embedder, text) if hasattr(embedder.embed_query, "__wrapped__") else _raw_embed_query(embedder, text)
    with _cache_lock:
        if len(_embed_query_cache) >= _EMBED_CACHE_MAX:
            # Evict oldest entry (insertion-order dict, Python 3.7+)
            _embed_query_cache.pop(next(iter(_embed_query_cache)))
        _embed_query_cache[text] = result
    return result


def _raw_embed_query(embedder: Embeddings, text: str) -> list[float]:
    """Call the underlying embed without cache (used by _cached_embed_query)."""
    if isinstance(embedder, RemoteViFashionCLIPTextEmbeddings):
        return embedder._embed_batch([text])[0]
    if isinstance(embedder, ViFashionCLIPTextEmbeddings):
        return embedder._encode([text])[0]
    # Fallback for BGEM3 or unknown
    return embedder.embed_query(text)


def get_rule_embeddings() -> BGEM3Embeddings:
    """Lazy singleton for Layer B BGE-M3 embeddings."""
    global _rule_embeddings
    if _rule_embeddings is None:
        with _embedding_lock:
            if _rule_embeddings is None:
                print("[INFO] Loading BGE-M3 embeddings for Layer B...")
                _rule_embeddings = BGEM3Embeddings()
    return _rule_embeddings


def get_product_embeddings() -> Embeddings:
    """Lazy singleton for Layer A ViFashionCLIP text embeddings."""
    global _product_embeddings, _local_product_embeddings
    if _product_embeddings is None:
        with _embedding_lock:
            if _product_embeddings is None:
                if PRODUCT_EMBEDDING_BACKEND == "local":
                    print("[INFO] Loading local ViFashionCLIP text embeddings for Layer A...")
                    _product_embeddings = ViFashionCLIPTextEmbeddings()
                elif PRODUCT_EMBEDDING_BACKEND == "remote":
                    print(f"[INFO] Using remote ViFashionCLIP embedding service: {VIFASHIONCLIP_SERVICE_URL}")
                    remote = RemoteViFashionCLIPTextEmbeddings()
                    if REMOTE_EMBEDDING_FALLBACK_LOCAL:
                        try:
                            remote.health()
                        except Exception as exc:
                            print(f"[WARN] Remote embedding service unavailable -> local fallback: {exc}")
                            _local_product_embeddings = ViFashionCLIPTextEmbeddings()
                            _product_embeddings = _local_product_embeddings
                        else:
                            _product_embeddings = remote
                    else:
                        _product_embeddings = remote
                else:
                    raise ValueError(
                        "PRODUCT_EMBEDDING_BACKEND must be 'remote' or 'local', "
                        f"got {PRODUCT_EMBEDDING_BACKEND!r}"
                    )
    return _product_embeddings
