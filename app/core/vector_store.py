"""Qdrant access, product retrieval, and Layer B indexing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import torch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    ENABLE_PRODUCT_RERANKER,
    LAYER_B_FEMALE_PATH,
    LAYER_B_MALE_PATH,
    LAYER_B_VECTOR_SIZE,
    PRODUCT_SEARCH_BRAND_LIMIT,
    PRODUCT_SEARCH_CANDIDATE_K,
    PRODUCT_SEARCH_PAGE_SIZE,
    PRODUCT_VECTOR_SIZE,
    QDRANT_COLLECTION_FASHION,
    QDRANT_COLLECTION_LAYER_B_F,
    QDRANT_COLLECTION_LAYER_B_M,
    QDRANT_URL,
    RERANKER_BATCH_SIZE,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    RETRIEVAL_RETRY_COUNT,
    RETRIEVAL_RETRY_SLEEP,
)
from app.core.embeddings import get_product_embeddings, get_rule_embeddings


_client: QdrantClient | None = None
_vector_db: QdrantVectorStore | None = None
_retriever: BaseRetriever | None = None
_product_reranker = None
_reranker_enabled = ENABLE_PRODUCT_RERANKER
# get_product_retriever() có thể gọi tiếp get_product_vector_db() trong cùng
# critical section, nên cần re-entrant lock để không tự deadlock ở request đầu.
_lock = RLock()
_layer_b_lock = Lock()
_layer_b_ready = False


def get_qdrant_client() -> QdrantClient:
    """Lazy Qdrant client. Network calls happen when a collection is queried."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(url=QDRANT_URL)
    return _client


client = get_qdrant_client()


def _load_layer_b(file_path: str | Path) -> list[dict]:
    with Path(file_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


layer_b_female = _load_layer_b(LAYER_B_FEMALE_PATH)
layer_b_male = _load_layer_b(LAYER_B_MALE_PATH)


def _collection_exists(collection_name: str) -> bool:
    qdrant = get_qdrant_client()
    try:
        return qdrant.collection_exists(collection_name)
    except AttributeError:
        return collection_name in [c.name for c in qdrant.get_collections().collections]


def build_layer_b_embedding_text(rule: dict) -> str:
    """Build the semantic text indexed for one Layer B styling rule."""
    return " ".join(
        [
            str(rule.get("rule_key", "")),
            str(rule.get("phong_cach", "")),
            str(rule.get("boi_canh", "")),
            str(rule.get("ly_do_tu_van", "")),
        ]
    ).strip()


def index_layer_b(data: list[dict], collection_name: str, recreate: bool = False) -> None:
    """Index outfit-rule knowledge into Qdrant if the collection is missing."""
    qdrant = get_qdrant_client()
    if _collection_exists(collection_name):
        if recreate:
            qdrant.delete_collection(collection_name)
        else:
            print(f"[SKIP] {collection_name} already exists.")
            return

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=LAYER_B_VECTOR_SIZE, distance=Distance.COSINE),
    )
    rule_embeddings = get_rule_embeddings()
    points = []
    for i, rule in enumerate(data):
        text = build_layer_b_embedding_text(rule)
        points.append(PointStruct(id=i, vector=rule_embeddings.embed_query(text), payload=rule))
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Indexed {len(points)} Layer B rules -> {collection_name}")


def ensure_layer_b_indexed() -> None:
    """Verify Layer B collections once per process and create them if missing."""
    global _layer_b_ready
    if _layer_b_ready:
        return
    with _layer_b_lock:
        if _layer_b_ready:
            return
        index_layer_b(layer_b_female, QDRANT_COLLECTION_LAYER_B_F)
        index_layer_b(layer_b_male, QDRANT_COLLECTION_LAYER_B_M)
        _layer_b_ready = True


def normalize_product_metadata(doc: Document) -> Document:
    """Ensure each product document has stable image paths for rendering."""
    images = doc.metadata.get("images", [])
    if isinstance(images, str):
        text = images.strip()
        if text.startswith(("[", "{")):
            try:
                images = json.loads(text)
            except json.JSONDecodeError:
                images = [images] if images else []
        else:
            images = [images] if images else []
    if isinstance(images, dict):
        images = [images]

    normalized_images: list[str] = []
    for image in images or []:
        if isinstance(image, dict):
            value = (
                image.get("large")
                or image.get("hi_res")
                or image.get("url")
                or image.get("image_url")
                or image.get("path")
            )
        else:
            value = image
        value = str(value or "").strip()
        if value and value not in normalized_images:
            normalized_images.append(value)

    main_image = str(
        doc.metadata.get("image_url")
        or doc.metadata.get("main_image_relpath")
        or doc.metadata.get("main_image_path")
        or ""
    ).strip()
    if not main_image and normalized_images:
        main_image = normalized_images[0]

    doc.metadata["images"] = normalized_images
    doc.metadata["image_url"] = main_image
    return doc


def diversity_filter_documents(
    docs: list[Document],
    max_docs: int = PRODUCT_SEARCH_PAGE_SIZE,
    max_per_brand: int = PRODUCT_SEARCH_BRAND_LIMIT,
) -> list[Document]:
    """Dedupe by product_id and limit repeated brands."""
    selected: list[Document] = []
    seen_product_ids: set[str] = set()
    brand_counts: dict[str, int] = {}

    for doc in docs:
        doc = normalize_product_metadata(doc)
        product_id = str(doc.metadata.get("product_id", "")).strip().lower()
        brand = str(doc.metadata.get("brand", "")).strip().lower()
        if product_id and product_id in seen_product_ids:
            continue
        if brand and brand_counts.get(brand, 0) >= max_per_brand:
            continue
        selected.append(doc)
        if product_id:
            seen_product_ids.add(product_id)
        if brand:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        if len(selected) >= max_docs:
            return selected

    for doc in docs:
        if len(selected) >= max_docs:
            break
        doc = normalize_product_metadata(doc)
        product_id = str(doc.metadata.get("product_id", "")).strip().lower()
        if product_id and product_id in seen_product_ids:
            continue
        selected.append(doc)
        if product_id:
            seen_product_ids.add(product_id)

    return selected


class ProductCrossEncoderReranker:
    """Lazy cross-encoder reranker with safe fallback."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[OK] Reranker loaded: {model_name} on {self.device}")

    def score_pairs(
        self,
        query: str,
        docs: list[Document],
        batch_size: int = RERANKER_BATCH_SIZE,
    ) -> list[float]:
        scores: list[float] = []
        pairs = [(query, doc.page_content[:1400]) for doc in docs]
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits
                batch_scores = logits[:, -1] if logits.ndim == 2 and logits.shape[-1] > 1 else logits.reshape(-1)
                scores.extend(batch_scores.detach().float().cpu().tolist())
        return scores


def get_product_reranker():
    """Load reranker lazily; disable it if unavailable."""
    global _product_reranker, _reranker_enabled
    if not _reranker_enabled:
        return None
    if _product_reranker is not None:
        return _product_reranker
    try:
        _product_reranker = ProductCrossEncoderReranker(RERANKER_MODEL_NAME)
        return _product_reranker
    except Exception as exc:
        _reranker_enabled = False
        print(f"[WARN] Reranker unavailable -> dense retrieval only: {exc}")
        return None


def is_reranker_enabled() -> bool:
    return bool(_reranker_enabled)


def rerank_documents(query: str, docs: list[Document], top_n: int = RERANKER_TOP_N) -> list[Document]:
    reranker = get_product_reranker()
    if reranker is None or not docs:
        return docs
    try:
        scores = reranker.score_pairs(query, docs)
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        output = []
        for rank, (doc, score) in enumerate(ranked[:top_n], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank
            output.append(doc)
        return output
    except Exception as exc:
        print(f"[WARN] Rerank failed -> dense order: {exc}")
        return docs


def safe_base_retrieve(base_retriever, query: str) -> list[Document]:
    """Retry Qdrant retrieval; return [] instead of crashing the chat."""
    last_error = None
    total_attempts = RETRIEVAL_RETRY_COUNT + 1
    for attempt in range(total_attempts):
        try:
            return base_retriever.invoke(query)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Retrieval failed {attempt + 1}/{total_attempts}: {type(exc).__name__}: {exc}")
            if attempt < total_attempts - 1:
                time.sleep(RETRIEVAL_RETRY_SLEEP * (attempt + 1))
    print(f"[ERROR] Retrieval unavailable for query={query!r}. Last error: {last_error}")
    return []


class DiversityFilteredRetriever(BaseRetriever):
    """Qdrant candidates -> optional rerank -> dedupe/diversify."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    max_docs: int = PRODUCT_SEARCH_PAGE_SIZE

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        raw_docs = safe_base_retrieve(self.base_retriever, query)
        if not raw_docs:
            return []
        ranked_docs = rerank_documents(query, raw_docs, top_n=RERANKER_TOP_N)
        return diversity_filter_documents(ranked_docs, max_docs=self.max_docs)


def get_product_vector_db() -> QdrantVectorStore:
    """Lazy Layer A ViFashionCLIP vector store."""
    global _vector_db
    if _vector_db is None:
        with _lock:
            if _vector_db is None:
                _vector_db = QdrantVectorStore(
                    client=get_qdrant_client(),
                    collection_name=QDRANT_COLLECTION_FASHION,
                    embedding=get_product_embeddings(),
                )
    return _vector_db


def get_product_retriever() -> BaseRetriever:
    """Lazy product retriever used by the RAG chain."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                base_retriever = get_product_vector_db().as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": PRODUCT_SEARCH_CANDIDATE_K},
                )
                _retriever = DiversityFilteredRetriever(
                    base_retriever=base_retriever,
                    max_docs=PRODUCT_SEARCH_PAGE_SIZE,
                )
    return _retriever
