"""FashionCLIP image retrieval for uploaded product images."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
from langchain_core.documents import Document
from PIL import Image
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from app.config import (
    IMAGE_EMBEDDING_BATCH_SIZE,
    IMAGE_SEARCH_MAX_PRODUCTS,
    IMAGE_SEARCH_SCORE_THRESHOLD,
    IMAGE_SEARCH_TOP_K,
    IMAGE_VECTOR_SIZE,
    METADATA_FILE,
    PRODUCT_IMAGE_ROOT,
    QDRANT_COLLECTION_PRODUCT_IMAGE,
    TEACHER_MODEL_NAME,
)
from app.core.product_data import build_product_metadata, build_product_page_content
from app.core.vector_store import get_qdrant_client, normalize_product_metadata


class FashionCLIPImageEmbeddings:
    """FashionCLIP image encoder used for image indexing and query search."""

    def __init__(self, model_name: str = TEACHER_MODEL_NAME, batch_size: int = IMAGE_EMBEDDING_BATCH_SIZE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_name = model_name

        try:
            # Ưu tiên cache local để buổi demo không phụ thuộc kết nối Hugging Face.
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            print("[WARN] FashionCLIP cache chưa đầy đủ; đang tải phần còn thiếu từ Hugging Face...")
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = int(getattr(self.model.config, "projection_dim", IMAGE_VECTOR_SIZE))
        print(f"[OK] FashionCLIP image encoder: {model_name}")
        print(f"     Device: {self.device} | Dim: {self.embedding_dim}")

    @staticmethod
    def _open_image(image_path: str | Path):
        return Image.open(image_path).convert("RGB")

    @torch.no_grad()
    def encode_image_paths(self, image_paths: list[str | Path]) -> list[list[float] | None]:
        results: list[list[float] | None] = [None] * len(image_paths)
        for start in tqdm(range(0, len(image_paths), self.batch_size), desc="FashionCLIP image embed", leave=False):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            valid_positions = []

            for offset, path in enumerate(batch_paths):
                try:
                    images.append(self._open_image(path))
                    valid_positions.append(start + offset)
                except Exception as exc:
                    print(f"[WARN] Cannot read image {path}: {exc}")

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            embeds = self.model.get_image_features(**inputs)
            if hasattr(embeds, "pooler_output"):
                embeds = embeds.pooler_output
            embeds = F.normalize(embeds, p=2, dim=-1).detach().cpu().tolist()
            for pos, vector in zip(valid_positions, embeds):
                results[pos] = vector

        return results

    def embed_image(self, image_path: str | Path) -> list[float] | None:
        return self.encode_image_paths([image_path])[0]


_image_embeddings_instance: FashionCLIPImageEmbeddings | None = None
_image_lock = Lock()


def get_image_embeddings() -> FashionCLIPImageEmbeddings:
    """Lazy singleton for the image encoder."""
    global _image_embeddings_instance
    if _image_embeddings_instance is None:
        with _image_lock:
            if _image_embeddings_instance is None:
                print("[INFO] Loading FashionCLIP image encoder for the first image request...")
                _image_embeddings_instance = FashionCLIPImageEmbeddings()
    return _image_embeddings_instance


def get_main_image_relative_path(item: dict) -> str:
    images = item.get("images", []) or []
    for image in images:
        if str(image.get("variant", "")).upper() == "MAIN" and image.get("large"):
            return image["large"]
    for image in images:
        large = str(image.get("large", ""))
        if "_MAIN" in large.upper():
            return large
    if images and images[0].get("large"):
        return images[0]["large"]
    product_id = item.get("product_id")
    return f"images/{product_id}_MAIN.jpg" if product_id else ""


def resolve_main_image_path(relative_path: str, image_root: str | Path = PRODUCT_IMAGE_ROOT) -> Path:
    rel = Path(str(relative_path).replace("\\", "/"))
    if rel.parts and rel.parts[0].lower() == "images":
        rel = Path(*rel.parts[1:])
    return Path(image_root) / rel


def iter_products_with_main_image(
    metadata_file: str | Path = METADATA_FILE,
    image_root: str | Path = PRODUCT_IMAGE_ROOT,
):
    metadata_file = Path(metadata_file)
    image_root = Path(image_root)
    with metadata_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = get_main_image_relative_path(item)
            img_path = resolve_main_image_path(rel_path, image_root)
            if img_path.exists():
                yield item, rel_path, img_path


def build_image_payload(item: dict, rel_path: str, img_path: Path) -> dict:
    metadata = build_product_metadata(item)
    metadata["main_image_path"] = str(img_path)
    metadata["main_image_relpath"] = rel_path
    metadata["image_url"] = metadata.get("image_url") or rel_path
    return {
        "product_id": metadata.get("product_id", ""),
        "title": metadata.get("title", ""),
        "category": metadata.get("category", ""),
        "department": metadata.get("department", ""),
        "brand": metadata.get("brand", ""),
        "price": metadata.get("price", 0),
        "image_url": metadata.get("image_url", ""),
        "main_image_path": str(img_path),
        "main_image_relpath": rel_path,
        "page_content": build_product_page_content(item),
        "metadata": metadata,
    }


def run_main_image_index_pipeline(
    metadata_file: str | Path = METADATA_FILE,
    image_root: str | Path = PRODUCT_IMAGE_ROOT,
    collection_name: str = QDRANT_COLLECTION_PRODUCT_IMAGE,
    batch_size: int = 64,
) -> None:
    """Index one MAIN image per product into Qdrant with FashionCLIP image vectors."""
    metadata_file = Path(metadata_file)
    image_root = Path(image_root)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    qdrant = get_qdrant_client()
    items = list(iter_products_with_main_image(metadata_file, image_root))
    print(f"[OK] Found {len(items)} products with MAIN images")

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
        )
        current_count = 0
    else:
        current_count = qdrant.count(collection_name).count
        print(f"[INFO] Image collection already has {current_count} vectors")

    remaining = items[current_count:]
    if not remaining:
        print("[OK] Image collection is already fully indexed.")
        return

    embeddings = get_image_embeddings()
    with tqdm(total=len(items), initial=current_count, desc="Image index", unit="img") as progress:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            image_paths = [img_path for _, _, img_path in batch]
            vectors = embeddings.encode_image_paths(image_paths)
            points = []
            for (item, rel_path, img_path), vector in zip(batch, vectors):
                if vector is None:
                    continue
                product_id = str(item.get("product_id", ""))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"fashion-main-image:{product_id}"))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=build_image_payload(item, rel_path, img_path),
                    )
                )
            if points:
                qdrant.upsert(collection_name=collection_name, points=points)
            progress.update(len(batch))

    print(f"[OK] Image indexing completed -> {collection_name}")


def image_point_to_document(point) -> Document:
    payload = point.payload or {}
    metadata = payload.get("metadata", {}) or {}
    metadata["image_search_score"] = getattr(point, "score", None)
    metadata["image_url"] = payload.get("image_url") or metadata.get("image_url", "")
    return Document(page_content=payload.get("page_content", ""), metadata=metadata)


def search_products_by_image(
    image_path: str | Path,
    collection_name: str = QDRANT_COLLECTION_PRODUCT_IMAGE,
    top_k: int = IMAGE_SEARCH_TOP_K,
    max_products: int = IMAGE_SEARCH_MAX_PRODUCTS,
    score_threshold: float | None = IMAGE_SEARCH_SCORE_THRESHOLD,
) -> list[Document]:
    """Search products similar to an uploaded image."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return []

    qdrant = get_qdrant_client()
    if not qdrant.collection_exists(collection_name):
        print(f"[WARN] Image collection does not exist: {collection_name}")
        return []

    query_vector = get_image_embeddings().embed_image(image_path)
    if query_vector is None:
        return []

    kwargs = {
        "collection_name": collection_name,
        "query": query_vector,
        "limit": top_k,
        "with_payload": True,
    }
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold

    response = qdrant.query_points(**kwargs)
    docs: list[Document] = []
    seen_ids: set[str] = set()
    for point in response.points:
        doc = image_point_to_document(point)
        product_id = doc.metadata.get("product_id", "")
        if product_id and product_id in seen_ids:
            continue
        docs.append(normalize_product_metadata(doc))
        if product_id:
            seen_ids.add(product_id)
        if len(docs) >= max_products:
            break
    return docs
