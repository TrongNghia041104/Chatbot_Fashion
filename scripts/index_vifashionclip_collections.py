"""Index research-demo v3 Qdrant collections.

Usage from Chatbot_Fashion:
    python scripts/index_vifashionclip_collections.py --text
    python scripts/index_vifashionclip_collections.py --image
    python scripts/index_vifashionclip_collections.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    METADATA_FILE,
    PRODUCT_IMAGE_ROOT,
    PRODUCT_VECTOR_SIZE,
    QDRANT_COLLECTION_FASHION,
    QDRANT_COLLECTION_PRODUCT_IMAGE,
)
from app.core.embeddings import get_product_embeddings  # noqa: E402
from app.core.image_search import run_main_image_index_pipeline  # noqa: E402
from app.core.product_data import process_fashion_metadata  # noqa: E402
from app.core.vector_store import get_qdrant_client  # noqa: E402


def index_text_collection(
    metadata_file: str | Path = METADATA_FILE,
    collection_name: str = QDRANT_COLLECTION_FASHION,
    batch_size: int = 128,
) -> None:
    """Index product text documents with ViFashionCLIP embeddings."""
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"[INFO] Reading metadata: {metadata_file}")
    all_docs = process_fashion_metadata(metadata_file)
    qdrant = get_qdrant_client()

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=PRODUCT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        current_count = 0
    else:
        current_count = qdrant.get_collection(collection_name).points_count
        print(f"[INFO] Text collection already has {current_count} points")

    remaining = all_docs[current_count:]
    if not remaining:
        print("[OK] Text collection is already fully indexed.")
        return

    vector_db = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        embedding=get_product_embeddings(),
    )
    with tqdm(total=len(all_docs), initial=current_count, desc="Text index", unit="product") as progress:
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            vector_db.add_documents(documents=batch)
            progress.update(len(batch))

    print(f"[OK] Text indexing completed -> {collection_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ViFashionCLIP text and FashionCLIP image collections.")
    parser.add_argument("--text", action="store_true", help="Index product text collection.")
    parser.add_argument("--image", action="store_true", help="Index product MAIN image collection.")
    parser.add_argument("--all", action="store_true", help="Index both text and image collections.")
    parser.add_argument("--metadata-file", default=METADATA_FILE, help="Product metadata JSONL path.")
    parser.add_argument("--image-root", default=PRODUCT_IMAGE_ROOT, help="Root folder containing product images.")
    parser.add_argument("--text-batch-size", type=int, default=128, help="Text indexing batch size.")
    parser.add_argument("--image-batch-size", type=int, default=64, help="Image indexing batch size.")
    args = parser.parse_args()

    run_text = args.all or args.text
    run_image = args.all or args.image
    if not run_text and not run_image:
        parser.error("Choose at least one of --text, --image, or --all.")

    if run_text:
        index_text_collection(
            metadata_file=args.metadata_file,
            collection_name=QDRANT_COLLECTION_FASHION,
            batch_size=args.text_batch_size,
        )
    if run_image:
        run_main_image_index_pipeline(
            metadata_file=args.metadata_file,
            image_root=args.image_root,
            collection_name=QDRANT_COLLECTION_PRODUCT_IMAGE,
            batch_size=args.image_batch_size,
        )


if __name__ == "__main__":
    main()
