"""
Quản lý kết nối và thao tác với Qdrant Vector Database.
Bao gồm:
  - Khởi tạo VectorStore cho truy vấn (inference)
  - Index dữ liệu mới vào database (ingestion)
"""
import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

from fashion_rag.config.settings import (
    QDRANT_URL,
    QDRANT_LOCAL_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_VECTOR_SIZE,
    INDEXING_BATCH_SIZE,
)
from fashion_rag.embeddings import BGEM3Embeddings


def load_vector_db(use_local: bool = True) -> QdrantVectorStore:
    """
    Khởi tạo QdrantVectorStore để truy vấn sản phẩm.

    Args:
        use_local: True để dùng local storage, False để dùng Docker Qdrant.

    Returns:
        QdrantVectorStore đã kết nối và sẵn sàng truy vấn.
    """
    print("[THÔNG BÁO] Đang khởi tạo mô hình Embedding...")
    custom_embeddings = BGEM3Embeddings()

    print("[THÔNG BÁO] Đang kết nối tới Qdrant Database...")
    if use_local:
        client = QdrantClient(path=QDRANT_LOCAL_PATH)
    else:
        client = QdrantClient(url=QDRANT_URL)

    vector_db = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=custom_embeddings,
    )

    print("✅ Đã kết nối thành công!")
    return vector_db


def index_documents_to_qdrant(
    all_docs: list,
    embeddings: BGEM3Embeddings = None,
    use_docker: bool = True,
) -> None:
    """
    Index toàn bộ documents vào Qdrant với khả năng resume.
    Hỗ trợ tiếp tục từ điểm dừng nếu quá trình bị gián đoạn.

    Args:
        all_docs: Danh sách LangChain Documents cần index.
        embeddings: Custom embedding model. Nếu None sẽ tự khởi tạo.
        use_docker: True để dùng Docker Qdrant, False để dùng local.
    """
    if embeddings is None:
        embeddings = BGEM3Embeddings()

    if use_docker:
        client = QdrantClient(url=QDRANT_URL)
    else:
        client = QdrantClient(path=QDRANT_LOCAL_PATH)

    collection_name = QDRANT_COLLECTION_NAME

    # Kiểm tra số lượng đã lưu
    if not client.collection_exists(collection_name):
        print(f"[THÔNG BÁO] Tạo mới collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_VECTOR_SIZE, distance=Distance.COSINE
            ),
        )
        current_count = 0
    else:
        collection_info = client.get_collection(collection_name)
        current_count = collection_info.points_count
        print(
            f"[THÔNG BÁO] Tìm thấy {current_count} sản phẩm đã được lưu trước đó."
        )

    # Cắt bỏ phần đã xử lý
    remaining_docs = all_docs[current_count:]

    if len(remaining_docs) == 0:
        print("\n[THÔNG BÁO] 🎉 Toàn bộ dữ liệu đã được xử lý xong!")
        return

    print(f"[THÔNG BÁO] Bắt đầu nhúng và lưu {len(remaining_docs)} sản phẩm còn lại...")

    vector_db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    batch_size = INDEXING_BATCH_SIZE

    with tqdm(
        total=len(all_docs),
        initial=current_count,
        desc="Tiến độ Vector hóa",
        unit="SP",
    ) as pbar:
        for i in range(0, len(remaining_docs), batch_size):
            batch = remaining_docs[i : i + batch_size]
            vector_db.add_documents(documents=batch)
            pbar.update(len(batch))

    print("\n[THÔNG BÁO] 🎉 ĐÃ LƯU HOÀN TẤT VÀO VECTOR DATABASE!")
