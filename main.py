"""
Fashion RAG - Chatbot Tư Vấn Thời Trang Multimodal
===================================================
Entry point chính của hệ thống.

Cách chạy:
    python main.py              → Chạy chatbot (chế độ mặc định)
    python main.py --index      → Index dữ liệu sản phẩm vào Qdrant
    python main.py --index-b    → Index Layer B knowledge vào Qdrant
"""
import argparse
import os
import sys
import time
import uuid

from langchain_core.callbacks import BaseCallbackHandler


def run_indexing():
    """Index toàn bộ metadata sản phẩm thời trang vào Qdrant."""
    from fashion_rag.config.settings import FASHION_METADATA_DIR
    from fashion_rag.data import process_all_directory
    from fashion_rag.database import index_documents_to_qdrant
    from fashion_rag.embeddings import BGEM3Embeddings

    if not os.path.exists(FASHION_METADATA_DIR):
        print(f"[LỖI] Không tìm thấy thư mục: {FASHION_METADATA_DIR}")
        sys.exit(1)

    print(f"[THÔNG BÁO] Bắt đầu quét thư mục: {FASHION_METADATA_DIR}")
    print("=" * 50)

    all_docs = process_all_directory(FASHION_METADATA_DIR)

    print("=" * 50)
    print(f"[THÔNG BÁO] Tổng số chunk: {len(all_docs)}")

    embeddings = BGEM3Embeddings()
    index_documents_to_qdrant(all_docs, embeddings=embeddings, use_docker=True)


def run_index_layer_b():
    """Index Layer B knowledge vào Qdrant."""
    from fashion_rag.config.settings import (
        QDRANT_LAYER_B_FEMALE_COLLECTION,
        QDRANT_LAYER_B_MALE_COLLECTION,
    )
    from fashion_rag.database import load_vector_db
    from fashion_rag.layer_b import load_layer_b_knowledge, index_layer_b

    vector_db = load_vector_db(use_local=True)
    layer_b_female, layer_b_male = load_layer_b_knowledge()

    index_layer_b(layer_b_female, QDRANT_LAYER_B_FEMALE_COLLECTION, vector_db)
    index_layer_b(layer_b_male, QDRANT_LAYER_B_MALE_COLLECTION, vector_db)
    print("[OK] Layer B sẵn sàng cho semantic search!")


def run_chatbot():
    """Chạy chatbot tương tác trong terminal."""
    from fashion_rag.database import load_vector_db
    from fashion_rag.rag import build_rag_pipeline, detect_intent, detect_gender
    from fashion_rag.layer_b import build_outfit_context
    from fashion_rag.vision import detect_image_type, analyze_person_image, caption_product_image

    # ── Khởi tạo hệ thống ─────────────────────────────────────────
    vector_db = load_vector_db(use_local=True)
    full_chat_chain, outfit_chain_with_history, llm = build_rag_pipeline(vector_db)

    SESSION_ID = str(uuid.uuid4())
    user_profile = {}

    class SpyRetrieverHandler(BaseCallbackHandler):
        def on_retriever_start(self, serialized, query, **kwargs):
            print(f"\n🕵️  [Câu hỏi sau rewrite]: {query}\n")

    print("=" * 60)
    print("  👗👔  CHATBOT TƯ VẤN THỜI TRANG  👔👗  ")
    print("     Nhập '0' để thoát | Nhập đường dẫn ảnh nếu có")
    print("=" * 60 + "\n")

    while True:
        # ── Nhận input ────────────────────────────────────────────
        user_input = input("👤 Bạn: ").strip()
        if user_input == "0":
            print("\n🤖 Chatbot: Hẹn gặp lại bạn nhé!")
            break
        if not user_input:
            continue

        image_path = None
        final_query = user_input

        # ── Xử lý ảnh ────────────────────────────────────────────
        raw_img = input("📎 Ảnh (Enter để bỏ qua): ").strip()
        if raw_img and os.path.exists(raw_img):
            image_path = raw_img
            print("🔍 [Đang phân tích ảnh...]")

            image_type = detect_image_type(image_path)
            print(f"   → Phát hiện: {image_type.upper()}")

            if image_type == "person":
                person_info = analyze_person_image(image_path)
                if person_info["dang_nguoi"]:
                    user_profile["dang_nguoi"] = person_info["dang_nguoi"]
                if person_info["tone_da"]:
                    user_profile["tone_da"] = person_info["tone_da"]

                print(f"\n🤖 Chatbot: ", end="")
                print(
                    f"Mình đã phân tích xong rồi nhé! "
                    f"Bạn có **{person_info['dang_nguoi']}** "
                    f"với **{person_info['tone_da']}**. "
                    f"{person_info['nhan_xet']} "
                    f"\n\nMình đã lưu thông tin này lại để tư vấn phối đồ "
                    f"phù hợp hơn cho bạn. Bạn muốn mình gợi ý outfit cho "
                    f"dịp nào — đi làm, đi chơi hay đi tiệc?"
                )
                print("\n" + "-" * 60 + "\n")
                continue
            else:
                print("   → Đang đọc mô tả sản phẩm...")
                caption = caption_product_image(image_path)
                print(f"   → Caption: {caption[:80]}...")
                final_query = caption
                if user_input:
                    final_query = f"{caption}. Yêu cầu: {user_input}"

        elif raw_img and not os.path.exists(raw_img):
            print(f"   ⚠️  Không tìm thấy file: {raw_img}")

        # ── Phát hiện intent ──────────────────────────────────────
        intent = detect_intent(final_query)
        gender = detect_gender(final_query)
        print(
            f"🔍 [Intent: {intent.upper()} | Gender: {gender} | "
            f"Profile: {'✅' if user_profile else '(chưa có)'}]"
        )

        # ── Chạy pipeline ─────────────────────────────────────────
        print("🤖 Chatbot: ", end="")
        start_time = time.time()
        first_token_time = None

        try:
            if intent == "outfit":
                outfit_context = build_outfit_context(
                    user_query=final_query,
                    gender=gender,
                    profile=user_profile,
                    vector_db=vector_db,
                )

                if not outfit_context:
                    print("[Layer B không khớp → dùng RAG thông thường]")
                    intent = "search"
                else:
                    for chunk in outfit_chain_with_history.stream(
                        {"input": user_input, "outfit_context": outfit_context},
                        config={"configurable": {"session_id": SESSION_ID}},
                    ):
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time()
                            print(token, end="", flush=True)

            if intent == "search":
                for chunk in full_chat_chain.stream(
                    {"input": final_query},
                    config={
                        "configurable": {"session_id": SESSION_ID},
                        "callbacks": [SpyRetrieverHandler()],
                    },
                ):
                    if "answer" in chunk:
                        if first_token_time is None:
                            first_token_time = time.time()
                        print(chunk["answer"], end="", flush=True)

            end_time = time.time()
            if first_token_time is None:
                first_token_time = end_time

            print(
                f"\n\n⏱️  TTFT: {first_token_time - start_time:.2f}s | "
                f"Total: {end_time - start_time:.2f}s"
            )

        except Exception as e:
            print(f"\n[LỖI] {e}")

        print("\n\n" + "-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fashion RAG Chatbot")
    parser.add_argument("--index", action="store_true", help="Index sản phẩm vào Qdrant")
    parser.add_argument("--index-b", action="store_true", help="Index Layer B vào Qdrant")
    args = parser.parse_args()

    if args.index:
        run_indexing()
    elif args.index_b:
        run_index_layer_b()
    else:
        run_chatbot()


if __name__ == "__main__":
    main()
