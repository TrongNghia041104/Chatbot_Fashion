"""
app/api.py — FastAPI backend cho Fashion RAG Chatbot Web Demo
=============================================================
Cách chạy từ thư mục gốc:
    uvicorn app.api:app --reload --port 8000

Yêu cầu:
    - Qdrant chạy trên localhost:6333
    - Redis chạy trên localhost:6379
    - Ollama chạy trên localhost:11434
"""

import asyncio
import json
import os
import queue
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    API_TITLE, API_VERSION,
    STATIC_DIR, IMAGES_DIR,
)

# ── Import từ app.core ────────────────────────────────────────────────────────
print("[INFO] Đang load app.core...")
from app.core import (
    detect_image_type, analyze_person_image, caption_product_image,
    detect_intent, detect_gender,
    get_greeting_response, get_chitchat_response,
    build_outfit_context,
    full_chat_chain, outfit_chain_with_history, vector_db,
)
print("[OK] app.core loaded!")

# ── Khởi tạo FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ thư mục ảnh sản phẩm
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ── In-memory session store ───────────────────────────────────────────────────
sessions: dict = {}

_initialized = True  # app.core đã load ở import time


# ── Helper: tạo SSE event string ─────────────────────────────────────────────
def make_event(data: dict) -> str:
    """Tạo SSE event theo chuẩn 'data: {json}\\n\\n'."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Helper: stream chain qua queue (tránh block async event loop) ─────────────
def _stream_chain_via_queue(
    chain,
    input_dict: dict,
    config: dict,
    token_queue: queue.Queue,
    chain_type: str,
) -> None:
    """
    Chạy LangChain chain trong thread riêng, đẩy token vào queue.

    chain_type:
        "outfit" → chunk.content
        "search" → chunk["answer"], chunk["context"]
    """
    try:
        for chunk in chain.stream(input_dict, config=config):
            if chain_type == "outfit":
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    token_queue.put({"ok": True, "item_type": "token", "content": token})
            else:  # search
                if not isinstance(chunk, dict):
                    continue
                # Lấy ảnh từ context (chỉ xuất hiện ở chunk đầu tiên)
                if "context" in chunk:
                    images = []
                    for doc in chunk["context"]:
                        pid      = doc.metadata.get("product_id", "")
                        doc_imgs = [u for u in doc.metadata.get("images", []) if u]
                        if doc_imgs:
                            images.append({
                                "product_id": pid,
                                "category":   doc.metadata.get("category", ""),
                                "images":     doc_imgs[:2],
                            })
                    if images:
                        token_queue.put({"ok": True, "item_type": "images", "images": images})
                # Stream token trả lời
                token = chunk.get("answer", "")
                if token:
                    token_queue.put({"ok": True, "item_type": "token", "content": token})
    except Exception as e:
        token_queue.put({"ok": False, "error": str(e)})
    finally:
        token_queue.put(None)  # sentinel — báo kết thúc


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/api/session")
async def create_session():
    """Tạo session mới, trả về session_id."""
    sid = str(uuid.uuid4())
    sessions[sid] = {"profile": {}, "last_bot_msg": ""}
    return {"session_id": sid}


@app.get("/api/profile/{session_id}")
async def get_profile(session_id: str):
    """Lấy profile hiện tại của session."""
    return sessions.get(session_id, {}).get("profile", {})


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Xóa session (dùng khi New Chat)."""
    sessions.pop(session_id, None)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    session_id: str = Form(...),
    image: UploadFile = File(None),
):
    """
    Endpoint chat chính — trả về SSE stream.
    Frontend dùng fetch + ReadableStream để đọc.
    """

    async def event_stream():
        # ── Lấy state của session ─────────────────────────────────────
        if session_id not in sessions:
            sessions[session_id] = {"profile": {}, "last_bot_msg": ""}

        state        = sessions[session_id]
        profile      = state["profile"]
        last_bot_msg = state["last_bot_msg"]

        final_query      = message
        start_time       = time.time()
        first_token_time = None

        # ══ Xử lý ảnh ═══════════════════════════════════════════════
        if image and image.filename:
            suffix   = os.path.splitext(image.filename)[1] or ".jpg"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await image.read())
                    tmp_path = tmp.name

                image_type = await asyncio.to_thread(detect_image_type, tmp_path, message)

                if image_type == "person":
                    person_info = await asyncio.to_thread(analyze_person_image, tmp_path)

                    if person_info.get("dang_nguoi"):
                        profile["dang_nguoi"] = person_info["dang_nguoi"]
                    if person_info.get("tone_da"):
                        profile["tone_da"] = person_info["tone_da"]
                    sessions[session_id]["profile"] = profile

                    yield make_event({"type": "person_analyzed", **person_info})

                    bot_reply = (
                        f"Mình đã phân tích xong rồi nhé! "
                        f"Bạn có **{person_info.get('dang_nguoi', '...')}** "
                        f"với **{person_info.get('tone_da', '...')}**. "
                        f"{person_info.get('nhan_xet', '')} "
                        f"\n\nMình đã lưu thông tin này lại để tư vấn phối đồ "
                        f"phù hợp hơn cho bạn. Bạn muốn mình gợi ý outfit cho "
                        f"dịp nào — đi làm, đi chơi hay đi tiệc?"
                    )
                    for word in bot_reply.split(" "):
                        yield make_event({"type": "token", "content": word + " "})
                        await asyncio.sleep(0.02)

                    sessions[session_id]["last_bot_msg"] = bot_reply
                    yield make_event({
                        "type": "done",
                        "ttft": 0.0,
                        "total": round(time.time() - start_time, 2),
                    })
                    return

                else:
                    caption     = await asyncio.to_thread(caption_product_image, tmp_path, message)
                    yield make_event({"type": "product_captioned", "caption": caption})
                    final_query = f"{caption}. Yêu cầu: {message}" if message else caption

            except Exception as e:
                yield make_event({"type": "error", "message": f"Lỗi xử lý ảnh: {str(e)}"})
                return
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # ══ Detect intent & gender ════════════════════════════════════
        try:
            intent         = await asyncio.to_thread(detect_intent, final_query, last_bot_msg)
            current_gender = await asyncio.to_thread(detect_gender, final_query)
        except Exception as e:
            yield make_event({"type": "error", "message": f"Lỗi detect intent: {str(e)}"})
            return

        if current_gender == "male":
            profile["gender"] = "male"
        gender = profile.get("gender", current_gender or "female")

        yield make_event({"type": "intent_detected", "intent": intent, "gender": gender})

        # ══ Routing ══════════════════════════════════════════════════
        response_tokens: list[str] = []

        if intent == "greeting":
            reply = await asyncio.to_thread(get_greeting_response)
            yield make_event({"type": "token", "content": reply})
            sessions[session_id]["last_bot_msg"] = reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        if intent == "chitchat":
            reply = await asyncio.to_thread(get_chitchat_response, final_query)
            yield make_event({"type": "token", "content": reply})
            sessions[session_id]["last_bot_msg"] = reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        config      = {"configurable": {"session_id": session_id}}
        token_queue: queue.Queue = queue.Queue()

        if intent == "outfit":
            try:
                outfit_context, outfit_images = await asyncio.to_thread(
                    build_outfit_context, final_query, gender, profile,
                )
            except Exception:
                outfit_context, outfit_images = None, []

            if not outfit_context:
                intent = "search"   # fallback
            else:
                if outfit_images:
                    yield make_event({"type": "product_images", "images": outfit_images})
                chain_input = {"input": message or final_query, "outfit_context": outfit_context}
                t = threading.Thread(
                    target=_stream_chain_via_queue,
                    args=(outfit_chain_with_history, chain_input, config, token_queue, "outfit"),
                    daemon=True,
                )
                t.start()

        if intent == "search":
            chain_input = {"input": final_query}
            t = threading.Thread(
                target=_stream_chain_via_queue,
                args=(full_chat_chain, chain_input, config, token_queue, "search"),
                daemon=True,
            )
            t.start()

        # ── Đọc token từ queue và yield SSE ──────────────────────────
        while True:
            try:
                item = await asyncio.to_thread(token_queue.get, timeout=60)
            except Exception:
                yield make_event({"type": "error", "message": "Timeout chờ phản hồi từ LLM."})
                break

            if item is None:
                break

            if not item.get("ok", False):
                yield make_event({"type": "error", "message": item.get("error", "Lỗi không xác định")})
                break

            item_type = item.get("item_type", "token")
            if item_type == "images":
                yield make_event({"type": "product_images", "images": item["images"]})
            else:
                token = item.get("content", item.get("token", ""))
                if first_token_time is None:
                    first_token_time = time.time()
                response_tokens.append(token)
                yield make_event({"type": "token", "content": token})
            await asyncio.sleep(0)

        # ── Lưu response cuối cùng ────────────────────────────────────
        full_response = "".join(response_tokens)
        sessions[session_id]["last_bot_msg"] = full_response

        ttft = round((first_token_time - start_time), 2) if first_token_time else 0.0
        yield make_event({
            "type": "done",
            "ttft": ttft,
            "total": round(time.time() - start_time, 2),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Serve index.html ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve file index.html từ app/static/."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "initialized": _initialized}
