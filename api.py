"""
api.py — FastAPI backend cho Fashion RAG Chatbot Web Demo
=========================================================
Cách chạy:
    /venv/main/bin/uvicorn api:app --reload --port 8000

Yêu cầu: chatbot_core.py phải nằm cùng thư mục
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

# ── Import từ chatbot_core (extracted từ Chatbot_RAG_MultiModal.ipynb) ────────
print("[INFO] Đang load chatbot_core...")
from chatbot_core import (
    detect_image_type, analyze_person_image, caption_product_image,
    detect_intent, detect_gender,
    get_greeting_response, get_chitchat_response,
    build_outfit_context,
    full_chat_chain, outfit_chain_with_history, vector_db,
)
# Alias để dùng chung tên biến nội bộ
_full_chat_chain      = full_chat_chain
_outfit_chain         = outfit_chain_with_history
_detect_intent        = detect_intent
_detect_gender        = detect_gender
_build_outfit_context = build_outfit_context
_detect_image_type    = detect_image_type
_analyze_person_image = analyze_person_image
_caption_product_image= caption_product_image
_get_greeting_response= get_greeting_response
_get_chitchat_response= get_chitchat_response
_vector_db            = vector_db
print("[OK] chatbot_core loaded!")

# ── Khởi tạo FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Fashion RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
sessions: dict = {}


# ── Helper: tạo SSE event string ─────────────────────────────────────────────
def make_event(data: dict) -> str:
    """Tạo SSE event theo chuẩn 'data: {json}\\n\\n'."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

_initialized = True  # chatbot_core đã load ở import time


# ── Helper: stream chain qua queue (tránh block async event loop) ─────────────
def _stream_chain_via_queue(chain, input_dict: dict, config: dict, token_queue: queue.Queue, chain_type: str):
    """
    Chạy LangChain chain trong thread riêng, đẩy token vào queue.
    chain_type: "outfit" → chunk.content  |  "search" → chunk["answer"]
    """
    try:
        for chunk in chain.stream(input_dict, config=config):
            if chain_type == "outfit":
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
            else:  # search
                token = chunk.get("answer", "") if isinstance(chunk, dict) else ""
            if token:
                token_queue.put({"ok": True, "token": token})
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

        state = sessions[session_id]
        profile = state["profile"]
        last_bot_msg = state["last_bot_msg"]

        final_query = message
        start_time = time.time()
        first_token_time = None

        # ══ Xử lý ảnh ═══════════════════════════════════════════════
        if image and image.filename:
            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            tmp_path = None
            try:
                # Lưu ảnh vào file tạm
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await image.read())
                    tmp_path = tmp.name

                # Detect loại ảnh: person hay product?
                image_type = await asyncio.to_thread(_detect_image_type, tmp_path, message)

                if image_type == "person":
                    # ── Phân tích vóc dáng ─────────────────────────
                    person_info = await asyncio.to_thread(_analyze_person_image, tmp_path)

                    # Cập nhật profile
                    if person_info.get("dang_nguoi"):
                        profile["dang_nguoi"] = person_info["dang_nguoi"]
                    if person_info.get("tone_da"):
                        profile["tone_da"] = person_info["tone_da"]
                    sessions[session_id]["profile"] = profile

                    # Gửi event phân tích về frontend
                    yield make_event({"type": "person_analyzed", **person_info})

                    # Stream response giả (split từng word để có cảm giác streaming)
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
                        "total": round(time.time() - start_time, 2)
                    })
                    return  # Kết thúc — không chạy RAG

                else:
                    # ── Caption ảnh sản phẩm ───────────────────────
                    caption = await asyncio.to_thread(_caption_product_image, tmp_path, message)
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
            intent = await asyncio.to_thread(_detect_intent, final_query, last_bot_msg)
            current_gender = await asyncio.to_thread(_detect_gender, final_query)
        except Exception as e:
            yield make_event({"type": "error", "message": f"Lỗi detect intent: {str(e)}"})
            return

        # Chỉ cập nhật gender nếu detect được "male" (mặc định female)
        if current_gender == "male":
            profile["gender"] = "male"
        gender = profile.get("gender", current_gender or "female")

        yield make_event({"type": "intent_detected", "intent": intent, "gender": gender})

        # ══ Routing ══════════════════════════════════════════════════
        response_tokens = []

        # ── Greeting ────────────────────────────────────────────────
        if intent == "greeting":
            reply = await asyncio.to_thread(_get_greeting_response)
            yield make_event({"type": "token", "content": reply})
            sessions[session_id]["last_bot_msg"] = reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        # ── Chitchat ─────────────────────────────────────────────────
        if intent == "chitchat":
            reply = await asyncio.to_thread(_get_chitchat_response, final_query)
            yield make_event({"type": "token", "content": reply})
            sessions[session_id]["last_bot_msg"] = reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        # ── Outfit / Search ───────────────────────────────────────────
        config = {"configurable": {"session_id": session_id}}
        token_queue: queue.Queue = queue.Queue()

        if intent == "outfit":
            try:
                outfit_context = await asyncio.to_thread(
                    _build_outfit_context, final_query, gender, profile, _vector_db
                )
            except Exception as e:
                outfit_context = None

            if not outfit_context:
                # Không có outfit context → fallback về search
                intent = "search"
            else:
                chain_input = {"input": message or final_query, "outfit_context": outfit_context}
                t = threading.Thread(
                    target=_stream_chain_via_queue,
                    args=(_outfit_chain, chain_input, config, token_queue, "outfit"),
                    daemon=True,
                )
                t.start()

        if intent == "search":
            chain_input = {"input": final_query}
            t = threading.Thread(
                target=_stream_chain_via_queue,
                args=(_full_chat_chain, chain_input, config, token_queue, "search"),
                daemon=True,
            )
            t.start()

        # ── Đọc token từ queue và yield SSE ─────────────────────────
        while True:
            try:
                item = await asyncio.to_thread(token_queue.get, timeout=60)
            except Exception:
                yield make_event({"type": "error", "message": "Timeout chờ phản hồi từ LLM."})
                break

            if item is None:
                break  # sentinel — chain đã xong

            if not item.get("ok", False):
                yield make_event({"type": "error", "message": item.get("error", "Lỗi không xác định")})
                break

            token = item["token"]
            if first_token_time is None:
                first_token_time = time.time()
            response_tokens.append(token)
            yield make_event({"type": "token", "content": token})
            await asyncio.sleep(0)  # yield control cho event loop

        # ── Lưu lại response cuối cùng ────────────────────────────
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


# ── Serve index.html từ thư mục cùng cấp ─────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve file index.html (nếu chạy qua uvicorn)."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "initialized": _initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
