"""FastAPI backend for the Fashion RAG chatbot web demo."""

from __future__ import annotations

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

from app.config import API_TITLE, API_VERSION, IMAGES_DIR, STATIC_DIR


app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


sessions: dict = {}


def make_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _docs_to_images(docs) -> list[dict]:
    images = []
    for doc in docs or []:
        doc_images = doc.metadata.get("images", [])
        if isinstance(doc_images, str):
            doc_images = [doc_images] if doc_images else []
        doc_images = [url for url in doc_images if url]
        if doc_images:
            images.append(
                {
                    "product_id": doc.metadata.get("product_id", ""),
                    "category": doc.metadata.get("category", ""),
                    "images": doc_images[:2],
                }
            )
    return images


def _is_reranker_enabled() -> bool:
    try:
        from app.core.vector_store import is_reranker_enabled

        return is_reranker_enabled()
    except Exception:
        return False


def _stream_chain_via_queue(
    chain,
    input_dict: dict,
    config: dict,
    token_queue: queue.Queue,
    chain_type: str,
) -> None:
    """Run a LangChain stream in a worker thread and push serializable chunks."""
    try:
        for chunk in chain.stream(input_dict, config=config):
            if chain_type == "search":
                if not isinstance(chunk, dict):
                    continue
                if "context" in chunk:
                    docs = chunk["context"]
                    token_queue.put(
                        {
                            "ok": True,
                            "item_type": "context",
                            "docs": docs,
                            "images": _docs_to_images(docs),
                        }
                    )
                token = chunk.get("answer", "")
                if token:
                    token_queue.put({"ok": True, "item_type": "token", "content": token})
            else:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    token_queue.put({"ok": True, "item_type": "token", "content": token})
    except Exception as exc:
        token_queue.put({"ok": False, "error": str(exc)})
    finally:
        token_queue.put(None)


async def _stream_text(reply: str, delay: float = 0.01):
    for word in reply.split(" "):
        yield make_event({"type": "token", "content": word + " "})
        await asyncio.sleep(delay)


@app.post("/api/session")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "profile": {},
        "last_bot_msg": "",
        "last_route_decision": None,
        "last_query": "",
        "unclear_count": 0,
    }
    return {"session_id": sid}


@app.get("/api/profile/{session_id}")
async def get_profile(session_id: str):
    return sessions.get(session_id, {}).get("profile", {})


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    session_id: str = Form(...),
    image: UploadFile = File(None),
):
    """Main chat endpoint. Returns Server-Sent Events for token streaming."""

    async def event_stream():
        from app.core.intent import (
            ROUTE_PRODUCT_SEARCH,
            detect_gender,
            get_chitchat_response,
            get_clarify_response,
            get_greeting_response,
            get_out_of_scope_response,
            get_profile_inquiry_response,
            route_user_request,
        )
        from app.core.security import (
            append_chat_turn_log,
            check_answer_grounding,
            extract_product_ids_from_docs,
            extract_product_ids_from_text,
            validate_user_query,
        )

        if session_id not in sessions:
            sessions[session_id] = {
                "profile": {},
                "last_bot_msg": "",
                "last_route_decision": None,
                "last_query": "",
                "unclear_count": 0,
            }

        state = sessions[session_id]
        profile = state["profile"]
        last_bot_msg = state.get("last_bot_msg", "")
        final_query = message or ""
        active_query = final_query
        image_search_docs = []
        force_image_search = False
        start_time = time.time()
        first_token_time = None
        response_tokens: list[str] = []
        retrieved_docs = []
        allowed_product_ids: set[str] = set()
        grounding_route = ""

        is_valid, validation_message = validate_user_query(message or "")
        if not is_valid:
            async for event in _stream_text(validation_message):
                yield event
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        if image and image.filename:
            from app.core.image_search import search_products_by_image
            from app.core.vision import analyze_person_image, caption_product_image, detect_image_type

            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await image.read())
                    tmp_path = tmp.name

                image_type = await asyncio.to_thread(detect_image_type, tmp_path, message)
                yield make_event({"type": "image_type_detected", "image_type": image_type})

                if image_type == "person":
                    person_info = await asyncio.to_thread(analyze_person_image, tmp_path)
                    if person_info.get("dang_nguoi"):
                        profile["dang_nguoi"] = person_info["dang_nguoi"]
                    if person_info.get("tone_da"):
                        profile["tone_da"] = person_info["tone_da"]
                    state["profile"] = profile
                    yield make_event({"type": "person_analyzed", **person_info})

                    reply = (
                        f"Mình đã phân tích xong! Bạn có dáng **{person_info.get('dang_nguoi', 'chưa rõ')}** "
                        f"với **{person_info.get('tone_da', 'chưa rõ')}**. "
                        f"{person_info.get('nhan_xet', '')}\n\n"
                        "Mình đã lưu thông tin này để tư vấn phối đồ phù hợp hơn. "
                        "Bạn muốn gợi ý outfit cho dịp nào?"
                    )
                    async for event in _stream_text(reply):
                        yield event
                    state["last_bot_msg"] = reply
                    yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
                    return

                image_search_docs = await asyncio.to_thread(search_products_by_image, tmp_path)
                if image_search_docs:
                    force_image_search = True
                    final_query = f"Find products similar to the uploaded image. Extra request: {message}"
                    yield make_event(
                        {
                            "type": "product_images",
                            "images": _docs_to_images(image_search_docs),
                        }
                    )
                    yield make_event(
                        {
                            "type": "image_search_ready",
                            "count": len(image_search_docs),
                        }
                    )
                else:
                    caption = await asyncio.to_thread(caption_product_image, tmp_path, message)
                    yield make_event({"type": "product_captioned", "caption": caption})
                    final_query = f"{caption}. Extra request: {message}" if message else caption
            except Exception as exc:
                yield make_event({"type": "error", "message": f"Lỗi xử lý ảnh: {exc}"})
                return
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not final_query.strip() and not image_search_docs:
            reply = get_clarify_response()
            async for event in _stream_text(reply):
                yield event
            state["last_bot_msg"] = reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        is_valid, validation_message = validate_user_query(final_query)
        if not is_valid:
            async for event in _stream_text(validation_message):
                yield event
            state["last_bot_msg"] = validation_message
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        decision = await asyncio.to_thread(
            route_user_request,
            final_query,
            last_bot_msg,
            state,
            force_image_search,
        )
        intent = decision.intent
        active_query = decision.rewrite_query or final_query
        grounding_route = decision.route

        current_gender = await asyncio.to_thread(detect_gender, active_query)
        if current_gender == "male":
            profile["gender"] = "male"
        gender = profile.get("gender") or "female"
        state["profile"] = profile

        yield make_event(
            {
                "type": "route_detected",
                "route": decision.route,
                "action": decision.action,
                "intent": intent,
                "source": decision.source,
                "gender": gender,
            }
        )

        if intent != "clarify":
            state["unclear_count"] = 0

        simple_reply = None
        if intent == "greeting":
            simple_reply = get_greeting_response()
        elif intent == "chitchat":
            simple_reply = get_chitchat_response(active_query)
        elif intent == "profile_inquiry":
            simple_reply = get_profile_inquiry_response(profile)
        elif intent == "out_of_scope":
            simple_reply = get_out_of_scope_response(active_query)
        elif intent == "clarify":
            state["unclear_count"] = state.get("unclear_count", 0) + 1
            if state["unclear_count"] >= 2:
                decision.route = ROUTE_PRODUCT_SEARCH
                decision.action = "fallback_after_unclear"
                intent = "search"
                grounding_route = decision.route
            else:
                simple_reply = get_clarify_response(decision)

        if simple_reply is not None:
            async for event in _stream_text(simple_reply):
                yield event
            state["last_bot_msg"] = simple_reply
            yield make_event({"type": "done", "ttft": 0.0, "total": round(time.time() - start_time, 2)})
            return

        chain = None
        chain_input = None
        chain_type = "llm"

        try:
            if intent == "image_search":
                from app.core.chains import get_product_answer_chain
                from app.core.llm import format_documents_for_llm

                retrieved_docs = image_search_docs
                allowed_product_ids = extract_product_ids_from_docs(image_search_docs)
                chain = get_product_answer_chain()
                chain_input = {
                    "input": active_query,
                    "context": format_documents_for_llm(image_search_docs),
                }
                chain_type = "llm"

            if intent == "outfit":
                from app.core.chains import get_outfit_chain
                from app.core.outfit import build_outfit_context

                outfit_context, outfit_images = await asyncio.to_thread(
                    build_outfit_context,
                    active_query,
                    gender,
                    profile,
                )
                if not outfit_context:
                    fallback = "Mình chưa có công thức phối đồ thật khớp cho yêu cầu này, nhưng để mình tìm sản phẩm phù hợp cho bạn nhé. "
                    yield make_event({"type": "token", "content": fallback})
                    response_tokens.append(fallback)
                    intent = "search"
                    decision.route = ROUTE_PRODUCT_SEARCH
                    decision.action = "fallback_search"
                    grounding_route = decision.route
                else:
                    if outfit_images:
                        yield make_event({"type": "product_images", "images": outfit_images})
                    allowed_product_ids = extract_product_ids_from_text(outfit_context)
                    chain = get_outfit_chain()
                    chain_input = {"input": active_query, "outfit_context": outfit_context}
                    chain_type = "llm"

            if intent == "search":
                # Use fast chain: active_query is already rewritten by the keyword router /
                # LLM router (decision.rewrite_query), so we skip the extra LLM rewrite step.
                from app.core.chains import get_fast_search_chain

                chain = get_fast_search_chain()
                chain_input = {"input": active_query}
                chain_type = "search"

            if chain is None or chain_input is None:
                yield make_event({"type": "error", "message": "Không xác định được luồng xử lý phù hợp."})
                return

            config = {"configurable": {"session_id": session_id}}
            token_queue: queue.Queue = queue.Queue()
            worker = threading.Thread(
                target=_stream_chain_via_queue,
                args=(chain, chain_input, config, token_queue, chain_type),
                daemon=True,
            )
            worker.start()

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
                if item_type == "context":
                    retrieved_docs = item.get("docs", [])
                    allowed_product_ids = extract_product_ids_from_docs(retrieved_docs)
                    if item.get("images"):
                        yield make_event({"type": "product_images", "images": item["images"]})
                else:
                    token = item.get("content", "")
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_tokens.append(token)
                    yield make_event({"type": "token", "content": token})
                await asyncio.sleep(0)

            worker.join(timeout=1)

            answer_text = "".join(response_tokens)
            if answer_text and intent in {"search", "image_search", "outfit"}:
                grounding_report = check_answer_grounding(
                    answer_text,
                    allowed_product_ids,
                    active_query,
                    grounding_route,
                )
            else:
                grounding_report = {"ok": True, "unknown_product_ids": []}

            end_time = time.time()
            if first_token_time is None:
                first_token_time = end_time

            append_chat_turn_log(
                {
                    "session_id": session_id,
                    "raw_query": message or "",
                    "query": active_query,
                    "route": decision.route,
                    "action": decision.action,
                    "intent": intent,
                    "source": decision.source,
                    "route_confidence": decision.confidence,
                    "route_reason": decision.reason,
                    "retrieved_count": len(retrieved_docs) if retrieved_docs else len(allowed_product_ids),
                    "retrieved_product_ids": sorted(allowed_product_ids),
                    "ttft_sec": round(first_token_time - start_time, 4),
                    "total_sec": round(end_time - start_time, 4),
                    "reranker_enabled": _is_reranker_enabled(),
                    "grounding_ok": grounding_report.get("ok", True),
                    "unknown_ids": grounding_report.get("unknown_product_ids", []),
                }
            )

            if intent in {"search", "image_search", "outfit"}:
                state["last_route_decision"] = decision
                state["last_query"] = active_query
            if answer_text:
                state["last_bot_msg"] = answer_text[-1200:]

            yield make_event(
                {
                    "type": "done",
                    "ttft": round(first_token_time - start_time, 2),
                    "total": round(end_time - start_time, 2),
                    "grounding_ok": grounding_report.get("ok", True),
                }
            )

        except Exception as exc:
            yield make_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as handle:
            return HTMLResponse(content=handle.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "initialized": True,
        "mode": "research_demo_v3_lazy",
    }
