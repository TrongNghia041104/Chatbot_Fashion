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

from app.config import (
    API_TITLE,
    API_VERSION,
    DEBUG_ROUTER_TRACE,
    IMAGES_DIR,
    PRELOAD_IMAGE_ENCODER,
    STATIC_DIR,
)


app = FastAPI(title=API_TITLE, version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _preload_image_encoder() -> None:
    """Warm FashionCLIP in the background before the first demo request."""
    try:
        from app.core.image_search import get_image_embeddings

        get_image_embeddings()
        print("[OK] FashionCLIP image encoder preloaded.")
    except Exception as exc:
        print(f"[WARN] FashionCLIP preload failed; the first image request will retry: {exc}")


@app.on_event("startup")
async def _warm_up_before_serving() -> None:
    if PRELOAD_IMAGE_ENCODER:
        # Chỉ báo API sẵn sàng sau khi import/model warm-up hoàn tất. Nếu chạy
        # nền, request text đầu tiên có thể tranh import lock với FashionCLIP.
        await asyncio.to_thread(_preload_image_encoder)


sessions: dict[str, dict] = {}


def _new_session_state() -> dict:
    """Create the complete state shape in one place."""
    return {
        "profile": {},
        "last_bot_msg": "",
        "last_route_decision": None,
        "last_query": "",
        "pending_profile_candidate": None,
        "pending_image_context": None,
        "pending_image_docs": [],
        "unclear_count": 0,
    }


def make_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _docs_to_images(docs) -> list[dict]:
    """Serialize retrieved documents into browser-ready commerce cards."""
    products = []
    for doc in docs or []:
        doc_images = doc.metadata.get("images", [])
        if isinstance(doc_images, str):
            doc_images = [doc_images] if doc_images else []
        main_image = doc.metadata.get("image_url", "")
        if main_image and main_image not in doc_images:
            doc_images = [main_image, *doc_images]
        doc_images = [url for url in doc_images if url]
        if doc_images:
            products.append(
                {
                    "product_id": doc.metadata.get("product_id", ""),
                    "title": doc.metadata.get("title", "") or "Sản phẩm thời trang",
                    "category": doc.metadata.get("category", ""),
                    "brand": doc.metadata.get("brand", "") or "Thương hiệu khác",
                    "price": doc.metadata.get("price", 0),
                    "images": [_browser_image_url(url) for url in doc_images],
                }
            )
    return products


def _normalize_product_cards(products: list[dict]) -> list[dict]:
    """Normalize product-card payloads produced outside the API module."""
    normalized = []
    for product in products or []:
        item = dict(product)
        images = item.get("images", [])
        if isinstance(images, str):
            images = [images]
        item["images"] = [_browser_image_url(url) for url in images if url]
        alternatives = item.get("alternatives", [])
        item["alternatives"] = _normalize_product_cards(alternatives) if alternatives else []
        item.setdefault("title", item.get("category") or "Sản phẩm thời trang")
        item.setdefault("brand", "Thương hiệu khác")
        item.setdefault("price", 0)
        normalized.append(item)
    return normalized


def _browser_image_url(value: str) -> str:
    """Convert local metadata paths into URLs served by FastAPI's /images mount."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith(("http://", "https://", "data:", "blob:", "/images/")):
        return raw
    normalized = raw.replace("\\", "/")
    try:
        relative = os.path.relpath(raw, IMAGES_DIR).replace("\\", "/")
        if not relative.startswith("../") and relative != "..":
            return f"/images/{relative.lstrip('/')}"
    except (TypeError, ValueError):
        pass
    if normalized.lower().startswith("images/"):
        normalized = normalized.split("/", 1)[1]
    return f"/images/{normalized.lstrip('/')}"


def _is_reranker_enabled() -> bool:
    try:
        from app.core.vector_store import is_reranker_enabled

        return is_reranker_enabled()
    except Exception:
        return False


def _retrieval_progress_message(entities: dict) -> str:
    """Build a short user-facing status from observable retrieval constraints."""
    parts = []
    categories = entities.get("categories") or []
    if categories:
        parts.append(str(categories[0]).replace("ao", "áo", 1))
    budget = str(entities.get("budget_text") or "").strip()
    if budget:
        budget = budget.replace("duoi", "dưới").replace("trieu", "triệu")
        parts.append(budget)
    sizes = entities.get("sizes") or []
    if sizes:
        parts.append(f"size {', '.join(map(str, sizes))}")
    if not parts:
        return "Mình đang xem những món phù hợp nhất trong cửa hàng..."
    return f"Mình đang tìm trong cửa hàng theo: {' · '.join(parts)}..."


def _image_identification_reply(image_context: dict) -> str:
    """Turn a VLM observation into a cautious, user-facing identification."""
    fashion_item = str(image_context.get("fashion_item") or "").strip()
    caption = str(image_context.get("caption") or "").strip()
    if fashion_item:
        sentences = [f"Mình nhận ra đây có vẻ là **{fashion_item}**."]
        if caption and fashion_item.casefold() not in caption.casefold():
            sentences.append(f"{caption.rstrip('.')}.")
        sentences.append(
            "Mình đã đặt những mẫu gần giống tìm được trong cửa hàng ở bên cạnh để bạn đối chiếu."
        )
        return " ".join(sentences)
    if caption:
        return (
            f"Qua ảnh, mình nhìn thấy **{caption.rstrip('.')}**. "
            "Mình chưa xác định chắc tên món đồ, nhưng vẫn đặt các kết quả gần nhất ở bên cạnh để bạn tham khảo."
        )
    return (
        "Mình chưa nhìn đủ rõ để gọi đúng tên món đồ trong ảnh. "
        "Bạn thử gửi ảnh sáng hơn, chụp trọn món và ít vật thể phía sau nhé."
    )


def _stream_chain_via_queue(
    chain,
    input_dict: dict,
    config: dict,
    token_queue: queue.Queue,
    chain_type: str,
) -> None:
    """Run a blocking LangChain stream in a worker thread."""
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


def _decision_event(decision, gender: str, include_trace: bool = False) -> dict:
    event = {
        "type": "route_detected",
        "intent": decision.intent,
        "modality": decision.modality,
        "action": decision.action,
        "route": decision.route,
        "handler": decision.handler,
        "source": decision.source,
        "certainty": decision.certainty,
        # Kept for older notebook consumers; certainty is the authoritative field.
        "confidence": decision.confidence,
        "reason": decision.reason,
        "entities": decision.entities,
        "missing_slots": decision.missing_slots,
        "workflow": decision.workflow,
        "gender": gender,
    }
    if DEBUG_ROUTER_TRACE or include_trace:
        event["trace"] = decision.trace
    return event


@app.post("/api/session")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = _new_session_state()
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
    developer_mode: bool = Form(False),
):
    """Route a request and stream observable results as Server-Sent Events."""

    async def event_stream():
        from app.core.intent import (
            ROUTE_IMAGE_OUTFIT_ADVICE,
            detect_gender,
            get_clarify_response,
            get_out_of_scope_response,
            get_social_response,
            route_user_request,
        )
        from app.core.profile import apply_profile_decision, sanitize_profile_candidate
        from app.core.security import (
            CommerceFactStreamFilter,
            append_chat_turn_log,
            check_answer_grounding,
            extract_product_ids_from_docs,
            extract_product_ids_from_text,
            validate_user_query,
        )
        from app.core.telemetry import TurnTelemetry

        state = sessions.setdefault(session_id, _new_session_state())
        profile = dict(state.get("profile") or {})
        last_bot_msg = state.get("last_bot_msg", "")
        image_present = bool(image and image.filename)
        start_time = time.time()
        telemetry = TurnTelemetry()
        first_token_time = None
        response_tokens: list[str] = []
        retrieved_docs = []
        image_search_docs = []
        allowed_product_ids: set[str] = set()
        temporary_profile = None
        identification_reply = ""

        def done_payload(ttft: float = 0.0, **extra) -> dict:
            """Build one consistent completion event, including developer telemetry."""
            payload = {
                "type": "done",
                "ttft": round(ttft, 2),
                "total": round(time.time() - start_time, 2),
                **extra,
            }
            if developer_mode:
                payload.update(telemetry.snapshot())
            return payload

        def decision_used_llm(current_decision) -> bool:
            """Detect both successful and failed LLM-router attempts."""
            if current_decision.source == "llm":
                return True
            return any(item.get("stage") == "llm_intent" for item in current_decision.trace or [])

        is_valid, validation_message = validate_user_query(message or "")
        if not is_valid:
            async for event in _stream_text(validation_message):
                yield event
            yield make_event(done_payload())
            return

        stage_started = telemetry.begin()
        decision = await asyncio.to_thread(
            route_user_request,
            message or "",
            last_bot_msg,
            state,
            False,
            image_present,
        )
        telemetry.finish("router", stage_started)
        if decision_used_llm(decision):
            telemetry.add_call("llm_router")
        final_query = decision.rewrite_query or message or ""
        yield make_event(
            {
                "type": "progress",
                "phase": "routing",
                "message": "Mình đang đọc kỹ điều bạn mong muốn...",
                "status": "active",
            }
        )

        if image_present:
            from app.core.image_search import search_products_by_image
            from app.core.vision import analyze_person_image, describe_image_for_routing

            suffix = os.path.splitext(image.filename)[1] or ".jpg"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(await image.read())
                    tmp_path = tmp.name

                if decision.action in {"inspect_image", "identify_image_item"} and not decision.image_context:
                    yield make_event(
                        {
                            "type": "progress",
                            "phase": "vision",
                            "message": "Mình đang xem món đồ trong ảnh của bạn...",
                            "status": "active",
                        }
                    )
                    stage_started = telemetry.begin()
                    image_context = await asyncio.to_thread(
                        describe_image_for_routing,
                        tmp_path,
                        message or "",
                    )
                    telemetry.finish("vlm_image_understanding", stage_started)
                    telemetry.add_call("vlm")
                    yield make_event({"type": "image_understood", **image_context})
                    stage_started = telemetry.begin()
                    decision = await asyncio.to_thread(
                        route_user_request,
                        message or "",
                        last_bot_msg,
                        state,
                        False,
                        True,
                        image_context,
                    )
                    telemetry.finish("router", stage_started)
                    if decision_used_llm(decision):
                        telemetry.add_call("llm_router")
                    final_query = decision.rewrite_query or message or ""

                    if decision.action == "identify_image_item":
                        identification_reply = _image_identification_reply(decision.image_context)
                        yield make_event(
                            _decision_event(
                                decision,
                                profile.get("gender", "unknown"),
                                developer_mode,
                            )
                        )
                        async for event in _stream_text(identification_reply):
                            yield event
                        response_tokens.append(identification_reply)

                if decision.handler == "profile_analysis":
                    yield make_event(
                        {
                            "type": "progress",
                            "phase": "profile",
                            "message": "Mình đang quan sát để hiểu vóc dáng và tone da của bạn...",
                            "status": "active",
                        }
                    )
                    stage_started = telemetry.begin()
                    person_info = await asyncio.to_thread(analyze_person_image, tmp_path)
                    telemetry.finish("vlm_profile_analysis", stage_started)
                    telemetry.add_call("vlm")
                    candidate = sanitize_profile_candidate(person_info)
                    state["pending_profile_candidate"] = candidate
                    yield make_event(
                        {
                            "type": "profile_candidate",
                            "candidate": candidate,
                            "comment": person_info.get("nhan_xet", ""),
                            "saved": False,
                        }
                    )
                    if decision.action != "analyze_then_style":
                        reply = (
                            f"Qua bức ảnh, mình thấy vóc dáng của bạn có thể là **{candidate.get('dang_nguoi', 'chưa rõ')}**, "
                            f"với **{candidate.get('tone_da', 'tone da chưa rõ')}**. "
                            f"{person_info.get('nhan_xet', '')}\n\n"
                            "Nhận xét từ ảnh đôi khi chưa hoàn toàn chính xác. Bạn thấy kết quả này có đúng với mình không? "
                            "Nếu bạn đồng ý, mình sẽ ghi nhớ để những lần tư vấn sau phù hợp hơn."
                        )
                        yield make_event(
                            _decision_event(
                                decision,
                                profile.get("gender", "unknown"),
                                developer_mode,
                            )
                        )
                        yield make_event(
                            {
                                "type": "clarification",
                                "question": "Bạn có đồng ý lưu kết quả phân tích này không?",
                                "options": [
                                    {"label": "Đúng rồi, hãy ghi nhớ", "value": "Đồng ý lưu thông tin"},
                                    {"label": "Chưa đúng, bỏ qua", "value": "Không lưu"},
                                ],
                            }
                        )
                        async for event in _stream_text(reply):
                            yield event
                        state["last_bot_msg"] = reply
                        yield make_event(done_payload())
                        return
                    temporary_profile = {**profile, **candidate}
                    final_query = message or "Gợi ý outfit phù hợp với profile vừa phân tích"
                elif not decision.needs_clarification:
                    yield make_event(
                        {
                            "type": "progress",
                            "phase": "image_retrieval",
                            "message": "Mình đang tìm những món có kiểu dáng gần giống...",
                            "status": "active",
                        }
                    )
                    stage_started = telemetry.begin()
                    image_search_docs = await asyncio.to_thread(search_products_by_image, tmp_path)
                    telemetry.finish("image_retrieval", stage_started)
                    telemetry.add_call("fashionclip_image")
                    state["pending_image_context"] = dict(decision.image_context or {})
                    state["pending_image_docs"] = image_search_docs
                    if image_search_docs:
                        is_outfit_image = decision.route == ROUTE_IMAGE_OUTFIT_ADVICE
                        card_docs = image_search_docs[:1] if is_outfit_image else image_search_docs
                        # Với outfit ảnh, chỉ giữ một catalog match làm món gốc.
                        yield make_event(
                            {
                                "type": "product_images",
                                "images": _docs_to_images(card_docs),
                                "source": "image_retrieval",
                                "group": "source_item" if is_outfit_image else "search_results",
                                "replace": True,
                            }
                        )
                        yield make_event(
                            {
                                "type": "image_search_ready",
                                "count": len(image_search_docs),
                                "route": decision.route,
                            }
                        )
                    if developer_mode:
                        yield make_event({"type": "stage_metrics", **telemetry.snapshot()})

                    if decision.action == "identify_image_item":
                        if decision.follow_up_question:
                            yield make_event(
                                {
                                    "type": "suggested_follow_up",
                                    "question": decision.follow_up_question,
                                    "options": decision.follow_up_options,
                                }
                            )
                        state["last_route_decision"] = decision
                        state["last_query"] = final_query
                        state["last_bot_msg"] = identification_reply
                        yield make_event(
                            done_payload(
                                retrieved_count=len(image_search_docs),
                                grounding_ok=True,
                            )
                        )
                        return
            except Exception as exc:
                yield make_event({"type": "error", "message": f"Lỗi xử lý ảnh: {exc}"})
                return
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Text follow-up có thể tiếp tục dùng candidates của ảnh ở lượt trước.
        if not image_present and decision.route == ROUTE_IMAGE_OUTFIT_ADVICE:
            image_search_docs = list(state.get("pending_image_docs") or [])

        active_query = final_query.strip()
        current_gender = await asyncio.to_thread(detect_gender, active_query)
        if current_gender in {"male", "female"}:
            profile["gender"] = current_gender
            state["profile"] = profile
        gender = profile.get("gender", "female")
        yield make_event(_decision_event(decision, gender, developer_mode))

        if decision.handler == "profile_management":
            reply, profile = apply_profile_decision(decision, state)
            yield make_event({"type": "profile_updated", "action": decision.action, "profile": profile})
            async for event in _stream_text(reply):
                yield event
            state["last_bot_msg"] = reply
            yield make_event(done_payload())
            return

        if decision.handler == "social":
            if decision.entities.get("clear_pending_image"):
                state["pending_image_context"] = None
                state["pending_image_docs"] = []
            reply = get_social_response(decision.action)
            async for event in _stream_text(reply):
                yield event
            state["last_bot_msg"] = reply
            yield make_event(done_payload())
            return

        if decision.handler == "out_of_scope":
            reply = get_out_of_scope_response(active_query)
            async for event in _stream_text(reply):
                yield event
            state["last_bot_msg"] = reply
            yield make_event(done_payload())
            return

        if decision.handler == "clarify":
            state["unclear_count"] = state.get("unclear_count", 0) + 1
            reply = get_clarify_response(decision)
            yield make_event(
                {
                    "type": "clarification",
                    "question": reply,
                    "options": decision.clarification_options,
                    "missing_slots": decision.missing_slots,
                }
            )
            async for event in _stream_text(reply):
                yield event
            state["last_bot_msg"] = reply
            yield make_event(done_payload())
            return

        state["unclear_count"] = 0
        execution_handler = "outfit" if decision.action == "analyze_then_style" else decision.handler
        chain = None
        chain_input = None
        chain_type = "llm"

        try:
            if execution_handler == "image_search":
                from app.core.chains import get_product_answer_chain
                from app.core.llm import format_documents_for_llm

                yield make_event(
                    {
                        "type": "progress",
                        "phase": "generation",
                        "message": "Mình đang chọn ra vài mẫu gần giống nhất...",
                        "status": "active",
                    }
                )

                if not image_search_docs:
                    reply = "Mình chưa tìm thấy sản phẩm đủ giống ảnh trong dữ liệu hiện tại."
                    async for event in _stream_text(reply):
                        yield event
                    state["last_bot_msg"] = reply
                    yield make_event(done_payload())
                    return
                retrieved_docs = image_search_docs
                allowed_product_ids = extract_product_ids_from_docs(retrieved_docs)
                chain = get_product_answer_chain()
                chain_input = {"input": active_query, "context": format_documents_for_llm(retrieved_docs)}

            elif execution_handler == "outfit":
                from app.core.chains import get_outfit_chain
                from app.core.outfit import build_outfit_context, build_outfit_context_from_image_docs

                yield make_event(
                    {
                        "type": "progress",
                        "phase": "outfit_retrieval",
                        "message": "Mình đang chọn các món có thể kết hợp thật hài hòa...",
                        "status": "active",
                    }
                )

                effective_profile = temporary_profile or profile
                outfit_metrics: dict = {}
                stage_started = telemetry.begin()
                if decision.route == ROUTE_IMAGE_OUTFIT_ADVICE and image_search_docs:
                    outfit_context, outfit_images, diagnostics = await asyncio.to_thread(
                        build_outfit_context_from_image_docs,
                        image_search_docs,
                        active_query,
                        gender,
                        effective_profile,
                        outfit_metrics,
                    )
                    yield make_event({"type": "image_item_context", **diagnostics})
                else:
                    outfit_context, outfit_images = await asyncio.to_thread(
                        build_outfit_context,
                        active_query,
                        gender,
                        effective_profile,
                        outfit_metrics,
                    )
                telemetry.finish("outfit_retrieval", stage_started)
                telemetry.merge_calls(outfit_metrics.get("model_calls"))
                telemetry.merge_vectors(outfit_metrics.get("model_vectors"))
                for name, elapsed in outfit_metrics.get("timings", {}).items():
                    telemetry.timings[f"outfit.{name}"] = elapsed
                if not outfit_context:
                    reply = "Mình chưa tìm được công thức phối đồ đủ khớp yêu cầu này. Bạn có thể nói rõ dịp sử dụng hoặc phong cách mong muốn."
                    async for event in _stream_text(reply):
                        yield event
                    yield make_event(done_payload())
                    return
                if outfit_images:
                    yield make_event(
                        {
                            "type": "product_images",
                            "images": _normalize_product_cards(outfit_images),
                            "source": "outfit_retrieval",
                            "group": "outfit",
                            "replace": True,
                        }
                    )
                if developer_mode:
                    yield make_event({"type": "stage_metrics", **telemetry.snapshot()})
                allowed_product_ids = extract_product_ids_from_text(outfit_context)
                chain = get_outfit_chain()
                chain_input = {"input": active_query, "outfit_context": outfit_context}

            elif execution_handler == "search":
                from app.core.chains import get_fast_search_chain

                filter_summary = decision.entities or {}
                yield make_event(
                    {
                        "type": "progress",
                        "phase": "product_retrieval",
                        "message": _retrieval_progress_message(filter_summary),
                        "status": "active",
                        "filters": filter_summary,
                    }
                )

                if decision.action == "stock_check":
                    notice = "Lưu ý: hệ thống chưa có dữ liệu tồn kho thời gian thực; mình chỉ có thể tìm sản phẩm liên quan để bạn tham khảo.\n\n"
                    response_tokens.append(notice)
                    yield make_event({"type": "token", "content": notice})
                chain = get_fast_search_chain()
                chain_input = {"input": active_query}
                chain_type = "search"

            if chain is None or chain_input is None:
                yield make_event({"type": "error", "message": "Không xác định được pipeline thực thi."})
                return

            if chain_type != "search":
                yield make_event(
                    {
                        "type": "progress",
                        "phase": "generation",
                        "message": "Mình đang chuẩn bị lời tư vấn dành riêng cho bạn...",
                        "status": "active",
                    }
                )

            config = {"configurable": {"session_id": session_id}}
            token_queue: queue.Queue = queue.Queue()
            fact_filter = CommerceFactStreamFilter()
            telemetry.add_call("llm_answer")
            generation_started = telemetry.begin()
            worker = threading.Thread(
                target=_stream_chain_via_queue,
                args=(chain, chain_input, config, token_queue, chain_type),
                daemon=True,
            )
            worker.start()
            while True:
                try:
                    item = await asyncio.to_thread(token_queue.get, True, 60)
                except queue.Empty:
                    yield make_event({"type": "error", "message": "Timeout chờ phản hồi từ LLM."})
                    break
                if item is None:
                    break
                if not item.get("ok", False):
                    yield make_event({"type": "error", "message": item.get("error", "Lỗi không xác định")})
                    break
                if item.get("item_type") == "context":
                    retrieved_docs = item.get("docs", [])
                    allowed_product_ids = extract_product_ids_from_docs(retrieved_docs)
                    if item.get("images"):
                        yield make_event(
                            {
                                "type": "product_images",
                                "images": item["images"],
                                "source": "text_retrieval",
                                "group": "search_results",
                                "replace": True,
                            }
                        )
                    yield make_event(
                        {
                            "type": "progress",
                            "phase": "generation",
                            "message": "Mình đang xem lại các lựa chọn trước khi gửi bạn...",
                            "status": "active",
                        }
                    )
                else:
                    token = item.get("content", "")
                    safe_token = fact_filter.feed(token)
                    if safe_token:
                        if first_token_time is None:
                            first_token_time = time.time()
                        response_tokens.append(safe_token)
                        yield make_event({"type": "token", "content": safe_token})
                await asyncio.sleep(0)
            worker.join(timeout=1)
            final_safe_token = fact_filter.finish()
            if final_safe_token:
                if first_token_time is None:
                    first_token_time = time.time()
                response_tokens.append(final_safe_token)
                yield make_event({"type": "token", "content": final_safe_token})
            telemetry.finish("answer_chain", generation_started)

            if developer_mode and fact_filter.removed_lines:
                yield make_event(
                    {
                        "type": "grounding_filter",
                        "message": "Đã ẩn các trường thương mại do LLM tự viết; product card là nguồn sự thật.",
                        "removed_lines": fact_filter.removed_lines,
                    }
                )

            if decision.follow_up_question:
                follow_up = "\n\n" + decision.follow_up_question
                response_tokens.append(follow_up)
                yield make_event({"type": "token", "content": follow_up})
                yield make_event(
                    {
                        "type": "suggested_follow_up",
                        "question": decision.follow_up_question,
                        "options": decision.follow_up_options,
                    }
                )

            if decision.action == "analyze_then_style":
                reminder = "\n\nKết quả profile vẫn chưa được lưu. Bạn có đồng ý lưu không?"
                response_tokens.append(reminder)
                yield make_event({"type": "token", "content": reminder})
                yield make_event(
                    {
                        "type": "clarification",
                        "question": "Bạn có đồng ý lưu profile vừa phân tích không?",
                        "options": [
                            {"label": "Đúng rồi, hãy ghi nhớ", "value": "Đồng ý lưu thông tin"},
                            {"label": "Chưa đúng, bỏ qua", "value": "Không lưu"},
                        ],
                    }
                )

            answer_text = "".join(response_tokens)
            grounding_report = check_answer_grounding(
                answer_text,
                allowed_product_ids,
                active_query,
                decision.route or "",
            ) if answer_text else {"ok": True, "unknown_product_ids": []}
            end_time = time.time()
            first_token_time = first_token_time or end_time

            telemetry_snapshot = telemetry.snapshot()
            append_chat_turn_log(
                {
                    "session_id": session_id,
                    "raw_query": message or "",
                    "query": active_query,
                    "intent": decision.intent,
                    "modality": decision.modality,
                    "action": decision.action,
                    "route": decision.route,
                    "handler": execution_handler,
                    "source": decision.source,
                    "route_certainty": decision.certainty,
                    "route_confidence_legacy": decision.confidence,
                    "route_reason": decision.reason,
                    "router_trace": decision.trace,
                    "retrieved_count": len(retrieved_docs) or len(allowed_product_ids),
                    "retrieved_product_ids": sorted(allowed_product_ids),
                    "ttft_sec": round(first_token_time - start_time, 4),
                    "total_sec": round(end_time - start_time, 4),
                    "reranker_enabled": _is_reranker_enabled(),
                    "grounding_ok": grounding_report.get("ok", True),
                    "unknown_ids": grounding_report.get("unknown_product_ids", []),
                    "filtered_commerce_fact_lines": fact_filter.removed_lines,
                    **telemetry_snapshot,
                }
            )

            state["last_route_decision"] = decision
            state["last_query"] = active_query
            if answer_text:
                state["last_bot_msg"] = answer_text[-1200:]
            yield make_event(done_payload(
                first_token_time - start_time,
                grounding_ok=grounding_report.get("ok", True),
                retrieved_count=len(retrieved_docs) or len(allowed_product_ids),
            ))
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
    return {"status": "ok", "initialized": True, "mode": "research_demo_v3_lazy"}
