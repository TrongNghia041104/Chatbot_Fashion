# Prompt: Tạo Web Demo cho Fashion RAG Chatbot

## Nhiệm vụ
Tôi có một chatbot tư vấn thời trang chạy local bằng Python (notebook đã hoàn chỉnh).
Hãy tạo **2 file**: `api.py` (FastAPI backend) và `index.html` (frontend single-file)
để demo hệ thống này qua giao diện web.

---

## PHẦN 1: Hiểu hệ thống hiện có

### Các hàm chính đã có sẵn (import từ notebook đã chạy)

```python
# ── Vision ──────────────────────────────────────────────────────
detect_image_type(image_path: str, user_query: str = "") -> str
# Trả về: "person" hoặc "product"

analyze_person_image(image_path: str) -> dict
# Trả về: {"dang_nguoi": "Dáng quả lê", "tone_da": "Da vàng", "nhan_xet": "..."}

caption_product_image(image_path: str, user_query: str = "") -> str
# Trả về: "Áo thun trắng basic, form oversize, phong cách casual..."

# ── Intent Detection ────────────────────────────────────────────
detect_intent(query: str, last_bot_msg: str = "") -> str
# Trả về: "outfit" | "search" | "greeting" | "chitchat"

detect_gender(query: str) -> str
# Trả về: "male" | "female"

get_greeting_response() -> str
get_chitchat_response(query: str) -> str

# ── Outfit Logic ─────────────────────────────────────────────────
build_outfit_context(user_query: str, gender: str, profile: dict) -> str
# Trả về context string (rỗng nếu không tìm được rule)

# ── LangChain Chains (stream được) ──────────────────────────────
full_chat_chain          # Luồng SEARCH — dùng RAG retriever
outfit_chain_with_history # Luồng OUTFIT — dùng context từ Layer B

# Cả 2 chain đều stream qua:
for chunk in chain.stream(input_dict, config={"configurable": {"session_id": sid}}):
    ...

# full_chat_chain trả về chunk["answer"]
# outfit_chain_with_history trả về chunk.content

# ── Redis History ────────────────────────────────────────────────
get_message_history(session_id: str) -> RedisChatMessageHistory
```

### Logic chính của chat loop (để hiểu cần convert gì sang API)

```python
# 1. Nhận user_input (text) + ảnh (optional)
# 2. Nếu có ảnh:
#    - detect_image_type(path, user_input) → "person" hoặc "product"
#    - Nếu "person": analyze_person_image() → cập nhật user_profile → trả lời ngay, KHÔNG chạy RAG
#    - Nếu "product": caption_product_image() → dùng caption làm final_query
# 3. detect_intent(final_query) → intent
# 4. Lưu gender vào user_profile nếu detect_gender() == "male"
#    gender = user_profile.get("gender", detect_gender(final_query))
# 5. Routing:
#    - "greeting"  → get_greeting_response(), không RAG
#    - "chitchat"  → get_chitchat_response(), không RAG
#    - "outfit"    → build_outfit_context() → outfit_chain_with_history.stream()
#                    (nếu context rỗng → fallback về search)
#    - "search"    → full_chat_chain.stream()
```

---

## PHẦN 2: Backend — file `api.py`

### Cấu trúc file

```python
# api.py
import asyncio, json, uuid, os, tempfile, time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import toàn bộ từ notebook (giả sử đã extract ra chatbot_core.py)
from chatbot_core import (
    detect_image_type, analyze_person_image, caption_product_image,
    detect_intent, detect_gender,
    get_greeting_response, get_chitchat_response,
    build_outfit_context,
    full_chat_chain, outfit_chain_with_history,
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# In-memory state (production dùng Redis)
sessions = {}
# sessions[session_id] = {
#     "profile": {"dang_nguoi": None, "tone_da": None, "gender": "female"},
#     "last_bot_msg": ""
# }
```

### Endpoints cần tạo

**`POST /api/session`** — Tạo session mới
```python
@app.post("/api/session")
async def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {"profile": {}, "last_bot_msg": ""}
    return {"session_id": sid}
```

**`GET /api/profile/{session_id}`** — Lấy profile hiện tại
```python
@app.get("/api/profile/{session_id}")
async def get_profile(session_id: str):
    return sessions.get(session_id, {}).get("profile", {})
```

**`POST /api/chat`** — Endpoint chính, trả về SSE stream
```python
@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    session_id: str = Form(...),
    image: UploadFile = File(None)
):
    async def event_stream():
        # Yield từng event theo format: "data: {json}\n\n"
        ...
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
```

### SSE Event Format — Backend phải yield đúng format này

```python
def make_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# Các event type và khi nào yield:

# 1. Sau khi phân tích ảnh person xong
yield make_event({
    "type": "person_analyzed",
    "dang_nguoi": "Dáng quả lê",
    "tone_da": "Da vàng",
    "nhan_xet": "Phần hông rộng hơn vai..."
})

# 2. Sau khi caption ảnh product xong
yield make_event({
    "type": "product_captioned",
    "caption": "Áo thun trắng basic..."
})

# 3. Sau khi detect intent
yield make_event({
    "type": "intent_detected",
    "intent": "outfit",   # outfit | search | greeting | chitchat
    "gender": "female"
})

# 4. Mỗi token LLM stream ra
yield make_event({
    "type": "token",
    "content": "Dạ, "
})

# 5. Khi hoàn thành — kèm thống kê tốc độ
yield make_event({
    "type": "done",
    "ttft": 1.23,      # time to first token (giây)
    "total": 8.45      # tổng thời gian (giây)
})

# 6. Khi có lỗi
yield make_event({
    "type": "error",
    "message": "Không thể phân tích ảnh"
})
```

### Logic đầy đủ của `event_stream()` trong `/api/chat`

```python
async def event_stream():
    state = sessions.get(session_id, {"profile": {}, "last_bot_msg": ""})
    profile = state["profile"]
    last_bot_msg = state["last_bot_msg"]

    final_query = message
    start_time = time.time()
    first_token_time = None

    # ── Xử lý ảnh ────────────────────────────────────────────────
    if image and image.filename:
        # Lưu ảnh tạm
        suffix = os.path.splitext(image.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        try:
            image_type = detect_image_type(tmp_path, message)

            if image_type == "person":
                person_info = analyze_person_image(tmp_path)
                # Cập nhật profile
                if person_info["dang_nguoi"]:
                    profile["dang_nguoi"] = person_info["dang_nguoi"]
                if person_info["tone_da"]:
                    profile["tone_da"] = person_info["tone_da"]
                sessions[session_id]["profile"] = profile

                # Yield kết quả phân tích
                yield make_event({"type": "person_analyzed", **person_info})

                # Tạo response cho lượt này (không chạy RAG)
                bot_reply = (
                    f"Mình đã phân tích xong! Bạn có **{person_info['dang_nguoi']}** "
                    f"với **{person_info['tone_da']}**. {person_info['nhan_xet']} "
                    f"\n\nMình đã lưu thông tin để tư vấn phối đồ phù hợp hơn. "
                    f"Bạn muốn gợi ý outfit cho dịp nào?"
                )
                # Stream từng token của bot_reply (giả lập streaming)
                for word in bot_reply.split(" "):
                    yield make_event({"type": "token", "content": word + " "})
                    await asyncio.sleep(0.01)

                sessions[session_id]["last_bot_msg"] = bot_reply
                yield make_event({"type": "done", "ttft": 0, "total": time.time() - start_time})
                return  # Kết thúc, không chạy RAG

            else:
                caption = caption_product_image(tmp_path, message)
                yield make_event({"type": "product_captioned", "caption": caption})
                final_query = f"{caption}. Yêu cầu: {message}" if message else caption

        finally:
            os.unlink(tmp_path)

    # ── Detect intent ─────────────────────────────────────────────
    intent = detect_intent(final_query, last_bot_msg)
    current_gender = detect_gender(final_query)
    if current_gender == "male":
        profile["gender"] = "male"
    gender = profile.get("gender", current_gender)

    yield make_event({"type": "intent_detected", "intent": intent, "gender": gender})

    # ── Routing ───────────────────────────────────────────────────
    response_tokens = []

    if intent == "greeting":
        reply = get_greeting_response()
        yield make_event({"type": "token", "content": reply})
        sessions[session_id]["last_bot_msg"] = reply
        yield make_event({"type": "done", "ttft": 0, "total": time.time() - start_time})
        return

    if intent == "chitchat":
        reply = get_chitchat_response(final_query)
        yield make_event({"type": "token", "content": reply})
        sessions[session_id]["last_bot_msg"] = reply
        yield make_event({"type": "done", "ttft": 0, "total": time.time() - start_time})
        return

    # outfit hoặc search
    config = {"configurable": {"session_id": session_id}}

    if intent == "outfit":
        outfit_context = build_outfit_context(final_query, gender, profile)
        if not outfit_context:
            intent = "search"  # fallback

    if intent == "outfit":
        chain_input = {"input": message or final_query, "outfit_context": outfit_context}
        for chunk in outfit_chain_with_history.stream(chain_input, config=config):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                response_tokens.append(token)
                yield make_event({"type": "token", "content": token})
                await asyncio.sleep(0)  # yield control

    if intent == "search":
        for chunk in full_chat_chain.stream({"input": final_query}, config=config):
            if "answer" in chunk:
                token = chunk["answer"]
                if token:
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_tokens.append(token)
                    yield make_event({"type": "token", "content": token})
                    await asyncio.sleep(0)

    full_response = "".join(response_tokens)
    sessions[session_id]["last_bot_msg"] = full_response

    ttft = (first_token_time - start_time) if first_token_time else 0
    yield make_event({
        "type": "done",
        "ttft": round(ttft, 2),
        "total": round(time.time() - start_time, 2)
    })
```

---

## PHẦN 3: Frontend — file `index.html`

### Yêu cầu kỹ thuật
- **Single HTML file** — CSS và JS inline, không file ngoài
- Chỉ dùng **Vanilla JS** (không React, không Vue, không jQuery)
- CDN được phép: Google Fonts, Phosphor Icons, marked.js (render markdown)
- Kết nối SSE: dùng **`fetch` + `ReadableStream`** (KHÔNG dùng `EventSource` vì không hỗ trợ POST + file upload)

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  HEADER: "👗 Fashion AI Advisor"          [New Chat]    │
├──────────────┬──────────────────────────────────────────┤
│              │                                           │
│   SIDEBAR    │         CHAT AREA                        │
│   (280px)    │  ┌───────────────────────────────────┐  │
│              │  │                                   │  │
│  [Profile]   │  │  Tin nhắn scroll ở đây            │  │
│  ─────────   │  │                                   │  │
│  Dáng: ...   │  │                                   │  │
│  Tone:  ...  │  └───────────────────────────────────┘  │
│  Gender: ... │                                           │
│              │  ┌───────────────────────────────────┐  │
│  [Intent]    │  │ [📎] [text input...........]  [→] │  │
│  ─────────   │  └───────────────────────────────────┘  │
│  🎨 OUTFIT   │                                           │
│              │                                           │
│  [Stats]     │                                           │
│  ─────────   │                                           │
│  TTFT: 1.2s  │                                           │
│  Total: 8.4s │                                           │
│              │                                           │
└──────────────┴──────────────────────────────────────────┘
```

### Chi tiết các component

**Chat bubbles:**
- User: bên phải, background tối
- Bot: bên trái, background trắng/card, render markdown với `marked.js`
- Streaming: hiện từng token realtime, có cursor `▌` nhấp nháy khi đang nhận
- Khi đang chờ: hiện typing indicator (3 chấm nhảy)

**Image upload:**
- Click icon 📎 → file picker (chỉ nhận image/*)
- Preview ảnh nhỏ trong input bar sau khi chọn (có nút ✕ để xóa)
- Khi send: hiện thumbnail ảnh trong bubble của user

**Profile card (sidebar):**
- Default: "Chưa có thông tin" với icon mờ
- Sau khi phân tích ảnh người: cập nhật dáng + tone + gender với animation highlight

**Intent badge (sidebar):**
- 4 màu khác nhau: outfit (tím), search (xanh), greeting (xanh lá), chitchat (vàng)
- Transition animation khi đổi

**Stats (sidebar):**
- Chỉ hiện sau lượt chat đầu tiên có response từ LLM
- TTFT và Total time

**Person analysis bubble (đặc biệt):**
Khi nhận event `person_analyzed`, hiện card:
```
┌─────────────────────────────────┐
│ 📊 Kết quả phân tích vóc dáng   │
│ Dáng người : Dáng quả lê        │
│ Tone da    : Da vàng            │
│ Nhận xét   : [nhan_xet text]   │
└─────────────────────────────────┘
```

### Xử lý SSE trong JavaScript

```javascript
async function sendMessage(message, imageFile) {
    const formData = new FormData();
    formData.append("message", message);
    formData.append("session_id", sessionId);
    if (imageFile) formData.append("image", imageFile);

    const response = await fetch("/api/chat", {
        method: "POST",
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let botBubble = null; // DOM element để append token vào

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Phần chưa hoàn chỉnh

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const event = JSON.parse(line.slice(6));

            switch (event.type) {
                case "person_analyzed":
                    showPersonAnalysisCard(event);
                    updateProfileSidebar(event);
                    break;

                case "product_captioned":
                    // Hiện caption nhỏ dưới ảnh (optional)
                    break;

                case "intent_detected":
                    updateIntentBadge(event.intent);
                    break;

                case "token":
                    if (!botBubble) botBubble = createBotBubble();
                    appendToken(botBubble, event.content);
                    break;

                case "done":
                    if (botBubble) renderMarkdown(botBubble); // render markdown cuối cùng
                    updateStats(event.ttft, event.total);
                    removeCursor(botBubble);
                    enableInput();
                    break;

                case "error":
                    showErrorBubble(event.message);
                    enableInput();
                    break;
            }
        }
    }
}
```

### Lưu ý quan trọng

1. `sessionId` được tạo khi load trang bằng cách gọi `POST /api/session`, lưu vào `localStorage` để reload không mất
2. Disable input + send button khi đang chờ response, enable lại khi nhận event `done` hoặc `error`
3. Auto scroll to bottom sau mỗi token mới
4. Khi gửi ảnh: sau khi nhận `person_analyzed` thì **không** tạo `botBubble` từ tokens ngay — vì response là streaming fake (split words). Vẫn dùng cùng logic `token` events.
5. Nút `[New Chat]` ở header: gọi `POST /api/session` để lấy session_id mới, xóa chat UI, reset sidebar

---

## PHẦN 4: Thẩm mỹ

### Định hướng: **Modern, Clean, Fashion-forward**

Không cần quá phức tạp. Ưu tiên:
- Tông màu nhẹ nhàng, thanh lịch (không dùng màu sặc sỡ)
- Font đẹp: Google Fonts — `Playfair Display` cho heading, `DM Sans` cho body
- Bubble chat bo tròn, có shadow nhẹ
- Sidebar có divider rõ ràng giữa các section
- Responsive tốt ở 1280px+ (demo trên laptop)
- Micro-interaction: hover nhẹ, transition 150-200ms

### Gợi ý màu sắc
```css
--bg: #F8F7F4;
--surface: #FFFFFF;
--text: #1A1A1A;
--text-muted: #6B6B6B;
--accent: #1A1A1A;
--bubble-user-bg: #1A1A1A;
--bubble-user-text: #FFFFFF;
--intent-outfit: #7C3AED;
--intent-search: #1D4ED8;
--intent-greeting: #059669;
--intent-chitchat: #B45309;
--highlight: #FEF3C7;
```

---

## PHẦN 5: Deliverables

Tạo **2 file hoàn chỉnh**, không có placeholder hay TODO:

### `api.py`
- Import từ `chatbot_core` (giả sử file này chứa toàn bộ hàm đã liệt kê ở Phần 1)
- Tất cả endpoints đã mô tả ở Phần 2
- Logic `event_stream()` đầy đủ như pseudocode ở trên
- Chạy được bằng: `uvicorn api:app --reload --port 8000`

### `index.html`
- Single file, tất cả CSS và JS inline
- Tất cả component đã mô tả ở Phần 3
- Kết nối đến `http://localhost:8000`
- Mở trực tiếp trong browser hoặc serve qua FastAPI

### Lưu ý thêm:
- Thêm comment giải thích các đoạn logic quan trọng
- Error handling: nếu backend không chạy → hiện thông báo rõ ràng trong UI
- Nếu LLM streaming trong `event_stream()` là synchronous blocking → wrap bằng `asyncio.to_thread()` để không block event loop:

```python
# Chạy synchronous LangChain stream trong thread pool
import asyncio

def run_chain_sync(chain, input_dict, config):
    """Chạy trong thread riêng để không block async event loop."""
    tokens = []
    for chunk in chain.stream(input_dict, config=config):
        tokens.append(chunk)
    return tokens

# Trong event_stream():
chunks = await asyncio.to_thread(run_chain_sync, full_chat_chain, {"input": q}, config)
for chunk in chunks:
    if "answer" in chunk:
        yield make_event({"type": "token", "content": chunk["answer"]})
```

Hoặc dùng queue để stream realtime:
```python
import queue, threading

token_queue = queue.Queue()

def stream_in_thread():
    for chunk in chain.stream(input_dict, config=config):
        token_queue.put(chunk)
    token_queue.put(None)  # sentinel

thread = threading.Thread(target=stream_in_thread)
thread.start()

while True:
    chunk = await asyncio.to_thread(token_queue.get)
    if chunk is None:
        break
    # process chunk
    yield make_event(...)
```
