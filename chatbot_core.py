"""
chatbot_core.py — Extracted từ Chatbot_RAG_MultiModal.ipynb
Chạy với: /venv/main/bin/python
"""
import json, uuid, os, time, base64

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchAny
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
import ollama

# ── EMBEDDING (qua Ollama SSH Tunnel → Vast.ai GPU) ──────────────────────────
class BGEM3Embeddings(Embeddings):
    """
    Wrapper cho BGE-M3 chạy qua Ollama (localhost:11434 via SSH tunnel lên Vast.ai).
    Vector 1024 chiều — khớp hoàn toàn với Qdrant collection đã tạo.
    """
    def __init__(self, model_name="bge-m3"):
        self.ollama_embeddings = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434",  # SSH tunnel → Vast.ai
        )
    def embed_documents(self, texts):
        return self.ollama_embeddings.embed_documents(texts)
    def embed_query(self, text):
        return self.ollama_embeddings.embed_query(text)

# ── KHỞI TẠO EMBEDDING & QDRANT (local) ──────────────────────────────────────
print("[INFO] Đang load Embedding model BGE-M3...")
custom_embeddings = BGEM3Embeddings()

print("[INFO] Đang kết nối Qdrant Docker (localhost:6333)...")
client = QdrantClient(url="http://localhost:6333")

vector_db = QdrantVectorStore(
    client=client,
    collection_name="fashion_products_bge_m3",
    embedding=custom_embeddings,
)
retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)
print("[OK] Qdrant + Retriever sẵn sàng!")

# ── LAYER B ───────────────────────────────────────────────────────────────────
def load_layer_b(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

_base_dir = os.path.dirname(os.path.abspath(__file__))
layer_b_female = load_layer_b(os.path.join(_base_dir, "Fashion_Stylists", "Layer_B_Female_Knowledge.json"))
layer_b_male   = load_layer_b(os.path.join(_base_dir, "Fashion_Stylists", "Layer_B_Male_Knowledge.json"))
print(f"[OK] Layer B: {len(layer_b_female)} rules Nữ | {len(layer_b_male)} rules Nam")

def index_layer_b(data: list, collection_name: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"[SKIP] {collection_name} đã tồn tại — bỏ qua index.")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    points = []
    for i, rule in enumerate(data):
        text   = f"{rule['rule_key']} {rule['phong_cach']} {rule['boi_canh']} {rule['ly_do_tu_van']}"
        vector = custom_embeddings.embed_query(text)
        points.append(PointStruct(id=i, vector=vector, payload=rule))
    client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Indexed {len(points)} rules → {collection_name}")

index_layer_b(layer_b_female, "layer_b_female")
index_layer_b(layer_b_male,   "layer_b_male")

# ── LAYER B MATCHING ──────────────────────────────────────────────────────────
def find_matching_rule(user_query: str, gender: str = "female", profile: dict = None):
    collection   = f"layer_b_{gender}"
    query_vector = custom_embeddings.embed_query(user_query)
    conditions   = []
    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(FieldCondition(key="dang_nguoi",
                match=MatchAny(any=[profile["dang_nguoi"], "Mọi vóc dáng"])))
        if profile.get("tone_da"):
            conditions.append(FieldCondition(key="tone_da",
                match=MatchAny(any=[profile["tone_da"], "Mọi tone da"])))
    search_filter = Filter(must=conditions) if conditions else None
    response = client.query_points(collection_name=collection, query=query_vector,
        query_filter=search_filter, limit=1, score_threshold=0.50)
    results = response.points
    # Fallback 1: bỏ tone_da
    if not results and profile and profile.get("tone_da") and profile.get("dang_nguoi"):
        conds2 = [FieldCondition(key="dang_nguoi",
            match=MatchAny(any=[profile["dang_nguoi"], "Mọi vóc dáng"]))]
        response = client.query_points(collection_name=collection, query=query_vector,
            query_filter=Filter(must=conds2), limit=1, score_threshold=0.50)
        results = response.points
    # Fallback 2: không lọc gì
    if not results:
        response = client.query_points(collection_name=collection, query=query_vector,
            limit=1, score_threshold=0.50)
        results = response.points
    return results[0].payload if results else None

def find_outfit_details(base_rule: dict, gender: str = "female") -> dict:
    knowledge    = layer_b_female if gender == "female" else layer_b_male
    outfit_rules = {}
    for category in base_rule.get("goi_y_phoi_cung", []):
        matched = [r for r in knowledge
                   if r["rule_key"].startswith(category)
                   and r["phong_cach"] == base_rule["phong_cach"]
                   and r["boi_canh"]   == base_rule["boi_canh"]]
        if matched:
            outfit_rules[category] = matched[0]
    return outfit_rules

# ── CATEGORY MAPPING (Layer B → Layer A) ─────────────────────────────────────
CATEGORY_MAPPING = {
    "Áo mặc trong (áo thun/sơ mi)": ["Áo"],
    "Áo khoác ngoài":               ["Áo khoác"],
    "Áo khoác nhẹ/Áo len":          ["Áo khoác"],
    "Quần/Chân váy":                 ["Quần", "Chân váy"],
    "Đầm/Jumpsuit":                  ["Đầm", "Jumpsuit"],
    "Giày dép":                      ["Giày"],
    "Túi xách":                      ["Túi xách"],
    "Phụ kiện":                       None,
}
PHU_KIEN_KEYWORD_ROUTER = {
    "Mũ":             ["beret","hat","cap","beanie","fedora","bucket","brim","flat cap"],
    "Găng tay":        ["gloves","glove","arm warmer"],
    "Kính mắt":        ["glasses","sunglasses","sunglass"],
    "Đồng hồ":         ["watch"],
    "Dây chuyền":      ["necklace","chain pendant","chain"],
    "Bông tai":        ["earring","earrings"],
    "Vòng tay":        ["bracelet"],
    "Nhẫn":            ["ring"],
    "Ghim cài áo":     ["brooch","pin","badge"],
    "Phụ kiện hỗ trợ": ["socks","sock","scarf","tie","belt","bandana","headband"],
}

def get_layer_a_categories(layer_b_category: str, product_type: str) -> list:
    if layer_b_category != "Phụ kiện":
        return CATEGORY_MAPPING.get(layer_b_category, [])
    ptype_lower = product_type.lower()
    for cat, keywords in PHU_KIEN_KEYWORD_ROUTER.items():
        if any(kw in ptype_lower for kw in keywords):
            return [cat]
    return ["Phụ kiện hỗ trợ"]

def get_products_for_outfit(product_type: str, layer_b_category: str,
                             phong_cach: str, vdb) -> list:
    target_categories = get_layer_a_categories(layer_b_category, product_type)
    search_filter = None
    if target_categories:
        search_filter = Filter(must=[FieldCondition(
            key="metadata.category", match=MatchAny(any=target_categories))])
    return vdb.similarity_search(
        query=f"{product_type} {phong_cach}", k=3, filter=search_filter)

# ── VISION (Qwen2.5-VL qua Ollama) ────────────────────────────────────────────
QWEN_VL_MODEL = "qwen2.5vl:3b"

def _call_vl(image_path: str, prompt: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(model=QWEN_VL_MODEL,
                       messages=[{"role": "user", "content": prompt, "images": [img_b64]}])
    return resp["message"]["content"].strip()

def detect_image_type(image_path: str, user_query: str = "") -> str:
    prompt = ("Ảnh này chứa gì? Trả lời đúng 1 chữ: "
              "PERSON nếu là ảnh chụp người, PRODUCT nếu là ảnh sản phẩm thời trang. "
              "Chỉ trả lời PERSON hoặc PRODUCT.")
    result = _call_vl(image_path, prompt).upper()
    return "person" if "PERSON" in result else "product"

def analyze_person_image(image_path: str) -> dict:
    prompt = """Bạn là chuyên gia tư vấn thời trang. Hãy phân tích người trong ảnh:
1. DÁNG NGƯỜI (chọn 1): Dáng chữ A | Dáng quả lê | Dáng táo | Dáng đồng hồ cát | Dáng chữ H | Dáng chữ V | Dáng thẳng
2. TONE DA (chọn 1): Da trắng | Da vàng | Da ngăm | Da tối
3. NHẬN XÉT: 1-2 câu về điểm nổi bật khi phối đồ.
Trả lời theo format:
DÁNG: [tên dáng]
TONE: [tên tone]
NHẬN XÉT: [nội dung]"""
    raw     = _call_vl(image_path, prompt)
    profile = {"dang_nguoi": None, "tone_da": None, "nhan_xet": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("DÁNG:"):
            profile["dang_nguoi"] = line.replace("DÁNG:", "").strip()
        elif line.startswith("TONE:"):
            profile["tone_da"] = line.replace("TONE:", "").strip()
        elif line.startswith("NHẬN XÉT:"):
            profile["nhan_xet"] = line.replace("NHẬN XÉT:", "").strip()
    return profile

def caption_product_image(image_path: str, user_query: str = "") -> str:
    prompt = """Mô tả sản phẩm thời trang trong ảnh bằng tiếng Việt.
Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu, phong cách. Ngắn gọn 1-2 câu."""
    return _call_vl(image_path, prompt)

print("[OK] Vision functions sẵn sàng!")

# ── LLM ───────────────────────────────────────────────────────────────────────
print("[INFO] Đang khởi tạo LLM Qwen local...")
llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.4,
                 timeout=120, num_predict=1024, num_ctx=8192)
print("[OK] LLM sẵn sàng!")

# ── PROMPTS ───────────────────────────────────────────────────────────────────
system_prompt = """Bạn là một chuyên viên tư vấn thời trang cao cấp, có gu thẩm mỹ tinh tế và giọng văn vô cùng thân thiện, thanh lịch.

QUY TẮC TỐI CAO (CHỐNG BỊA ĐẶT - ANTI-HALLUCINATION):
1. BẠN PHẢI TÌM TRONG phần "DỮ LIỆU SẢN PHẨM" bên dưới để trả lời khách.
2. TUYỆT ĐỐI KHÔNG bịa ra tên, giá tiền, hay đặc điểm sản phẩm nếu không có trong dữ liệu.
3. NẾU KHÔNG CÓ DỮ LIỆU KHỚP: Xin lỗi duyên dáng là shop tạm hết mẫu này và chủ động hỏi khách có muốn đổi sang phong cách khác không.

CÁCH TRÌNH BÀY (Mượt mà, tự nhiên, có xAI):
- Mở đầu bằng một câu chào hoặc nhận xét nhẹ nhàng về gu của khách.
- Khi giới thiệu sản phẩm, hãy lồng ghép thông tin khéo léo thành đoạn văn thay vì gạch đầu dòng khô khan.
- Bắt buộc in đậm **Tên Sản Phẩm** và kèm (Mã SP: [MÃ_SP]) - [Giá] VNĐ.
- xAI (GIẢI THÍCH LÝ DO - BẮT BUỘC): Sau mỗi sản phẩm, THÊM 1 câu giải thích ngắn tại sao sản phẩm này phù hợp với yêu cầu của khách (dựa vào màu sắc, chất liệu, dịp mặc, hoặc vóc dáng).
- Nếu có ẢNH trong dữ liệu: đính kèm ảnh đầu tiên theo format ![ảnh](URL_ẢNH) để khách xem trực quan.
- Trả lời súc tích, không vượt quá 300 từ.

DỮ LIỆU SẢN PHẨM:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """Nhiệm vụ của bạn là NGƯỜI VIẾT LẠI CÂU HỎI.
Dựa vào lịch sử trò chuyện, hãy làm rõ nghĩa của câu hỏi mới nhất để nó có thể đứng độc lập mà ai đọc cũng hiểu được.

QUY TẮC SỐNG CÒN:
- TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI CỦA KHÁCH.
- CHỈ IN RA DUY NHẤT CÂU HỎI ĐÃ ĐƯỢC VIẾT LẠI. Không giải thích, không dạ thưa.
- Nếu câu hỏi đã quá rõ ràng rồi, hãy in lại y nguyên.

VÍ DỤ: Khách: "Có màu khác không?" -> CHỈ IN RA: "Áo thun đỏ ở trên có màu khác không?"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

doc_prompt = PromptTemplate.from_template(
    "\n[MÃ_SP: {product_id}]"
    "\nẢNH: {images}"
    "\nTHÔNG TIN CHI TIẾT: {page_content}\n"
)

OUTFIT_SYSTEM_PROMPT = """Bạn là một chuyên gia tạo dáng (Personal Stylist) cực kỳ chuyên nghiệp và tâm lý.

NHIỆM VỤ: Dựa vào "CÔNG THỨC PHỐI ĐỒ" và "SẢN PHẨM GỢI Ý" bên dưới, hãy "hô biến" một bộ outfit hoàn hảo cho khách hàng.

QUY TẮC:
1. Khéo léo xâu chuỗi các món đồ thành một bức tranh tổng thể.
2. TUYỆT ĐỐI không giới thiệu đồ ngoài danh sách "SẢN PHẨM GỢI Ý". Không tự bịa thêm đồ.
3. Nhớ in đậm **Tên Sản Phẩm**, kèm (Mã SP: [MÃ_SP]) và [Giá] VNĐ ở mỗi món.
4. xAI - TÍNH MINH BẠCH (BẮT BUỘC): ở mỗi món đồ, BẠN PHẢI GIẢI THÍCH TẠI SAO món này phù hợp (dựa vào vóc dáng, tone da, hoặc lý do có trong công thức).
5. Nếu có ẢNH trong dữ liệu sản phẩm: đính kèm ![ảnh](URL_ẢNH) để khách xem trực quan.
6. Giọng điệu nịnh khách, sang trọng nhưng gần gũi. Lồng ghép thành các đoạn văn mượt mà, tránh dùng gạch đầu dòng liệt kê như hóa đơn.
7. Kết thúc bằng 1 câu chốt sale/hỏi han thân thiện. Giới hạn 350 từ.

{outfit_context}"""

outfit_prompt = ChatPromptTemplate.from_messages([
    ("system", OUTFIT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── REDIS HISTORY ─────────────────────────────────────────────────────────────
REDIS_URL = "redis://localhost:6379"

# def get_message_history(session_id: str):
#     history = RedisChatMessageHistory(session_id, url=REDIS_URL)
#     current = history.messages
#     if len(current) > 6:
#         kept = current[-6:]
#         history.clear()
#         history.add_messages(kept)
#     return history
# Cell 21 — thay get_message_history() bằng version có summarization

SUMMARIZE_PROMPT = """Tóm tắt cuộc hội thoại mua sắm thời trang sau thành 3-5 câu ngắn.
Giữ lại: sản phẩm đã hỏi, phong cách khách thích, thông tin vóc dáng/tone da (nếu có).
Bỏ qua: lời chào, câu xã giao.
Chỉ trả về đoạn tóm tắt, không thêm gì khác.

Hội thoại:
{history_text}"""

def summarize_history(messages: list) -> str:
    """Dùng LLM tóm tắt lịch sử hội thoại cũ."""
    history_text = "\n".join([
        f"{'Khách' if m.type == 'human' else 'Bot'}: {m.content[:300]}"
        for m in messages
    ])
    resp = ollama.chat(
        model   = "qwen3:4b-instruct",
        messages= [{"role": "user",
                    "content": SUMMARIZE_PROMPT.format(history_text=history_text)}],
        options = {"temperature": 0, "num_predict": 150}
    )
    return resp["message"]["content"].strip()

def get_message_history(session_id: str):
    from langchain_core.messages import SystemMessage

    history  = RedisChatMessageHistory(session_id, url=REDIS_URL)
    messages = history.messages

    # Dưới 8 message → giữ nguyên, chưa cần tóm tắt
    if len(messages) <= 8:
        return history

    # Trên 8 message → tóm tắt phần cũ, giữ 4 message gần nhất
    old_messages    = messages[:-4]
    recent_messages = messages[-4:]

    summary_text = summarize_history(old_messages)

    # Rebuild history: [summary message] + [4 recent messages]
    history.clear()
    history.add_message(SystemMessage(
        content=f"[TÓM TẮT HỘI THOẠI TRƯỚC]: {summary_text}"
    ))
    history.add_messages(recent_messages)

    return history
print("[OK] Redis history sẵn sàng!")

# ── RAG PIPELINE ──────────────────────────────────────────────────────────────
print("[INFO] Đang lắp ráp RAG Pipeline...")
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
document_chain          = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT,
                                                        document_prompt=doc_prompt)
rag_chain               = create_retrieval_chain(history_aware_retriever, document_chain)

full_chat_chain = RunnableWithMessageHistory(
    rag_chain, get_message_history,
    input_messages_key="input", history_messages_key="chat_history",
    output_messages_key="answer",
)

outfit_llm_chain = outfit_prompt | llm
outfit_chain_with_history = RunnableWithMessageHistory(
    outfit_llm_chain, get_message_history,
    input_messages_key="input", history_messages_key="chat_history",
)
print("[OK] RAG Pipeline sẵn sàng!")

# ── INTENT DETECTION (Hybrid: keyword + LLM) ─────────────────────────────────
DEFINITE_GREETING = ["xin chào", "hello", "hi bạn", "chào bạn", "hey", "alo",
                     "chào buổi sáng", "chào buổi chiều"]
DEFINITE_CHITCHAT = ["cảm ơn", "cảm on", "thank you", "thanks", "tạm biệt",
                     "bye", "hẹn gặp lại", "bái bai"]
DEFINITE_OUTFIT   = ["phối đồ", "mix match", "mặc với gì", "mặc cùng gì",
                     "kết hợp với gì", "phối với gì", "outfit cho",
                     "gợi ý outfit", "tư vấn phối"]
DEFINITE_SEARCH   = ["còn hàng không", "còn size", "giá bao nhiêu", "mã sp",
                     "có bán không", "tìm giúp", "cho xem", "shop có"]
MALE_KEYWORDS     = ["nam", "con trai", "anh", "bạn trai", "chàng", "đàn ông"]

INTENT_CLASSIFY_PROMPT = """Bạn là bộ phân loại intent cho chatbot tư vấn thời trang.
Phân loại câu hỏi vào đúng 1 trong 4 nhóm:
OUTFIT   → Hỏi cách phối đồ, mix-match, tư vấn mặc gì cho dịp/vóc dáng/phong cách
SEARCH   → Tìm sản phẩm cụ thể, hỏi giá, còn hàng, so sánh, xem ảnh sản phẩm
CHITCHAT → Cảm ơn, tạm biệt, hỏi thăm, câu xã giao không liên quan mua sắm
GREETING → Chào hỏi, bắt đầu cuộc trò chuyện
{context_block}
Câu cần phân loại: "{query}"
Chỉ trả lời đúng 1 từ: OUTFIT / SEARCH / CHITCHAT / GREETING"""

def detect_intent_llm(query: str, last_bot_msg: str = "") -> str:
    context_block = ""
    if last_bot_msg:
        context_block = f'\nContext — Bot vừa nói: "{last_bot_msg[:120]}..."\n'
    try:
        resp = ollama.chat(
            model="qwen3:4b-instruct",
            messages=[{"role": "user",
                        "content": INTENT_CLASSIFY_PROMPT.format(
                            query=query, context_block=context_block)}],
            options={"temperature": 0, "num_predict": 10},
        )
        result = resp["message"]["content"].strip().upper()
        for intent in ["OUTFIT", "SEARCH", "CHITCHAT", "GREETING"]:
            if intent in result:
                return intent.lower()
    except Exception as e:
        print(f"[WARN] LLM intent lỗi: {e} → fallback search")
    return "search"

def detect_intent(query: str, last_bot_msg: str = "") -> str:
    q = query.lower().strip()
    if any(kw in q for kw in DEFINITE_OUTFIT):   return "outfit"
    if any(kw in q for kw in DEFINITE_SEARCH):   return "search"
    if any(kw in q for kw in DEFINITE_GREETING): return "greeting"
    if any(kw in q for kw in DEFINITE_CHITCHAT): return "chitchat"
    return detect_intent_llm(query, last_bot_msg)

def detect_gender(query: str) -> str:
    return "male" if any(kw in query.lower() for kw in MALE_KEYWORDS) else "female"

def get_greeting_response() -> str:
    return ("Xin chào! Mình là trợ lý tư vấn thời trang của shop. "
            "Bạn cần tìm sản phẩm hay muốn được gợi ý phối đồ hôm nay? 😊")

def get_chitchat_response(query: str) -> str:
    return "Rất vui được hỗ trợ bạn! Bạn còn muốn hỏi thêm gì về thời trang không?"

# ── BUILD OUTFIT CONTEXT ──────────────────────────────────────────────────────
def build_outfit_context(user_query: str, gender: str = "female",
                          profile: dict = None) -> tuple:
    """
    Trả về tuple (context_str, images_data):
    - context_str: chuỗi context gửi vào LLM
    - images_data: list[dict] {product_id, category, images: [url...]}
    """
    base_rule = find_matching_rule(user_query, gender, profile)
    if not base_rule:
        return "", []
    outfit_rules = find_outfit_details(base_rule, gender)
    if not outfit_rules:
        return "", []

    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        products     = get_products_for_outfit(
            product_type, layer_b_category, base_rule["phong_cach"], vector_db)
        outfit_products[layer_b_category] = {
            "product_type": product_type,
            "ly_do":        rule["ly_do_tu_van"],
            "products":     products,
        }

    lines = ["CÔNG THỨC PHỐI ĐỒ:",
             f"  Phong cách : {base_rule['phong_cach']}",
             f"  Bối cảnh   : {base_rule['boi_canh']}",
             f"  Lý do chính: {base_rule['ly_do_tu_van']}"]
    if profile and profile.get("dang_nguoi"):
        lines.append(f"  Dáng người : {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):
        lines.append(f"  Tone da    : {profile['tone_da']}")
    lines += ["", "SẢN PHẨM GỢI Ý:"]

    images_data = []  # Thu thập ảnh sản phẩm để trả về frontend

    for cat, data in outfit_products.items():
        lines.append(f"\n[{cat} – {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")
        if data["products"]:
            for doc in data["products"]:
                pid = doc.metadata.get("product_id", "N/A")
                price_raw = doc.metadata.get("price", "N/A")
                try:
                    price_fmt = f"{int(price_raw):,}".replace(",", ".")
                except Exception:
                    price_fmt = price_raw
                lines.append(f"  • (Mã SP: {pid} | Giá: {price_fmt} VNĐ)")
                lines.append(f"    {doc.page_content[:600]}")

                # Thu thập ảnh sản phẩm
                doc_images = [url for url in doc.metadata.get("images", []) if url]
                if doc_images:
                    images_data.append({
                        "product_id": pid,
                        "category":   cat,
                        "images":     doc_images[:2],  # Tối đa 2 ảnh/sản phẩm
                    })
        else:
            lines.append("  • (Chưa có sản phẩm phù hợp trong kho)")

    return "\n".join(lines), images_data

print("[OK] chatbot_core đã load xong — sẵn sàng phục vụ!")
