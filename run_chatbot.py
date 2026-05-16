"""
Fashion RAG Chatbot - Script chạy gọn (thay cho notebook Version4)
Thứ tự đã được sắp xếp lại đúng dependency.
Chạy: python run_chatbot.py
"""

# ═══════════════════════════════════════════════════════════════════
# PHẦN 1: IMPORT THƯ VIỆN
# ═══════════════════════════════════════════════════════════════════
import json, sys, uuid, os, time, base64
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
import ollama

# ═══════════════════════════════════════════════════════════════════
# PHẦN 2: MÔ HÌNH EMBEDDING (BGE-M3)
# ═══════════════════════════════════════════════════════════════════
class BGEM3Embeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-m3"):
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
    def embed_documents(self, texts):
        return self.hf_embeddings.embed_documents(texts)
    def embed_query(self, text):
        return self.hf_embeddings.embed_query(text)

# ═══════════════════════════════════════════════════════════════════
# PHẦN 3: KẾT NỐI QDRANT & KHỞI TẠO VECTOR STORE
# ═══════════════════════════════════════════════════════════════════
print("[THÔNG BÁO] Đang khởi tạo mô hình Embedding...")
custom_embeddings = BGEM3Embeddings()

print("[THÔNG BÁO] Đang kết nối tới Qdrant Database...")
client = QdrantClient(path="./qdrant_data")
vector_db = QdrantVectorStore(
    client=client,
    collection_name="fashion_products_bge_m3",
    embedding=custom_embeddings,
)
print("[OK] Đã kết nối thành công!")

retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)
print("[OK] Retriever đã sẵn sàng!")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 4: NẠP LAYER B KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════
def load_layer_b(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

layer_b_female = load_layer_b("Layer_B_Female_Knowledge.json")
layer_b_male = load_layer_b("Layer_B_Male_Knowledge.json")
print(f"[OK] Đã nạp {len(layer_b_female)} quy tắc Nữ và {len(layer_b_male)} quy tắc Nam.")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 5: LAYER B - INDEX & SEMANTIC MATCHING
# ═══════════════════════════════════════════════════════════════════
def index_layer_b(data, collection_name):
    qdrant_client = vector_db.client
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing:
        print(f"[SKIP] {collection_name} đã tồn tại ({qdrant_client.count(collection_name).count} rules)")
        return
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    points = []
    for i, rule in enumerate(data):
        text = f"{rule['rule_key']} {rule['phong_cach']} {rule['boi_canh']} {rule['ly_do_tu_van']}"
        vector = vector_db.embeddings.embed_query(text)
        points.append(PointStruct(id=i, vector=vector, payload=rule))
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Indexed {len(points)} rules -> {collection_name}")

index_layer_b(layer_b_female, "layer_b_female")
index_layer_b(layer_b_male, "layer_b_male")

def find_matching_rule(user_query, gender="female", profile=None):
    collection = f"layer_b_{gender}"
    qdrant_client = vector_db.client
    query_vector = vector_db.embeddings.embed_query(user_query)
    conditions = []
    if profile:
        if profile.get("dang_nguoi"):
            conditions.append(FieldCondition(key="dang_nguoi", match=MatchAny(any=[profile["dang_nguoi"], "Mọi dáng người"])))
        if profile.get("tone_da"):
            conditions.append(FieldCondition(key="tone_da", match=MatchAny(any=[profile["tone_da"], "Mọi tone da"])))
    search_filter = Filter(must=conditions) if conditions else None
    results = qdrant_client.search(collection_name=collection, query_vector=query_vector,
                                   query_filter=search_filter, limit=1, score_threshold=0.50)
    if not results and search_filter:
        results = qdrant_client.search(collection_name=collection, query_vector=query_vector,
                                       limit=1, score_threshold=0.50)
    return results[0].payload if results else None

def find_outfit_details(base_rule, gender="female"):
    knowledge = layer_b_female if gender == "female" else layer_b_male
    outfit_rules = {}
    for category in base_rule.get("goi_y_phoi_cung", []):
        matched = [r for r in knowledge
                   if r["rule_key"].startswith(category)
                   and r["phong_cach"] == base_rule["phong_cach"]
                   and r["boi_canh"] == base_rule["boi_canh"]]
        if matched:
            outfit_rules[category] = matched[0]
    return outfit_rules

# ═══════════════════════════════════════════════════════════════════
# PHẦN 6: LAYER B -> LAYER A CATEGORY MAPPING
# ═══════════════════════════════════════════════════════════════════
CATEGORY_MAPPING = {
    "Áo mặc trong (áo thun/sơ mi)": ["Áo"],
    "Áo khoác ngoài":                ["Áo khoác"],
    "Áo khoác nhẹ/Áo len":           ["Áo khoác"],
    "Quần/Chân váy":                  ["Quần", "Chân váy"],
    "Đầm/Jumpsuit":                   ["Đầm", "Jumpsuit"],
    "Giày dép":                       ["Giày"],
    "Túi xách":                       ["Túi xách"],
    "Phụ kiện":                       None,
}
PHU_KIEN_KEYWORD_ROUTER = {
    "Mũ":             ["beret","hat","cap","beanie","fedora","bucket","brim","flat cap","earflap","snood","balaclava","trapper"],
    "Găng tay":        ["gloves","glove","arm warmer"],
    "Kính mắt":        ["glasses","sunglasses","sunglass"],
    "Đồng hồ":         ["watch"],
    "Dây chuyền":      ["necklace","chain pendant","chain"],
    "Bông tai":        ["earring","earrings"],
    "Vòng tay":        ["bracelet"],
    "Nhẫn":            ["ring"],
    "Ghim cài áo":     ["brooch","pin","badge"],
    "Phụ kiện hỗ trợ": ["socks","sock","scarf","tie","belt","bandana","headband","suspender"],
}

def get_layer_a_categories(layer_b_category, product_type):
    if layer_b_category != "Phụ kiện":
        return CATEGORY_MAPPING.get(layer_b_category, [])
    ptype_lower = product_type.lower()
    for layer_a_cat, keywords in PHU_KIEN_KEYWORD_ROUTER.items():
        if any(kw in ptype_lower for kw in keywords):
            return [layer_a_cat]
    return ["Phụ kiện hỗ trợ"]

def get_products_for_outfit(product_type, layer_b_category, phong_cach, vector_db):
    target_categories = get_layer_a_categories(layer_b_category, product_type)
    search_filter = None
    if target_categories:
        search_filter = Filter(must=[FieldCondition(key="metadata.category", match=MatchAny(any=target_categories))])
    query = f"{product_type} {phong_cach}"
    return vector_db.similarity_search(query=query, k=3, filter=search_filter)

# ═══════════════════════════════════════════════════════════════════
# PHẦN 7: XỬ LÝ ẢNH VỚI QWEN2.5-VL
# ═══════════════════════════════════════════════════════════════════
QWEN_VL_MODEL = "qwen2.5vl:3b"

def _call_vl(image_path, prompt):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(model=QWEN_VL_MODEL,
                       messages=[{"role":"user","content":prompt,"images":[img_b64]}])
    return resp["message"]["content"].strip()

def detect_image_type(image_path):
    prompt = ("Ảnh này chứa gì? Trả lời đúng 1 chữ: "
              "PERSON nếu là ảnh chụp người, PRODUCT nếu là ảnh sản phẩm thời trang. "
              "Chỉ trả lời PERSON hoặc PRODUCT.")
    result = _call_vl(image_path, prompt).upper()
    return "person" if "PERSON" in result else "product"

def analyze_person_image(image_path):
    prompt = """Bạn là chuyên gia tư vấn thời trang. Hãy phân tích người trong ảnh:
1. DÁNG NGƯỜI (chọn 1): Dáng chữ A | Dáng quả lê | Dáng táo | Dáng đồng hồ cát | Dáng chữ H | Dáng chữ V | Dáng thẳng
2. TONE DA (chọn 1): Da trắng | Da vàng | Da ngăm | Da tối
3. NHẬN XÉT: 1-2 câu về điểm nổi bật khi phối đồ.
Trả lời theo format:
DÁNG: [tên dáng]
TONE: [tên tone]
NHẬN XÉT: [nội dung]"""
    raw = _call_vl(image_path, prompt)
    profile = {"dang_nguoi": None, "tone_da": None, "nhan_xet": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("DÁNG:"): profile["dang_nguoi"] = line.replace("DÁNG:","").strip()
        elif line.startswith("TONE:"): profile["tone_da"] = line.replace("TONE:","").strip()
        elif line.startswith("NHẬN XÉT:"): profile["nhan_xet"] = line.replace("NHẬN XÉT:","").strip()
    return profile

def caption_product_image(image_path):
    prompt = """Mô tả sản phẩm thời trang trong ảnh bằng tiếng Việt.
Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu, phong cách. Ngắn gọn 1-2 câu."""
    return _call_vl(image_path, prompt)

print("[OK] Vision functions đã sẵn sàng!")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 8: KHỞI TẠO LLM
# ═══════════════════════════════════════════════════════════════════
print("[THÔNG BÁO] Đang khởi tạo mô hình Qwen local...")
llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0.4,
                 timeout=120, num_predict=350, num_ctx=8192)
print("[OK] LLM đã sẵn sàng!")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 9: PROMPT TEMPLATES & REDIS
# ═══════════════════════════════════════════════════════════════════
system_prompt = """Bạn là chuyên viên tư vấn thời trang cao cấp của shop.

QUY TẮC TỐI CAO (CHỐNG BỊA ĐẶT - ANTI-HALLUCINATION):
1. CHỈ SỬ DỤNG thông tin có trong phần "DỮ LIỆU SẢN PHẨM" bên dưới để trả lời.
2. TUYỆT ĐỐI KHÔNG tự bịa ra tên sản phẩm, không bịa ID, giá tiền, màu sắc hay chất liệu. 
3. NẾU KHÔNG TÌM THẤY SẢN PHẨM KHỚP NHU CẦU: Khẳng định ngay là shop chưa có mẫu này.

QUY TẮC TRÍCH XUẤT VÀ ĐỊNH DẠNG:
4. BẮT BUỘC TRÍCH XUẤT CHÍNH XÁC 100%.
5. TRÌNH BÀY BẮT BUỘC theo khuôn mẫu sau:
   - **[TÊN_SP]** (Mã SP: [MÃ_SP])
   - Giá: [GIÁ_TIỀN] VNĐ
   - Đặc điểm chi tiết: VIẾT THÀNH MỘT CÂU HOÀN CHỈNH.
   - (1-2 câu tư vấn thân thiện)
6. GIỚI HẠN: không vượt quá 200 từ.

DỮ LIỆU SẢN PHẨM:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """BẠN LÀ MỘT CÔNG CỤ VIẾT LẠI CÂU TRUY VẤN, BẠN KHÔNG PHẢI LÀ CHATBOT.
Nhiệm vụ: Đọc "Lịch sử trò chuyện" và "Câu hỏi mới", VIẾT nó thành MỘT CÂU TRUY VẤN HOÀN CHỈNH.
QUY TẮC: CHỈ IN RA CÂU TRUY VẤN HOÀN CHỈNH. Nếu không chắc chắn thì TRẢ VỀ CÂU GỐC."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

doc_prompt = PromptTemplate.from_template("\n[MÃ_SP: {product_id}]\nTHÔNG TIN CHI TIẾT: {page_content}\n")

OUTFIT_SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn thời trang cao cấp của shop.
NHIỆM VỤ: Dựa vào "CÔNG THỨC PHỐI ĐỒ" và "SẢN PHẨM GỢI Ý" bên dưới, tư vấn cho khách một bộ outfit hoàn chỉnh.
QUY TẮC:
1. CHỈ giới thiệu sản phẩm có trong phần "SẢN PHẨM GỢI Ý". KHÔNG tự bịa.
2. Với mỗi món đồ: • **[TÊN SP]** (Mã SP: [MÃ_SP]) – [GIÁ] VNĐ → [1 câu mô tả]
3. Kết thúc bằng 1-2 câu tổng kết phong cách. Giới hạn 250 từ.

{outfit_context}"""

outfit_prompt = ChatPromptTemplate.from_messages([
    ("system", OUTFIT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

REDIS_URL = "redis://localhost:6379"

def get_message_history(session_id):
    history = RedisChatMessageHistory(session_id, url=REDIS_URL)
    current_messages = history.messages
    if len(current_messages) > 6:
        kept = current_messages[-6:]
        history.clear()
        history.add_messages(kept)
    return history

print("[OK] Prompts & Redis đã cấu hình xong!")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 10: XÂY DỰNG RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════
print("[THÔNG BÁO] Đang lắp ráp RAG Pipeline...")

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
document_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT, document_prompt=doc_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

full_chat_chain = RunnableWithMessageHistory(
    rag_chain, get_message_history,
    input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer",
)

outfit_llm_chain = outfit_prompt | llm
outfit_chain_with_history = RunnableWithMessageHistory(
    outfit_llm_chain, get_message_history,
    input_messages_key="input", history_messages_key="chat_history",
)

print("[OK] RAG Pipeline đã sẵn sàng!")

# ═══════════════════════════════════════════════════════════════════
# PHẦN 11: INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════
OUTFIT_KEYWORDS = ["phối","phối đồ","mặc với","kết hợp","outfit","mix",
                   "mặc gì","đi với","hợp với","bộ đồ","mặc cùng","style với"]
MALE_KEYWORDS = ["nam","con trai","anh","bạn trai","chàng","đàn ông"]

def detect_intent(query):
    return "outfit" if any(kw in query.lower() for kw in OUTFIT_KEYWORDS) else "search"

def detect_gender(query):
    return "male" if any(kw in query.lower() for kw in MALE_KEYWORDS) else "female"

# ═══════════════════════════════════════════════════════════════════
# PHẦN 12: BUILD OUTFIT CONTEXT (Layer B -> Layer A -> LLM)
# ═══════════════════════════════════════════════════════════════════
def build_outfit_context(user_query, gender="female", profile=None):
    base_rule = find_matching_rule(user_query, gender, profile)
    if not base_rule: return ""
    outfit_rules = find_outfit_details(base_rule, gender)
    if not outfit_rules: return ""

    outfit_products = {}
    for layer_b_category, rule in outfit_rules.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        products = get_products_for_outfit(product_type, layer_b_category, base_rule["phong_cach"], vector_db)
        outfit_products[layer_b_category] = {"product_type": product_type, "ly_do": rule["ly_do_tu_van"], "products": products}

    lines = ["CÔNG THỨC PHỐI ĐỒ:",
             f"  Phong cách : {base_rule['phong_cach']}",
             f"  Bối cảnh   : {base_rule['boi_canh']}",
             f"  Lý do chính: {base_rule['ly_do_tu_van']}", ""]
    if profile and profile.get("dang_nguoi"): lines.append(f"  Dáng người : {profile['dang_nguoi']}")
    if profile and profile.get("tone_da"):    lines.append(f"  Tone da    : {profile['tone_da']}")
    lines += ["", "SẢN PHẨM GỢI Ý:"]

    for cat, data in outfit_products.items():
        lines.append(f"\n[{cat} – {data['product_type']}]")
        lines.append(f"  Lý do: {data['ly_do']}")
        if data["products"]:
            for doc in data["products"]:
                pid = doc.metadata.get("product_id", "N/A")
                price = doc.metadata.get("price", "N/A")
                lines.append(f"  • (Mã SP: {pid} | Giá: {price} VNĐ)")
                lines.append(f"    {doc.page_content[:200]}")
        else:
            lines.append("  • (Chưa có sản phẩm phù hợp trong kho)")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# PHẦN 13: LUỒNG CHẠY CHÍNH (CHAT LOOP)
# ═══════════════════════════════════════════════════════════════════
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
    user_input = input("👤 Bạn: ").strip()
    if user_input == "0":
        print("\n🤖 Chatbot: Hẹn gặp lại bạn nhé!")
        break
    if not user_input: continue

    final_query = user_input

    # Xử lý ảnh
    raw_img = input("📎 Ảnh (Enter để bỏ qua): ").strip()
    if raw_img and os.path.exists(raw_img):
        print("🔍 [Đang phân tích ảnh...]")
        image_type = detect_image_type(raw_img)
        print(f"   → Phát hiện: {image_type.upper()}")
        if image_type == "person":
            person_info = analyze_person_image(raw_img)
            if person_info["dang_nguoi"]: user_profile["dang_nguoi"] = person_info["dang_nguoi"]
            if person_info["tone_da"]:    user_profile["tone_da"] = person_info["tone_da"]
            print(f"\n🤖 Chatbot: Mình đã phân tích xong! Bạn có **{person_info['dang_nguoi']}** "
                  f"với **{person_info['tone_da']}**. {person_info['nhan_xet']} "
                  f"\nBạn muốn mình gợi ý outfit cho dịp nào?")
            print("\n" + "-" * 60 + "\n")
            continue
        else:
            caption = caption_product_image(raw_img)
            print(f"   → Caption: {caption[:80]}...")
            final_query = f"{caption}. Yêu cầu: {user_input}" if user_input else caption
    elif raw_img:
        print(f"   ⚠️  Không tìm thấy file: {raw_img}")

    # Detect intent
    intent = detect_intent(final_query)
    gender = detect_gender(final_query)
    print(f"🔍 [Intent: {intent.upper()} | Gender: {gender} | Profile: {'✅' if user_profile else '(chưa có)'}]")

    # Chạy pipeline
    print("🤖 Chatbot: ", end="")
    start_time = time.time()
    first_token_time = None

    try:
        if intent == "outfit":
            outfit_context = build_outfit_context(final_query, gender, user_profile)
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
                        if first_token_time is None: first_token_time = time.time()
                        print(token, end="", flush=True)

        if intent == "search":
            for chunk in full_chat_chain.stream(
                {"input": final_query},
                config={"configurable": {"session_id": SESSION_ID}, "callbacks": [SpyRetrieverHandler()]},
            ):
                if "answer" in chunk:
                    if first_token_time is None: first_token_time = time.time()
                    print(chunk["answer"], end="", flush=True)

        end_time = time.time()
        if first_token_time is None: first_token_time = end_time
        print(f"\n\n⏱️  TTFT: {first_token_time - start_time:.2f}s | Total: {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"\n[LỖI] {e}")

    print("\n\n" + "-" * 60 + "\n")
