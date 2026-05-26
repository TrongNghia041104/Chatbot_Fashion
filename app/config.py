"""
config.py — Cấu hình tập trung cho Fashion RAG Chatbot
=======================================================
Tất cả constants, URLs, model names, keyword lists đều ở đây.
Khi cần thay đổi, chỉ cần sửa file này.
"""
import os

# ── Đường dẫn gốc của project ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Đường dẫn dữ liệu ─────────────────────────────────────────────────────────
DATA_DIR           = os.path.join(BASE_DIR, "data")
STYLISTS_DIR       = os.path.join(DATA_DIR, "stylists")
METADATA_DIR       = os.path.join(DATA_DIR, "metadata")
STATIC_DIR         = os.path.join(BASE_DIR, "app", "static")
IMAGES_DIR         = os.path.join(BASE_DIR, "images")

LAYER_B_FEMALE_PATH = os.path.join(STYLISTS_DIR, "Layer_B_Female_Knowledge.json")
LAYER_B_MALE_PATH   = os.path.join(STYLISTS_DIR, "Layer_B_Male_Knowledge.json")

# ── Cấu hình Ollama ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"   # SSH tunnel → Vast.ai GPU
EMBEDDING_MODEL  = "bge-m3"
LLM_MODEL        = "qwen3:4b-instruct"
VISION_MODEL     = "qwen2.5vl:3b"

# ── Cấu hình LLM ─────────────────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.4
LLM_TIMEOUT      = 120
LLM_NUM_PREDICT  = 1024
LLM_NUM_CTX      = 8192

# ── Cấu hình Qdrant ───────────────────────────────────────────────────────────
QDRANT_URL                   = "http://localhost:6333"
QDRANT_COLLECTION_FASHION    = "fashion_products_bge_m3"
QDRANT_COLLECTION_LAYER_B_F  = "layer_b_female"
QDRANT_COLLECTION_LAYER_B_M  = "layer_b_male"
QDRANT_VECTOR_SIZE           = 1024

# ── Cấu hình Retriever ────────────────────────────────────────────────────────
RETRIEVER_K               = 5
RETRIEVER_SCORE_THRESHOLD = 0.7
LAYER_B_SCORE_THRESHOLD   = 0.50
LAYER_B_LIMIT             = 1

# ── Cấu hình Redis ───────────────────────────────────────────────────────────
REDIS_URL                 = "redis://localhost:6379"
HISTORY_MAX_MESSAGES      = 8    # Số message trước khi summarize
HISTORY_RECENT_KEEP       = 4    # Số message gần nhất giữ lại sau summarize
SUMMARIZE_MAX_TOKENS      = 150

# ── Cấu hình API ─────────────────────────────────────────────────────────────
API_TITLE   = "Fashion RAG Chatbot API"
API_VERSION = "1.0.0"
API_HOST    = "0.0.0.0"
API_PORT    = 8000

# ── Intent Keywords ───────────────────────────────────────────────────────────
DEFINITE_GREETING = [
    "xin chào", "hello", "hi bạn", "chào bạn", "hey", "alo",
    "chào buổi sáng", "chào buổi chiều",
]
DEFINITE_CHITCHAT = [
    "cảm ơn", "cảm on", "thank you", "thanks", "tạm biệt",
    "bye", "hẹn gặp lại", "bái bai",
]
DEFINITE_OUTFIT = [
    "phối đồ", "mix match", "mặc với gì", "mặc cùng gì",
    "kết hợp với gì", "phối với gì", "outfit cho",
    "gợi ý outfit", "tư vấn phối",
]
DEFINITE_SEARCH = [
    "còn hàng không", "còn size", "giá bao nhiêu", "mã sp",
    "có bán không", "tìm giúp", "cho xem", "shop có",
]
MALE_KEYWORDS = ["nam", "con trai", "anh", "bạn trai", "chàng", "đàn ông"]

# ── Category Mapping (Layer B → Layer A / Qdrant) ────────────────────────────
CATEGORY_MAPPING = {
    "Áo mặc trong (áo thun/sơ mi)": ["Áo"],
    "Áo khoác ngoài":               ["Áo khoác"],
    "Áo khoác nhẹ/Áo len":          ["Áo khoác"],
    "Quần/Chân váy":                 ["Quần", "Chân váy"],
    "Đầm/Jumpsuit":                  ["Đầm", "Jumpsuit"],
    "Giày dép":                      ["Giày"],
    "Túi xách":                      ["Túi xách"],
    "Phụ kiện":                      None,
}

PHU_KIEN_KEYWORD_ROUTER = {
    "Mũ":             ["beret", "hat", "cap", "beanie", "fedora", "bucket", "brim", "flat cap"],
    "Găng tay":        ["gloves", "glove", "arm warmer"],
    "Kính mắt":        ["glasses", "sunglasses", "sunglass"],
    "Đồng hồ":         ["watch"],
    "Dây chuyền":      ["necklace", "chain pendant", "chain"],
    "Bông tai":        ["earring", "earrings"],
    "Vòng tay":        ["bracelet"],
    "Nhẫn":            ["ring"],
    "Ghim cài áo":     ["brooch", "pin", "badge"],
    "Phụ kiện hỗ trợ": ["socks", "sock", "scarf", "tie", "belt", "bandana", "headband"],
}
