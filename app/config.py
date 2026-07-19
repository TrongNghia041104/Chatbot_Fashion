"""Central configuration for the Fashion RAG chatbot backend.

Research-demo v3 uses:
- Layer A text search: ViFashionCLIP, 512-dim Qdrant collection.
- Layer A image search: FashionCLIP image vectors, 512-dim Qdrant collection.
- Layer B outfit rules: BGE-M3 through Ollama, 1024-dim Qdrant collections.
"""

from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STYLISTS_DIR = os.path.join(DATA_DIR, "stylists")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
# IMAGES_DIR = os.path.join(BASE_DIR, "images")
# PRODUCT_IMAGE_ROOT = IMAGES_DIR
IMAGES_DIR = "D:/KHÓA LUẬN/WORKSPACE/Amazon_Lazada_Fashion_Metadata_65k/images"
PRODUCT_IMAGE_ROOT = "D:/KHÓA LUẬN/WORKSPACE/Amazon_Lazada_Fashion_Metadata_65k/images"

METADATA_FILE = os.path.join(METADATA_DIR, "meta_Amazon_Lazada_Fashion_65k.jsonl")
LAYER_B_FEMALE_PATH = os.path.join(STYLISTS_DIR, "Layer_B_Female_Knowledge.json")
LAYER_B_MALE_PATH = os.path.join(STYLISTS_DIR, "Layer_B_Male_Knowledge.json")

VIFASHIONCLIP_CHECKPOINT = os.path.join(
    BASE_DIR,
    "Vietnamese",
    "vifashionclip_aiteamvn_embedding_v2_projection_336k",
    "stage2_last_layers",
    "best_stage2_model.pt",
)


# Ollama models
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
QWEN_VL_MODEL = VISION_MODEL


# ViFashionCLIP model configuration
TEACHER_MODEL_NAME = "patrickjohncyh/fashion-clip"
STUDENT_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
STUDENT_MAX_LENGTH = 256
PROJECTION_HIDDEN_DIM = 1024
PROJECTION_NUM_LAYERS = 3
PROJECTION_DROPOUT = 0.05
PRODUCT_EMBEDDING_BATCH_SIZE = 32
IMAGE_EMBEDDING_BATCH_SIZE = 32
PRODUCT_EMBEDDING_BACKEND = os.getenv("PRODUCT_EMBEDDING_BACKEND", "remote").lower()
# PRODUCT_EMBEDDING_BACKEND = "local"
VIFASHIONCLIP_SERVICE_URL = os.getenv("VIFASHIONCLIP_SERVICE_URL", "http://localhost:18080")
VIFASHIONCLIP_SERVICE_TIMEOUT = float(os.getenv("VIFASHIONCLIP_SERVICE_TIMEOUT", "120"))
REMOTE_EMBEDDING_FALLBACK_LOCAL = os.getenv("REMOTE_EMBEDDING_FALLBACK_LOCAL", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# LLM generation
LLM_TEMPERATURE = 0.4
LLM_TIMEOUT = 120
LLM_NUM_PREDICT = 1024
LLM_NUM_CTX = 4096  # Reduced from 8192 to cut prefill latency (~30% faster TTFT)

# Router diagnostics and multimodal policy
# Keep detailed traces available for notebooks/logs, but do not expose them to
# the web client unless explicitly enabled.
DEBUG_ROUTER_TRACE = os.getenv("DEBUG_ROUTER_TRACE", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
IMAGE_ROUTER_CONFIDENCE_THRESHOLD = float(
    os.getenv("IMAGE_ROUTER_CONFIDENCE_THRESHOLD", "0.55")
)


# Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_FASHION = "fashion_products_vifashionclip_vi_65k_structured_vi"
QDRANT_COLLECTION_PRODUCT_IMAGE = "fashion_products_fashionclip_image_main_65k"
QDRANT_COLLECTION_LAYER_B_F = "layer_b_female"
QDRANT_COLLECTION_LAYER_B_M = "layer_b_male"
PRODUCT_VECTOR_SIZE = 512
IMAGE_VECTOR_SIZE = 512
LAYER_B_VECTOR_SIZE = 1024


# Retrieval
PRODUCT_SEARCH_CANDIDATE_K = 15
PRODUCT_SEARCH_PAGE_SIZE = 5
PRODUCT_SEARCH_BRAND_LIMIT = 2
ENABLE_PRODUCT_RERANKER = os.getenv("ENABLE_PRODUCT_RERANKER", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 8
RERANKER_BATCH_SIZE = 8
RETRIEVAL_RETRY_COUNT = 2
RETRIEVAL_RETRY_SLEEP = 1.0
IMAGE_SEARCH_TOP_K = 12
IMAGE_SEARCH_MAX_PRODUCTS = 3
IMAGE_SEARCH_SCORE_THRESHOLD = 0.15
LAYER_B_SCORE_THRESHOLD = 0.50
LAYER_B_FALLBACK_SCORE_THRESHOLD = 0.40
LAYER_B_LIMIT = 1
OUTFIT_MAX_PRODUCT_SLOTS = int(os.getenv("OUTFIT_MAX_PRODUCT_SLOTS", "3"))


# Redis history. The demo keeps only recent turns by default so a request never
# waits for an extra LLM summarization round-trip.
REDIS_URL = "redis://localhost:6379"
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "6"))
HISTORY_RECENT_KEEP = int(os.getenv("HISTORY_RECENT_KEEP", "4"))
HISTORY_ENABLE_SUMMARIZATION = os.getenv("HISTORY_ENABLE_SUMMARIZATION", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARIZE_MAX_TOKENS = 150


# API
API_TITLE = "Fashion RAG Chatbot API"
API_VERSION = "3.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000


# Vision
VL_MAX_SIZE = 512
PRELOAD_IMAGE_ENCODER = os.getenv("PRELOAD_IMAGE_ENCODER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LAYER_B_DANG_NGUOI = [
    "Dáng quả lê",
    "Dáng quả táo",
    "Dáng đồng hồ cát",
    "Dáng chữ nhật",
    "Dáng cân đối",
    "Người thấp bé",
    "Người ngoại cỡ",
    "Người mảnh",
]
LAYER_B_TONE_DA = ["Da sáng", "Da trung bình", "Da ngăm", "Da ấm"]
LAYER_B_WILDCARD_DANG = "Mọi vóc dáng"
LAYER_B_WILDCARD_TONE = "Mọi tone da"


# Intent/router keywords
DEFINITE_GREETING = [
    "xin chào",
    "hello",
    "hi bạn",
    "chào bạn",
    "hey",
    "alo",
    "chào buổi sáng",
    "chào buổi chiều",
]
DEFINITE_CHITCHAT = [
    "cảm ơn",
    "cảm on",
    "thank you",
    "thanks",
    "tạm biệt",
    "bye",
    "hẹn gặp lại",
    "bái bai",
]
DEFINITE_OUTFIT = [
    "phối đồ",
    "mix match",
    "mặc với gì",
    "mặc cùng gì",
    "kết hợp với gì",
    "phối với gì",
    "outfit cho",
    "gợi ý outfit",
    "tư vấn phối",
    # Expanded to reduce LLM router fallback
    "nên mặc gì",
    "mặc gì",
    "mặc đồ gì",
    "chọn đồ",
    "ăn mặc thế nào",
    "mặc như thế nào",
    "diện gì",
    "trang phục gì",
    "mặc gì cho",
    "mặc gì đi",
]
DEFINITE_SEARCH = [
    "còn hàng không",
    "còn size",
    "giá bao nhiêu",
    "mã sp",
    "có bán không",
    "tìm giúp",
    "cho xem",
    "shop có",
    "xem thêm",
    "xem them",
    "cho xem thêm",
    "cho xem them",
]
FOLLOWUP_MORE_KEYWORDS = [
    "xem them",
    "cho xem them",
    "them san pham",
    "them lua chon",
    "san pham khac",
    "mau khac",
    "goi y them",
]
DEFINITE_PROFILE_INQUIRY = [
    "dang nguoi toi",
    "toi dang nguoi gi",
    "toi thuoc dang",
    "body type cua toi",
    "tone da cua toi",
    "mau da cua toi",
    "profile cua toi",
    "thong tin cua toi",
    "ban da luu gi ve toi",
    "toi da gui thong tin gi",
]
STRICT_OUT_OF_SCOPE_PATTERNS = [
    r"^\s*\d+\s*[+\-*/]\s*\d+\s*=*\s*\??\s*$",
    r"\b(thoi tiet hom nay|du bao thoi tiet|gia bitcoin|bong da|lap trinh python)\b",
]
MALE_KEYWORDS = ["nam", "con trai", "bạn trai", "chàng", "đàn ông", "bố", "chồng"]


# Security and logging
MAX_QUERY_CHARS = 500
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"bo\s+qua\s+(moi\s+)?(huong\s+dan|lenh|prompt)",
    r"bỏ\s+qua\s+(mọi\s+)?(hướng\s+dẫn|lệnh|prompt)",
    r"quen\s+(het|cac)\s+(huong\s+dan|lenh)",
    r"quên\s+(hết|các)\s+(hướng\s+dẫn|lệnh)",
    r"dong\s+vai",
    r"đóng\s+vai",
    r"system\s+prompt",
    r"developer\s+message",
    r"tiet\s+lo\s+(prompt|huong\s+dan|lenh)",
    r"tiết\s+lộ\s+(prompt|hướng\s+dẫn|lệnh)",
]
CHATBOT_LOG_DIR = os.path.join(BASE_DIR, "logs")
CHAT_TURN_LOG_FILE = os.path.join(CHATBOT_LOG_DIR, "chat_turns_research_demo_v3.jsonl")
HALLUCINATION_LOG_FILE = os.path.join(
    CHATBOT_LOG_DIR,
    "hallucination_warnings_v3.jsonl",
)


# Category mapping from Layer B to Layer A shelves.
CATEGORY_MAPPING = {
    "Áo mặc trong (áo thun/sơ mi)": ["Áo"],
    "Áo khoác ngoài": ["Áo khoác"],
    "Áo khoác nhẹ/Áo len": ["Áo khoác"],
    "Quần/Chân váy": ["Quần", "Chân váy"],
    "Đầm/Jumpsuit": ["Đầm", "Jumpsuit"],
    "Giày dép": ["Giày"],
    "Túi xách": ["Túi xách"],
    "Phụ kiện": None,
}

PHU_KIEN_KEYWORD_ROUTER = {
    "Mũ": ["beret", "hat", "cap", "beanie", "fedora", "bucket", "brim", "flat cap"],
    "Găng tay": ["gloves", "glove", "arm warmer"],
    "Kính mắt": ["glasses", "sunglasses", "sunglass"],
    "Đồng hồ": ["watch"],
    "Dây chuyền": ["necklace", "chain pendant", "chain"],
    "Bông tai": ["earring", "earrings"],
    "Vòng tay": ["bracelet"],
    "Nhẫn": ["ring"],
    "Ghim cài áo": ["brooch", "pin", "badge"],
    "Phụ kiện hỗ trợ": ["socks", "sock", "scarf", "tie", "belt", "bandana", "headband"],
}
