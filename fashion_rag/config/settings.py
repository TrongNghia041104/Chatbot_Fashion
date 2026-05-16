"""
Cấu hình trung tâm cho toàn bộ hệ thống Fashion RAG.
Tất cả các hằng số, đường dẫn, tham số mô hình được quản lý tại đây.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# ĐƯỜNG DẪN DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FASHION_METADATA_DIR = os.path.join(BASE_DIR, "Fashion_Metadata")
LAYER_B_FEMALE_PATH = os.path.join(BASE_DIR, "Layer_B_Female_Knowledge.json")
LAYER_B_MALE_PATH = os.path.join(BASE_DIR, "Layer_B_Male_Knowledge.json")

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"  # Chuyển thành 'cuda' nếu bạn có GPU
EMBEDDING_VECTOR_SIZE = 1024

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH QDRANT
# ═══════════════════════════════════════════════════════════════════
QDRANT_URL = "http://localhost:6333"
QDRANT_LOCAL_PATH = os.path.join(BASE_DIR, "qdrant_data")
QDRANT_COLLECTION_NAME = "fashion_products_bge_m3"
QDRANT_LAYER_B_FEMALE_COLLECTION = "layer_b_female"
QDRANT_LAYER_B_MALE_COLLECTION = "layer_b_male"

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH LLM (Ollama)
# ═══════════════════════════════════════════════════════════════════
LLM_MODEL = "qwen2.5:3b-instruct"
LLM_TEMPERATURE = 0.4
LLM_TIMEOUT = 120
LLM_NUM_PREDICT = 350
LLM_NUM_CTX = 8192

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH VISION MODEL (Qwen2.5-VL qua Ollama)
# ═══════════════════════════════════════════════════════════════════
QWEN_VL_MODEL = "qwen2.5vl:3b"

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH REDIS
# ═══════════════════════════════════════════════════════════════════
REDIS_URL = "redis://localhost:6379"
MAX_HISTORY_MESSAGES = 6

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH RETRIEVER
# ═══════════════════════════════════════════════════════════════════
RETRIEVER_SEARCH_TYPE = "similarity_score_threshold"
RETRIEVER_TOP_K = 5
RETRIEVER_SCORE_THRESHOLD = 0.7

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH INDEXING
# ═══════════════════════════════════════════════════════════════════
INDEXING_BATCH_SIZE = 128

# Layer B semantic search
LAYER_B_SCORE_THRESHOLD = 0.50
