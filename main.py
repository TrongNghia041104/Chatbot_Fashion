"""
main.py — Entry point cho Fashion RAG Chatbot
==============================================
Chạy server:
    python main.py

Hoặc dùng uvicorn trực tiếp (cần PYTHONPATH trỏ vào src/):
    PYTHONPATH=src uvicorn apps.api.api:app --reload --port 8000
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)
# uvicorn's --reload spawns a subprocess that only inherits os.environ, not sys.path.
os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn
from fashion_rag.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "apps.api.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
