"""
main.py — Entry point cho Fashion RAG Chatbot
==============================================
Chạy server:
    python main.py
    
Hoặc dùng uvicorn trực tiếp:
    uvicorn app.api:app --reload --port 8000
"""

import uvicorn
from app.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
