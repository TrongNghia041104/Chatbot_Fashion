"""Input validation, grounding checks, and lightweight chat logging."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from langchain_core.documents import Document

from app.config import (
    CHAT_TURN_LOG_FILE,
    CHATBOT_LOG_DIR,
    HALLUCINATION_LOG_FILE,
    MAX_QUERY_CHARS,
    PROMPT_INJECTION_PATTERNS,
)


os.makedirs(CHATBOT_LOG_DIR, exist_ok=True)

PRODUCT_ID_EXTRACT_PATTERNS = [
    r"(?:Mã SP|Ma SP|MÃ_SP|MA_SP|product_id|Product ID)[:：\s]*([A-Za-z0-9_\-]{4,40})",
    r"\[MÃ_SP:\s*([A-Za-z0-9_\-]{4,40})\]",
    r"\[MA_SP:\s*([A-Za-z0-9_\-]{4,40})\]",
]

GENERATED_COMMERCE_FACT_PREFIXES = (
    "mã sp:",
    "ma sp:",
    "mã sản phẩm:",
    "ma san pham:",
    "product id:",
    "giá:",
    "gia:",
    "thương hiệu:",
    "thuong hieu:",
    "ảnh:",
    "anh:",
)


def validate_user_query(query: str) -> tuple[bool, str]:
    """Validate text length and common prompt-injection patterns."""
    clean_query = (query or "").strip()
    if len(clean_query) > MAX_QUERY_CHARS:
        return False, f"Tin nhắn hơi dài rồi ạ. Bạn rút gọn dưới {MAX_QUERY_CHARS} ký tự giúp mình nhé."
    lowered = clean_query.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS):
        return False, "Mình chỉ hỗ trợ tư vấn thời trang và sản phẩm trong shop. Bạn gửi lại nhu cầu mua sắm cụ thể giúp mình nhé."
    return True, ""


def normalize_product_id(product_id: str) -> str:
    return str(product_id or "").strip().lower()


def extract_product_ids_from_docs(docs: list[Document]) -> set[str]:
    ids = set()
    for doc in docs or []:
        product_id = normalize_product_id(doc.metadata.get("product_id", ""))
        if product_id:
            ids.add(product_id)
    return ids


def extract_product_ids_from_text(text: str) -> set[str]:
    ids = set()
    for pattern in PRODUCT_ID_EXTRACT_PATTERNS:
        for match in re.findall(pattern, text or "", flags=re.IGNORECASE):
            product_id = normalize_product_id(match)
            if product_id and product_id not in {"ma_sp", "mã_sp"}:
                ids.add(product_id)
    return ids


def check_answer_grounding(answer: str, allowed_product_ids: set[str], query: str, route: str) -> dict:
    """Warn when the model mentions product IDs that were not in retrieved context."""
    mentioned = extract_product_ids_from_text(answer)
    allowed = {normalize_product_id(product_id) for product_id in allowed_product_ids if product_id}
    unknown = sorted(product_id for product_id in mentioned if product_id not in allowed)
    report = {
        "ok": len(unknown) == 0,
        "route": route,
        "query": query,
        "mentioned_product_ids": sorted(mentioned),
        "allowed_product_ids": sorted(allowed),
        "unknown_product_ids": unknown,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if unknown:
        with open(HALLUCINATION_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
        print(f"[WARN] Grounding check: unknown product IDs {unknown}")
    return report


class CommerceFactStreamFilter:
    """Remove generated commerce-fact lines before they reach the browser.

    Product cards are serialized directly from retrieved documents and are the
    single source of truth for ID, price, brand and images. The LLM may explain
    why an item fits, but it must not restate those fragile fields. Buffering is
    line-based so a label split across streaming chunks is still caught.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self.removed_lines: list[str] = []

    @staticmethod
    def _is_generated_fact_line(line: str) -> bool:
        normalized = line.strip().lower().lstrip("-*• ").strip()
        return normalized.startswith(GENERATED_COMMERCE_FACT_PREFIXES)

    def feed(self, chunk: str) -> str:
        """Accept a stream chunk and return only completed, safe lines."""
        self._buffer += str(chunk or "")
        safe_parts: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            complete_line = line + "\n"
            if self._is_generated_fact_line(line):
                self.removed_lines.append(line.strip())
            else:
                safe_parts.append(complete_line)
        return "".join(safe_parts)

    def finish(self) -> str:
        """Flush the final partial line after applying the same policy."""
        line, self._buffer = self._buffer, ""
        if not line:
            return ""
        if self._is_generated_fact_line(line):
            self.removed_lines.append(line.strip())
            return ""
        return line


def append_chat_turn_log(record: dict) -> None:
    record = dict(record)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(CHAT_TURN_LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
