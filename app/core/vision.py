"""Vision helpers for uploaded person/product images."""

from __future__ import annotations

import base64
import io
import json
import os
import re

import ollama
from PIL import Image

from app.config import (
    LAYER_B_DANG_NGUOI,
    LAYER_B_TONE_DA,
    OLLAMA_BASE_URL,
    QWEN_VL_MODEL,
    VL_MAX_SIZE,
)

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

PERSON_QUERY_HINTS = (
    "dáng người",
    "vóc dáng",
    "body",
    "tone da",
    "màu da",
    "phân tích người",
    "phân tích dáng",
    "tư vấn phong cách",
    "tư vấn phối đồ",
    "hợp màu",
    "hợp kiểu",
)

PRODUCT_QUERY_HINTS = (
    "tìm sản phẩm",
    "tìm đồ",
    "tìm áo",
    "tìm quần",
    "tìm váy",
    "mua",
    "giống ảnh",
    "giống hình",
    "sản phẩm tương tự",
    "món đồ",
    "chiếc áo",
    "chiếc quần",
    "chiếc váy",
)


def _preprocess_image(image_path: str) -> str:
    """Resize and JPEG-encode images before sending them to Qwen-VL."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > VL_MAX_SIZE:
        ratio = VL_MAX_SIZE / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def _call_vl(image_path: str, prompt: str) -> str:
    """Call Qwen-VL through Ollama and return an empty string on model errors."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    img_b64 = _preprocess_image(image_path)
    try:
        response = ollama_client.chat(
            model=QWEN_VL_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        print(f"[VISION ERROR] {exc}")
        return ""


def _query_has_any(query: str, hints: tuple[str, ...]) -> bool:
    query = query.casefold()
    return any(hint.casefold() in query for hint in hints)


def _parse_image_type(raw: str, default: str = "person") -> str:
    """Parse the final PRODUCT/PERSON label without being fooled by explanations."""
    text = str(raw or "").strip().upper()
    if not text:
        return default

    first_token = re.sub(r"[^A-Z]", "", text.split()[0]) if text.split() else ""
    if first_token == "PERSON":
        return "person"
    if first_token == "PRODUCT":
        return "product"

    labels = re.findall(r"\b(PERSON|PRODUCT)\b", text)
    if labels:
        return "person" if labels[-1] == "PERSON" else "product"

    return default


def _extract_json_object(raw: str) -> dict:
    """Read the first JSON object returned by the VLM, including fenced JSON."""
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def describe_image_for_routing(image_path: str, user_query: str = "") -> dict:
    """Describe an ambiguous upload before choosing a business intent.

    This function only performs visual understanding. It does *not* choose an
    execution route. The intent module combines this structured observation
    with the user's text and applies deterministic routing policy.
    """
    prompt = (
        "Bạn đang quan sát ảnh cho một trợ lý thời trang. Hãy mô tả khách quan, "
        "không tự quyết định người dùng muốn tìm sản phẩm hay phối đồ.\n\n"
        f'Câu hỏi đi kèm: "{user_query}"\n\n'
        "Trả về đúng một JSON object, không Markdown, gồm:\n"
        "{\n"
        '  "subject": "product|person|mixed|unclear",\n'
        '  "caption": "mô tả ngắn nội dung ảnh",\n'
        '  "fashion_item": "món thời trang nổi bật hoặc chuỗi rỗng",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "lý do ngắn cho subject"\n'
        "}\n"
        "Dùng product khi ảnh tập trung vào một món đồ; person khi chủ thể chính "
        "là người và không có món đồ riêng biệt rõ ràng; mixed khi có người mặc "
        "trang phục đủ rõ để vừa nhận diện người vừa nhận diện món đồ."
    )
    raw = _call_vl(image_path, prompt)
    data = _extract_json_object(raw)
    subject = str(data.get("subject", "unclear")).strip().lower()
    if subject not in {"product", "person", "mixed", "unclear"}:
        subject = "unclear"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "subject": subject,
        "caption": str(data.get("caption", "")).strip(),
        "fashion_item": str(data.get("fashion_item", "")).strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason", "")).strip(),
    }


def detect_image_type(image_path: str, user_query: str = "") -> str:
    """Return product or person, using both image content and the user's request."""
    if _query_has_any(user_query, PERSON_QUERY_HINTS):
        return "person"
    if _query_has_any(user_query, PRODUCT_QUERY_HINTS):
        return "product"

    prompt = (
        "Bạn là bộ phân loại ý định cho chatbot thời trang.\n"
        "Nhiệm vụ: dựa vào ảnh và câu hỏi, chọn đúng nhãn cuối cùng.\n\n"
        f'Câu hỏi của người dùng: "{user_query}"\n\n'
        "Chọn PERSON nếu người dùng muốn phân tích người trong ảnh: vóc dáng, tone da, "
        "đặc điểm cơ thể, tư vấn phong cách hoặc phối đồ theo người đó.\n"
        "Chọn PRODUCT nếu người dùng muốn tìm/mua/hỏi về quần áo, phụ kiện, giày dép "
        "hoặc món đồ trong ảnh.\n\n"
        "Nếu câu hỏi trống: ảnh chủ yếu là selfie/toàn thân/người -> PERSON; "
        "ảnh chụp sát một món đồ riêng lẻ -> PRODUCT.\n\n"
        "Trả lời đúng một dòng theo format:\n"
        "LABEL: PERSON\n"
        "hoặc\n"
        "LABEL: PRODUCT"
    )
    raw = _call_vl(image_path, prompt)
    return _parse_image_type(raw)


def analyze_person_image(image_path: str) -> dict:
    """Analyze body shape and skin tone using labels that match Layer B filters."""
    dang_list = " | ".join(LAYER_B_DANG_NGUOI)
    tone_list = " | ".join(LAYER_B_TONE_DA)
    prompt = (
        "Bạn là chuyên gia tư vấn thời trang. Phân tích người trong ảnh:\n\n"
        f"1. DÁNG NGƯỜI (chọn đúng 1 trong danh sách): {dang_list}\n"
        f"2. TONE DA (chọn đúng 1 trong danh sách): {tone_list}\n"
        "3. NHẬN XÉT: 1-2 câu về điểm nổi bật có thể khai thác khi phối đồ.\n\n"
        "Trả lời theo đúng format (không thêm gì khác):\n"
        "DÁNG: [tên dáng]\n"
        "TONE: [tên tone]\n"
        "NHẬN XÉT: [nội dung]"
    )
    raw = _call_vl(image_path, prompt)
    profile = {"dang_nguoi": None, "tone_da": None, "nhan_xet": ""}
    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if "DÁNG" in upper and ":" in line:
            value = line.split(":", 1)[1].strip()
            profile["dang_nguoi"] = value or None
        elif "TONE" in upper and ":" in line:
            value = line.split(":", 1)[1].strip()
            profile["tone_da"] = value or None
        elif "NHẬN XÉT" in upper and ":" in line:
            profile["nhan_xet"] = line.split(":", 1)[1].strip()
    return profile


def caption_product_image(image_path: str, user_query: str = "") -> str:
    """Caption a product image as a fallback query when vector image search fails."""
    prompt = (
        f'Câu hỏi của người dùng: "{user_query}"\n\n'
        "Mô tả chi tiết MÓN ĐỒ THỜI TRANG mà người dùng đang quan tâm trong ảnh bằng tiếng Việt.\n"
        "Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu (nếu nhận ra), phong cách.\n"
        "Ngắn gọn 1-2 câu. TUYỆT ĐỐI KHÔNG mô tả người mẫu."
    )
    return _call_vl(image_path, prompt)
