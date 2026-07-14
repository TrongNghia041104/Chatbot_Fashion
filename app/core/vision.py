"""Vision helpers for uploaded person/product images."""

from __future__ import annotations

import base64
import io
import os

import ollama
from PIL import Image

from app.config import (
    LAYER_B_DANG_NGUOI,
    LAYER_B_TONE_DA,
    QWEN_VL_MODEL,
    VL_MAX_SIZE,
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
        response = ollama.chat(
            model=QWEN_VL_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        print(f"[VISION ERROR] {exc}")
        return ""


def detect_image_type(image_path: str, user_query: str = "") -> str:
    """Return product or person, using both image content and the user's request."""
    prompt = (
        f'Ảnh này chứa người hay chứa sản phẩm?\n'
        f'Câu hỏi của người dùng: "{user_query}"\n\n'
        f"Dựa vào câu hỏi và ảnh, xác định MỤC ĐÍCH của người dùng:\n"
        f"1. Muốn tìm/hỏi về quần áo/món đồ trong ảnh (dù có người mẫu mặc) -> PRODUCT\n"
        f"2. Muốn phân tích vóc dáng/tone da của người trong ảnh -> PERSON\n"
        f"3. Không có câu hỏi + ảnh chủ yếu là người -> PERSON\n"
        f"4. Không có câu hỏi + ảnh chụp sát sản phẩm -> PRODUCT\n\n"
        f"Chỉ trả lời đúng 1 chữ: PERSON hoặc PRODUCT."
    )
    result = _call_vl(image_path, prompt).upper()
    return "product" if "PRODUCT" in result else "person"


def analyze_person_image(image_path: str) -> dict:
    """Analyze body shape and skin tone using labels that match Layer B filters."""
    dang_list = " | ".join(LAYER_B_DANG_NGUOI)
    tone_list = " | ".join(LAYER_B_TONE_DA)
    prompt = (
        f"Bạn là chuyên gia tư vấn thời trang. Phân tích người trong ảnh:\n\n"
        f"1. DÁNG NGƯỜI (chọn đúng 1 trong danh sách): {dang_list}\n"
        f"2. TONE DA (chọn đúng 1 trong danh sách): {tone_list}\n"
        f"3. NHẬN XÉT: 1-2 câu về điểm nổi bật có thể khai thác khi phối đồ.\n\n"
        f"Trả lời theo đúng format (không thêm gì khác):\n"
        f"DÁNG: [tên dáng]\n"
        f"TONE: [tên tone]\n"
        f"NHẬN XÉT: [nội dung]"
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
        f"Mô tả chi tiết MÓN ĐỒ THỜI TRANG mà người dùng đang quan tâm trong ảnh bằng tiếng Việt.\n"
        f"Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu (nếu nhận ra), phong cách.\n"
        f"Ngắn gọn 1-2 câu. TUYỆT ĐỐI KHÔNG mô tả người mẫu."
    )
    return _call_vl(image_path, prompt)
