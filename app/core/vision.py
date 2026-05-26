"""
app/core/vision.py — Vision Functions (Qwen2.5-VL)
====================================================
Xử lý ảnh: phân loại ảnh, phân tích vóc dáng, caption sản phẩm.
Sử dụng model Qwen2.5-VL qua Ollama.
"""

import base64
import os

import ollama

from app.config import VISION_MODEL


def _call_vl(image_path: str, prompt: str) -> str:
    """
    Gọi Vision-Language model với ảnh và prompt.

    Args:
        image_path: Đường dẫn tuyệt đối đến file ảnh.
        prompt: Câu hỏi/hướng dẫn cho model.

    Returns:
        Chuỗi phản hồi từ model.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file ảnh.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
    )
    return resp["message"]["content"].strip()


def detect_image_type(image_path: str, user_query: str = "") -> str:
    """
    Xác định ảnh là người (person) hay sản phẩm (product).

    Returns:
        "person" hoặc "product"
    """
    prompt = (
        "Ảnh này chứa gì? Trả lời đúng 1 chữ: "
        "PERSON nếu là ảnh chụp người, PRODUCT nếu là ảnh sản phẩm thời trang. "
        "Chỉ trả lời PERSON hoặc PRODUCT."
    )
    result = _call_vl(image_path, prompt).upper()
    return "person" if "PERSON" in result else "product"


def analyze_person_image(image_path: str) -> dict:
    """
    Phân tích vóc dáng và tone da của người trong ảnh.

    Returns:
        dict với keys: dang_nguoi, tone_da, nhan_xet
    """
    prompt = """\
Bạn là chuyên gia tư vấn thời trang. Hãy phân tích người trong ảnh:
1. DÁNG NGƯỜI (chọn 1): Dáng chữ A | Dáng quả lê | Dáng táo | Dáng đồng hồ cát | Dáng chữ H | Dáng chữ V | Dáng thẳng
2. TONE DA (chọn 1): Da trắng | Da vàng | Da ngăm | Da tối
3. NHẬN XÉT: 1-2 câu về điểm nổi bật khi phối đồ.
Trả lời theo format:
DÁNG: [tên dáng]
TONE: [tên tone]
NHẬN XÉT: [nội dung]\
"""
    raw = _call_vl(image_path, prompt)
    profile: dict = {"dang_nguoi": None, "tone_da": None, "nhan_xet": ""}

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
    """
    Tạo mô tả (caption) cho ảnh sản phẩm thời trang bằng tiếng Việt.

    Returns:
        Chuỗi mô tả sản phẩm.
    """
    prompt = (
        "Mô tả sản phẩm thời trang trong ảnh bằng tiếng Việt. "
        "Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu, phong cách. Ngắn gọn 1-2 câu."
    )
    return _call_vl(image_path, prompt)


print("[OK] Vision functions sẵn sàng!")
