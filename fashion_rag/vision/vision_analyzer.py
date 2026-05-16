"""
Xử lý ảnh đầu vào bằng Qwen2.5-VL qua Ollama.
Hỗ trợ:
  - Phân loại ảnh (person / product)
  - Phân tích dáng người + tone da
  - Caption sản phẩm thời trang
"""
import os
import base64
import ollama

from fashion_rag.config.settings import QWEN_VL_MODEL


def _call_vl(image_path, prompt):
    """Gọi Qwen2.5-VL qua Ollama với 1 ảnh và 1 prompt."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=QWEN_VL_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
    )
    return resp["message"]["content"].strip()


def detect_image_type(image_path):
    """Phân biệt ảnh người hay ảnh sản phẩm. Trả về 'person' hoặc 'product'."""
    prompt = (
        "Ảnh này chứa gì? Trả lời đúng 1 chữ: "
        "PERSON nếu là ảnh chụp người (có mặt, thân người, chân dung), "
        "PRODUCT nếu là ảnh sản phẩm thời trang (quần áo, giày, túi trên nền phẳng). "
        "Chỉ trả lời PERSON hoặc PRODUCT, không thêm gì khác."
    )
    result = _call_vl(image_path, prompt).upper()
    return "person" if "PERSON" in result else "product"


def analyze_person_image(image_path):
    """Phân tích dáng người và tone da. Trả về dict: {dang_nguoi, tone_da, nhan_xet}."""
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
        if line.startswith("DÁNG:"):
            profile["dang_nguoi"] = line.replace("DÁNG:", "").strip()
        elif line.startswith("TONE:"):
            profile["tone_da"] = line.replace("TONE:", "").strip()
        elif line.startswith("NHẬN XÉT:"):
            profile["nhan_xet"] = line.replace("NHẬN XÉT:", "").strip()
    return profile


def caption_product_image(image_path):
    """Mô tả sản phẩm thời trang trong ảnh bằng tiếng Việt."""
    prompt = """Mô tả sản phẩm thời trang trong ảnh bằng tiếng Việt.
Bao gồm: loại sản phẩm, màu sắc, kiểu dáng, chất liệu (nếu nhận ra), phong cách.
Ngắn gọn trong 1-2 câu. Chỉ mô tả sản phẩm, không mô tả người."""
    return _call_vl(image_path, prompt)
