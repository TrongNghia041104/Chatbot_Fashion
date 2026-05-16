"""
Intent Detection: phân biệt user đang hỏi phối đồ hay tìm sản phẩm.
"""

OUTFIT_KEYWORDS = [
    "phối", "phối đồ", "mặc với", "kết hợp", "outfit", "mix",
    "mặc gì", "đi với", "hợp với", "bộ đồ", "mặc cùng", "style với",
]

MALE_KEYWORDS = ["nam", "con trai", "anh", "bạn trai", "chàng", "đàn ông"]


def detect_intent(query: str) -> str:
    """Trả về 'outfit' nếu user hỏi phối đồ, ngược lại trả về 'search'."""
    q = query.lower()
    return "outfit" if any(kw in q for kw in OUTFIT_KEYWORDS) else "search"


def detect_gender(query: str) -> str:
    """Trả về 'male' nếu câu hỏi có từ chỉ giới tính nam, mặc định 'female'."""
    return "male" if any(kw in query.lower() for kw in MALE_KEYWORDS) else "female"
