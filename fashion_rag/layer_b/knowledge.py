"""
Nạp dữ liệu tri thức Layer B từ file JSON.
Layer B chứa các quy tắc phối đồ cho Nam và Nữ.
"""
import json

from fashion_rag.config.settings import LAYER_B_FEMALE_PATH, LAYER_B_MALE_PATH


def _load_json(file_path: str) -> list[dict]:
    """Đọc file JSON và trả về danh sách quy tắc."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Cache dữ liệu Layer B (singleton pattern)
_layer_b_cache: dict[str, list[dict]] = {}


def load_layer_b_knowledge() -> tuple[list[dict], list[dict]]:
    """
    Nạp 2 bộ tri thức Layer B (Nữ và Nam).

    Returns:
        Tuple (layer_b_female, layer_b_male)
    """
    if "female" not in _layer_b_cache:
        _layer_b_cache["female"] = _load_json(LAYER_B_FEMALE_PATH)
        _layer_b_cache["male"] = _load_json(LAYER_B_MALE_PATH)
        print(
            f"Đã nạp {len(_layer_b_cache['female'])} quy tắc Nữ "
            f"và {len(_layer_b_cache['male'])} quy tắc Nam."
        )

    return _layer_b_cache["female"], _layer_b_cache["male"]


def get_knowledge_by_gender(gender: str = "female") -> list[dict]:
    """Trả về bộ tri thức theo giới tính."""
    female, male = load_layer_b_knowledge()
    return female if gender == "female" else male
