"""Small, deterministic profile state transitions.

VLM observations are candidates, not facts. They live in
``pending_profile_candidate`` until the user explicitly confirms them.
"""

from __future__ import annotations

from app.core.intent import IntentDecision, get_profile_inquiry_response


PROFILE_FIELDS = {"gender", "dang_nguoi", "tone_da"}
PROFILE_FIELD_LABELS = {
    "gender": "giới tính",
    "dang_nguoi": "dáng người",
    "tone_da": "tone da",
}


def _describe_profile_values(values: dict) -> str:
    return ", ".join(
        f"{PROFILE_FIELD_LABELS.get(key, key)} là {value}"
        for key, value in values.items()
    )


def sanitize_profile_candidate(candidate: dict | None) -> dict:
    """Keep only supported, non-empty fields from a user/VLM candidate."""
    candidate = candidate or {}
    return {
        key: value
        for key, value in candidate.items()
        if key in PROFILE_FIELDS and value not in {None, ""}
    }


def apply_profile_decision(decision: IntentDecision, state: dict) -> tuple[str, dict]:
    """Apply one profile-management decision and return ``(reply, profile)``."""
    profile = dict(state.get("profile") or {})
    action = decision.action

    if action == "read":
        return get_profile_inquiry_response(profile), profile

    if action == "update":
        updates = sanitize_profile_candidate(decision.entities.get("profile_updates"))
        profile.update(updates)
        state["profile"] = profile
        changed = _describe_profile_values(updates)
        return f"Mình đã ghi nhớ {changed} để tư vấn phù hợp hơn nhé." if changed else "Mình chưa thấy thông tin mới để ghi nhớ.", profile

    if action == "delete_field":
        fields = [
            field
            for field in decision.entities.get("profile_delete_fields", [])
            if field in PROFILE_FIELDS
        ]
        for field in fields:
            profile.pop(field, None)
        state["profile"] = profile
        removed = ", ".join(PROFILE_FIELD_LABELS.get(field, field) for field in fields)
        return f"Mình đã bỏ thông tin {removed} như bạn yêu cầu." if removed else "Mình chưa rõ bạn muốn bỏ thông tin nào.", profile

    if action == "clear_all":
        state["profile"] = {}
        state.pop("pending_profile_candidate", None)
        return "Mình đã đặt lại toàn bộ thông tin cá nhân. Khi nào muốn, bạn có thể chia sẻ lại với mình nhé.", {}

    if action == "confirm_candidate":
        candidate = sanitize_profile_candidate(state.get("pending_profile_candidate"))
        profile.update(candidate)
        state["profile"] = profile
        state.pop("pending_profile_candidate", None)
        changed = _describe_profile_values(candidate)
        return f"Cảm ơn bạn. Mình đã ghi nhớ {changed} cho những lần tư vấn sau." if changed else "Mình không còn nhận xét nào đang chờ bạn xác nhận.", profile

    if action == "reject_candidate":
        state.pop("pending_profile_candidate", None)
        return "Được rồi, mình sẽ bỏ qua nhận xét vừa rồi và không ghi nhớ thông tin đó.", profile

    return "Mình chưa rõ bạn muốn thay đổi điều gì. Bạn nói lại giúp mình nhé.", profile
