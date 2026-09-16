import unittest

from fashion_rag.core.intent import IntentDecision, INTENT_PROFILE_MANAGEMENT
from fashion_rag.application.chat.profile import apply_profile_decision


class ProfileStateTests(unittest.TestCase):
    def decision(self, action, entities=None):
        return IntentDecision(
            intent=INTENT_PROFILE_MANAGEMENT,
            action=action,
            route="profile_state_handler",
            entities=entities or {},
        )

    def test_candidate_is_saved_only_after_confirmation(self):
        state = {
            "profile": {"gender": "female"},
            "pending_profile_candidate": {
                "dang_nguoi": "Tam giác",
                "tone_da": "Ấm",
                "nhan_xet": "must not persist",
            },
        }
        _, profile = apply_profile_decision(self.decision("confirm_candidate"), state)
        self.assertEqual(profile["dang_nguoi"], "Tam giác")
        self.assertEqual(profile["tone_da"], "Ấm")
        self.assertNotIn("nhan_xet", profile)
        self.assertNotIn("pending_profile_candidate", state)

    def test_reject_does_not_change_profile(self):
        state = {"profile": {"gender": "male"}, "pending_profile_candidate": {"tone_da": "Lạnh"}}
        _, profile = apply_profile_decision(self.decision("reject_candidate"), state)
        self.assertEqual(profile, {"gender": "male"})
        self.assertNotIn("pending_profile_candidate", state)

    def test_update_delete_and_clear(self):
        state = {"profile": {"gender": "female"}}
        apply_profile_decision(
            self.decision("update", {"profile_updates": {"tone_da": "Ấm"}}),
            state,
        )
        self.assertEqual(state["profile"]["tone_da"], "Ấm")
        apply_profile_decision(
            self.decision("delete_field", {"profile_delete_fields": ["tone_da"]}),
            state,
        )
        self.assertNotIn("tone_da", state["profile"])
        apply_profile_decision(self.decision("clear_all"), state)
        self.assertEqual(state["profile"], {})


if __name__ == "__main__":
    unittest.main()
