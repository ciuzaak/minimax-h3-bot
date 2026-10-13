from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from minimax_h3 import telegram_bot as bot
from minimax_h3.preferences import GenerationSettings, PreferenceStore


class UserSettingsHandlersTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = PreferenceStore(Path(temporary.name) / "settings.json", GenerationSettings())
        for name, value in (("PREFERENCES", self.store), ("TURBO_UNAVAILABLE", False),
                            ("ALLOWLIST", {1, 2}), ("PENDING_COUNT", {})):
            patcher = patch.object(bot, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def update(self, user=1, callback="settings:1:seconds:15"):
        return SimpleNamespace(
            effective_user=SimpleNamespace(id=user), effective_chat=SimpleNamespace(id=11),
            callback_query=SimpleNamespace(data=callback, answer=AsyncMock(), edit_message_text=AsyncMock()),
            message=SimpleNamespace(text="She says <hello>", reply_text=AsyncMock()),
        )

    async def test_own_panel_updates_only_its_user(self):
        update = self.update()
        await bot.on_mode(update, SimpleNamespace())
        self.assertEqual(self.store.get(1).seconds, 15)
        self.assertEqual(self.store.get(2).seconds, 5)
        keyboard = update.callback_query.edit_message_text.call_args.kwargs["reply_markup"]
        self.assertTrue(all(button.callback_data.startswith("settings:1:")
                            for row in keyboard.inline_keyboard for button in row))

    async def test_other_user_cannot_change_settings_through_a_shared_panel(self):
        update = self.update(user=2)
        await bot.on_mode(update, SimpleNamespace())
        self.assertEqual(self.store.get(1).seconds, 5)
        self.assertEqual(self.store.get(2).seconds, 5)
        update.callback_query.answer.assert_awaited_once_with("这不是你的设置面板，请使用 /mode。", show_alert=True)
        update.callback_query.edit_message_text.assert_not_awaited()

    async def test_legacy_panel_expires_without_changing_preferences(self):
        update = self.update(callback="settings:seconds:15")
        await bot.on_mode(update, SimpleNamespace())
        self.assertEqual(self.store.get(1).seconds, 5)
        self.assertTrue(update.callback_query.answer.call_args.kwargs["show_alert"])

    async def test_failed_fast_preflight_does_not_change_preferences(self):
        self.store.update(1, mode="standard")
        update = self.update(callback="settings:1:mode:fast")
        with patch.object(bot.submit, "check_turbo_support", return_value=["missing node"]):
            await bot.on_mode(update, SimpleNamespace())
        self.assertEqual(self.store.get(1).mode, "standard")

    async def test_prompt_freezes_user_mode_duration_and_language(self):
        self.store.update(1, mode="standard", seconds=10, dialogue_language="Japanese")
        state = {"chat_id": 11, "user_id": 1, "image_path": "unused", "width": 576, "height": 768}
        with patch.object(bot, "ACTIVE_IMAGES", {(11, 1): state}), \
             patch.object(bot, "acquire_image"), \
             patch.object(bot, "send_text_with_retry", new_callable=AsyncMock):
            await bot.on_prompt_text(self.update(), SimpleNamespace(bot=Mock()))
        self.store.update(1, mode="fast", seconds=15, dialogue_language="English")
        pending = next(iter(bot.PENDING_COUNT.values()))
        self.assertFalse(pending["accelerated"])
        self.assertEqual(pending["seconds"], 10)
        self.assertIn("<d>[Japanese] hello</d>", pending["prompt"])
        self.assertNotIn("prompt", state)

    def test_runtime_fallback_preserves_saved_fast_preference(self):
        with patch.object(bot, "TURBO_UNAVAILABLE", True):
            self.assertEqual(bot.generation_settings(1).mode, "standard")
        self.assertEqual(self.store.get(1).mode, "fast")
        self.assertEqual(bot.generation_settings(1).mode, "fast")
