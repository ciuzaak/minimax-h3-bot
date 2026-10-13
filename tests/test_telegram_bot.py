import unittest

from minimax_h3.telegram_bot import expand_dialogue_prompt, mode_keyboard, validate_angle_brackets
from minimax_h3.preferences import GenerationSettings


class TelegramBotDialogueTest(unittest.TestCase):
    def test_dialogue_uses_selected_h3_language_tag(self):
        for language in ("Chinese", "Japanese", "Korean", "English"):
            with self.subTest(language=language):
                prompt, changed = expand_dialogue_prompt("She says <hello>", language)
                self.assertEqual(prompt, f"She says <d>[{language}] hello</d>")
                self.assertTrue(changed)

    def test_none_preserves_prompt_exactly(self):
        prompt = "<d>[Chinese] 你好</d> then <d>[Japanese] こんにちは</d> and <unfinished"
        self.assertEqual(expand_dialogue_prompt(prompt, "none"), (prompt, False))

    def test_existing_h3_tags_are_preserved(self):
        prompt = (
            "<Picture 1> <Subject 2> <Video 3> <Audio 4> "
            "<d>[English] hello<scenetrans> again<cutoff></d> <new line>"
        )
        expanded, changed = expand_dialogue_prompt(prompt, "Korean")
        self.assertEqual(
            expanded,
            "<Picture 1> <Subject 2> <Video 3> <Audio 4> "
            "<d>[English] hello<scenetrans> again<cutoff></d> <d>[Korean] new line</d>",
        )
        self.assertTrue(changed)

    def test_angle_bracket_validation(self):
        self.assertTrue(validate_angle_brackets("<first> and <second>"))
        self.assertFalse(validate_angle_brackets("<unfinished"))

    def test_settings_keyboard_uses_compact_button_rows(self):
        rows = mode_keyboard(GenerationSettings(), 123).inline_keyboard
        self.assertEqual([button.text.lstrip("✓ ") for button in rows[1]], ["5s", "10s", "15s"])
        self.assertEqual([button.text.lstrip("✓ ") for button in rows[2]], ["🇨🇳", "🇯🇵", "🇰🇷", "🇬🇧", "🚫"])


if __name__ == "__main__":
    unittest.main()
