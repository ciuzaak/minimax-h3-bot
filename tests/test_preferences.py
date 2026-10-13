import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minimax_h3.preferences import GenerationSettings, PreferenceStore


class PreferenceStoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bot_settings.json"
        self.default = GenerationSettings()

    def test_legacy_shared_settings_become_inherited_defaults(self):
        legacy = {"generation_mode": "standard", "seconds": 15, "dialogue_language": "Japanese"}
        self.path.write_text(json.dumps(legacy))
        store = PreferenceStore(self.path, self.default)
        inherited = GenerationSettings("standard", 15, "Japanese")
        self.assertEqual(store.get(1), inherited)
        self.assertEqual(store.get(2), inherited)
        self.assertEqual(json.loads(self.path.read_text()), legacy)
        store.update(1, seconds=5)
        restored = PreferenceStore(self.path, self.default)
        self.assertEqual(restored.get(1), GenerationSettings("standard", 5, "Japanese"))
        self.assertEqual(restored.get(2), inherited)
        self.assertEqual(restored.get(3), inherited)
        self.assertEqual(json.loads(self.path.read_text())["schema_version"], 2)

    def test_changes_are_isolated_and_survive_restart(self):
        store = PreferenceStore(self.path, self.default)
        store.update(1, seconds=10)
        store.update(2, dialogue_language="English")
        store.update(1, mode="standard")
        restored = PreferenceStore(self.path, self.default)
        self.assertEqual(restored.get(1), GenerationSettings("standard", 10, "Chinese"))
        self.assertEqual(restored.get(2), GenerationSettings("fast", 5, "English"))
        self.assertEqual(restored.get(3), self.default)

    def test_snapshot_does_not_change_with_later_preferences(self):
        store = PreferenceStore(self.path, self.default)
        before = store.get(1)
        store.update(1, seconds=15)
        self.assertEqual(before.seconds, 5)

    def test_corrupt_files_and_invalid_fields_fall_back_to_defaults(self):
        for content in ('invalid JSON', '[]', '{"seconds": "15"}', '{"seconds": 5.0}',
                        '{"generation_mode": "unknown", "dialogue_language": []}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                self.assertEqual(PreferenceStore(self.path, self.default).get(1), self.default)

    def test_failed_write_preserves_memory_disk_and_removes_temporary_file(self):
        store = PreferenceStore(self.path, self.default)
        store.update(1, seconds=10)
        original = self.path.read_bytes()
        with patch("minimax_h3.preferences.Path.replace", side_effect=OSError("test failure")):
            with self.assertRaises(OSError):
                store.update(1, seconds=15)
        self.assertEqual(store.get(1).seconds, 10)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [self.path.name])

    def test_invalid_updates_do_not_create_a_file(self):
        store = PreferenceStore(self.path, self.default)
        for changes in ({"seconds": 7}, {"mode": "unknown"}, {"dialogue_language": "unknown"}):
            with self.assertRaises(ValueError):
                store.update(1, **changes)
        self.assertFalse(self.path.exists())
