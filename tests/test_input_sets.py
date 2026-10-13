import json
from pathlib import Path
import tempfile
import unittest

from minimax_h3.input_sets import InputStore, collection_state, image_paths
from minimax_h3.preferences import GenerationSettings, PreferenceStore


class InputSetsTest(unittest.TestCase):
    def test_order_and_chat_ownership_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"1_{i}.jpg" for i in range(3)]
            for path in paths:
                path.write_bytes(b"image")
            images = [{"image_path": str(p), "width": 576, "height": 576} for p in paths]
            state = collection_state(-10, 1, images[::-1])
            store = InputStore(root / "manifest.json", root)
            store.save({(-10, 1): state})
            restored = store.load({1})
            self.assertEqual(image_paths(restored[(-10, 1)]), tuple(str(p) for p in paths[::-1]))
            self.assertEqual(store.load({2}), {})
            paths[0].unlink()
            self.assertEqual(store.load({1}), {})

    def test_manifest_cannot_restore_other_users_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "2_private.jpg"
            path.write_bytes(b"image")
            state = collection_state(1, 1, ({"image_path": str(path), "width": 576, "height": 576},))
            store = InputStore(root / "manifest.json", root)
            store.save({(1, 1): state})
            self.assertEqual(store.load({1}), {})
            self.assertTrue(path.exists())

    def test_clear_manifest_does_not_restore_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InputStore(Path(directory) / "manifest.json", directory)
            store.save({})
            self.assertEqual(store.load({1}), {})
            self.assertEqual(json.loads(store.path.read_text())["collections"], [])

    def test_input_mode_persists_without_changing_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"generation_mode":"fast","seconds":10}')
            store = PreferenceStore(path, GenerationSettings())
            self.assertEqual(store.get(1).input_mode, "i2v")
            store.update(1, input_mode="reference")
            restored = PreferenceStore(path, GenerationSettings())
            self.assertEqual(restored.get(1).input_mode, "reference")
            self.assertEqual(restored.get(2).input_mode, "i2v")
