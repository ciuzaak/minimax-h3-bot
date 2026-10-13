from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import download_reference_turbo as download


class ReferenceDownloadTest(unittest.TestCase):
    def run_download(self, destination, expected):
        def snapshot(repo, **kwargs):
            self.assertEqual(repo, "Comfy-Org/MiniMax-H3")
            self.assertEqual(kwargs["allow_file_pattern"], [f"loras/{download.REFERENCE_TURBO_LORA_NAME}"])
            source = Path(kwargs["local_dir"]) / "loras" / download.REFERENCE_TURBO_LORA_NAME
            source.parent.mkdir()
            source.write_bytes(b"verified model")
        with patch.dict("sys.modules", {"modelscope": SimpleNamespace(snapshot_download=snapshot)}), \
             patch("sys.argv", ["download", "--loras-dir", str(destination)]), \
             patch.object(download, "EXPECTED_HASH", expected), patch("builtins.print"):
            download.main()

    def test_custom_directory_only_publishes_verified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "custom"
            self.run_download(destination, sha256(b"verified model").hexdigest())
            self.assertEqual((destination / download.REFERENCE_TURBO_LORA_NAME).read_bytes(), b"verified model")
            self.assertEqual(len(list(destination.iterdir())), 1)

    def test_failed_checksum_preserves_existing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            target = destination / download.REFERENCE_TURBO_LORA_NAME
            target.write_bytes(b"existing model")
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                self.run_download(destination, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing model")
            self.assertEqual(len(list(destination.iterdir())), 1)
