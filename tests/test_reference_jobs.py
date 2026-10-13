import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minimax_h3 import jobs


class ReferenceJobsTest(unittest.TestCase):
    def graph(self, accelerated=False, paths=None):
        job = dict(task="reference", image_paths=paths or ["one.jpg", "two.jpg"],
                   prompt="<Picture 1> and <Picture 2>", seconds=5, seed=42,
                   width=576, height=576, accelerated=accelerated)
        with patch.object(jobs, "stage_image", side_effect=lambda path: "input/" + path):
            return jobs.build_prompt(job, 1)

    def test_all_references_are_connected_in_order(self):
        graph, meta = self.graph(paths=[f"{i}.jpg" for i in range(9)])
        self.assertEqual(meta["reference_count"], 9)
        self.assertEqual(meta["task"], "reference")
        self.assertEqual(graph["2"]["inputs"]["unet_name"], jobs.REFERENCE_MODEL_NAME)
        self.assertNotIn("first_frame", graph["6"]["inputs"])
        for i in range(9):
            node = graph["6"]["inputs"][f"ref_images.ref_image_{i}"][0]
            self.assertEqual(graph[node]["inputs"]["image"], f"input/{i}.jpg")

    def test_reference_fast_uses_dedicated_lora_and_native_sampler(self):
        graph, meta = self.graph(accelerated=True)
        self.assertEqual(meta["steps"], 4)
        self.assertEqual(graph["16"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["16"]["inputs"]["lora_name"], jobs.REFERENCE_TURBO_LORA_NAME)
        self.assertEqual(graph["9"]["class_type"], "KSamplerSelect")
        self.assertNotIn("MiniMaxH3TurboLoRA", [node["class_type"] for node in graph.values()])
        self.assertEqual(graph["10"]["inputs"]["model"], ["16", 0])

    def test_legacy_single_image_fast_keeps_existing_model_and_sampler(self):
        job = dict(image_path="one.jpg", prompt="Move", seconds=5, seed=42,
                   width=576, height=576, accelerated=True)
        with patch.object(jobs, "stage_image", return_value="one.jpg"):
            graph, meta = jobs.build_prompt(job, 1)
        self.assertEqual(meta["steps"], 8)
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["16"]["inputs"]["lora_name"], jobs.TURBO_LORA_NAME)
        self.assertEqual(graph["9"]["class_type"], "MiniMaxH3TurboSampler")

    def test_invalid_reference_lists_never_stage_files(self):
        for paths in ([], "one.jpg", [None], ["one.jpg"] * 10):
            with self.subTest(paths=paths), patch.object(jobs, "stage_image") as stage:
                with self.assertRaises(ValueError):
                    jobs.build_prompt(dict(task="reference", image_paths=paths), 1)
                stage.assert_not_called()
        with self.assertRaises(ValueError):
            jobs.job_image_paths(dict(image_paths=["one.jpg", "two.jpg"]))

    def test_jsonl_resolves_every_reference_and_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / f"{i}.jpg" for i in range(2)]
            for image in images:
                image.write_bytes(b"input")
            row = dict(task="reference", image_paths=[str(p) for p in images], prompt="Scene", seconds=5)
            source = root / "jobs.jsonl"
            source.write_text(json.dumps(row) + "\n")
            loaded = jobs.load_jobs(source)[0]
            self.assertEqual(loaded["image_paths"], row["image_paths"])
            images[1].unlink()
            with self.assertRaisesRegex(ValueError, "image does not exist"):
                jobs.load_jobs(source)

    def test_preflight_requires_correct_reference_lora(self):
        def request(url, **kwargs):
            if "/object_info/" in url:
                return {url.rsplit("/", 1)[1]: {}}
            if url.endswith("diffusion_models"):
                return [jobs.REFERENCE_MODEL_NAME]
            return [jobs.TURBO_LORA_NAME]
        with patch.object(jobs, "request_json", side_effect=request):
            self.assertEqual(jobs.check_reference_support(["http://worker"]), [])
            issues = jobs.check_reference_support(["http://worker"], accelerated=True)
            self.assertTrue(any("4 步 LoRA" in issue for issue in issues))
