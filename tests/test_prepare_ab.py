import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from minimax_h3.jobs import load_jobs
from tools.prepare_ab import prepare_jobs


class PrepareAttentionABTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image = self.root / "input.png"
        Image.new("RGB", (64, 64)).save(image)
        self.source = self.root / "scenes.jsonl"
        self.scenes = [
            {"image_path": str(image), "prompt": "A person waves at the camera.",
             "seconds": 5, "accelerated": True, "width": 576, "height": 768},
            {"image_path": str(image), "prompt": "A bird lands on a branch.",
             "seconds": 15, "steps": 25, "sampler": "res_multistep", "scheduler": "simple"},
        ]
        self.source.write_text("".join(json.dumps(scene) + "\n" for scene in self.scenes))
        self.destination = self.root / "experiment"

    def test_variants_have_identical_inputs_and_pairing(self):
        plan = prepare_jobs(self.source, self.destination, [42, 99])
        on = (self.destination / "sage_on/jobs.jsonl").read_bytes()
        off = (self.destination / "sage_off/jobs.jsonl").read_bytes()
        self.assertEqual(on, off)
        self.assertEqual(hashlib.sha256(on).hexdigest(), plan["jobs_sha256"])
        jobs = load_jobs(self.destination / "sage_on/jobs.jsonl")
        self.assertEqual([job["seed"] for job in jobs], [42, 99, 42, 99])
        self.assertEqual([pair["scene"] for pair in plan["pairs"]], [1, 1, 2, 2])
        self.assertEqual(plan["inputs"][0]["image_sha256"],
                         hashlib.sha256(Path(self.scenes[0]["image_path"]).read_bytes()).hexdigest())
        self.assertEqual((plan["inputs"][0]["width"], plan["inputs"][0]["height"]),
                         (576, 768))
        for index, job in enumerate(jobs):
            for key, value in self.scenes[index // 2].items():
                self.assertEqual(job[key], value)
        self.assertEqual(plan["variants"]["sage_on"]["environment"],
                         {"MINIMAX_USE_SAGE_ATTENTION": "1"})
        self.assertEqual(plan["variants"]["sage_off"]["environment"],
                         {"MINIMAX_USE_SAGE_ATTENTION": "0"})

    def test_preparation_is_reproducible_without_source_seeds(self):
        prepare_jobs(self.source, self.destination, [42])
        second = self.root / "second"
        prepare_jobs(self.source, second, [42])
        self.assertEqual((self.destination / "sage_on/jobs.jsonl").read_bytes(),
                         (second / "sage_on/jobs.jsonl").read_bytes())

    def test_existing_experiment_is_never_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_jobs(self.source, self.destination, [42])
        self.assertEqual(marker.read_text(), "keep")

    def test_invalid_seeds_do_not_create_output(self):
        for seeds in ([], [42, 42], [-1], [2 ** 64]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                prepare_jobs(self.source, self.destination, seeds)
            self.assertFalse(self.destination.exists())

    def test_invalid_input_does_not_create_output(self):
        self.source.write_text('{"prompt": "missing fields"}\n')
        with self.assertRaises(ValueError):
            prepare_jobs(self.source, self.destination, [42])
        self.assertFalse(self.destination.exists())

    def test_invalid_canvas_does_not_create_output(self):
        scene = dict(self.scenes[0], width=575)
        self.source.write_text(json.dumps(scene) + "\n")
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            prepare_jobs(self.source, self.destination, [42])
        self.assertFalse(self.destination.exists())
