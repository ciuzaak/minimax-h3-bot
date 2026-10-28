import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from minimax_h3 import jobs


class WorkflowRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image = Path(self.temporary.name) / "input.png"
        Image.new("RGB", (768, 1024)).save(self.image)
        self.job = {"image_path": str(self.image), "prompt": "A person waves. Audio: room ambience.",
                    "seconds": 5, "seed": 42, "steps": 25}

    def build(self, job):
        # Building the graph must not stage test images into a real ComfyUI install.
        with patch.object(jobs, "stage_image", return_value="test.png"):
            return jobs.build_prompt(job, 1)

    def test_standard_keeps_native_sampling_and_joint_audio_video(self):
        graph, metadata = self.build(self.job)
        self.assertNotIn("16", graph)
        self.assertEqual(graph["8"]["inputs"]["model"], ["2", 0])
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph["10"]["inputs"]["steps"], 25)
        self.assertEqual(graph["10"]["inputs"]["scheduler"], "simple")
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["frames"] % 17, 5)
        self.assertEqual(graph["12"]["inputs"]["samples"], ["11", 0])
        self.assertEqual(graph["13"]["inputs"]["samples"], ["11", 0])
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertTrue(graph["6"]["inputs"]["prompt"].endswith(jobs.NO_MUSIC_FIELD))

    def test_turbo_uses_matching_loader_sampler_and_fixed_schedule(self):
        graph, metadata = self.build(dict(self.job, accelerated=True, steps=50,
                                           sampler="euler", scheduler="normal"))
        self.assertEqual(graph["16"]["inputs"]["lora_name"], jobs.TURBO_LORA_NAME)
        self.assertEqual(graph["16"]["inputs"]["strength"], 1.0)
        self.assertEqual(graph["8"]["inputs"]["model"], ["16", 0])
        self.assertEqual(graph["9"]["class_type"], "MiniMaxH3TurboSampler")
        self.assertEqual(graph["10"]["inputs"]["model"], ["16", 0])
        self.assertEqual(graph["10"]["inputs"]["steps"], 8)
        self.assertEqual(graph["10"]["inputs"]["scheduler"], "simple")
        self.assertEqual(metadata["steps"], 8)
        self.assertTrue(metadata["accelerated"])

    def test_explicit_canvas_and_duration_match_metadata(self):
        graph, metadata = self.build(dict(self.job, width=768, height=1024, seconds=15))
        inputs = graph["6"]["inputs"]
        self.assertEqual((inputs["width"], inputs["height"]), (768, 1024))
        self.assertEqual(inputs["length"], metadata["frames"])
        self.assertEqual(metadata["actual_seconds"], inputs["length"] / 24)

    def test_auto_canvas_follows_exif_orientation_for_real_jpeg_input(self):
        source = Path(self.temporary.name) / "phone.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (1200, 800)).save(source, exif=exif)
        self.assertEqual(jobs.image_resolution(source), (576, 864))
        self.assertEqual(jobs.image_resolution(source, profile="native"), (768, 1152))
        self.assertEqual(jobs.image_resolution(source, 1024, 576), (1024, 576))
