import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tools.prepare_ab import prepare_jobs
from tools.run_attention_ab import attention_evidence, completed_variant, load_plan, owned_worker, require_idle_gpu


class AttentionPlanValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "input.png"
        Image.new("RGB", (64, 64)).save(self.image)
        source = self.root / "scenes.jsonl"
        source.write_text(json.dumps({"image_path": str(self.image), "prompt": "A bird lands.",
                                      "seconds": 5, "accelerated": True}) + "\n")
        self.output = self.root / "paired"
        prepare_jobs(source, self.output, [42])
        self.plan = self.output / "plan.json"

    def test_unchanged_plan_loads_paired_jobs(self):
        plan, variants = load_plan(self.plan)
        self.assertEqual(plan["job_count_per_variant"], 1)
        self.assertEqual(variants["sage_on"], variants["sage_off"])

    def test_task_tampering_is_rejected_before_worker_start(self):
        (self.output / "sage_off/jobs.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Task file changed"):
            load_plan(self.plan)

    def test_image_tampering_is_rejected(self):
        Image.new("RGB", (64, 64), "red").save(self.image)
        with self.assertRaisesRegex(ValueError, "Input image changed"):
            load_plan(self.plan)

    def test_worker_setting_tampering_is_rejected(self):
        plan = json.loads(self.plan.read_text())
        plan["variants"]["sage_off"]["environment"]["MINIMAX_USE_SAGE_ATTENTION"] = "1"
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Unexpected worker settings"):
            load_plan(self.plan)

    def test_resume_accepts_only_complete_matching_records(self):
        path = self.root / "results.jsonl"
        self.assertFalse(completed_variant(path, [{"seed": 42}]))
        output = self.root / "video.mp4"
        output.write_bytes(b"test")
        path.write_text(json.dumps({"index": 1, "seed": 42, "status": "completed",
                                     "outputs": [str(output)]}) + "\n")
        self.assertTrue(completed_variant(path, [{"seed": 42}]))
        for batch in ([{"seed": 99}], [{"seed": 42}, {"seed": 99}]):
            with self.assertRaisesRegex(ValueError, "partially completed"):
                completed_variant(path, batch)
        output.unlink()
        with self.assertRaisesRegex(ValueError, "outputs are missing"):
            completed_variant(path, [{"seed": 42}])


class OwnedWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.server = f"http://127.0.0.1:{probe.getsockname()[1]}"

    def worker(self, timeout=10):
        return owned_worker(["unused"], self.root, {}, self.root / "worker.log", self.server, timeout)

    def test_worker_is_terminated_on_exception(self):
        process = Mock()
        process.poll.return_value = None
        with patch("tools.run_attention_ab.subprocess.Popen", return_value=process), \
             patch("tools.run_attention_ab.jobs.request_json", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with self.worker():
                    raise RuntimeError("test failure")
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)
        process.kill.assert_not_called()

    def test_unresponsive_owned_worker_is_killed_after_grace_period(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("unused", 15), 0]
        with patch("tools.run_attention_ab.subprocess.Popen", return_value=process), \
             patch("tools.run_attention_ab.jobs.request_json", return_value={}):
            with self.worker():
                pass
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_startup_timeout_still_cleans_up_owned_process(self):
        process = Mock()
        process.poll.return_value = None
        with patch("tools.run_attention_ab.subprocess.Popen", return_value=process):
            with self.assertRaises(TimeoutError):
                with self.worker(timeout=0):
                    self.fail("Unready worker must never be yielded")
        process.terminate.assert_called_once()

    def test_occupied_port_never_starts_or_stops_another_process(self):
        port = int(self.server.rsplit(":", 1)[1])
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", port))
            with patch("tools.run_attention_ab.subprocess.Popen") as launch:
                with self.assertRaises(OSError):
                    with self.worker():
                        pass
            launch.assert_not_called()

    def test_busy_gpu_is_rejected(self):
        with patch("tools.run_attention_ab.subprocess.check_output", return_value="1234\n"):
            with self.assertRaisesRegex(RuntimeError, "existing compute processes"):
                require_idle_gpu(1)

    def test_backend_evidence_reports_fallback(self):
        log = self.root / "worker.log"
        log.write_text("Using sage attention\nError running sage attention: test, using pytorch attention instead.\n")
        evidence = attention_evidence(log)
        self.assertTrue(evidence["sage_selected"])
        self.assertFalse(evidence["pytorch_selected"])
        self.assertEqual(evidence["fallback_log_count"], 1)
