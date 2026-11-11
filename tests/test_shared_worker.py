"""Image and video must share one queue and every control must preserve one GPU."""

import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from minimax_h3 import image_jobs, jobs, telegram_bot as bot


class SharedWorkerTest(unittest.TestCase):
    def test_discovery_never_falls_back_to_additional_video_workers(self):
        with patch.object(jobs, "SHARED_WORKER", True), \
             patch.object(jobs, "request_json", return_value={}) as request:
            self.assertEqual(jobs.check_servers(), [image_jobs.SERVER])
            request.assert_called_once_with(image_jobs.SERVER + "/system_stats", timeout=0.5)
            request.reset_mock(side_effect=True)
            request.side_effect = OSError("offline")
            with self.assertRaises(jobs.NoWorkersAvailable):
                jobs.check_servers()
            self.assertEqual(request.call_count, 1)

    def test_image_and_video_graphs_are_submitted_to_same_queue(self):
        requests = []

        def request(url, data=None, **kwargs):
            requests.append((url, data))
            return {"prompt_id": "test"}

        with patch.object(jobs, "SERVERS", [image_jobs.SERVER]), \
             patch.object(jobs, "select_server", return_value=image_jobs.SERVER), \
             patch.object(jobs, "stage_image", return_value="reference.png"), \
             patch.object(jobs, "request_json", side_effect=request), \
             patch.object(image_jobs, "check_support", return_value=[]):
            video = jobs.submit_job(dict(image_path="reference.png", prompt="A red mug", seed=1,
                                         seconds=5, width=576, height=576), 1)
            image = image_jobs.submit_job(dict(prompt="A red mug", seed=2))
        self.assertEqual(video["server"], image["server"])
        self.assertEqual([url for url, _ in requests], [image_jobs.SERVER + "/prompt"] * 2)
        self.assertEqual(requests[0][1]["prompt"]["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(requests[1][1]["prompt"]["6"]["class_type"], "MiniMaxH3ImageConditioning")

    def test_cli_refuses_multiple_gpus_before_starting_anything(self):
        script = Path(__file__).resolve().parents[1] / "bin/minimax-h3-service"
        env = dict(os.environ, MINIMAX_ENV_FILE="/dev/null", MINIMAX_SHARED_WORKER="1")
        for action in ("start", "restart", "workers-start", "workers-restart"):
            with self.subTest(action=action):
                result = subprocess.run([str(script), action, "2"], env=env,
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn("exactly one GPU", result.stderr)

    def test_shared_launcher_loads_video_custom_nodes_and_loras_from_original_base(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("image", "video/custom_nodes/ComfyUI-MiniMax-H3-Turbo", "bin"):
                (root / name).mkdir(parents=True)
            (root / "image/main.py").touch()
            (root / "video/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py").touch()
            (root / "bin/tmux").write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if sys.argv[1] in ('has-session', 'list-windows'):
    sys.exit(1)
if sys.argv[1] == 'new-session':
    Path(os.environ['LAUNCH_REPORT']).write_text(sys.argv[-1])
''')
            (root / "bin/nvidia-smi").write_text("#!/bin/sh\nprintf '1\\n'\n")
            (root / "bin/curl").write_text("#!/bin/sh\nexit 0\n")
            for file in (root / "bin").iterdir():
                file.chmod(0o700)
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       MINIMAX_ENV_FILE="/dev/null", MINIMAX_SHARED_WORKER="1", MINIMAX_IMAGE_GPU="0",
                       COMFY_DIR=str(root / "video"), MINIMAX_IMAGE_COMFY_DIR=str(root / "image"),
                       MINIMAX_IMAGE_PYTHON=os.sys.executable, LAUNCH_REPORT=str(root / "launch"),
                       MINIMAX_MODEL_PATHS_CONFIG=str(root / "models.yaml"),
                       MINIMAX_LOG_DIR=str(root / "logs"), MINIMAX_OUTPUT_DIR=str(root / "output"))
            script = Path(__file__).resolve().parents[1] / "bin/minimax-h3-image-service"
            result = subprocess.run([str(script), "start"], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            launch = (root / "launch").read_text()
            self.assertIn("CUDA_VISIBLE_DEVICES=0", launch)
            self.assertIn("--base-directory " + str(root / "video"), launch)
            self.assertIn("--whitelist-custom-nodes ComfyUI-MiniMax-H3-Turbo", launch)

    def test_shared_start_rejects_legacy_workers_and_image_only_runtime(self):
        script = Path(__file__).resolve().parents[1] / "bin/minimax-h3-image-service"
        for legacy, expected in ((True, "Stop legacy video workers"), (False, "restart it to apply shared mode")):
            with self.subTest(legacy_video=legacy), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "tmux").write_text('''#!/bin/sh
if [ "$1" = list-windows ]; then
    [ "$LEGACY_WORKER" = 0 ] || printf 'gpu0\\n'
fi
exit 0
''')
                (root / "curl").write_text("#!/bin/sh\nprintf '{}\\n'\n")
                for file in root.iterdir():
                    file.chmod(0o700)
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                           MINIMAX_ENV_FILE="/dev/null", MINIMAX_SHARED_WORKER="1",
                           MINIMAX_IMAGE_PYTHON=os.sys.executable, LEGACY_WORKER="1" if legacy else "0")
                result = subprocess.run([str(script), "start"], env=env, capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)


class SharedBotTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_modes_control_the_shared_worker_without_gpu_picker(self):
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1))
        with patch.object(bot, "SHARED_WORKER", True), \
             patch.object(bot, "image_worker_control", new_callable=AsyncMock) as control, \
             patch.object(bot, "request_service_gpu_count", new_callable=AsyncMock) as picker:
            await bot.service_start_command(update, None)
            await bot.service_restart_command(update, None)
            await bot.service_stop_command(update, None)
            self.assertEqual([call.args[1] for call in control.await_args_list], ["start", "restart", "stop"])
            picker.assert_not_awaited()

    async def test_stale_multigpu_panel_cannot_launch_additional_workers(self):
        query = SimpleNamespace(answer=AsyncMock())
        with patch.object(bot, "SHARED_WORKER", True), \
             patch.object(bot, "run_worker_control", new_callable=AsyncMock) as control:
            await bot.on_service_gpu_count(SimpleNamespace(callback_query=query), None)
            query.answer.assert_awaited_once()
            control.assert_not_awaited()

    def test_progress_counts_shared_queue_once(self):
        def request(url, **kwargs):
            return {"queue_running": [[0]], "queue_pending": [[1], [2]]} if url.endswith("/queue") else {}

        with patch.object(bot, "SHARED_WORKER", True), \
             patch.object(jobs, "check_servers", return_value=[image_jobs.SERVER]), \
             patch.object(jobs, "request_json", side_effect=request), \
             patch.object(bot, "sampler_progress", return_value=25):
            snapshot = bot.service_progress_snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual((snapshot[0]["running"], snapshot[0]["pending"]), (1, 2))
        self.assertEqual(snapshot[0]["gpu"], "共享单卡")
