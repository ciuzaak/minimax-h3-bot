import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from minimax_h3 import image_jobs, jobs, pool_bot, telegram_bot as bot, worker_pools as pools


class FakeRuntime:
    def __init__(self, cards=(0, 1, 2)):
        self.cards = set(cards)
        self.alive = {}
        self.running = set()
        self.started = []
        self.stopped = []
        self.external = set()

    def free_gpus(self):
        return sorted(self.cards - set(self.alive) - self.external)

    def exists(self, worker):
        return worker["gpu"] in self.alive

    def ready(self, worker):
        return self.exists(worker)

    def busy(self, worker):
        return worker["gpu"] in self.running

    def launch(self, role, gpu):
        assert gpu in self.free_gpus(), "Cannot place two workers on the same GPU"
        worker = pools.new_worker(role, gpu)
        self.alive[gpu] = worker
        self.started.append((role, gpu))
        return worker

    def stop(self, worker):
        assert worker["gpu"] not in self.running, "Must never interrupt accepted jobs"
        self.stopped.append((worker["role"], worker["gpu"]))
        self.alive.pop(worker["gpu"])


class WorkerPoolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.patch = patch.object(pools, "STATE_PATH", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.runtime = FakeRuntime()

    def apply(self, video=None, image=None):
        pools.set_requested(video, image)
        pools.reconcile(self.runtime)
        return pools.reconcile(self.runtime)

    def test_initial_boot_allocates_no_cards_and_explicit_counts_are_persistent(self):
        state = pools.reconcile(self.runtime)
        self.assertEqual(state["requested"], dict(video=0, image=0))
        self.assertEqual(self.runtime.started, [])
        self.apply(1, 1)
        self.assertEqual(pools.read_state()["requested"], dict(video=1, image=1))
        self.assertEqual([w["role"] for w in pools.read_state()["workers"]], ["image", "video"])

    def test_scarcity_prioritizes_video_and_never_claims_external_gpus(self):
        self.runtime.external = {2}
        state = self.apply(2, 2)
        self.assertEqual(sorted(self.runtime.started), [("video", 0), ("video", 1)])
        self.assertEqual(len(state["workers"]), 2)
        self.assertEqual(pools.servers("image"), [])

    def test_expanding_image_pool_keeps_running_video_untouched(self):
        self.apply(1, 0)
        self.runtime.running = {0}
        self.apply(image=2)
        self.assertEqual(self.runtime.stopped, [])
        self.assertEqual(self.runtime.started, [("video", 0), ("image", 1), ("image", 2)])
        self.assertEqual(len(pools.servers("video")), 1)

    def test_video_reclaims_idle_image_before_busy_image(self):
        self.runtime = FakeRuntime((0, 1))
        self.apply(0, 2)
        self.runtime.running = {0}
        self.apply(video=1, image=1)
        self.assertEqual(self.runtime.stopped, [("image", 1)])
        self.assertEqual(self.runtime.alive[0]["role"], "image")
        self.assertEqual(self.runtime.alive[1]["role"], "video")

    def test_busy_image_drains_before_video_reuses_its_card(self):
        self.runtime = FakeRuntime((0, 1))
        self.apply(0, 2)
        self.runtime.running = {0, 1}
        state = self.apply(1, 1)
        self.assertEqual(self.runtime.stopped, [])
        self.assertEqual(len(pools.servers("image")), 1)
        self.assertEqual(len(pools.servers("image", include_draining=True)), 2)
        self.assertEqual([w["gpu"] for w in state["workers"] if w["status"] == "draining"], [1])
        self.runtime.running.remove(1)
        pools.reconcile(self.runtime)
        pools.reconcile(self.runtime)
        self.assertEqual(self.runtime.alive[0]["role"], "image")
        self.assertEqual(self.runtime.alive[1]["role"], "video")
        self.assertEqual(self.runtime.stopped, [("image", 1)])

    def test_zero_disables_new_jobs_and_releases_workers_after_drain(self):
        self.apply(1, 1)
        self.runtime.running = {0, 1}
        state = self.apply(0, 0)
        self.assertTrue(all(w["status"] == "draining" for w in state["workers"]))
        self.assertEqual(pools.servers("video") + pools.servers("image"), [])
        self.assertEqual(self.runtime.stopped, [])
        self.runtime.running.clear()
        state = pools.reconcile(self.runtime)
        self.assertEqual(state["workers"], [])

    def test_restarting_image_does_not_restart_or_drain_video(self):
        self.apply(1, 1)
        self.runtime.running = {0, 1}
        with patch.object(pools, "Runtime", return_value=self.runtime):
            pools.restart_pool("image")
        state = pools.read_state()
        self.assertEqual(next(w for w in state["workers"] if w["role"] == "video")["status"], "ready")
        self.assertEqual(next(w for w in state["workers"] if w["role"] == "image")["status"], "draining")
        self.runtime.running.remove(1)
        pools.reconcile(self.runtime)
        self.assertEqual(self.runtime.stopped, [("image", 1)])

    def test_bad_counts_and_stale_versions_do_not_overwrite_allocations(self):
        for count in (-1, 9, True, "1"):
            with self.assertRaises(ValueError):
                pools.set_requested(video=count)
        pools.set_requested(1, 0, expected_version=0)
        with self.assertRaises(ValueError):
            pools.set_requested(0, 1, expected_version=0)
        self.assertEqual(pools.read_state()["requested"], dict(video=1, image=0))

    def test_partial_launch_failure_keeps_ownership_journal_for_recovery(self):
        original = self.runtime.launch

        def partial(role, gpu):
            original(role, gpu)
            raise TimeoutError("tmux response timed out after spawn")

        pools.set_requested(1, 0)
        with patch.object(self.runtime, "launch", side_effect=partial):
            state = pools.reconcile(self.runtime)
        self.assertEqual(len(state["workers"]), 1)
        state = pools.reconcile(self.runtime)
        self.assertEqual(state["workers"][0]["status"], "ready")
        self.assertEqual(len(self.runtime.started), 1)

    def test_disabled_pools_never_fall_back_to_legacy_or_each_other(self):
        with patch.object(jobs, "MANAGED_POOLS", True), \
             patch.object(jobs, "SHARED_WORKER", True), \
             patch.object(image_jobs, "MANAGED_POOLS", True), \
             patch.object(jobs, "request_json") as request:
            for discover in (jobs.check_servers, image_jobs.check_servers):
                with self.assertRaises(jobs.NoWorkersAvailable):
                    discover()
            request.assert_not_called()

    def test_failed_image_start_does_not_back_off_video_allocation(self):
        original = self.runtime.launch

        def launch(role, gpu):
            if role == "image":
                raise RuntimeError("Missing image weights")
            return original(role, gpu)

        with patch.object(self.runtime, "launch", side_effect=launch):
            self.apply(0, 1)
            state = self.apply(video=1)
        self.assertEqual(len(pools.servers("video")), 1)
        self.assertEqual(pools.servers("image"), [])
        self.assertEqual(state["requested"], dict(video=1, image=1))

    def test_api_failure_treats_a_ready_worker_as_busy(self):
        with patch.object(pools.Runtime, "request", side_effect=OSError("disconnected")):
            self.assertTrue(pools.Runtime().busy(dict(status="ready")))

    def test_submissions_use_only_their_pool_and_images_choose_the_shorter_queue(self):
        self.apply(1, 2)
        video_server = pools.servers("video")[0]
        image_servers = pools.servers("image")
        submissions = []

        def request(url, data=None, **kwargs):
            if url.endswith("/system_stats"):
                return {}
            if url.endswith("/queue"):
                return {"queue_running": [], "queue_pending": [[1]] * (3 if url.startswith(image_servers[0]) else 0)}
            if "/object_info/" in url:
                return {url.rsplit("/", 1)[1]: {}}
            if "/models/" in url:
                return list(image_jobs.QUANTS.values()) + [image_jobs.CLIP, image_jobs.VAE]
            if url.endswith("/prompt"):
                submissions.append(url)
                return {"prompt_id": "test"}
            raise AssertionError(url)

        with patch.object(jobs, "MANAGED_POOLS", True), \
             patch.object(image_jobs, "MANAGED_POOLS", True), \
             patch.object(jobs, "stage_image", return_value="fixture.png"), \
             patch.object(jobs, "request_json", side_effect=request):
            video = jobs.submit_job(dict(image_path="fixture.png", prompt="A red mug", width=576,
                                         height=576, seconds=5, seed=1), 1)
            image = image_jobs.submit_job(dict(prompt="A blue mug", seed=2))
        self.assertEqual(video["server"], video_server)
        self.assertEqual(image["server"], image_servers[1])
        self.assertEqual(submissions, [video_server + "/prompt", image_servers[1] + "/prompt"])


class PoolBotTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.multiple(pools, STATE_PATH=Path(self.temp.name) / "state.json")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.allowed = patch.object(bot, "ALLOWLIST", {1, 2})
        self.allowed.start()
        self.addCleanup(self.allowed.stop)
        pool_bot.PANELS.clear()
        self.update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=11),
                                      message=SimpleNamespace(reply_text=AsyncMock()))

    async def click(self, token, action, count=0, owner=1):
        self.update.callback_query = SimpleNamespace(data=f"pools:{owner}:{token}:{action}:{count}",
                                                    answer=AsyncMock(), edit_message_text=AsyncMock())
        await pool_bot.on_gpu(self.update, None)

    async def test_draft_requires_apply_and_both_counts_include_zero(self):
        await pool_bot.gpu_command(self.update, None)
        token = next(iter(pool_bot.PANELS))
        await self.click(token, "video", 2)
        await self.click(token, "image", 0)
        self.assertEqual(pools.read_state()["requested"], dict(video=0, image=0))
        with patch.object(pools, "reconcile", side_effect=pools.read_state):
            await self.click(token, "apply")
        self.assertEqual(pools.read_state()["requested"], dict(video=2, image=0))
        keyboard = self.update.callback_query.edit_message_text.call_args.kwargs["reply_markup"]
        buttons = [b.callback_data for row in keyboard.inline_keyboard for b in row]
        self.assertIn(f"pools:1:{token}:video:0", buttons)
        self.assertIn(f"pools:1:{token}:image:0", buttons)

    async def test_foreign_owner_and_stale_global_settings_are_rejected(self):
        await pool_bot.gpu_command(self.update, None)
        token = next(iter(pool_bot.PANELS))
        self.update.effective_user.id = 2
        await self.click(token, "video", 8)
        self.assertEqual(pool_bot.PANELS[token]["draft"]["video"], 0)
        self.update.effective_user.id = 1
        pools.set_requested(1, 1)
        await self.click(token, "apply")
        self.assertEqual(pools.read_state()["requested"], dict(video=1, image=1))
        self.assertTrue(self.update.callback_query.answer.call_args.kwargs["show_alert"])

    async def test_stop_current_image_pool_preserves_video_request(self):
        pools.set_requested(2, 1)
        with patch.object(bot, "MANAGED_POOLS", True), \
             patch.object(bot, "generation_settings", return_value=SimpleNamespace(input_mode="image")), \
             patch.object(pools, "reconcile", side_effect=pools.read_state):
            await bot.service_stop_command(self.update, None)
        self.assertEqual(pools.read_state()["requested"], dict(video=2, image=0))
