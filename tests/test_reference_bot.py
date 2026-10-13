import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from minimax_h3 import telegram_bot as bot
from minimax_h3.input_sets import InputStore, collection_state, image_paths
from minimax_h3.preferences import GenerationSettings, PreferenceStore


class ReferenceBotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.preferences = PreferenceStore(self.root / "settings.json", GenerationSettings())
        self.preferences.update(1, input_mode="reference", mode="standard")
        self.tasks = []
        def create_task(coroutine):
            task = asyncio.create_task(coroutine)
            self.tasks.append(task)
            return task
        self.context = SimpleNamespace(args=[], bot=SimpleNamespace(send_photo=AsyncMock(),
            send_media_group=AsyncMock(), send_message=AsyncMock()),
            application=SimpleNamespace(create_task=create_task))
        for name, value in (("ACTIVE_IMAGES", {}), ("UPLOAD_BATCHES", {}), ("INPUT_EPOCHS", {}),
                            ("PENDING_COUNT", {}), ("IMAGE_USE_COUNT", {}), ("RETIRED_IMAGES", set()),
                            ("ALLOWLIST", {1, 2}), ("PREFERENCES", self.preferences),
                            ("BOT_INPUT_DIR", self.root), ("ALBUM_SETTLE_SECONDS", 0.001),
                            ("INPUT_STORE", InputStore(self.root / "inputs.json", self.root)),
                            ("REFERENCE_TURBO_UNAVAILABLE", False), ("TURBO_UNAVAILABLE", False)):
            p = patch.object(bot, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(bot.submit, "discard_staged_image")
        p.start()
        self.addCleanup(p.stop)

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def update(self, message_id=1, delay=0, album="album", user=1):
        async def download(path):
            await asyncio.sleep(delay)
            Path(path).write_bytes(str(message_id).encode())
        file = SimpleNamespace(download_to_drive=AsyncMock(side_effect=download))
        photo = SimpleNamespace(width=512, height=512, get_file=AsyncMock(return_value=file))
        return SimpleNamespace(effective_user=SimpleNamespace(id=user), effective_chat=SimpleNamespace(id=11),
            message=SimpleNamespace(message_id=message_id, media_group_id=album, photo=[photo],
                                    delete=AsyncMock(), reply_text=AsyncMock(), text="Scene"))

    async def upload(self, updates):
        await asyncio.gather(*(bot.on_photo(update, self.context) for update in updates))
        await asyncio.gather(*self.tasks)

    def state(self, count):
        images = []
        for i in range(count):
            path = self.root / f"1_old{i}.jpg"
            path.write_bytes(b"old")
            images.append(dict(image_path=str(path), width=576, height=576))
        return collection_state(11, 1, images)

    async def test_album_order_survives_out_of_order_download_completion(self):
        first, second = self.update(10, delay=0.04), self.update(20)
        await self.upload([second, first])
        state = bot.ACTIVE_IMAGES[(11, 1)]
        self.assertEqual([Path(p).read_text() for p in image_paths(state)], ["10", "20"])
        self.context.bot.send_media_group.assert_awaited_once()
        media = self.context.bot.send_media_group.call_args.kwargs["media"]
        self.assertIn("<Picture 1>", media[0].caption)
        self.assertIn("<Picture 2>", media[1].caption)
        self.assertTrue(all(item.has_spoiler for item in media))

    async def test_over_limit_rejects_whole_batch_and_preserves_old_inputs(self):
        original = self.state(8)
        bot.ACTIVE_IMAGES[(11, 1)] = original
        updates = [self.update(10), self.update(20)]
        await self.upload(updates)
        self.assertIs(bot.ACTIVE_IMAGES[(11, 1)], original)
        self.assertEqual(len(list(self.root.glob('1_*.jpg'))), 8)
        for update in updates:
            update.message.delete.assert_not_awaited()

    async def test_single_image_mode_rejects_album_and_replaces_normal_photo(self):
        self.preferences.update(1, input_mode="i2v")
        await self.upload([self.update()])
        self.assertNotIn((11, 1), bot.ACTIVE_IMAGES)
        self.tasks.clear()
        await self.upload([self.update(10, album=None), self.update(20, delay=0.02, album=None)])
        self.assertEqual([Path(p).read_text() for p in image_paths(bot.ACTIVE_IMAGES[(11, 1)])], ["20"])

    async def test_prompt_during_upload_cannot_capture_partial_collection(self):
        update = self.update(10, delay=0.04)
        uploading = asyncio.create_task(bot.on_photo(update, self.context))
        await asyncio.sleep(0.001)
        await bot.on_prompt_text(self.update(20), self.context)
        self.assertEqual(bot.PENDING_COUNT, {})
        await uploading
        await asyncio.gather(*self.tasks)

    async def test_repeated_update_does_not_append_duplicate_reference(self):
        update = self.update(10)
        await self.upload([update])
        await bot.on_photo(update, self.context)
        self.assertEqual(len(image_paths(bot.ACTIVE_IMAGES[(11, 1)])), 1)
        update.message.photo[-1].get_file.assert_awaited_once()

    async def test_replacement_during_preview_keeps_files_until_preview_finishes(self):
        async def clear_during_send(*args, **kwargs):
            paths = image_paths(bot.ACTIVE_IMAGES[(11, 1)])
            bot.replace_collection((11, 1), ())
            self.assertTrue(all(Path(p).exists() for p in paths))
        self.context.bot.send_media_group.side_effect = clear_during_send
        await self.upload([self.update(10), self.update(20)])
        self.assertEqual(bot.ACTIVE_IMAGES, {})
        self.assertEqual(bot.IMAGE_USE_COUNT, {})
        self.assertEqual(list(self.root.glob('1_*.jpg')), [])

    async def test_failed_manifest_save_does_not_replace_existing_inputs(self):
        old = self.state(1)
        bot.ACTIVE_IMAGES[(11, 1)] = old
        update = self.update(10)
        with patch.object(bot.INPUT_STORE, "save", side_effect=OSError("disk unavailable")):
            with self.assertLogs(bot.log, level="ERROR"):
                await self.upload([update])
        self.assertIs(bot.ACTIVE_IMAGES[(11, 1)], old)
        self.assertEqual(len(list(self.root.glob('1_*.jpg'))), 1)
        update.message.delete.assert_not_awaited()

    async def test_prompt_freezes_all_references_and_retirement_waits_for_cancel(self):
        state = self.state(2)
        bot.ACTIVE_IMAGES[(11, 1)] = state
        await bot.on_prompt_text(self.update(), self.context)
        pending = next(iter(bot.PENDING_COUNT.values()))
        self.assertEqual(pending["task"], "reference")
        self.assertEqual(pending["image_paths"], image_paths(state))
        bot.replace_collection((11, 1), ())
        self.assertTrue(all(Path(p).exists() for p in pending["image_paths"]))
        bot.discard_pending(lambda batch: True)
        self.assertTrue(all(not Path(p).exists() for p in pending["image_paths"]))
        self.assertEqual(bot.IMAGE_USE_COUNT, {})

    async def test_stale_or_foreign_image_panel_cannot_change_collection(self):
        state = self.state(2)
        bot.ACTIVE_IMAGES[(11, 1)] = state
        update = self.update(user=2)
        update.callback_query = SimpleNamespace(data=f"images:1:{state['revision']}:clear:0", answer=AsyncMock())
        await bot.on_images(update, self.context)
        self.assertIs(bot.ACTIVE_IMAGES[(11, 1)], state)
        update = self.update()
        update.callback_query = SimpleNamespace(data="images:1:00000000:clear:0", answer=AsyncMock())
        await bot.on_images(update, self.context)
        self.assertIs(bot.ACTIVE_IMAGES[(11, 1)], state)

    async def test_move_up_changes_future_inputs_but_preserves_prompt_snapshot(self):
        state = self.state(2)
        bot.ACTIVE_IMAGES[(11, 1)] = state
        await bot.on_prompt_text(self.update(), self.context)
        pending = next(iter(bot.PENDING_COUNT.values()))
        update = self.update()
        update.callback_query = SimpleNamespace(data=f"images:1:{state['revision']}:up:1", answer=AsyncMock())
        await bot.on_images(update, self.context)
        self.assertEqual(image_paths(bot.ACTIVE_IMAGES[(11, 1)]), image_paths(state)[::-1])
        self.assertEqual(pending["image_paths"], image_paths(state))

    async def test_clear_during_upload_does_not_resurrect_inputs(self):
        update = self.update(10, delay=0.04)
        uploading = asyncio.create_task(bot.on_photo(update, self.context))
        await asyncio.sleep(0.001)
        self.context.args = ["clear"]
        await bot.images_command(self.update(), self.context)
        await uploading
        await asyncio.gather(*self.tasks)
        self.assertEqual(bot.ACTIVE_IMAGES, {})
        self.assertEqual(list(self.root.glob('1_*.jpg')), [])

    async def test_missing_picture_number_is_rejected_before_acquiring_images(self):
        state = self.state(1)
        bot.ACTIVE_IMAGES[(11, 1)] = state
        update = self.update()
        update.message.text = "Use <Picture 2>"
        await bot.on_prompt_text(update, self.context)
        self.assertEqual(bot.PENDING_COUNT, {})
        self.assertEqual(bot.IMAGE_USE_COUNT, {})

    async def test_batch_submission_uses_frozen_reference_list(self):
        state = self.state(2)
        bot.ACTIVE_IMAGES[(11, 1)] = state
        await bot.on_prompt_text(self.update(), self.context)
        pending = next(iter(bot.PENDING_COUNT.values()))
        bot.replace_collection((11, 1), ())
        application = SimpleNamespace(bot=self.context.bot)
        def submit(job, index):
            return dict(index=index, seed=job["seed"], status="error")
        with patch.object(bot, "SUBMIT_LOCK", asyncio.Lock()), \
             patch.object(bot.submit, "check_servers", return_value=["http://worker"]), \
             patch.object(bot.submit, "SERVERS", []), \
             patch.object(bot.submit, "submit_job", side_effect=submit) as queued, \
             patch.object(bot, "wait_for_job", new=AsyncMock(side_effect=lambda record: record)):
            await bot.submit_batch(application, pending, 5, 2)
        self.assertEqual(queued.call_count, 2)
        for call in queued.call_args_list:
            job = call.args[0]
            self.assertEqual(job["task"], "reference")
            self.assertEqual(job["image_paths"], list(image_paths(state)))
        self.assertEqual(bot.IMAGE_USE_COUNT, {})
        self.assertTrue(all(not Path(p).exists() for p in image_paths(state)))

    def test_reference_fast_fallback_is_independent_of_single_image_fast(self):
        self.preferences.update(1, mode="fast", reference_mode="fast")
        with patch.object(bot, "REFERENCE_TURBO_UNAVAILABLE", True):
            self.assertEqual(bot.generation_settings(1).mode, "standard")
            self.assertEqual(bot.generation_settings(2).mode, "fast")
        self.assertEqual(self.preferences.get(1).mode, "fast")

    async def test_reference_starts_standard_and_preserves_single_image_fast(self):
        self.preferences.update(1, mode="fast")
        self.assertEqual(bot.generation_settings(1).mode, "standard")
        update = self.update()
        update.callback_query = SimpleNamespace(data="settings:1:mode:fast", answer=AsyncMock(), edit_message_text=AsyncMock())
        with patch.object(bot.submit, "check_reference_support", return_value=[]):
            await bot.on_mode(update, self.context)
        self.assertEqual(bot.generation_settings(1).mode, "fast")
        self.preferences.update(1, input_mode="i2v")
        self.assertEqual(bot.generation_settings(1).mode, "fast")
