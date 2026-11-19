import asyncio
from codecs import BOM_UTF16_BE, BOM_UTF16_LE
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from telegram import Update
from telegram.error import TimedOut

from minimax_h3 import telegram_bot as bot
from minimax_h3.input_sets import collection_state
from minimax_h3.preferences import GenerationSettings, PreferenceStore
from minimax_h3.prompt_file import parse_prompt_file


class PromptFileParserTest(unittest.TestCase):
    def test_supported_encodings_and_blank_line_groups(self):
        text = "\r\n人物挥手 x2\r\n镜头固定 X3  \r\n \t\r\n\r\n人物走到窗边。\r\n"
        for content in (text.encode("utf-8"), text.encode("utf-8-sig"),
                        BOM_UTF16_LE + text.encode("utf-16-le"),
                        BOM_UTF16_BE + text.encode("utf-16-be")):
            with self.subTest(bom=content[:4]):
                self.assertEqual(parse_prompt_file(bytearray(content)),
                                 [("人物挥手 x2\n镜头固定", 3), ("人物走到窗边。", 1)])

    def test_count_token_only_applies_at_end_of_group(self):
        self.assertEqual(parse_prompt_file(b"Scene helix2\n\nSecond scene\nx12"),
                         [("Scene helix2", 1), ("Second scene", 12)])

    def test_invalid_counts_and_missing_prompt_report_the_group(self):
        for group in (b"Scene x0", b"Scene x-2", b"Scene x01", b"x3", b"Scene x18446744073709551617"):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "第 2 组"):
                parse_prompt_file(b"Valid scene\n\n" + group)

    def test_empty_and_invalid_text(self):
        for content, error in ((b"\n\t\n", "没有提示词"), (b"\xff", "UTF-8"),
                               (b"\xff\xfeA", "UTF-8"), ("Scene".encode("utf-16-le"), "UTF-8")):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, error):
                parse_prompt_file(content)


class PromptFileBotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = PreferenceStore(self.root / "settings.json", GenerationSettings())
        self.tasks = []
        self.context = SimpleNamespace(
            application=SimpleNamespace(
                bot=SimpleNamespace(send_message=AsyncMock()), create_task=self.create_task,
            ),
        )
        patcher = patch.multiple(bot, ALLOWLIST={1}, PREFERENCES=self.store, BOT_INPUT_DIR=self.root,
                                 ACTIVE_IMAGES={}, UPLOAD_BATCHES={}, IMAGE_USE_COUNT={},
                                 RETIRED_IMAGES=set(), PENDING_COUNT={}, SUBMIT_LOCK=asyncio.Lock(),
                                 TURBO_UNAVAILABLE=False, REFERENCE_TURBO_UNAVAILABLE=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def create_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def attachment(self, content, user=1):
        telegram_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(content)))
        document = SimpleNamespace(file_name="任意文件名.TXT", file_size=len(content),
                                   get_file=AsyncMock(return_value=telegram_file))
        message = SimpleNamespace(document=document, reply_text=AsyncMock(), delete=AsyncMock())
        update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=user),
                                 effective_chat=SimpleNamespace(id=11))
        return update, telegram_file

    def reference(self):
        path = self.root / "reference.jpg"
        path.touch()
        bot.ACTIVE_IMAGES[(11, 1)] = collection_state(
            11, 1, [dict(image_path=str(path), width=576, height=576)],
        )
        return path

    async def test_image_attachment_runs_all_groups_through_existing_submission(self):
        self.store.update(1, input_mode="image")
        first = "A red mug. " * 400 + "Soft daylight."
        update, telegram_file = self.attachment(f"{first} x2\n\nA blue mug.".encode("utf-8-sig"))

        def submit_job(job, index):
            return dict(seed=job["seed"], status="completed", outputs=[f"{job['seed']}.png"])

        with patch.object(bot.image_jobs, "submit_job", side_effect=submit_job) as submit, \
             patch.object(bot, "wait_for_job", new_callable=AsyncMock, side_effect=lambda record: record), \
             patch.object(bot, "deliver_image", new_callable=AsyncMock) as deliver:
            await bot.on_prompt_file(update, self.context)
            await asyncio.gather(*self.tasks)
        self.assertCountEqual([call.args[0]["prompt"] for call in submit.call_args_list],
                              [first, first, "A blue mug."])
        self.assertEqual(len({call.args[0]["seed"] for call in submit.call_args_list
                              if call.args[0]["prompt"] == first}), 2)
        self.assertEqual(deliver.await_count, 3)
        telegram_file.download_as_bytearray.assert_awaited_once()
        update.message.delete.assert_not_awaited()
        self.assertIn("2 组提示词，共 3 张图像", update.message.reply_text.call_args.args[0])
        self.assertEqual({path.name for path in self.root.iterdir()}, {"settings.json"})

    async def test_video_attachment_keeps_reference_and_settings_at_receipt(self):
        self.store.update(1, input_mode="reference", seconds=10, dialogue_language="Japanese")
        reference = self.reference()
        text = "<Picture 1> says <你好>.\nCamera follows. x3\n\nA quiet scene."
        update, telegram_file = self.attachment(text.encode("utf-16"))

        async def download():
            self.store.update(1, input_mode="i2v", seconds=5, dialogue_language="English")
            del bot.ACTIVE_IMAGES[(11, 1)]
            bot.retire_image(reference)
            self.assertTrue(reference.exists())
            return bytearray(text.encode("utf-16"))

        async def submit_batch(application, batch, seconds, count):
            self.assertTrue(reference.exists())
            bot.release_image(reference)

        telegram_file.download_as_bytearray.side_effect = download
        with patch.object(bot, "submit_batch", side_effect=submit_batch) as submit:
            await bot.on_prompt_file(update, self.context)
            await asyncio.gather(*self.tasks)
        first, second = [call.args for call in submit.call_args_list]
        self.assertEqual(first[1]["prompt"], "<Picture 1> says <d>[Japanese] 你好</d>.\nCamera follows.")
        self.assertEqual(second[1]["prompt"], "A quiet scene.")
        self.assertEqual([first[2:], second[2:]], [(10, 3), (10, 1)])
        self.assertTrue(all(call[1]["task"] == "reference" for call in (first, second)))
        self.assertEqual(bot.IMAGE_USE_COUNT, {})
        self.assertFalse(reference.exists())

    async def test_invalid_later_group_prevents_all_submissions_in_each_mode(self):
        for mode in ("image", "i2v", "reference"):
            with self.subTest(mode=mode):
                self.store.update(1, input_mode=mode)
                if mode != "image":
                    self.reference()
                update, _ = self.attachment(b"Valid scene x2\n\nUse <Picture 2> x3")
                await bot.on_prompt_file(update, self.context)
                self.assertIn("整批未提交。第 2 组", update.message.reply_text.call_args.args[0])
                self.assertEqual(self.tasks, [])
                self.assertEqual(bot.IMAGE_USE_COUNT, {})
                self.assertEqual(list(self.root.glob("*.txt")), [])

    async def test_invalid_encoding_requests_utf8_without_retaining_attachment(self):
        self.store.update(1, input_mode="image")
        update, _ = self.attachment(b"\xff")
        await bot.on_prompt_file(update, self.context)
        self.assertIn("另存为 UTF-8", update.message.reply_text.call_args.args[0])
        self.assertEqual(self.tasks, [])
        self.assertEqual({path.name for path in self.root.iterdir()}, {"settings.json"})

    async def test_download_failure_releases_reference_and_does_not_write_a_file(self):
        reference = self.reference()
        update, telegram_file = self.attachment(b"A scene.")
        telegram_file.download_as_bytearray.side_effect = TimedOut()
        await bot.on_prompt_file(update, self.context)
        self.assertIn("下载失败", update.message.reply_text.call_args.args[0])
        self.assertEqual(self.tasks, [])
        self.assertEqual(bot.IMAGE_USE_COUNT, {})
        self.assertEqual(list(self.root.iterdir()), [reference])

    async def test_rejected_files_do_not_download(self):
        self.store.update(1, input_mode="image")
        unauthorized, _ = self.attachment(b"A scene.", user=2)
        oversized, _ = self.attachment(b"A scene.")
        oversized.message.document.file_size = bot.TELEGRAM_DOWNLOAD_LIMIT + 1
        for update, reason in ((unauthorized, "白名单"), (oversized, "20MB")):
            await bot.on_prompt_file(update, self.context)
            update.message.document.get_file.assert_not_awaited()
            self.assertIn(reason, update.message.reply_text.call_args.args[0])
        self.assertEqual(self.tasks, [])


class PromptFileRoutingTest(unittest.TestCase):
    def test_txt_documents_route_before_image_documents_and_plain_text(self):
        with patch.object(bot, "TOKEN", "123:dummy"), \
             patch.object(bot.Application, "run_polling", autospec=True) as run:
            bot.main()
        application = run.call_args.args[0]
        for fields, expected in (
            ({"document": {"file_name": "任意名称.TXT", "file_id": "f", "file_unique_id": "u"}},
             bot.on_prompt_file),
            ({"document": {"file_name": "reference.jpg", "file_id": "f", "file_unique_id": "u"}},
             bot.on_photo),
            ({"text": "Scene 1 x2\n\nScene 2 x3"}, bot.on_prompt_text),
        ):
            with self.subTest(fields=fields):
                update = Update.de_json(
                    {"update_id": 1, "message": {
                        "message_id": 1, "date": 0, "chat": {"id": 11, "type": "private"}, **fields,
                    }},
                    application.bot,
                )
                handler = next(handler for handler in application.handlers[0] if handler.check_update(update))
                self.assertIs(handler.callback, expected)
