import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from PIL import Image

from minimax_h3 import image_jobs, telegram_bot as bot
from minimax_h3.input_sets import collection_state
from minimax_h3.preferences import GenerationSettings, PreferenceStore


class ImageJobsTest(unittest.TestCase):
    def test_distilled_schedule_and_image_components_are_used_without_video_audio(self):
        prompt, metadata = image_jobs.build_prompt({"prompt": 'A sign reading "hello" <你好>', "seed": 42})
        self.assertEqual(prompt["6"]["inputs"]["prompt"], 'A sign reading "hello" <你好>')
        self.assertEqual(prompt["10"]["inputs"]["steps"], 6)
        self.assertEqual(prompt["9"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(prompt["8"]["class_type"], "BasicGuider")
        self.assertEqual(prompt["15"]["class_type"], "SaveImage")
        self.assertEqual(metadata["image_quant"], "w4a8")
        self.assertEqual((metadata["width"], metadata["height"]), (4096, 4096))
        self.assertEqual(prompt["12"]["class_type"], "VAEDecode")
        self.assertEqual(metadata["prompt_enhancement"], "off")
        self.assertFalse(any("Video" in node["class_type"] or "Audio" in node["class_type"] for node in prompt.values()))
        self.assertEqual(prompt["3"]["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")

    def test_profiles_stay_on_32_pixel_grid_at_requested_area_and_ratio(self):
        for profile, (_, edge) in image_jobs.PROFILES.items():
            for ratio, (w, h) in image_jobs.RATIOS.items():
                width, height = image_jobs.canvas(profile, ratio)
                self.assertEqual(width % 32 + height % 32, 0)
                self.assertAlmostEqual(width / height, w / h, delta=0.03)
                self.assertAlmostEqual(width * height / edge ** 2, 1, delta=0.025)
        self.assertEqual(image_jobs.canvas("preview"), (2048, 2048))
        self.assertEqual(image_jobs.canvas("native"), (4096, 4096))

    def test_edit_keeps_picture_order_and_invalid_settings_never_stage_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.png" for i in range(16)]
            for path in paths:
                Image.new("RGB", (32, 32)).save(path)
            job = {"prompt": "Edit <Picture 1> using <Picture 16>.", "image_paths": list(map(str, paths))}
            with patch.object(image_jobs.jobs, "stage_image", side_effect=lambda path: Path(path).name) as stage:
                graph, _ = image_jobs.build_prompt(job)
                self.assertEqual(graph["21"]["inputs"]["image"], "0.png")
                self.assertEqual(graph["36"]["inputs"]["image"], "15.png")
                self.assertEqual(graph["6"]["inputs"]["images.image_16"], ["36", 0])
                stage.reset_mock()
                for changes in ({"steps": 20}, {"cfg": 7}, {"image_quant": "fp16"},
                                {"image_paths": list(map(str, paths)) * 2}, {"prompt": "<Picture 17>"}):
                    with self.assertRaises(ValueError):
                        image_jobs.build_prompt(dict(job, **changes))
                stage.assert_not_called()

    def test_edit_output_matches_first_reference_exactly_without_forcing_square(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "base.png", Path(directory) / "donor.png"
            Image.new("RGB", (1001, 777)).save(first)
            Image.new("RGB", (600, 900)).save(second)
            with patch.object(image_jobs.jobs, "stage_image", side_effect=lambda path: Path(path).name):
                graph, metadata = image_jobs.build_prompt({"prompt": "Edit <Picture 1> using <Picture 2>",
                    "image_paths": [str(first), str(second)], "image_profile": "native", "image_ratio": "1:1"})
            self.assertEqual((metadata["width"], metadata["height"]), (1001, 777))
            generation = graph["6"]["inputs"]
            self.assertEqual(generation["width"] % 32 + generation["height"] % 32, 0)
            self.assertAlmostEqual(generation["width"] / generation["height"], 1001 / 777, delta=0.02)
            self.assertEqual((graph["14"]["inputs"]["width"], graph["14"]["inputs"]["height"]), (1001, 777))
            self.assertEqual(graph["14"]["inputs"]["crop"], "disabled")
            self.assertEqual(graph["15"]["inputs"]["images"], ["14", 0])

    def test_edit_dimensions_follow_exif_and_allow_explicit_canvas_override(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (101, 77))
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)
            job = dict(prompt="Edit <Picture 1>", image_paths=[str(source)], image_profile="preview", image_ratio="16:9")
            matched = image_jobs.normalize_job(job)
            self.assertEqual((matched["width"], matched["height"]), (77, 101))
            explicit = image_jobs.normalize_job(dict(job, image_edit_size="canvas"))
            self.assertEqual((explicit["width"], explicit["height"]), image_jobs.canvas("preview", "16:9"))
            resized = image_jobs.normalize_job(dict(job, output_width=1281, output_height=719))
            self.assertEqual((resized["width"], resized["height"]), (1281, 719))
            with self.assertRaises(ValueError):
                image_jobs.normalize_job(dict(job, output_width=1281))

    def test_preferences_round_trip_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = PreferenceStore(path, GenerationSettings())
            store.update(1, input_mode="image", image_quant="int8", image_profile="preview", image_ratio="9:16", image_edit_size="canvas")
            restored = PreferenceStore(path, GenerationSettings())
            self.assertEqual(restored.get(1), store.get(1))
            self.assertEqual(restored.get(2).input_mode, "i2v")
            self.assertEqual(restored.get(1).image_profile, "preview")
            self.assertEqual(restored.get(2).image_profile, "native")
            self.assertEqual(restored.get(1).image_edit_size, "canvas")
            self.assertEqual(restored.get(2).image_edit_size, "input")


class ImageBotTest(unittest.IsolatedAsyncioTestCase):
    async def test_sixteen_references_preview_in_legal_telegram_album_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            images = []
            for i in range(16):
                path = Path(directory) / f"{i}.png"
                Image.new("RGB", (32, 32)).save(path)
                images.append(dict(image_path=str(path), width=2048, height=2048))
            state = collection_state(11, 1, images)
            context = SimpleNamespace(bot=SimpleNamespace(send_media_group=AsyncMock(), send_message=AsyncMock()))
            store = PreferenceStore(Path(directory) / "settings.json", GenerationSettings(input_mode="image", image_profile="preview"))
            with patch.multiple(bot, PREFERENCES=store, IMAGE_USE_COUNT={}, RETIRED_IMAGES=set()):
                await bot.show_images(context, (11, 1), state)
            self.assertEqual([len(call.kwargs["media"]) for call in context.bot.send_media_group.call_args_list], [10, 6])

    async def test_original_png_is_sent_as_document_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"test PNG")
            app = SimpleNamespace(bot=SimpleNamespace(send_document=AsyncMock(), send_photo=AsyncMock()))
            record = dict(width=4096, height=4096, image_quant="w4a8", seed=42, prompt="A red mug")
            await bot.deliver_image(app, 11, record, str(path))
            app.bot.send_document.assert_awaited_once()
            app.bot.send_photo.assert_not_awaited()
            self.assertTrue(path.exists())

    async def test_text_to_image_needs_no_upload_and_never_expands_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(Path(directory) / "settings.json", GenerationSettings())
            store.update(1, input_mode="image", image_profile="balanced", image_ratio="16:9")
            update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=11),
                message=SimpleNamespace(text='A poster reading <你好>', reply_text=AsyncMock()))
            with patch.multiple(bot, PREFERENCES=store, ALLOWLIST={1}, ACTIVE_IMAGES={}, UPLOAD_BATCHES={}, PENDING_COUNT={}):
                await bot.on_prompt_text(update, SimpleNamespace())
                batch = next(iter(bot.PENDING_COUNT.values()))
                self.assertEqual(batch["prompt"], 'A poster reading <你好>')
                self.assertEqual(batch["image_paths"], ())
                self.assertEqual(batch["image_profile"], "balanced")
                store.update(1, image_profile="native")
                self.assertEqual(batch["image_profile"], "balanced")

    async def test_batch_routes_to_image_worker_and_delivers_png_document(self):
        record = dict(task="image", seed=42, width=2048, height=2048, image_quant="w4a8", prompt="A red mug",
                      status="completed", outputs=["image.png"], server=image_jobs.SERVER, prompt_id="abc")
        batch = dict(chat_id=11, user_id=1, task="image", image_paths=(), prompt="A red mug", display_prompt="A red mug")
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        with patch.object(bot, "SUBMIT_LOCK", asyncio.Lock()), \
             patch.object(image_jobs, "submit_job", return_value=record) as submit, \
             patch.object(bot, "wait_for_job", new_callable=AsyncMock, return_value=record), \
             patch.object(bot, "deliver_image", new_callable=AsyncMock) as deliver, \
             patch.object(bot.submit, "submit_job") as video_submit:
            await bot.submit_batch(app, batch, 0, 1)
            submit.assert_called_once()
            video_submit.assert_not_called()
            deliver.assert_awaited_once_with(app, 11, record, "image.png")

    async def test_image_ratio_callback_and_foreign_panel_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(Path(directory) / "settings.json", GenerationSettings(input_mode="image"))
            query = SimpleNamespace(data="settings:1:image_ratio:9-16", answer=AsyncMock(), edit_message_text=AsyncMock())
            update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=11), callback_query=query)
            with patch.multiple(bot, PREFERENCES=store, ALLOWLIST={1, 2}, UPLOAD_BATCHES={}):
                await bot.on_mode(update, SimpleNamespace())
                self.assertEqual(store.get(1).image_ratio, "9:16")
                update.effective_user.id = 2
                query.data = "settings:1:image_profile:native"
                await bot.on_mode(update, SimpleNamespace())
                self.assertEqual(store.get(1).image_profile, "native")

    async def test_one_poll_failure_does_not_abandon_other_image_results(self):
        batch = dict(chat_id=11, user_id=1, task="image", image_paths=("reference.png",),
                     prompt="A red mug", display_prompt="A red mug")
        records = [dict(seed=seed, status="completed", outputs=[f"{seed}.png"]) for seed in (1, 2)]
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        survivor_started = asyncio.Event()

        async def poll(record):
            if record["seed"] == 1:
                await survivor_started.wait()
                raise ConnectionError("worker disconnected")
            survivor_started.set()
            await asyncio.sleep(0.01)
            release.assert_not_called()
            return record

        with patch.object(bot, "SUBMIT_LOCK", asyncio.Lock()), \
             patch.object(image_jobs, "submit_job", side_effect=records), \
             patch.object(bot, "wait_for_job", side_effect=poll), \
             patch.object(bot, "deliver_image", new_callable=AsyncMock) as deliver, \
             patch.object(bot, "release_image") as release:
            await bot.submit_image_batch(app, batch, 2)
            deliver.assert_awaited_once_with(app, 11, records[1], "2.png")
            release.assert_called_once_with("reference.png")
        self.assertTrue(any("状态查询失败" in call.args[1] for call in app.bot.send_message.call_args_list))


if __name__ == "__main__":
    unittest.main()
