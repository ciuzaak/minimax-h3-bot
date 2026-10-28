from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from minimax_h3 import jobs, telegram_bot as bot
from minimax_h3.preferences import GenerationSettings, PreferenceStore


class OfflinePreflightTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = PreferenceStore(Path(temp.name) / 'settings.json', GenerationSettings())
        self.store.update(1, mode='fast')
        self.store.update(2, input_mode='reference', reference_mode='fast')
        self.application = SimpleNamespace(bot=SimpleNamespace(
            delete_my_commands=AsyncMock(), set_my_commands=AsyncMock(),
            get_me=AsyncMock(return_value=SimpleNamespace(username='offline-test'))))
        for name, value in [('PREFERENCES', self.store), ('TURBO_UNAVAILABLE', False),
                            ('REFERENCE_TURBO_UNAVAILABLE', False), ('SUBMIT_LOCK', None)]:
            p = patch.object(bot, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(bot, 'restore_private_chat_images', return_value=0)
        p.start()
        self.addCleanup(p.stop)

    async def test_offline_start_preserves_fast_and_recovered_workers_submit_original_recipes(self):
        before = sha256(self.store.path.read_bytes()).hexdigest()
        with patch.object(jobs, 'request_json', side_effect=OSError('offline')), self.assertLogs(bot.log, level='INFO') as logs:
            await bot.post_init(self.application)
        self.assertFalse(bot.TURBO_UNAVAILABLE)
        self.assertFalse(bot.REFERENCE_TURBO_UNAVAILABLE)
        self.assertEqual(bot.generation_settings(1).mode, 'fast')
        self.assertEqual(bot.generation_settings(2).mode, 'fast')
        self.assertEqual(sha256(self.store.path.read_bytes()).hexdigest(), before)
        self.assertTrue(any('PREFLIGHT_DEFERRED' in line for line in logs.output))
        self.assertFalse(any('PREFLIGHT_OK' in line or 'PREFLIGHT_FAILED' in line for line in logs.output))

        submitted = []
        def online(url, data=None, **kwargs):
            if url.endswith('/prompt'):
                submitted.append(data)
                return {'prompt_id': 'recovered'}
            if '/object_info/' in url:
                return {url.rsplit('/', 1)[1]: {}}
            if url.endswith('diffusion_models'):
                return [jobs.REFERENCE_MODEL_NAME]
            if url.endswith('loras'):
                return [jobs.TURBO_LORA_NAME, jobs.REFERENCE_TURBO_LORA_NAME]
            raise AssertionError(url)
        with patch.object(jobs, 'request_json', side_effect=online), \
             patch.object(jobs, 'select_server', return_value='http://worker'), \
             patch.object(jobs, 'stage_image', return_value='fixture.jpg'):
            for user, expected_steps in ((1, 8), (2, 4)):
                settings = bot.generation_settings(user)
                row = dict(task=settings.input_mode, image_path='fixture.jpg', prompt='Scene',
                           seconds=5, seed=42, width=576, height=576,
                           accelerated=settings.mode == 'fast')
                if settings.input_mode == 'reference':
                    row['reference_mode'] = settings.mode
                record = jobs.submit_job(row, 1)
                self.assertEqual(record['steps'], expected_steps)
                self.assertTrue(record['accelerated'])
        self.assertEqual(submitted[0]['prompt']['16']['inputs']['lora_name'], jobs.TURBO_LORA_NAME)
        self.assertEqual(submitted[1]['prompt']['16']['inputs']['lora_name'], jobs.REFERENCE_TURBO_LORA_NAME)

    async def test_real_missing_components_still_fall_back_without_rewriting_preferences(self):
        def broken(url, **kwargs):
            if url.endswith('system_stats'):
                return {'system': {}}
            return {} if '/object_info/' in url else []
        with patch.object(jobs, 'request_json', side_effect=broken):
            await bot.post_init(self.application)
        self.assertTrue(bot.TURBO_UNAVAILABLE)
        self.assertTrue(bot.REFERENCE_TURBO_UNAVAILABLE)
        self.assertEqual(bot.generation_settings(1).mode, 'standard')
        self.assertEqual(bot.generation_settings(2).mode, 'standard')
        self.assertEqual(self.store.get(1).mode, 'fast')
        self.assertEqual(self.store.get(2).reference_mode, 'fast')

    def test_offline_submission_remains_an_explicit_error(self):
        with patch.object(jobs, 'request_json', side_effect=OSError('offline')):
            with self.assertRaises(jobs.NoWorkersAvailable):
                jobs.check_servers()
