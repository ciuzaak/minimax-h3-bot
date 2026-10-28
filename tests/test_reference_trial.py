import asyncio
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from minimax_h3 import jobs, reference_trial as trial, telegram_bot as bot
from minimax_h3.input_sets import collection_state
from minimax_h3.preferences import GenerationSettings, PreferenceStore
from tools import download_reference_trial as download


class ReferenceTrialTest(unittest.TestCase):
    def row(self, mode):
        return dict(task="reference", reference_mode=mode, image_paths=["one.jpg", "two.jpg"],
                    prompt="Scene", seed=42, seconds=5, width=768, height=768,
                    steps=99, sampler="invalid-override")

    def test_three_recipes_and_history_preserve_actual_selection(self):
        for mode, steps, lora in (("fast",4,jobs.REFERENCE_TURBO_LORA_NAME),
                                  ("trial8",8,trial.LORA_NAME), ("standard",20,None)):
            row=self.row(mode)
            if mode != "trial8":row.update(steps=20,sampler="res_multistep")
            def request(url, data=None, **kwargs):
                if url.endswith('/prompt'):
                    self.assertEqual(data['extra_data']['minimax_h3_recipe']['reference_mode'],mode)
                    self.assertEqual(data['extra_data']['minimax_h3_recipe']['steps'],steps)
                    return {'prompt_id':'test'}
            with patch.object(jobs,'stage_image',side_effect=lambda p:p), \
                 patch.object(jobs,'select_server',return_value='http://worker'), \
                 patch.object(jobs,'check_reference_support',return_value=[]) as preflight, \
                 patch.object(jobs,'request_json',side_effect=request):
                graph,meta=jobs.build_prompt(row,1)
                record=jobs.submit_job(row,1)
            self.assertEqual(meta['steps'],steps)
            self.assertEqual(record['lora'],lora)
            self.assertEqual(meta['diffusion_model'],jobs.REFERENCE_MODEL_NAME)
            self.assertEqual(graph['4']['inputs']['vae_name'],'minimax_h3_video_vae_fp16.safetensors')
            preflight.assert_called_once_with(['http://worker'],mode!='standard',mode)
            if mode=='trial8':
                self.assertEqual(meta['sampler'],'euler')
                self.assertEqual(graph['30']['inputs'],{'model':['16',0],'shift_video':6.,'shift_audio':3.})
                for node in ('8','10'):self.assertEqual(graph[node]['inputs']['model'],['30',0])
                self.assertEqual(graph['10']['inputs']['steps'],8)
                self.assertEqual(meta['lora_sha256'],trial.EXPECTED_HASH)
            else:self.assertNotIn('30',graph)

    def test_invalid_trial_never_stages_and_jsonl_can_select_without_legacy_boolean(self):
        for row in (dict(task='i2v',reference_mode='trial8',image_path='one'),
                    dict(task='reference',reference_mode='unknown',image_paths=['one'])):
            with patch.object(jobs,'stage_image') as stage:
                with self.assertRaises(ValueError):jobs.build_prompt(row,1)
                stage.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'one.png'
            from PIL import Image
            Image.new('RGB',(768,512)).save(image)
            path=root/'jobs.jsonl';path.write_text(json.dumps(dict(task='reference',reference_mode='trial8',image_paths=[str(image)],prompt='Scene',seconds=5)))
            row=jobs.load_jobs(path)[0]
            with patch.object(jobs,'stage_image',return_value='one'):
                _,meta=jobs.build_prompt(row,1)
            self.assertEqual(meta['steps'],8);self.assertTrue(meta['accelerated'])
            self.assertEqual(min(meta['width'],meta['height']),576)

    def test_missing_or_shadowed_bad_trial_does_not_affect_old_recipes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);managed=root/'models';comfy=root/'comfy'
            def request(url,**kwargs):
                if '/object_info/' in url:return {url.rsplit('/',1)[1]:{}}
                if url.endswith('diffusion_models'):return [jobs.REFERENCE_MODEL_NAME]
                if url.endswith('loras'):return [jobs.REFERENCE_TURBO_LORA_NAME,trial.LORA_NAME]
                if url.endswith('system_stats'):return {'system':{'comfyui_version':'0.31.0'}}
                return ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','minimax_h3_video_vae_fp16.safetensors','minimax_h3_audio_vae_fp32.safetensors']
            with patch.object(jobs,'MODEL_DIR',managed),patch.object(jobs,'COMFY_DIR',comfy), \
                 patch.object(jobs,'request_json',side_effect=request), \
                 patch.object(trial,'EXPECTED_HASH',sha256(b'valid').hexdigest()):
                self.assertTrue(any('Missing Ref8' in x for x in jobs.check_reference_support(['worker'],reference_mode='trial8')))
                self.assertEqual(jobs.check_reference_support(['worker'],accelerated=True),[])
                self.assertEqual(jobs.check_reference_support(['worker']),[])
                target=managed/'loras'/trial.LORA_NAME;target.parent.mkdir(parents=True);target.write_bytes(b'valid')
                self.assertEqual(jobs.check_reference_support(['worker'],reference_mode='trial8'),[])
                shadow=comfy/'models/loras'/trial.LORA_NAME;shadow.parent.mkdir(parents=True);shadow.write_bytes(b'wrong')
                self.assertTrue(any('mismatch' in x for x in jobs.check_reference_support(['worker'],reference_mode='trial8')))
                self.assertEqual(jobs.check_reference_support(['worker'],accelerated=True),[])
                previous=shadow.stat()
                shadow.write_bytes(b'valid')
                import os
                os.utime(shadow,ns=(previous.st_atime_ns,previous.st_mtime_ns+1))
                self.assertEqual(jobs.check_reference_support(['worker'],reference_mode='trial8'),[])

    def test_old_worker_and_missing_node_or_model_fail_trial_preflight(self):
        def request(url,**kwargs):
            if url.endswith('system_stats'):return {'system':{'comfyui_version':'0.30.0'}}
            return {} if '/object_info/' in url else []
        issues=trial.check_worker('worker',request)
        self.assertTrue(any('v0.31.0' in x for x in issues))
        self.assertTrue(any('MiniMaxH3SigmaShift' in x for x in issues))
        self.assertTrue(any('video_vae_fp16' in x for x in issues))

    def test_download_reuses_verified_and_bad_download_preserves_existing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(download,'EXPECTED_HASH',sha256(b'valid').hexdigest()):
            root=Path(folder);dest=root/'loras'/trial.LORA_NAME;dest.parent.mkdir();dest.write_bytes(b'valid')
            with patch.dict('sys.modules',{'modelscope':None}):
                self.assertEqual(download.download_asset(root),dest)
            dest.write_bytes(b'original')
            def snapshot(repo,**kwargs):
                self.assertEqual(repo,'lightx2v/Minimax-h3-Turbo')
                self.assertEqual(kwargs['allow_file_pattern'],[trial.LORA_NAME])
                self.assertNotIn('HTTPS_PROXY',__import__('os').environ)
                (Path(kwargs['local_dir'])/trial.LORA_NAME).write_bytes(b'invalid')
            with patch.dict('sys.modules',{'modelscope':SimpleNamespace(snapshot_download=snapshot)}), \
                 patch.dict('os.environ',{'HTTPS_PROXY':'http://example.invalid'}):
                with self.assertRaisesRegex(RuntimeError,'Checksum'):download.download_asset(root)
                self.assertEqual(__import__('os').environ['HTTPS_PROXY'],'http://example.invalid')
            self.assertEqual(dest.read_bytes(),b'original')

    def test_exif_canvas_matches_loadimage_orientation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rotated.jpg';image=Image.new('RGB',(768,512));exif=Image.Exif();exif[274]=6;image.save(path,exif=exif)
            width,height=jobs.image_resolution(path)
            self.assertLess(width,height)
            self.assertEqual(jobs.image_resolution(path,768,512),(768,512))


class TrialBotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.store=PreferenceStore(self.root/'settings.json',GenerationSettings())
        for name,value in [('PREFERENCES',self.store),('ALLOWLIST',{1,2}),('PENDING_COUNT',{}),
                           ('REFERENCE_TURBO_UNAVAILABLE',False),('TURBO_UNAVAILABLE',False),
                           ('IMAGE_USE_COUNT',{}),('RETIRED_IMAGES',set()),('UPLOAD_BATCHES',{})]:
            p=patch.object(bot,name,value);p.start();self.addCleanup(p.stop)

    def update(self,callback='settings:1:mode:trial8'):
        return SimpleNamespace(effective_user=SimpleNamespace(id=1),effective_chat=SimpleNamespace(id=11),
                               callback_query=SimpleNamespace(data=callback,answer=AsyncMock(),edit_message_text=AsyncMock()),
                               message=SimpleNamespace(text='Move naturally',reply_text=AsyncMock()))

    async def test_trial_is_per_user_persistent_and_cannot_change_old_fast_capability(self):
        self.store.update(1,input_mode='reference');self.store.update(2,input_mode='reference',reference_mode='fast')
        with patch.object(bot,'REFERENCE_TURBO_UNAVAILABLE',True),patch.object(jobs,'check_reference_support',return_value=[]) as check:
            await bot.on_mode(self.update(),SimpleNamespace())
            check.assert_called_once_with(accelerated=True,reference_mode='trial8')
            self.assertTrue(bot.REFERENCE_TURBO_UNAVAILABLE)
            self.assertEqual(bot.generation_settings(2).mode,'standard')
            self.assertEqual(bot.generation_settings(1).mode,'trial8')
        restored=PreferenceStore(self.store.path,GenerationSettings())
        self.assertEqual(restored.get(1).reference_mode,'trial8');self.assertEqual(restored.get(1).mode,'fast')
        self.assertEqual(restored.get(2).reference_mode,'fast')
        buttons=[b for row in bot.mode_keyboard(bot.generation_settings(1),1).inline_keyboard for b in row]
        self.assertTrue({'settings:1:mode:fast','settings:1:mode:trial8','settings:1:mode:standard'} <= {b.callback_data for b in buttons})
        self.assertTrue(any('试用' in b.text and '较慢' in b.text for b in buttons))
        self.store.update(1,input_mode='i2v');self.assertEqual(bot.generation_settings(1).mode,'fast')
        keys=[b.callback_data for row in bot.mode_keyboard(bot.generation_settings(1),1).inline_keyboard for b in row]
        self.assertNotIn('settings:1:mode:trial8',keys)

    async def test_missing_trial_does_not_change_saved_mode_and_i2v_rejects_it(self):
        self.store.update(1,input_mode='reference')
        with patch.object(jobs,'check_reference_support',return_value=['Missing Ref8 LoRA']):
            await bot.on_mode(self.update(),SimpleNamespace())
        self.assertEqual(self.store.get(1).reference_mode,'standard')
        self.store.update(1,input_mode='i2v');update=self.update()
        await bot.on_mode(update,SimpleNamespace());self.assertEqual(self.store.get(1).mode,'fast')
        update.callback_query.answer.assert_awaited_once_with('8 步试用仅用于多图参考',show_alert=True)

    async def test_prompt_freezes_trial_through_actual_batch_graph_after_preference_changes(self):
        self.store.update(1,input_mode='reference',reference_mode='trial8')
        source=self.root/'one.jpg';source.write_bytes(b'fixture')
        state=collection_state(11,1,[dict(image_path=str(source),width=576,height=576)])
        context=SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        graphs=[]
        def submit(row,index):
            graph,meta=jobs.build_prompt(row,index);graphs.append(graph)
            return dict(meta,status='completed',outputs=[],seed=row['seed'])
        with patch.object(bot,'ACTIVE_IMAGES',{(11,1):state}),patch.object(bot,'send_text_with_retry',new_callable=AsyncMock):
            await bot.on_prompt_text(self.update(),context)
            pending=next(iter(bot.PENDING_COUNT.values()))
            self.store.update(1,reference_mode='fast')
            with patch.object(bot,'SUBMIT_LOCK',asyncio.Lock()),patch.object(jobs,'check_servers',return_value=['worker']), \
                 patch.object(jobs,'SERVERS',[]),patch.object(jobs,'submit_job',side_effect=submit), \
                 patch.object(jobs,'stage_image',return_value='one.jpg'),patch.object(jobs,'discard_staged_image'), \
                 patch.object(bot,'wait_for_job',new=AsyncMock(side_effect=lambda r:r)):
                await bot.submit_batch(SimpleNamespace(bot=context.bot),pending,5,1)
        self.assertEqual(pending['reference_mode'],'trial8')
        self.assertEqual(graphs[0]['16']['inputs']['lora_name'],trial.LORA_NAME)
        self.assertEqual(graphs[0]['10']['inputs']['steps'],8)
        self.assertIn('8 步试用',bot.mode_label(bot.batch_mode(pending),'reference'))

    def test_old_schema_users_keep_existing_reference_and_i2v_modes(self):
        self.store.path.write_text(json.dumps({'schema_version':2,'defaults':{},'users':{
            '1':{'mode':'fast','input_mode':'reference','reference_mode':'fast'},
            '2':{'mode':'standard','input_mode':'reference','reference_mode':'standard'}}}))
        restored=PreferenceStore(self.store.path,GenerationSettings())
        self.assertEqual(restored.get(1).reference_mode,'fast');self.assertEqual(restored.get(1).mode,'fast')
        self.assertEqual(restored.get(2).reference_mode,'standard');self.assertEqual(restored.get(2).mode,'standard')

    async def test_input_mode_change_during_trial_preflight_cannot_change_i2v(self):
        self.store.update(1,input_mode='reference')
        def check(**kwargs):
            self.store.update(1,input_mode='i2v')
            return []
        update=self.update()
        with patch.object(jobs,'check_reference_support',side_effect=check):
            await bot.on_mode(update,SimpleNamespace())
        self.assertEqual(self.store.get(1).mode,'fast')
        self.assertEqual(self.store.get(1).reference_mode,'standard')
        update.callback_query.answer.assert_awaited_once_with('输入模式已改变，请重新打开 /mode',show_alert=True)
