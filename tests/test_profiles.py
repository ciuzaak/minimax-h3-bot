from hashlib import sha256
from pathlib import Path
import tempfile
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minimax_h3 import assets, jobs
from minimax_h3.preferences import GenerationSettings, PreferenceStore
from minimax_h3.profiles import NATIVE_TURBO_LORA_NAME, NATIVE_REFERENCE_LORA_NAME, VIDEO_VAES, resolution_for_size
from tools import download_upgrades as download, validate_profiles as validate


class RecipeTest(unittest.TestCase):
    def build(self, **options):
        row = dict(image_path="input.png", width=576, height=768, seed=42,
                   seconds=5, prompt="A paper boat floats on water.", accelerated=True)
        row.update(options)
        with patch.object(jobs, "stage_image", return_value="input.png"):
            return jobs.build_prompt(row, 1)

    def test_native_i2v_binds_adapter_sampler_schedule_and_metadata(self):
        graph, meta = self.build(turbo_backend="native", video_vae="int8",
                                 steps=50, sampler="euler", scheduler="normal")
        self.assertEqual(graph["16"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["16"]["inputs"]["lora_name"], NATIVE_TURBO_LORA_NAME)
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["30"]["inputs"], {"model": ["16", 0], "shift_video": 6.0, "shift_audio": 3.0})
        self.assertEqual(graph["8"]["inputs"]["model"], ["30", 0])
        self.assertEqual(graph["10"]["inputs"]["model"], ["30", 0])
        self.assertEqual(graph["10"]["inputs"]["steps"], 8)
        self.assertEqual(graph["10"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["4"]["inputs"]["vae_name"], VIDEO_VAES["int8"])
        self.assertEqual(meta["lora"], NATIVE_TURBO_LORA_NAME)
        self.assertEqual(meta["video_vae"], "int8")
        self.assertFalse(any(n["class_type"].startswith("MiniMaxH3Turbo") for n in graph.values()))

    def test_reference_recipe_does_not_inherit_i2v_backend_or_sampler(self):
        with patch.object(jobs, "I2V_TURBO_BACKEND", "native"):
            graph, meta = self.build(task="reference", sampler="euler")
        self.assertEqual(meta["steps"], 4)
        self.assertEqual(meta["turbo_backend"], "legacy")
        self.assertEqual(meta["lora"], jobs.REFERENCE_TURBO_LORA_NAME)
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "res_multistep")

    def test_invalid_recipes_fail_before_staging(self):
        for options in ({"turbo_backend": "pdd"}, {"video_vae": "unknown"},
                        {"resolution": "unknown"}, {"video_vae": []}, {"resolution": {}},
                        {"turbo_backend": "disabled", "accelerated": True},
                        {"task": "reference", "turbo_backend": "pdd"}):
            with self.subTest(options=options), patch.object(jobs, "stage_image") as stage:
                with self.assertRaises(ValueError):
                    jobs.build_prompt(dict(image_path="input.png", **options), 1)
                stage.assert_not_called()

    def test_native_reference_binds_its_own_lora_and_eight_steps(self):
        graph, meta = self.build(task="reference", turbo_backend="native")
        self.assertEqual(meta["steps"], 8)
        self.assertEqual(meta["lora"], NATIVE_REFERENCE_LORA_NAME)
        self.assertEqual(graph["2"]["inputs"]["unet_name"], jobs.REFERENCE_MODEL_NAME)
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ReferenceToVideo")

    def test_submission_records_effective_recipe_in_worker_history(self):
        graph, meta = self.build(turbo_backend="native", steps=50)
        with patch.object(jobs, "build_prompt", return_value=(graph, meta)), \
             patch.object(jobs, "select_server", return_value="http://worker"), \
             patch.object(jobs, "check_job_support", return_value=[]), \
             patch.object(jobs, "request_json", return_value={"prompt_id": "test"}) as request:
            record = jobs.submit_job({"steps": 50}, 1)
        sent = request.call_args.args[1]["extra_data"]["minimax_h3_recipe"]
        self.assertEqual(sent["steps"], 8)
        self.assertEqual(sent["lora"], NATIVE_TURBO_LORA_NAME)
        self.assertEqual(record["sampler"], sent["sampler"])

    def test_preflight_does_not_treat_legacy_lora_as_native(self):
        def request(url, **kwargs):
            return [jobs.TURBO_LORA_NAME] if url.endswith("/loras") else {url.rsplit("/", 1)[1]: {}}
        with patch.object(jobs, "request_json", side_effect=request):
            issues = jobs.check_turbo_support(["http://worker"], "native")
        self.assertTrue(any(NATIVE_TURBO_LORA_NAME in issue for issue in issues))

    def test_int8_worker_preflight_rejects_old_version_even_if_file_exists(self):
        with patch.object(jobs, "verify_asset", return_value=[]), patch.object(jobs, "request_json", return_value={"system": {"comfyui_version": "0.30.0"}}):
            self.assertTrue(jobs.check_video_vae_support("http://worker", "int8"))
        def request(url, **kwargs):
            return {"system": {"comfyui_version": "0.43.0"}} if url.endswith("system_stats") else [VIDEO_VAES["int8"]]
        with patch.object(jobs, "verify_asset", return_value=[]), patch.object(jobs, "request_json", side_effect=request):
            self.assertEqual(jobs.check_video_vae_support("http://worker", "int8"), [])

    def test_canvas_profiles_preserve_balanced_and_match_native_target(self):
        self.assertEqual(resolution_for_size(1920, 1080), (1024, 576))
        self.assertEqual(resolution_for_size(1920, 1080, "native"), (1344, 768))
        self.assertEqual(resolution_for_size(1080, 1920, "native"), (768, 1344))
        self.assertEqual(resolution_for_size(1, 100000, "native")[0], 32)

    def test_old_preferences_keep_balanced_and_new_preference_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"seconds": 10}')
            store = PreferenceStore(path, GenerationSettings())
            self.assertEqual(store.get(1).resolution, "balanced")
            store.update(1, resolution="native")
            restored = PreferenceStore(path, GenerationSettings())
            self.assertEqual(restored.get(1).resolution, "native")
            self.assertEqual(restored.get(2).resolution, "balanced")


class UpgradeDownloadTest(unittest.TestCase):
    def test_service_validate_allows_standard_when_optional_turbo_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "comfy"
            comfy.mkdir()
            (comfy / "comfyui_version.py").write_text('__version__ = "0.43.0"')
            env = dict(os.environ, MINIMAX_ENV_FILE="/dev/null", MINIMAX_BOT_ENV_FILE="",
                       COMFY_DIR=str(comfy), COMFY_PYTHON=sys.executable,
                       MINIMAX_USE_SAGE_ATTENTION="0", MINIMAX_I2V_TURBO_BACKEND="legacy",
                       MINIMAX_REFERENCE_TURBO_BACKEND="legacy", MINIMAX_VIDEO_VAE="fp16",
                       MINIMAX_MODEL_DIR=str(root / "models"), MINIMAX_MODEL_PATHS_CONFIG=str(root / "paths.yaml"))
            script = Path(__file__).resolve().parents[1] / "bin/minimax-h3-service"
            result = subprocess.run(["bash", str(script), "validate"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("standard", result.stderr)
            self.assertTrue((root / "paths.yaml").exists())

    def test_network_failure_keeps_versioned_partial_data_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def snapshot(repo, **kwargs):
                stage = Path(kwargs["local_dir"])
                calls.append(stage)
                fragment = stage / "part"
                if len(calls) == 1:
                    fragment.write_bytes(b"partial")
                    raise OSError("network interrupted")
                self.assertEqual(fragment.read_bytes(), b"partial")
                source = stage / "vae/test.safetensors"
                source.parent.mkdir()
                source.write_bytes(b"verified")
            with patch.dict(download.ASSETS, {"test": ("vae/test.safetensors", sha256(b"verified").hexdigest())}), \
                 patch.dict("sys.modules", {"modelscope": SimpleNamespace(snapshot_download=snapshot)}):
                with self.assertRaisesRegex(OSError, "network interrupted"):
                    download.download_asset("test", root)
                result = download.download_asset("test", root)
                self.assertEqual(result.read_bytes(), b"verified")
                self.assertEqual(calls[0], calls[1])

    def test_shadowed_corrupt_native_weights_block_only_selected_recipe_and_cache_invalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir, comfy_dir = root / "models", root / "comfy"
            relative = "loras/native.safetensors"
            expected = sha256(b"verified").hexdigest()
            good, shadow = model_dir / relative, comfy_dir / "models" / relative
            good.parent.mkdir(parents=True); good.write_bytes(b"verified")
            shadow.parent.mkdir(parents=True); shadow.write_bytes(b"corrupt")
            with patch.dict(assets.ASSETS, {"native-turbo": (relative, expected)}), \
                 patch.object(jobs, "MODEL_DIR", model_dir), patch.object(jobs, "COMFY_DIR", comfy_dir):
                graph, meta = RecipeTest().build(accelerated=False)
                def request(url, **kwargs):
                    if url.endswith("system_stats"):
                        return {"system": {"comfyui_version": "0.43.0"}}
                    if "/object_info/" in url:
                        return {url.rsplit("/", 1)[1]: {}}
                    return [meta["diffusion_model"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", *VIDEO_VAES.values(), "minimax_h3_audio_vae_fp32.safetensors", NATIVE_TURBO_LORA_NAME]
                with patch.object(jobs, "request_json", side_effect=request):
                    self.assertEqual(jobs.check_job_support("http://worker", graph, meta), [])
                    graph, meta = RecipeTest().build(turbo_backend="native")
                    self.assertTrue(any("Checksum mismatch" in issue for issue in jobs.check_job_support("http://worker", graph, meta)))
                    shadow.write_bytes(b"verified")
                    self.assertEqual(jobs.check_job_support("http://worker", graph, meta), [])

    def test_verified_existing_file_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "vae/test.safetensors"
            target.parent.mkdir()
            target.write_bytes(b"verified")
            asset = {"test": ("vae/test.safetensors", sha256(b"verified").hexdigest())}
            with patch.dict(download.ASSETS, asset), patch.dict("sys.modules", {"modelscope": None}):
                self.assertEqual(download.download_asset("test", root), target)

    def test_bad_download_keeps_original_and_direct_connection_restores_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "vae/test.safetensors"
            target.parent.mkdir()
            target.write_bytes(b"original")
            def snapshot(repo, **kwargs):
                self.assertNotIn("HTTPS_PROXY", download.os.environ)
                source = Path(kwargs["local_dir"]) / "vae/test.safetensors"
                source.parent.mkdir()
                source.write_bytes(b"corrupt")
            with patch.dict(download.ASSETS, {"test": ("vae/test.safetensors", "0" * 64)}), \
                 patch.dict("sys.modules", {"modelscope": SimpleNamespace(snapshot_download=snapshot)}), \
                 patch.dict("os.environ", {"HTTPS_PROXY": "http://test.invalid"}):
                with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                    download.download_asset("test", root)
                self.assertEqual(download.os.environ["HTTPS_PROXY"], "http://test.invalid")
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_int8_preflight_rejects_old_comfy_before_loading_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comfyui_version.py").write_text('__version__ = "0.30.0"')
            with patch.object(validate, "COMFY_DIR", root), patch.object(validate, "VIDEO_VAE", "int8"):
                with self.assertRaisesRegex(RuntimeError, "v0.36.0"):
                    validate.validate_assets()
