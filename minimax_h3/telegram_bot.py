#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import re
import secrets
import urllib.request
from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path

from .config import BASE_PORT, BOT_INPUT_DIR, LOG_DIR, PROJECT_DIR
from .preferences import GenerationSettings, PreferenceStore
from .input_sets import InputStore, collection_state, image_entries, image_paths
from .profiles import resolution_for_size

from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeDefault, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest


SERVICE_SCRIPT = PROJECT_DIR / "bin" / "minimax-h3-service"
TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
ALLOWLIST = {int(value) for value in os.environ.get("TG_ALLOWLIST", "").replace(",", " ").split() if value.isdigit()}

from . import jobs as submit


logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger("minimax-h3-telegram")

ACTIVE_IMAGES = {}
UPLOAD_BATCHES = {}
INPUT_EPOCHS = {}
ALBUM_SETTLE_SECONDS = 1.0
PENDING_COUNT = {}
IMAGE_USE_COUNT = {}
RETIRED_IMAGES = set()
PENDING_SERVICE_ACTIONS = {}
SUBMIT_LOCK = None
SETTINGS_PATH = PROJECT_DIR / ".runtime" / "bot_settings.json"
INPUT_STORE = InputStore(PROJECT_DIR / ".runtime" / "bot_inputs.json", BOT_INPUT_DIR)
DIALOGUE_LANGUAGES = {
    "Chinese": "中文",
    "Japanese": "日语",
    "Korean": "韩语",
    "English": "英语",
    "none": "无",
}
DIALOGUE_LANGUAGE_BUTTONS = {
    "Chinese": "🇨🇳",
    "Japanese": "🇯🇵",
    "Korean": "🇰🇷",
    "English": "🇬🇧",
    "none": "🚫",
}
BOT_DURATION_OPTIONS = (5, 10, 15)
H3_CONTROL_TAG_PATTERN = re.compile(
    r"^(?:/?d|scenetrans|cutoff|(?:Picture|Subject|Video|Audio)\s+[1-9][0-9]*)$"
)


def default_generation_settings():
    mode = os.environ.get("MINIMAX_DEFAULT_MODE", "fast").strip().lower()
    mode = mode if mode in {"fast", "standard"} else "fast"
    try:
        seconds = int(os.environ.get("MINIMAX_DEFAULT_SECONDS", "5"))
    except ValueError:
        seconds = 5
    seconds = seconds if seconds in BOT_DURATION_OPTIONS else 5
    language = os.environ.get("MINIMAX_DEFAULT_DIALOGUE_LANGUAGE", "Chinese").strip()
    language = language if language in DIALOGUE_LANGUAGES else "Chinese"
    return GenerationSettings(mode, seconds, language)


PREFERENCES = PreferenceStore(SETTINGS_PATH, default_generation_settings())
TURBO_UNAVAILABLE = False
REFERENCE_TURBO_UNAVAILABLE = False


def generation_settings(user_id):
    settings = PREFERENCES.get(user_id)
    if settings.input_mode == "reference":
        settings = replace(settings, mode=settings.reference_mode)
    unavailable = REFERENCE_TURBO_UNAVAILABLE if settings.input_mode == "reference" else TURBO_UNAVAILABLE
    if unavailable and settings.mode == "fast":
        return replace(settings, mode="standard")
    return settings


def mode_label(mode="fast", task="i2v"):
    if task == "reference":
        fast_label = "768p 试验 · 8 步" if submit.REFERENCE_TURBO_BACKEND == "native" else "Ref2VA · 4 步"
        return f"⚡ 多图参考 Fast（{fast_label}）" if mode == "fast" else "多图参考（Ref2VA · 20 步）"
    backend = "768p 试验 Turbo" if submit.I2V_TURBO_BACKEND == "native" else "Turbo LoRA"
    return f"⚡ 单图首帧 Fast（{backend} · 8 步）" if mode == "fast" else "单图首帧（20 步）"


def settings_label(settings):
    resolution = "原生 768p" if settings.resolution == "native" else "均衡 576p"
    return f"{mode_label(settings.mode, settings.input_mode)} · {settings.seconds} 秒 · {resolution} · 台词：{DIALOGUE_LANGUAGES[settings.dialogue_language]}"


def dialogue_conversion_description(settings):
    if settings.dialogue_language == "none":
        return "自动台词转换已关闭；尖括号内容会保持原样，适合手写多语言 H3 对白标签。"
    return f"每段 <台词> 会自动转换为 H3 {DIALOGUE_LANGUAGES[settings.dialogue_language]}对白格式。"


def allowed(user_id):
    return user_id in ALLOWLIST


def validate_angle_brackets(prompt):
    """Require every angle-bracket span to be complete, non-empty, and unnested."""
    opening = None
    for index, character in enumerate(prompt):
        if character == "<":
            if opening is not None:
                return False
            opening = index
        elif character == ">":
            if opening is None:
                return False
            if not prompt[opening + 1:index].strip():
                return False
            opening = None
    return opening is None


def expand_dialogue_prompt(prompt, language="Chinese"):
    """Convert every simple <dialogue> span to H3 syntax, preserving all other text."""
    if language == "none":
        return prompt, False
    if language not in DIALOGUE_LANGUAGES:
        raise ValueError(f"Unsupported dialogue language: {language}")
    changed = False

    def replace(match):
        nonlocal changed
        dialogue = match.group(1).strip()
        if not dialogue or H3_CONTROL_TAG_PATTERN.fullmatch(dialogue):
            return match.group(0)
        changed = True
        return f"<d>[{language}] {dialogue}</d>"

    expanded = re.sub(r"<([^<>]+)>", replace, prompt)
    return expanded, changed


def user_key(update):
    return update.effective_chat.id, update.effective_user.id


def resolution_for(width, height, profile="balanced"):
    return resolution_for_size(width, height, profile)


def jpeg_dimensions(path):
    """Read JPEG dimensions without adding Pillow to the lightweight bot environment."""
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG image")
        while True:
            byte = stream.read(1)
            while byte and byte != b"\xff":
                byte = stream.read(1)
            if not byte:
                raise ValueError("JPEG has no size marker")
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            marker_value = marker[0]
            if marker_value in {0x01, *range(0xD0, 0xDA)}:
                continue
            segment_length = int.from_bytes(stream.read(2), "big")
            if segment_length < 2:
                raise ValueError("invalid JPEG segment")
            if marker_value in start_of_frame:
                dimensions = stream.read(5)
                return int.from_bytes(dimensions[3:5], "big"), int.from_bytes(dimensions[1:3], "big")
            stream.seek(segment_length - 2, 1)


def restore_private_chat_images():
    """Restore each allowlisted user's newest image after a bot-only restart."""
    if INPUT_STORE.path.exists():
        try:
            states = INPUT_STORE.load(ALLOWLIST)
            ACTIVE_IMAGES.update(states)
            return len(states)
        except (OSError, ValueError):
            log.warning("Could not restore image collections; please re-upload")
            return 0
    restored = 0
    for user_id in ALLOWLIST:
        candidates = list(BOT_INPUT_DIR.glob(f"{user_id}_*.jpg"))
        if not candidates:
            continue
        image_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        try:
            source_width, source_height = jpeg_dimensions(image_path)
        except (OSError, ValueError):
            log.warning("Could not restore bot input %s", image_path)
            continue
        width, height = resolution_for(source_width, source_height)
        ACTIVE_IMAGES[(user_id, user_id)] = collection_state(user_id, user_id, (
            {"image_path": str(image_path), "width": width, "height": height,
             "source_width": source_width, "source_height": source_height},))
        restored += 1
    return restored


def count_keyboard(batch_key):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{count} 条", callback_data=f"count:{batch_key}:{count}")
            for count in range(start, start + 4)
        ]
        for start in (1, 5)
    ])


def mode_keyboard(settings, user_id):
    fast_steps = 4 if settings.input_mode == "reference" and submit.REFERENCE_TURBO_BACKEND != "native" else 8
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"⚡ Fast（{fast_steps} 步）", callback_data=f"settings:{user_id}:mode:fast"),
            InlineKeyboardButton("原始模型（20 步）", callback_data=f"settings:{user_id}:mode:standard"),
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if settings.seconds == seconds else ''}{seconds}s", callback_data=f"settings:{user_id}:seconds:{seconds}")
            for seconds in BOT_DURATION_OPTIONS
        ],
        [
            InlineKeyboardButton(
                f"{'✓ ' if settings.dialogue_language == language else ''}{DIALOGUE_LANGUAGE_BUTTONS[language]}",
                callback_data=f"settings:{user_id}:language:{language}",
            )
            for language in DIALOGUE_LANGUAGES
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if settings.input_mode == 'i2v' else ''}单图首帧", callback_data=f"settings:{user_id}:input_mode:i2v"),
            InlineKeyboardButton(f"{'✓ ' if settings.input_mode == 'reference' else ''}多图参考", callback_data=f"settings:{user_id}:input_mode:reference"),
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if settings.resolution == 'balanced' else ''}均衡 576p", callback_data=f"settings:{user_id}:resolution:balanced"),
            InlineKeyboardButton(f"{'✓ ' if settings.resolution == 'native' else ''}原生 768p（较慢）", callback_data=f"settings:{user_id}:resolution:native"),
        ],
    ])


def service_gpu_keyboard(action, action_key):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{count} 卡", callback_data=f"service:{action}:{action_key}:{count}")
            for count in range(start, start + 4)
        ]
        for start in (1, 5)
    ])


def discard_image(path):
    submit.discard_staged_image(path)
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        log.exception("Could not remove bot input %s", path)


def acquire_image(path):
    path = str(path)
    IMAGE_USE_COUNT[path] = IMAGE_USE_COUNT.get(path, 0) + 1


def release_image(path):
    path = str(path)
    remaining = IMAGE_USE_COUNT.get(path, 0) - 1
    if remaining > 0:
        IMAGE_USE_COUNT[path] = remaining
        return
    IMAGE_USE_COUNT.pop(path, None)
    submit.discard_staged_image(path)
    if path in RETIRED_IMAGES:
        RETIRED_IMAGES.discard(path)
        discard_image(path)


def retire_image(path):
    if not path:
        return
    path = str(path)
    RETIRED_IMAGES.add(path)
    if IMAGE_USE_COUNT.get(path, 0) == 0:
        RETIRED_IMAGES.discard(path)
        discard_image(path)


def discard_pending(predicate):
    discarded = 0
    for pending in (PENDING_COUNT,):
        for batch_key, batch in list(pending.items()):
            if predicate(batch):
                pending.pop(batch_key)
                for path in image_paths(batch):
                    release_image(path)
                discarded += 1
    return discarded


def sampler_progress(server):
    """Read the latest sampler percentage from the worker's tqdm log output."""
    port = int(server.rsplit(":", 1)[1])
    gpu = port - BASE_PORT
    log_path = LOG_DIR / f"minimax-h3-service-gpu{gpu}.log"
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 2 * 1024 * 1024))
            data = stream.read()
    except OSError:
        return None
    starts = list(re.finditer(rb"(?<!\d)0%\|", data))
    if not starts:
        return None
    percentages = re.findall(rb"(\d{1,3})%\|", data[starts[-1].start():])
    return int(percentages[-1]) if percentages else None


def service_progress_snapshot():
    snapshot = []
    for server in submit.check_servers():
        queue = submit.request_json(f"{server}/queue")
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        snapshot.append({
            "server": server,
            "gpu": int(server.rsplit(":", 1)[1]) - BASE_PORT,
            "running": running,
            "pending": pending,
            "percent": sampler_progress(server) if running else None,
        })
    return snapshot


def post_empty_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}")


def stop_all_service_jobs():
    stopped = []
    errors = []
    for server in submit.check_servers():
        try:
            queue = submit.request_json(f"{server}/queue")
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
            post_empty_json(f"{server}/queue", {"clear": True})
            if running:
                post_empty_json(f"{server}/interrupt", {})
            stopped.append({"server": server, "running": running, "pending": pending})
        except Exception as error:
            errors.append(f"{server}: {error}")
    return stopped, errors


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings = generation_settings(update.effective_user.id)
    status = "你已在使用白名单中。" if allowed(update.effective_user.id) else f"你的 ID 是 {update.effective_user.id}，目前不在使用白名单中。"
    await update.message.reply_text(
        "MiniMax H3 图生视频 Bot\n\n"
        "1. 默认先发一张图片作为首帧；多图请先在 /mode 选择多图参考，再发送相册或逐张上传（最多 9 张）。\n"
        "2. 收到提醒后发送一个提示词；整条消息视为同一个提示词。\n"
        f"   {dialogue_conversion_description(settings)}\n"
        f"3. 你的当前设置：{settings_label(settings)}；使用 /mode 修改模型、时长和自动台词语言，/language 可直接打开同一设置面板。\n"
        "4. 选择生成 1–8 条；也可在提示词末尾写 xN（如 x12）直接指定任意正整数数量。\n"
        "5. 每条视频生成好后会立即发回，并显示生成和排队耗时。\n"
        "6. 多图参考会追加图片；单图首帧会替换图片。/images 查看编号、删除与排序，/images clear 清空；等待编号预览后再发提示词。\n"
        "7. 使用 /progress 查看各 GPU worker 的生成和排队进度。\n"
        "8. 使用 /stop 中断所有生成并清空队列。\n"
        "9. 使用 /service_start、/service_stop、/service_restart 管理 GPU worker。\n"
        "10. 所有任务固定不生成背景音乐；环境声、人物声音和对白仍可在提示词中指定。\n\n" + status
    )


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = "已在白名单" if allowed(update.effective_user.id) else "不在白名单"
    await update.message.reply_text(f"你的 Telegram ID：{update.effective_user.id}\n{status}")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    settings = generation_settings(update.effective_user.id)
    await update.message.reply_text(
        f"你的 Bot 设置：{settings_label(settings)}\n{dialogue_conversion_description(settings)}\n修改后将用于随后收到的新提示词。",
        reply_markup=mode_keyboard(settings, update.effective_user.id),
    )


async def on_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TURBO_UNAVAILABLE, REFERENCE_TURBO_UNAVAILABLE
    query = update.callback_query
    if not allowed(update.effective_user.id):
        await query.answer("你不在使用白名单中", show_alert=True)
        return
    parts = query.data.split(":")
    if len(parts) != 4:
        await query.answer("设置面板已失效，请重新发送 /mode。", show_alert=True)
        return
    _, owner, setting, value = parts
    if owner != str(update.effective_user.id):
        await query.answer("这不是你的设置面板，请使用 /mode。", show_alert=True)
        return
    fields = {"mode": ("mode", {"fast", "standard"}), "seconds": ("seconds", {"5", "10", "15"}),
              "language": ("dialogue_language", set(DIALOGUE_LANGUAGES)),
              "input_mode": ("input_mode", {"i2v", "reference"}),
              "resolution": ("resolution", {"balanced", "native"})}
    if setting not in fields or value not in fields[setting][1]:
        await query.answer("无效的设置选项", show_alert=True)
        return
    if setting == "input_mode" and user_key(update) in UPLOAD_BATCHES:
        await query.answer("图片收集中，请等待编号预览后再切换模式。", show_alert=True)
        return
    if setting == "mode" and value == "fast":
        try:
            reference = PREFERENCES.get(update.effective_user.id).input_mode == "reference"
            check = submit.check_reference_support if reference else submit.check_turbo_support
            turbo_issues = await asyncio.to_thread(check, accelerated=True) if reference else await asyncio.to_thread(check)
        except Exception as error:
            turbo_issues = [str(error)]
        if turbo_issues:
            await query.answer("Fast 组件当前不可用", show_alert=True)
            await query.edit_message_text("无法切换到 Fast：\n" + "\n".join(turbo_issues))
            return
        if reference:
            REFERENCE_TURBO_UNAVAILABLE = False
        else:
            TURBO_UNAVAILABLE = False
    field = fields[setting][0]
    if setting == "mode" and PREFERENCES.get(update.effective_user.id).input_mode == "reference":
        field = "reference_mode"
    PREFERENCES.update(update.effective_user.id, **{field: int(value) if setting == "seconds" else value})
    settings = generation_settings(update.effective_user.id)
    await query.answer("你的设置已修改")
    await query.edit_message_text(
        f"你的 Bot 设置：{settings_label(settings)}\n{dialogue_conversion_description(settings)}\n可继续修改其他选项。",
        reply_markup=mode_keyboard(settings, update.effective_user.id),
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    discard_pending(
        lambda batch: batch["user_id"] == update.effective_user.id
        and batch["chat_id"] == update.effective_chat.id
    )
    await update.message.reply_text("已取消尚未提交的选项。当前图片集合仍会继续使用。")


async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await send_text_with_retry(context.bot, update.effective_chat.id, "你不在使用白名单中。")
        return
    try:
        snapshot = await asyncio.to_thread(service_progress_snapshot)
    except Exception as error:
        log.exception("Could not query service progress")
        await send_text_with_retry(context.bot, update.effective_chat.id, f"查询进度失败：{error}")
        return

    running = sum(item["running"] for item in snapshot)
    pending = sum(item["pending"] for item in snapshot)
    choosing = sum(
        batch["user_id"] == update.effective_user.id and batch["chat_id"] == update.effective_chat.id
        for batch in PENDING_COUNT.values()
    )
    lines = [f"任务进度：运行中 {running}，排队 {pending}，等待选择 {choosing}"]
    for item in snapshot:
        if not item["running"]:
            state = "空闲"
        elif item["percent"] is None or item["percent"] == 0:
            state = "模型加载或开始采样"
        elif item["percent"] >= 100:
            state = "采样完成，正在解码或保存"
        else:
            state = f"采样 {item['percent']}%"
        if item["pending"]:
            state += f"，后面排队 {item['pending']}"
        lines.append(f"GPU {item['gpu']}：{state}")
    await send_text_with_retry(context.bot, update.effective_chat.id, "\n".join(lines))


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await send_text_with_retry(context.bot, update.effective_chat.id, "你不在使用白名单中。")
        return

    discard_pending(lambda batch: True)
    try:
        async with SUBMIT_LOCK:
            stopped, errors = await asyncio.to_thread(stop_all_service_jobs)
    except Exception as error:
        log.exception("Could not stop service jobs")
        await send_text_with_retry(context.bot, update.effective_chat.id, f"停止任务失败：{error}")
        return

    running = sum(item["running"] for item in stopped)
    pending = sum(item["pending"] for item in stopped)
    message = f"已发送停止指令：中断 {running} 个运行中任务，清除 {pending} 个排队任务，并取消所有未完成的选项。"
    if errors:
        message += "\n部分 worker 操作失败：\n" + "\n".join(errors)
    await send_text_with_retry(context.bot, update.effective_chat.id, message)


async def run_worker_control(action, gpu_count=None):
    command = [str(SERVICE_SCRIPT), f"workers-{action}"]
    if gpu_count is not None:
        command.append(str(gpu_count))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=150)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return False, "操作等待超时，请查看服务日志确认 worker 状态。"
    output = stdout.decode("utf-8", "replace").strip()
    return process.returncode == 0, output


async def request_service_gpu_count(update: Update, action):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    for action_key, pending in list(PENDING_SERVICE_ACTIONS.items()):
        if pending["user_id"] == update.effective_user.id and pending["chat_id"] == update.effective_chat.id:
            PENDING_SERVICE_ACTIONS.pop(action_key, None)
    action_key = secrets.token_hex(6)
    PENDING_SERVICE_ACTIONS[action_key] = {
        "action": action,
        "user_id": update.effective_user.id,
        "chat_id": update.effective_chat.id,
    }
    warning = "\n重启会中断当前正在生成和排队的任务。" if action == "restart" else ""
    await update.message.reply_text(
        f"请选择要使用的显卡数量：{warning}",
        reply_markup=service_gpu_keyboard(action, action_key),
    )


async def service_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await request_service_gpu_count(update, "start")


async def service_restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await request_service_gpu_count(update, "restart")


async def service_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    message = await update.message.reply_text("正在停止 GPU worker，Bot 会继续保持在线……")
    async with SUBMIT_LOCK:
        success, detail = await run_worker_control("stop")
    if success:
        text = "GPU worker 已停止，Bot 仍在线。需要恢复时使用 /service_start。"
    else:
        text = f"停止 GPU worker 失败：\n{detail or '服务脚本没有返回详细信息。'}"
    try:
        await message.edit_text(text)
    except (NetworkError, TimedOut):
        await send_text_with_retry(context.bot, update.effective_chat.id, text)


async def on_service_gpu_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    _, action, action_key, count_text = query.data.split(":", 3)
    pending = PENDING_SERVICE_ACTIONS.get(action_key)
    if pending is None or pending["action"] != action:
        await query.answer("这个选择已经失效", show_alert=True)
        return
    if pending["user_id"] != update.effective_user.id or pending["chat_id"] != update.effective_chat.id:
        await query.answer("这不是你的操作", show_alert=True)
        return
    PENDING_SERVICE_ACTIONS.pop(action_key, None)
    gpu_count = int(count_text)
    verb = "启动" if action == "start" else "重启"
    await query.answer(f"正在{verb} {gpu_count} 张卡")
    try:
        await query.edit_message_text(f"正在{verb} {gpu_count} 个 GPU worker，请稍候……")
    except (NetworkError, TimedOut):
        pass
    async with SUBMIT_LOCK:
        success, detail = await run_worker_control(action, gpu_count)
    if success:
        text = f"已按 {gpu_count} 张卡{verb}服务。\n{detail}" if detail else f"已按 {gpu_count} 张卡{verb}服务。"
    else:
        text = f"服务{verb}失败：\n{detail or '服务脚本没有返回详细信息。'}"
    await send_text_with_retry(context.bot, update.effective_chat.id, text)


def replace_collection(key, images):
    previous = ACTIVE_IMAGES.get(key)
    states = dict(ACTIVE_IMAGES)
    if images:
        states[key] = collection_state(key[0], key[1], images)
    else:
        states.pop(key, None)
    INPUT_STORE.save(states)
    ACTIVE_IMAGES.clear()
    ACTIVE_IMAGES.update(states)
    keep = set(image_paths(states[key])) if key in states else set()
    if previous:
        for path in image_paths(previous):
            if path not in keep:
                retire_image(path)
    return states.get(key)


def images_keyboard(state):
    prefix = f"images:{state['user_id']}:{state['revision']}"
    rows = []
    for index in range(len(image_paths(state))):
        row = [InlineKeyboardButton(f"删除图 {index + 1}", callback_data=f"{prefix}:remove:{index}")]
        if index:
            row.append(InlineKeyboardButton(f"图 {index + 1} 上移", callback_data=f"{prefix}:up:{index}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("清空图片", callback_data=f"{prefix}:clear:0")])
    return InlineKeyboardMarkup(rows)


async def show_images(context, key, state):
    paths = image_paths(state)
    for path in paths:
        acquire_image(path)
    try:
        with ExitStack() as stack:
            media = [InputMediaPhoto(stack.enter_context(open(path, "rb")),
                                     caption=f"🔒 <Picture {index + 1}>（点击查看）", has_spoiler=True)
                     for index, path in enumerate(paths)]
            if len(media) == 1:
                await context.bot.send_photo(key[0], photo=media[0].media,
                                             caption=media[0].caption, has_spoiler=True)
            else:
                await context.bot.send_media_group(key[0], media=media)
        settings = generation_settings(key[1])
        text = (f"当前 {len(paths)} 张图片 · {settings_label(settings)}\n"
                f"画布 {state['width']}×{state['height']}，按图 1 比例确定；更换图 1 会改变画布比例。\n"
                "编号按上方预览顺序对应。用 <Picture 1>、<Picture 2> 明确各图用途；不要写 <图1>。\n"
                "多图参考会追加新图片，最多 9 张；单图首帧会用新图片替换。用下方按钮删除、上移或清空。\n"
                "请发送提示词；整条消息作为一个任务，背景音乐关闭。")
        await send_text_with_retry(context.bot, key[0], text, reply_markup=images_keyboard(state))
    except (OSError, NetworkError, TimedOut):
        log.warning("Could not show input previews")
        await send_text_with_retry(context.bot, key[0], "图片已保存，预览发送失败；请用 /images 重新查看编号。")
    finally:
        for path in paths:
            release_image(path)


async def images_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    key = user_key(update)
    if context.args == ["clear"]:
        replace_collection(key, ())
        INPUT_EPOCHS[key] = INPUT_EPOCHS.get(key, 0) + 1
        UPLOAD_BATCHES.pop(key, None)
        await update.message.reply_text("图片已清空；已收到提示词的任务仍使用原图片快照。")
        return
    if key in UPLOAD_BATCHES:
        await update.message.reply_text("图片收集中，请等待编号预览。")
        return
    state = ACTIVE_IMAGES.get(key)
    if state is None:
        await update.message.reply_text("请先上传图片；多图参考请先在 /mode 中选择。")
        return
    await show_images(context, key, state)


async def on_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    parts = query.data.split(":")
    key = user_key(update)
    state = ACTIVE_IMAGES.get(key)
    if not allowed(key[1]) or len(parts) != 5 or parts[1] != str(key[1]):
        await query.answer("这不是你的图片面板", show_alert=True)
        return
    if state is None or parts[2] != state.get("revision"):
        await query.answer("图片顺序已改变，请使用 /images 打开新面板。", show_alert=True)
        return
    if key in UPLOAD_BATCHES:
        await query.answer("图片收集中，请稍后操作。", show_alert=True)
        return
    action, index = parts[3], int(parts[4])
    images = list(image_entries(state))
    if action == "clear":
        images = []
    elif 0 <= index < len(images) and action == "remove":
        images.pop(index)
    elif 0 < index < len(images) and action == "up":
        images[index - 1], images[index] = images[index], images[index - 1]
    else:
        await query.answer("无效的图片操作", show_alert=True)
        return
    try:
        new_state = replace_collection(key, images)
    except OSError:
        await query.answer("图片设置保存失败，请重试。", show_alert=True)
        return
    paths = image_paths(new_state) if new_state else ()
    for path in paths:
        acquire_image(path)
    try:
        await query.answer("图片集合已更新")
        if new_state:
            await show_images(context, key, new_state)
        else:
            await query.edit_message_text("图片已清空；已收到提示词的任务仍使用原图片快照。")
    finally:
        for path in paths:
            release_image(path)


async def flush_uploads(context, key, batch):
    try:
        loop = asyncio.get_running_loop()
        while batch["pending"] or loop.time() - batch["touched"] < ALBUM_SETTLE_SECONDS:
            await asyncio.sleep(0.1)
        if batch["epoch"] != INPUT_EPOCHS.get(key, 0):
            return
        images = [batch["images"][index] for index in sorted(batch["images"])]
        if not images:
            return
        if batch["input_mode"] == "i2v":
            if batch["album"]:
                await send_text_with_retry(context.bot, key[0], "单图首帧模式不接收相册。请先在 /mode 选择多图参考，再发送相册；这批图片未加入。")
                return
            images = images[-1:]
        elif key in ACTIVE_IMAGES:
            images = list(image_entries(ACTIVE_IMAGES[key])) + images
        if len(images) > submit.MAX_REFERENCE_IMAGES:
            await send_text_with_retry(context.bot, key[0], "最多支持 9 张参考图；这批图片未加入，请用 /images 删除或清空后重发。")
            return
        state = replace_collection(key, images)
        batch["committed"] = set(image_paths(state))
        batch["preview_paths"] = image_paths(state)
        for path in batch["preview_paths"]:
            acquire_image(path)
        if UPLOAD_BATCHES.get(key) is batch:
            UPLOAD_BATCHES.pop(key)
        # Preserve original messages if the batch is rejected or cannot be saved.
        for message in batch["messages"]:
            try:
                await message.delete()
            except Exception:
                log.warning("Could not delete original input message")
        await show_images(context, key, state)
    except OSError:
        log.exception("Could not save input collection")
        await send_text_with_retry(context.bot, key[0], "图片保存失败，原消息已保留，请重试。")
    finally:
        if UPLOAD_BATCHES.get(key) is batch:
            UPLOAD_BATCHES.pop(key)
        for path in batch.get("preview_paths", ()):
            release_image(path)
        for image in batch["images"].values():
            if image["image_path"] not in batch.get("committed", set()):
                discard_image(image["image_path"])


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    key = user_key(update)
    loop = asyncio.get_running_loop()
    message_id = update.message.message_id
    if key in ACTIVE_IMAGES and any(image.get("message_id") == message_id for image in image_entries(ACTIVE_IMAGES[key])):
        return
    batch = UPLOAD_BATCHES.get(key)
    if batch is None:
        batch = {"images": {}, "seen": set(), "messages": [], "pending": 0,
                 "input_mode": PREFERENCES.get(key[1]).input_mode, "album": False,
                 "epoch": INPUT_EPOCHS.get(key, 0), "touched": loop.time()}
        UPLOAD_BATCHES[key] = batch
        context.application.create_task(flush_uploads(context, key, batch))
    if message_id in batch["seen"]:
        return
    batch["seen"].add(message_id)
    batch["pending"] += 1
    batch["album"] |= bool(update.message.media_group_id)
    batch["touched"] = loop.time()
    image_path = BOT_INPUT_DIR / f"{key[1]}_{secrets.token_hex(8)}.jpg"
    try:
        photo = update.message.photo[-1]
        width, height = resolution_for(photo.width, photo.height)
        BOT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                telegram_file = await photo.get_file()
                await telegram_file.download_to_drive(image_path)
                break
            except (NetworkError, TimedOut):
                if attempt == 2:
                    await update.message.reply_text("图片下载超时，请稍后重新发送；这张图片未加入。")
                    return
                await asyncio.sleep(2 ** attempt)
        batch["images"][message_id] = {"image_path": str(image_path), "width": width, "height": height,
                                       "source_width": photo.width, "source_height": photo.height,
                                       "message_id": message_id}
        batch["messages"].append(update.message)
    except (OSError, ValueError):
        log.exception("Could not save incoming photo")
        await update.message.reply_text("图片保存失败；这张图片未加入，请重新发送。")
    finally:
        batch["pending"] -= 1
        batch["touched"] = loop.time()
        if message_id not in batch["images"]:
            discard_image(image_path)


async def on_prompt_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text("你不在使用白名单中。")
        return
    settings = generation_settings(update.effective_user.id)
    key = user_key(update)
    if key in UPLOAD_BATCHES:
        await update.message.reply_text("图片收集中，请等待编号预览后再发送提示词。")
        return
    state = ACTIVE_IMAGES.get(key)
    if state is None:
        await update.message.reply_text("请先发送一张图片。")
        return
    paths = image_paths(state)
    if settings.input_mode == "i2v" and len(paths) != 1:
        await update.message.reply_text("单图首帧只能使用一张图。请在 /mode 切回多图参考，或用 /images 删除多余图片。")
        return

    raw_prompt, requested_count = submit.extract_prompt_count(update.message.text)
    if not raw_prompt:
        await send_text_with_retry(context.bot, update.effective_chat.id, "没有读到有效提示词，请重新输入。")
        return
    if any(not 1 <= int(number) <= len(paths) for number in re.findall(r"<Picture\s+([0-9]+)>", raw_prompt)):
        await update.message.reply_text("提示词引用了尚未上传的图片编号，请用 /images 核对。")
        return
    if re.search(r"<(?:Video|Audio)\s+[1-9][0-9]*>", raw_prompt):
        await update.message.reply_text("当前 Bot 尚未接入参考视频或参考音频；请先使用图片参考，不能只靠标签传入视频或音频。")
        return
    if settings.dialogue_language != "none" and not validate_angle_brackets(raw_prompt):
        await send_text_with_retry(
            context.bot,
            update.effective_chat.id,
            "检测到未严格匹配的 < >。请确保每段台词都写成完整的 <台词>，不能缺少、多余、嵌套或留空。也可以在 /language 中选择“无”后手写 H3 对白标签。",
        )
        return
    prompt, dialogue_expanded = expand_dialogue_prompt(raw_prompt, settings.dialogue_language)

    batch_key = secrets.token_hex(6)
    state = dict(state)
    state["prompt"] = prompt
    state["display_prompt"] = raw_prompt
    state["requested_count"] = requested_count
    state["image_paths"] = paths
    state["task"] = settings.input_mode
    state["accelerated"] = settings.mode == "fast"
    state["seconds"] = settings.seconds
    first_image = image_entries(state)[0]
    state["width"], state["height"] = resolution_for(
        first_image.get("source_width", first_image["width"]),
        first_image.get("source_height", first_image["height"]), settings.resolution)
    state["resolution"] = settings.resolution
    state["turbo_backend"] = submit.REFERENCE_TURBO_BACKEND if settings.input_mode == "reference" else submit.I2V_TURBO_BACKEND
    state["video_vae"] = submit.VIDEO_VAE
    for path in paths:
        acquire_image(path)
    selected_mode = "fast" if state["accelerated"] else "standard"
    if requested_count is not None:
        context.application.create_task(
            submit_batch(context.application, state, state["seconds"], requested_count)
        )
        await send_text_with_retry(
            context.bot,
            update.effective_chat.id,
            (f"已自动转换其中的{DIALOGUE_LANGUAGES[settings.dialogue_language]}台词。" if dialogue_expanded else "提示词已收到。")
            + f"使用 {mode_label(selected_mode, state['task'])} · {len(paths)} 张图 · {state['seconds']} 秒 · 共 {requested_count} 条；"
            "正在提交，每条使用不同随机 seed……",
        )
        return

    PENDING_COUNT[batch_key] = state
    await send_text_with_retry(
        context.bot,
        update.effective_chat.id,
        (f"已自动转换其中的{DIALOGUE_LANGUAGES[settings.dialogue_language]}台词。" if dialogue_expanded else "提示词已收到。")
        + f"当前使用 {mode_label(selected_mode, state['task'])} · {len(paths)} 张图 · {state['seconds']} 秒。请选择生成多少条视频：",
        reply_markup=count_keyboard(batch_key),
    )


async def send_text_with_retry(bot, chat_id, text, attempts=5, **kwargs):
    delay = 2
    for attempt in range(attempts):
        try:
            return await bot.send_message(chat_id, text, **kwargs)
        except (NetworkError, TimedOut):
            if attempt == attempts - 1:
                log.warning("Telegram text delivery failed after %d attempts", attempts)
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)


def format_elapsed(seconds):
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours} 小时")
    if minutes:
        parts.append(f"{minutes} 分")
    if secs or not parts:
        parts.append(f"{secs} 秒")
    return " ".join(parts)


async def deliver_video(application, chat_id, record, output_path):
    mode = mode_label("fast" if record.get("accelerated") else "standard", record.get("task", "i2v"))
    caption_lines = [f"完成 · {mode} · 视频 {record['actual_seconds']:.3f} 秒 · seed {record['seed']}"]
    if all(key in record for key in ("width", "height", "steps", "video_vae")):
        caption_lines.append(f"{record['width']}×{record['height']} · {record['steps']} 步 · VAE {record['video_vae'].upper()}")
    if record.get("render_seconds") is not None:
        timing = f"生成耗时 {format_elapsed(record['render_seconds'])}"
        if record.get("queue_seconds") is not None:
            timing += f" · 排队 {format_elapsed(record['queue_seconds'])}"
        caption_lines.append(timing)
    caption_lines.append(record["prompt_text"])
    caption = "\n".join(caption_lines)
    delay = 2
    while True:
        try:
            with open(output_path, "rb") as video:
                await application.bot.send_video(
                    chat_id,
                    video=video,
                    caption=caption[:1024],
                    supports_streaming=True,
                    has_spoiler=True,
                    read_timeout=120,
                    write_timeout=120,
                    connect_timeout=30,
                )
            log.info("VIDEO_DELIVERED file=%s seed=%d", Path(output_path).name, record["seed"])
            try:
                Path(output_path).unlink(missing_ok=True)
                log.info("VIDEO_REMOVED file=%s", Path(output_path).name)
            except OSError:
                log.exception("Could not remove delivered video %s", output_path)
            return
        except (NetworkError, TimedOut):
            log.warning("Video upload interrupted; retrying file=%s in %ds", Path(output_path).name, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as error:
            log.error("Video delivery failed permanently file=%s error=%s", Path(output_path).name, error)
            await send_text_with_retry(application.bot, chat_id, f"视频发送失败，seed：{record['seed']}，文件已保留在服务器。")
            return


async def wait_for_job(record):
    while True:
        result = await asyncio.to_thread(submit.poll_job, record)
        if result is not None:
            return result
        await asyncio.sleep(3)


async def submit_batch(application, batch, seconds, count):
    chat_id = batch["chat_id"]
    records = []
    try:
        try:
            async with SUBMIT_LOCK:
                servers = await asyncio.to_thread(submit.check_servers)
                submit.SERVERS[:] = servers
                seeds = set()
                while len(seeds) < count:
                    seeds.add(secrets.randbelow(2 ** 64))
                for index, seed in enumerate(seeds, 1):
                    job = {
                        "image_path": batch["image_path"],
                        "prompt": batch["prompt"],
                        "seed": seed,
                        "seconds": seconds,
                        "steps": 20,
                        "accelerated": batch["accelerated"],
                        "width": batch["width"],
                        "height": batch["height"],
                        "task": batch.get("task", "i2v"),
                        "resolution": batch.get("resolution", "balanced"),
                        "video_vae": batch.get("video_vae", submit.VIDEO_VAE),
                    }
                    if job["task"] == "reference":
                        job["image_paths"] = list(image_paths(batch))
                    job["turbo_backend"] = batch.get("turbo_backend", submit.REFERENCE_TURBO_BACKEND if job["task"] == "reference" else submit.I2V_TURBO_BACKEND)
                    record = await asyncio.to_thread(submit.submit_job, job, index)
                    record["prompt_text"] = batch.get("display_prompt", batch["prompt"])
                    records.append(record)
        except Exception as error:
            log.exception("Batch submission failed")
            await send_text_with_retry(application.bot, chat_id, f"部分任务提交失败：{error}")

        if not records:
            return
        mode = mode_label("fast" if batch["accelerated"] else "standard", batch.get("task", "i2v"))
        log.info(
            "BATCH_SUBMITTED count=%d seconds=%d accelerated=%s workers=%d",
            len(records),
            seconds,
            batch["accelerated"],
            len(submit.SERVERS),
        )
        await send_text_with_retry(
            application.bot,
            chat_id,
            f"已提交 {len(records)} 个任务 · {mode} · 使用 {len(submit.SERVERS)} 个 worker。每完成一个就会立即发回并显示耗时。",
        )

        tasks = [asyncio.create_task(wait_for_job(record)) for record in records]
        delivery_tasks = []
        for completed_task in asyncio.as_completed(tasks):
            try:
                record = await completed_task
                if record["status"] != "completed" or not record.get("outputs"):
                    await send_text_with_retry(application.bot, chat_id, f"有一个任务生成失败，seed：{record['seed']}")
                    continue
                for output_path in record["outputs"]:
                    log.info("VIDEO_READY file=%s seed=%d", Path(output_path).name, record["seed"])
                    delivery_tasks.append(asyncio.create_task(deliver_video(application, chat_id, record, output_path)))
            except Exception as error:
                log.exception("Rendering failed")
                await send_text_with_retry(application.bot, chat_id, f"视频生成失败：{error}")
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks)
    finally:
        for path in image_paths(batch):
            release_image(path)


async def on_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    _, batch_key, count_text = query.data.split(":", 2)
    batch = PENDING_COUNT.get(batch_key)
    if batch is None:
        await query.answer("这个选择已经失效", show_alert=True)
        return
    if batch["user_id"] != update.effective_user.id or batch["chat_id"] != update.effective_chat.id:
        await query.answer("这不是你的任务", show_alert=True)
        return
    batch = PENDING_COUNT.pop(batch_key)
    count = int(count_text)
    context.application.create_task(submit_batch(context.application, batch, batch["seconds"], count))
    selected_mode = "fast" if batch["accelerated"] else "standard"
    try:
        await query.answer(f"将生成 {count} 条")
        await query.edit_message_text(
            f"已选择 {batch['seconds']} 秒 · {mode_label(selected_mode, batch.get('task', 'i2v'))} · 共 {count} 条；"
            "正在提交，每条使用不同随机 seed……"
        )
    except (NetworkError, TimedOut):
        log.warning("Count acknowledgement failed, but batch submission continues")


async def post_init(application):
    global TURBO_UNAVAILABLE, REFERENCE_TURBO_UNAVAILABLE, SUBMIT_LOCK
    SUBMIT_LOCK = asyncio.Lock()
    commands = [
        BotCommand("start", "使用说明"),
        BotCommand("mode", "修改你的模型、时长和台词语言"),
        BotCommand("language", "修改自动台词语言"),
        BotCommand("images", "查看、排序或清空参考图片"),
        BotCommand("progress", "查看生成和排队进度"),
        BotCommand("id", "查看 Telegram ID"),
        BotCommand("cancel", "取消尚未提交的任务"),
        BotCommand("service_start", "选择卡数并启动 GPU 服务"),
        BotCommand("service_stop", "停止 GPU 服务并保持 Bot 在线"),
        BotCommand("service_restart", "选择卡数并重启 GPU 服务"),
        BotCommand("stop", "危险：停止全部生成并清空队列"),
    ]
    for scope in (BotCommandScopeDefault(), BotCommandScopeAllPrivateChats()):
        await application.bot.delete_my_commands(scope=scope)
        await application.bot.set_my_commands(commands, scope=scope)
    bot = await application.bot.get_me()
    restored = restore_private_chat_images()
    try:
        turbo_issues = await asyncio.to_thread(submit.check_turbo_support)
    except Exception as error:
        turbo_issues = [str(error)]
    if turbo_issues:
        log.warning("TURBO_PREFLIGHT_FAILED %s", " | ".join(turbo_issues))
        TURBO_UNAVAILABLE = True
        log.warning("GENERATION_MODE_FALLBACK mode=standard preferences_preserved=true")
    else:
        TURBO_UNAVAILABLE = False
        log.info("TURBO_PREFLIGHT_OK lora=%s", submit.TURBO_LORA_NAME)
    try:
        reference_issues = await asyncio.to_thread(submit.check_reference_support, accelerated=True)
    except Exception as error:
        reference_issues = [str(error)]
    REFERENCE_TURBO_UNAVAILABLE = bool(reference_issues)
    if reference_issues:
        log.warning("REFERENCE_TURBO_PREFLIGHT_FAILED %s", " | ".join(reference_issues))
    log.info("BOT_READY username=%s allowlist_size=%d restored_images=%d", bot.username, len(ALLOWLIST), restored)


async def on_error(update, context):
    log.error("Telegram update failed: %s: %s", type(context.error).__name__, context.error)


def main():
    if not TOKEN:
        raise RuntimeError("TG_BOT_TOKEN is not configured")
    request = HTTPXRequest(connection_pool_size=16, connect_timeout=30, read_timeout=60, write_timeout=60, pool_timeout=30, media_write_timeout=120)
    get_updates_request = HTTPXRequest(connection_pool_size=2, connect_timeout=30, read_timeout=70, write_timeout=60, pool_timeout=30)
    application = Application.builder().token(TOKEN).request(request).get_updates_request(get_updates_request).concurrent_updates(True).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("language", mode_command))
    application.add_handler(CommandHandler("images", images_command))
    application.add_handler(CommandHandler("id", id_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("progress", progress_command))
    application.add_handler(CommandHandler("status", progress_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("service_start", service_start_command))
    application.add_handler(CommandHandler("service_stop", service_stop_command))
    application.add_handler(CommandHandler("service_restart", service_restart_command))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_prompt_text))
    application.add_handler(CallbackQueryHandler(on_mode, pattern=r"^settings:(?:[1-9][0-9]*:)?(mode:(fast|standard)|seconds:(5|10|15)|language:(Chinese|Japanese|Korean|English|none)|input_mode:(i2v|reference))$"))
    application.add_handler(CallbackQueryHandler(on_images, pattern=r"^images:[1-9][0-9]*:[0-9a-f]{8}:(remove|up|clear):[0-8]$"))
    application.add_handler(CallbackQueryHandler(on_count, pattern=r"^count:[0-9a-f]+:[1-8]$"))
    application.add_handler(CallbackQueryHandler(on_service_gpu_count, pattern=r"^service:(start|restart):[0-9a-f]+:[1-8]$"))
    application.add_error_handler(on_error)
    application.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1)


if __name__ == "__main__":
    main()
