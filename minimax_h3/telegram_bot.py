#!/usr/bin/env python3
import asyncio
import json
import logging
import math
import os
import re
import secrets
import urllib.request
from pathlib import Path

from .config import BASE_PORT, BOT_INPUT_DIR, LOG_DIR, PROJECT_DIR

from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeDefault, InlineKeyboardButton, InlineKeyboardMarkup, Update
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
PENDING_COUNT = {}
IMAGE_USE_COUNT = {}
RETIRED_IMAGES = set()
PENDING_SERVICE_ACTIONS = {}
SUBMIT_LOCK = None
SETTINGS_PATH = PROJECT_DIR / ".runtime" / "bot_settings.json"


def load_generation_settings():
    default = os.environ.get("MINIMAX_DEFAULT_MODE", "fast").strip().lower()
    default = default if default in {"fast", "standard"} else "fast"
    try:
        default_seconds = int(os.environ.get("MINIMAX_DEFAULT_SECONDS", "5"))
    except ValueError:
        default_seconds = 5
    default_seconds = default_seconds if default_seconds in {5, 10, 15} else 5
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default, default_seconds
    mode = data.get("generation_mode") if data.get("generation_mode") in {"fast", "standard"} else default
    seconds = data.get("seconds") if data.get("seconds") in {5, 10, 15} else default_seconds
    return mode, seconds


def save_generation_settings():
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"generation_mode": GENERATION_MODE, "seconds": GENERATION_SECONDS},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(SETTINGS_PATH)


GENERATION_MODE, GENERATION_SECONDS = load_generation_settings()


def mode_label(mode=None):
    return "⚡ Fast（Turbo LoRA · 8 步）" if (mode or GENERATION_MODE) == "fast" else "原始模型（20 步）"


def settings_label():
    return f"{mode_label()} · {GENERATION_SECONDS} 秒"


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


def expand_dialogue_prompt(prompt):
    """Convert every simple <dialogue> span to H3 syntax, preserving all other text."""
    changed = False

    def replace(match):
        nonlocal changed
        dialogue = match.group(1).strip()
        if not dialogue or dialogue in {"d", "/d"} or dialogue.startswith("Picture"):
            return match.group(0)
        changed = True
        return f"<d>[Chinese] {dialogue}</d>"

    expanded = re.sub(r"<([^<>]+)>", replace, prompt)
    return expanded, changed


def user_key(update):
    return update.effective_chat.id, update.effective_user.id


def resolution_for(width, height):
    ratio = width / height
    if ratio >= 1:
        target_width, target_height = 576 * ratio, 576
    else:
        target_width, target_height = 576, 576 / ratio
    if target_width * target_height > 576 * 1024:
        scale = math.sqrt((576 * 1024) / (target_width * target_height))
        target_width *= scale
        target_height *= scale
    return round(target_width / 32) * 32, round(target_height / 32) * 32


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
    restored = 0
    for user_id in ALLOWLIST:
        candidates = list(BOT_INPUT_DIR.glob(f"{user_id}_*.jpg"))
        if not candidates:
            continue
        image_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        for stale_path in candidates:
            if stale_path != image_path:
                discard_image(stale_path)
        try:
            source_width, source_height = jpeg_dimensions(image_path)
        except (OSError, ValueError):
            log.warning("Could not restore bot input %s", image_path)
            continue
        width, height = resolution_for(source_width, source_height)
        ACTIVE_IMAGES[(user_id, user_id)] = {
            "chat_id": user_id,
            "user_id": user_id,
            "image_path": str(image_path),
            "width": width,
            "height": height,
        }
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


def mode_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚡ Fast（8 步）", callback_data="settings:mode:fast"),
            InlineKeyboardButton("原始模型（20 步）", callback_data="settings:mode:standard"),
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if GENERATION_SECONDS == seconds else ''}{seconds} 秒", callback_data=f"settings:seconds:{seconds}")
            for seconds in (5, 10, 15)
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
                release_image(batch["image_path"])
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
    status = "你已在使用白名单中。" if allowed(update.effective_user.id) else f"你的 ID 是 {update.effective_user.id}，目前不在使用白名单中。"
    await update.message.reply_text(
        "MiniMax H3 图生视频 Bot\n\n"
        "1. 先发一张图片。\n"
        "2. 收到提醒后发送一个提示词；整条消息视为同一个提示词。\n"
        "   提示词中的每段 <中文台词> 都会自动转换为 H3 中文对白格式。\n"
        f"3. 当前全局设置：{settings_label()}；使用 /mode 随时修改模型和时长。\n"
        "4. 选择生成 1–8 条；也可在提示词末尾写 xN（如 x12）直接指定任意正整数数量。\n"
        "5. 每条视频生成好后会立即发回，并显示生成和排队耗时。\n"
        "6. 在发送新图片前，后续所有普通文字都会继续使用最近一张图片。\n"
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
    await update.message.reply_text(
        f"当前全局生成设置：{settings_label()}\n修改后将用于随后创建的新任务。",
        reply_markup=mode_keyboard(),
    )


async def on_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global GENERATION_MODE, GENERATION_SECONDS
    query = update.callback_query
    if not allowed(update.effective_user.id):
        await query.answer("你不在使用白名单中", show_alert=True)
        return
    _, setting, value = query.data.split(":", 2)
    if setting == "mode" and value == "fast":
        try:
            turbo_issues = await asyncio.to_thread(submit.check_turbo_support)
        except Exception as error:
            turbo_issues = [str(error)]
        if turbo_issues:
            await query.answer("Fast 组件当前不可用", show_alert=True)
            await query.edit_message_text("无法切换到 Fast：\n" + "\n".join(turbo_issues))
            return
    if setting == "mode":
        GENERATION_MODE = value
    else:
        GENERATION_SECONDS = int(value)
    save_generation_settings()
    await query.answer("全局设置已修改")
    await query.edit_message_text(
        f"当前全局生成设置：{settings_label()}\n可继续修改另一项。",
        reply_markup=mode_keyboard(),
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    discard_pending(
        lambda batch: batch["user_id"] == update.effective_user.id
        and batch["chat_id"] == update.effective_chat.id
    )
    await update.message.reply_text("已取消尚未提交的选项。最近一张图片仍会继续使用。")


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


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update.effective_user.id):
        await update.message.reply_text(f"你不在使用白名单中。你的 Telegram ID：{update.effective_user.id}")
        return

    key = user_key(update)
    photo = update.message.photo[-1]
    width, height = resolution_for(photo.width, photo.height)
    BOT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = BOT_INPUT_DIR / f"{update.effective_user.id}_{secrets.token_hex(8)}.jpg"
    for attempt in range(3):
        try:
            telegram_file = await photo.get_file()
            await telegram_file.download_to_drive(image_path)
            break
        except (NetworkError, TimedOut):
            if attempt == 2:
                log.warning("Photo download failed after three attempts")
                discard_image(image_path)
                try:
                    await update.message.reply_text("图片下载超时，请稍后重新发送。")
                except (NetworkError, TimedOut):
                    pass
                return
            await asyncio.sleep(2 ** attempt)

    try:
        await update.message.delete()
    except Exception:
        log.warning("Could not delete the original Telegram photo message")
    try:
        with image_path.open("rb") as image:
            await context.bot.send_photo(
                update.effective_chat.id,
                photo=image,
                has_spoiler=True,
                caption="🔒 原图（点击查看）",
            )
    except (NetworkError, TimedOut):
        log.warning("Could not send spoiler preview for incoming photo")
    previous_image = ACTIVE_IMAGES.get(key, {}).get("image_path")
    ACTIVE_IMAGES[key] = {
        "chat_id": update.effective_chat.id,
        "user_id": update.effective_user.id,
        "image_path": str(image_path),
        "width": width,
        "height": height,
    }
    if previous_image != str(image_path):
        retire_image(previous_image)
    await context.bot.send_message(
        update.effective_chat.id,
        "图片已收到。请发送一个提示词，整条消息（包括换行）会作为同一个提示词。提示词中的每段 <中文台词> 都会自动转换为 H3 中文对白格式，其余文字保持不变。之后在发送新图片前，其他普通文字也会继续使用这张图片。背景音乐固定关闭，仍可指定环境声、人物声音或对白。",
    )


async def on_prompt_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    key = user_key(update)
    state = ACTIVE_IMAGES.get(key)
    if state is None:
        await update.message.reply_text("请先发送一张图片。")
        return

    raw_prompt, requested_count = submit.extract_prompt_count(update.message.text)
    if not raw_prompt:
        await send_text_with_retry(context.bot, update.effective_chat.id, "没有读到有效提示词，请重新输入。")
        return
    if not validate_angle_brackets(raw_prompt):
        await send_text_with_retry(
            context.bot,
            update.effective_chat.id,
            "检测到未严格匹配的 < >。请确保每段台词都写成完整的 <中文台词>，不能缺少、多余、嵌套或留空。",
        )
        return
    prompt, dialogue_expanded = expand_dialogue_prompt(raw_prompt)

    batch_key = secrets.token_hex(6)
    state = dict(state)
    state["prompt"] = prompt
    state["display_prompt"] = raw_prompt
    state["requested_count"] = requested_count
    state["accelerated"] = GENERATION_MODE == "fast"
    state["seconds"] = GENERATION_SECONDS
    acquire_image(state["image_path"])
    selected_mode = "fast" if state["accelerated"] else "standard"
    if requested_count is not None:
        context.application.create_task(
            submit_batch(context.application, state, state["seconds"], requested_count)
        )
        await send_text_with_retry(
            context.bot,
            update.effective_chat.id,
            ("已自动转换其中的中文台词。" if dialogue_expanded else "提示词已收到。")
            + f"使用 {mode_label(selected_mode)} · {state['seconds']} 秒 · 共 {requested_count} 条；"
            "正在提交，每条使用不同随机 seed……",
        )
        return

    PENDING_COUNT[batch_key] = state
    await send_text_with_retry(
        context.bot,
        update.effective_chat.id,
        ("已自动转换其中的中文台词。" if dialogue_expanded else "提示词已收到。")
        + f"当前使用 {mode_label(selected_mode)} · {state['seconds']} 秒。请选择生成多少条视频：",
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
    mode = mode_label("fast" if record.get("accelerated") else "standard")
    caption_lines = [f"完成 · {mode} · 视频 {record['actual_seconds']:.3f} 秒 · seed {record['seed']}"]
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
                    }
                    record = await asyncio.to_thread(submit.submit_job, job, index)
                    record["prompt_text"] = batch.get("display_prompt", batch["prompt"])
                    records.append(record)
        except Exception as error:
            log.exception("Batch submission failed")
            await send_text_with_retry(application.bot, chat_id, f"部分任务提交失败：{error}")

        if not records:
            return
        mode = mode_label("fast" if batch["accelerated"] else "standard")
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
        release_image(batch["image_path"])


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
            f"已选择 {batch['seconds']} 秒 · {mode_label(selected_mode)} · 共 {count} 条；"
            "正在提交，每条使用不同随机 seed……"
        )
    except (NetworkError, TimedOut):
        log.warning("Count acknowledgement failed, but batch submission continues")


async def post_init(application):
    global GENERATION_MODE, SUBMIT_LOCK
    SUBMIT_LOCK = asyncio.Lock()
    commands = [
        BotCommand("start", "使用说明"),
        BotCommand("mode", "修改全局模型和视频时长"),
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
        if GENERATION_MODE == "fast":
            GENERATION_MODE = "standard"
            save_generation_settings()
            log.warning("GENERATION_MODE_FALLBACK mode=standard")
    else:
        log.info("TURBO_PREFLIGHT_OK lora=%s", submit.TURBO_LORA_NAME)
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
    application.add_handler(CallbackQueryHandler(on_mode, pattern=r"^settings:(mode:(fast|standard)|seconds:(5|10|15))$"))
    application.add_handler(CallbackQueryHandler(on_count, pattern=r"^count:[0-9a-f]+:[1-8]$"))
    application.add_handler(CallbackQueryHandler(on_service_gpu_count, pattern=r"^service:(start|restart):[0-9a-f]+:[1-8]$"))
    application.add_error_handler(on_error)
    application.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1)


if __name__ == "__main__":
    main()
