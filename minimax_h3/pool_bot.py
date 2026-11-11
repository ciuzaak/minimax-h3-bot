"""Owner-bound Bot panels for global, independent video/image GPU counts."""

import asyncio
import logging
import secrets
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest

from .config import MAX_GPU_COUNT
from . import worker_pools as pools

PANELS = {}
log = logging.getLogger(__name__)


def keyboard(owner, token, draft):
    rows = []
    for role, label in (("video", "视频"), ("image", "图像")):
        for start in range(0, MAX_GPU_COUNT + 1, 3):
            rows.append([InlineKeyboardButton(f"{'✓ ' if draft[role] == n else ''}{label} {n} 张",
                         callback_data=f"pools:{owner}:{token}:{role}:{n}")
                         for n in range(start, min(start + 3, MAX_GPU_COUNT + 1))])
    rows.append([InlineKeyboardButton("应用分配", callback_data=f"pools:{owner}:{token}:apply:0"),
                 InlineKeyboardButton("刷新", callback_data=f"pools:{owner}:{token}:refresh:0")])
    return InlineKeyboardMarkup(rows)


def panel_text(state, draft):
    return pools.describe(state) + f"\n\n待应用：视频 {draft['video']} 张 + 图像 {draft['image']} 张。\n选择两类卡数后点“应用分配”；仅修改数字不会更改服务。"


async def gpu_command(update, context):
    from .telegram_bot import allowed
    owner, chat = update.effective_user.id, update.effective_chat.id
    if not allowed(owner):
        await update.message.reply_text("你不在使用白名单中。")
        return
    state = await asyncio.to_thread(pools.read_state)
    for key, panel in list(PANELS.items()):
        if time.time() - panel["created"] > 1800 or (panel["owner"], panel["chat"]) == (owner, chat):
            PANELS.pop(key, None)
    token = secrets.token_hex(6)
    panel = dict(owner=owner, chat=chat, created=time.time(), version=state["version"],
                 draft=dict(state["requested"]), lock=asyncio.Lock())
    PANELS[token] = panel
    await update.message.reply_text(panel_text(state, panel["draft"]), reply_markup=keyboard(owner, token, panel["draft"]))


async def on_gpu(update, context):
    from .telegram_bot import allowed
    query = update.callback_query
    _, owner, token, action, number = query.data.split(":")
    panel = PANELS.get(token)
    if not allowed(update.effective_user.id) or owner != str(update.effective_user.id):
        await query.answer("这不是你的资源面板。", show_alert=True)
        return
    if panel is None or panel["chat"] != update.effective_chat.id or time.time() - panel["created"] > 1800:
        await query.answer("面板已失效，请用 /gpu 重新打开。", show_alert=True)
        return
    async with panel["lock"]:
        if action in pools.ROLES and number.isdecimal() and 0 <= int(number) <= MAX_GPU_COUNT:
            panel["draft"][action] = int(number)
        elif action == "apply":
            try:
                await asyncio.to_thread(pools.set_requested, **panel["draft"], expected_version=panel["version"])
            except ValueError as error:
                await query.answer(str(error), show_alert=True)
                return
            await query.answer("目标已保存，按视频优先分配；已有任务先完成。")
            # Reconciliation is also retried by the background task when resources become free.
            try:
                await asyncio.to_thread(pools.reconcile)
            except Exception:
                log.exception("Could not immediately reconcile GPU allocation")
        elif action != "refresh":
            await query.answer("无效的资源选项。", show_alert=True)
            return
        state = await asyncio.to_thread(pools.read_state)
        if action in ("apply", "refresh"):
            panel["draft"] = dict(state["requested"])
            panel["version"] = state["version"]
        if action != "apply":
            await query.answer("尚未应用" if action in pools.ROLES else "已刷新")
        try:
            await query.edit_message_text(panel_text(state, panel["draft"]),
                                          reply_markup=keyboard(panel["owner"], token, panel["draft"]))
        except BadRequest as error:
            if "message is not modified" not in str(error).lower():
                raise


async def reconcile_loop():
    while True:
        try:
            await asyncio.to_thread(pools.reconcile)
        except Exception:
            log.exception("GPU allocation reconciliation failed")
        await asyncio.sleep(5)


async def shutdown(application):
    task = application.bot_data.get("gpu_pool_task")
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
