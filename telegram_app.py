"""Handler bot Telegram (dipakai api.py dan bot_jamu.py)."""
import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler,
                          ContextTypes, filters)

from jamu_engine import answer_cached

log = logging.getLogger("telegram_app")


WELCOME = (
    "Halo! Saya bot tanya jawab Jamu Madura 🌿\n\n"
    "Tanyakan khasiat atau jamu untuk keluhan tertentu, contoh:\n"
    "• Jamu apa yang bisa mengatasi keputihan?\n"
    "• Jamu untuk asam urat?\n"
    "• Anak saya susah makan, jamu apa yang cocok?\n\n"
    "Informasi bersifat referensi, bukan pengganti saran dokter."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        return
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        res = await asyncio.to_thread(answer_cached, q, "telegram")
        answer, kw = res['answer'], res['keywords']
        log.info("Q=%r | NER=%r", q, kw)
    except Exception:  # noqa: BLE001
        log.exception("Gagal memproses pertanyaan")
        answer = "Maaf, terjadi kesalahan saat memproses pertanyaan. Coba lagi sebentar."
    for i in range(0, len(answer), 4000):   # batas pesan Telegram 4096 karakter
        await update.message.reply_text(answer[i:i + 4000])


def build_application(token):
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app
