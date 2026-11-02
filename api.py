"""
API + web UI + bot Telegram dalam SATU proses (model hanya dimuat sekali).
Jalankan:  uvicorn api:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1
"""
import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jamu_engine import answer_cached, kb_size
from telegram_app import build_application

log = logging.getLogger("api")
STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app):
    tg = None
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        tg = build_application(token)
        await tg.initialize()
        await tg.start()
        await tg.updater.start_polling(drop_pending_updates=True)
        log.info("Bot Telegram aktif di dalam proses API")
    else:
        log.warning("TELEGRAM_BOT_TOKEN kosong: hanya website yang aktif")
    yield
    if tg:
        await tg.updater.stop()
        await tg.stop()
        await tg.shutdown()


app = FastAPI(title="Jamu Madura QA", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:   # hanya perlu jika frontend dipisah dari API
    app.add_middleware(CORSMiddleware, allow_origins=_origins,
                       allow_methods=["POST", "GET"], allow_headers=["Content-Type"])

# rate limit per IP (IP asli dibaca uvicorn dari X-Forwarded-For milik nginx)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "12"))
RATE_WINDOW = 60
_hits = {}


def _check_rate(ip):
    now = time.time()
    q = [t for t in _hits.get(ip, []) if now - t < RATE_WINDOW]
    if len(q) >= RATE_LIMIT:
        _hits[ip] = q
        raise HTTPException(429, "Terlalu banyak pertanyaan, coba lagi sebentar.")
    q.append(now)
    _hits[ip] = q
    if len(_hits) > 5000:   # bersihkan IP lama agar memori tidak membengkak
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW]:
            _hits.pop(k, None)


class ChatIn(BaseModel):
    message: str = Field(min_length=2, max_length=500)


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": kb_size()}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    _check_rate(request.client.host if request.client else "?")
    text = body.message.strip()
    if len(text) < 2:
        raise HTTPException(400, "Pertanyaan kosong.")
    return await asyncio.to_thread(answer_cached, text, "web")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
