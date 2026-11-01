"""FastAPI server: serves the client page and does the WebRTC offer/answer.

Run with:  python server.py   then open http://127.0.0.1:7860
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

import bot
import bot_realtime

load_dotenv(override=True)

# Two bots: "pipeline" (transcribe -> chat -> TTS, bot.py) and "realtime" (one
# speech-to-speech model, bot_realtime.py). The page can pick one per call;
# otherwise BOT_MODE in .env decides.
BOTS = {"pipeline": bot.run_bot, "realtime": bot_realtime.run_bot}

# Pipecat logs a lot at DEBUG; INFO keeps the per-turn latency lines readable.
logger.remove()
logger.add(sys.stderr, level="INFO")

# No STUN/TURN servers: browser and bot run on the same machine, so the
# connection never leaves it.
webrtc_handler = SmallWebRTCRequestHandler(ice_servers=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await webrtc_handler.close()  # hang up any open calls on shutdown


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Take the browser's SDP offer, start a bot for the new connection, return the answer.

    The page may send per-call settings (bot, voice, prompt, silence wait) as request_data.
    """
    options = request.request_data if isinstance(request.request_data, dict) else {}
    run_bot = BOTS.get(options.get("mode") or os.getenv("BOT_MODE", "pipeline"), bot.run_bot)

    async def start_bot(connection: SmallWebRTCConnection):
        # Runs after the answer is sent, for as long as the call lasts.
        background_tasks.add_task(run_bot, connection, options)

    return await webrtc_handler.handle_web_request(request, start_bot)


if __name__ == "__main__":
    # 127.0.0.1 only: the page is reachable from this laptop and nowhere else.
    uvicorn.run(app, host="127.0.0.1", port=7860)
