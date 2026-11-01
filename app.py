"""Streamlit front end for the voice bot.

Run with:  streamlit run app.py   then open http://127.0.0.1:8501

Streamlit shows the settings and embeds the voice page. The Pipecat bot server
(server.py) runs inside this process on 127.0.0.1:7860 and serves that page, so
microphone audio goes straight from the browser to the bot over WebRTC.
"""

import os
import socket
import threading
import time
from urllib.parse import urlencode

import streamlit as st
import uvicorn

import bot
import server

BOT_HOST, BOT_PORT = "127.0.0.1", 7860
BOT_LABELS = {
    "pipeline": "Pipeline: transcribe → chat → voice",
    "realtime": "Realtime speech model",
}


def bot_server_listening() -> bool:
    with socket.socket() as s:
        return s.connect_ex((BOT_HOST, BOT_PORT)) == 0


@st.cache_resource
def start_bot_server() -> bool:
    """Start the bot server once per Streamlit process; reruns reuse it.

    Returns False when another bot server already holds the port (e.g. `python
    server.py` in a terminal), which is then used as is.
    """
    if bot_server_listening():
        return False
    config = uvicorn.Config(server.app, host=BOT_HOST, port=BOT_PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, name="bot-server", daemon=True).start()
    for _ in range(50):  # wait up to 5 s so the embedded page doesn't load too early
        if bot_server_listening():
            break
        time.sleep(0.1)
    return True


st.set_page_config(page_title="Voice Bot", page_icon=":material/mic:")
if not start_bot_server():
    st.info(f"Using the bot server already running on {BOT_HOST}:{BOT_PORT}. Restart it to pick up code changes.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Bot",
        list(BOT_LABELS),
        index=list(BOT_LABELS).index(os.getenv("BOT_MODE", "pipeline")),
        format_func=BOT_LABELS.get,
        help="The realtime bot needs a realtime model deployment (AZURE_OPENAI_REALTIME_DEPLOYMENT in .env).",
    )
    default_voice = os.getenv("TTS_VOICE", "alloy")
    voice = st.selectbox("Voice", bot.VOICES, index=bot.VOICES.index(default_voice) if default_voice in bot.VOICES else 0)
    silence = st.slider(
        "Pause before the bot replies (seconds)",
        min_value=0.2,
        max_value=1.5,
        value=bot.END_OF_TURN_SILENCE_SECS,
        step=0.1,
        help="Silence the bot waits for before answering. Higher: fewer interruptions mid-sentence. Lower: faster replies.",
    )
    prompt = st.text_area("System prompt", bot.SYSTEM_PROMPT, height=170, max_chars=2000)
    st.caption("Changing a setting reloads the call panel. Click Connect again to use it.")

st.title("Voice Bot")
st.caption(
    "Pick your microphone, click Connect and talk. Your audio goes from this browser to the bot "
    "on this laptop, and from there only to your company's Azure OpenAI resource."
)

# The settings travel to the bot in the page's URL; the page sends them with each call.
query = urlencode({"mode": mode, "voice": voice, "silence": silence, "prompt": prompt})
st.iframe(f"http://{BOT_HOST}:{BOT_PORT}/?{query}", height=560)
