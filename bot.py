"""Real-time voice bot pipeline (Pipecat 1.15).

Browser mic --WebRTC--> Silero VAD + transcribe model --> chat model
--> TTS model --WebRTC--> browser speaker.

All three models are deployments on one Azure OpenAI resource (one key, one endpoint).

server.py calls run_bot() once per browser connection.
"""

import os

from loguru import logger
from openai import AsyncAzureOpenAI, DefaultAsyncHttpxClient

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.http import connection_limits
from pipecat.workers.runner import WorkerRunner

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Keep replies short (1–2 sentences), "
    "conversational, and with no markdown or lists since everything is spoken aloud."
)

# Silence to wait, on top of Silero's own 0.2 s, before treating the user's turn
# as finished. Raise it if the bot cuts in on pauses; lower it for faster replies.
# Silero also needs 0.2 s of speech to report "talking again", so mid-sentence
# pauses up to about this long are tolerated.
END_OF_TURN_SILENCE_SECS = 0.6

# Voices offered by both gpt-4o-mini-tts and the realtime models.
VOICES = ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")


def call_settings(options: dict | None) -> tuple[str, str, float]:
    """Return (voice, system prompt, silence wait) for one call.

    `options` comes from the page (e.g. the Streamlit sidebar); anything missing or
    invalid falls back to the defaults above and in .env.
    """
    options = options or {}
    default_voice = os.getenv("TTS_VOICE", "alloy")
    voice = options.get("voice") if options.get("voice") in VOICES else default_voice
    prompt = str(options.get("prompt") or SYSTEM_PROMPT).strip()[:2000] or SYSTEM_PROMPT
    try:
        silence = min(2.0, max(0.2, float(options.get("silence", END_OF_TURN_SILENCE_SECS))))
    except (TypeError, ValueError):
        silence = END_OF_TURN_SILENCE_SECS
    return voice, prompt, silence


# Azure API version used for the TTS deployment (see azure_audio_client).
AZURE_AUDIO_API_VERSION = "2025-03-01-preview"


def azure_resource_url() -> str:
    """Return the Azure OpenAI resource URL, e.g. https://<name>.openai.azure.com.

    Also accepts a full request URL copied from the portal
    (".../openai/deployments/<name>/chat/completions?api-version=...").
    """
    return os.environ["AZURE_OPENAI_ENDPOINT"].strip().split("/openai/")[0].rstrip("/")


def azure_audio_client() -> AsyncAzureOpenAI:
    """Client for Azure's per-deployment API (/openai/deployments/<name>/audio/...).

    Azure's /openai/v1 surface answered 404 for REST audio calls, so TTS uses this
    dated API. Connections stay open between turns: reconnecting costs ~0.7 s here.
    """
    return AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=azure_resource_url(),
        api_version=AZURE_AUDIO_API_VERSION,
        http_client=DefaultAsyncHttpxClient(limits=connection_limits(keepalive_expiry=None)),
    )


class AzureRealtimeSTTService(OpenAIRealtimeSTTService):
    """Streaming transcription that marks a transcript final once it is the newest one.

    Audio streams up while the user talks; at each VAD stop the service commits the
    buffer and the server returns one complete transcript for it. When that answers
    the latest commit, nothing more is coming, so the end-of-turn logic can stop
    waiting instead of sitting out its STT safety timeout (~1.5 s). If the counts
    ever drift (a failed commit), transcripts just aren't marked: slower, still correct.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._commits = 0
        self._results = 0

    async def _commit_audio_buffer(self):
        self._commits += 1
        await super()._commit_audio_buffer()

    async def _handle_transcription_completed(self, evt: dict):
        self._results += 1
        await super()._handle_transcription_completed(evt)

    async def _handle_transcription_failed(self, evt: dict):
        self._results += 1
        await super()._handle_transcription_failed(evt)

    async def push_frame(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, TranscriptionFrame):
            frame.finalized = self._results == self._commits
        await super().push_frame(frame, direction)


def browser_notifier(transport: SmallWebRTCTransport):
    """Return an async function that sends a JSON message to the page over the data channel."""

    async def notify(message: dict):
        await transport.output().send_message(OutputTransportMessageUrgentFrame(message=message))

    return notify


class CallObserver(BaseObserver):
    """Streams the conversation to the page and logs per-turn latency.

    Latency is how long after the user stopped talking each stage responded; it goes
    to the console and, as a "latency" message, to the page.
    """

    def __init__(self, notify=None):
        # Only see each frame once: when the processor that created it pushes it.
        super().__init__(observe_every_push=False)
        self._notify = notify
        self._turn = 0
        self._speech_end = None  # pipeline clock (ns) when the user stopped talking
        self._marks = {}

    async def _send(self, message: dict):
        if self._notify:
            await self._notify(message)

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        # Conversation text for the page: user transcripts and the bot's reply as it streams.
        if isinstance(frame, TranscriptionFrame):
            await self._send({"type": "transcript", "role": "user", "text": frame.text})
        elif isinstance(frame, LLMTextFrame):
            await self._send({"type": "transcript", "role": "bot", "text": frame.text})
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._send({"type": "reply-end"})

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The user is talking (again), so any earlier stop was just a pause.
            self._speech_end = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # VAD reports the stop `stop_secs` after the last word; back-date it.
            self._speech_end = data.timestamp - int(frame.stop_secs * 1e9)
            self._marks = {}
        elif self._speech_end is None:
            return
        elif isinstance(frame, TranscriptionFrame):
            self._marks["STT final"] = data.timestamp  # keep the turn's last final
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._marks.setdefault("turn end", data.timestamp)
        elif isinstance(frame, LLMTextFrame):
            self._marks.setdefault("LLM first token", data.timestamp)
        elif isinstance(frame, TTSAudioRawFrame):
            self._marks["TTS first byte"] = data.timestamp
            self._turn += 1
            stages = " | ".join(
                f"{name} +{(t - self._speech_end) / 1e6:.0f}ms" for name, t in self._marks.items()
            )
            logger.info(f"[latency] turn {self._turn}: {stages} (from end of user speech)")
            await self._send(
                {
                    "type": "latency",
                    "stages": {name: round((t - self._speech_end) / 1e6) for name, t in self._marks.items()},
                }
            )
            self._speech_end = None


async def run_bot(webrtc_connection: SmallWebRTCConnection, options: dict | None = None):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    voice, prompt, silence = call_settings(options)

    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    resource_url = azure_resource_url()

    # On Azure, every "model" below is the name of a deployment on your resource.
    stt = AzureRealtimeSTTService(
        api_key=api_key,
        base_url=f"{resource_url.replace('https://', 'wss://')}/openai/v1/realtime",
        settings=OpenAIRealtimeSTTService.Settings(
            model=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-mini-transcribe"),
        ),
    )

    # Azure's /openai/v1 API is OpenAI-compatible. Pipecat's OpenAI service keeps its
    # connections open between turns; AzureLLMService reconnects after 5 s idle.
    llm = OpenAILLMService(
        api_key=api_key,
        base_url=f"{resource_url}/openai/v1/",
        settings=OpenAILLMService.Settings(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            system_instruction=prompt,
        ),
    )

    tts = OpenAITTSService(
        api_key=api_key,
        settings=OpenAITTSService.Settings(
            model=os.getenv("AZURE_OPENAI_TTS_DEPLOYMENT", "gpt-4o-mini-tts"),
            voice=voice,
        ),
        # Speak each sentence as soon as the LLM finishes it, while it keeps streaming the rest.
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )
    tts._client = azure_audio_client()

    # The shared context holds the whole conversation, so the bot remembers earlier turns.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                # Barge-in: as soon as Silero hears the user, interrupt the bot,
                # cancelling LLM + TTS and flushing queued audio. This replaces
                # PipelineParams(allow_interruptions=True), removed in Pipecat 1.x.
                # The transcription strategy is a fallback for speech too soft for VAD.
                start=[
                    VADUserTurnStartStrategy(enable_interruptions=True),
                    TranscriptionUserTurnStartStrategy(enable_interruptions=True),
                ],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=silence)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,  # records what the bot actually said
        ]
    )

    worker = PipelineWorker(
        pipeline,
        observers=[CallObserver(notify=browser_notifier(transport))],
        enable_rtvi=False,  # index.html is plain WebRTC and doesn't use the RTVI protocol
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected (pipeline, voice={voice}, silence wait={silence}s)")
        # Run the LLM on the still-empty conversation: it opens with a greeting.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)  # uvicorn handles Ctrl+C
    await runner.add_workers(worker)
    await runner.run()
