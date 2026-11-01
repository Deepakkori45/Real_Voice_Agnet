"""Realtime variant of the voice bot (Pipecat 1.15).

Browser mic --WebRTC--> Silero VAD --> realtime speech-to-speech model
--WebRTC--> browser speaker.

One model hears the user and answers in its own voice, so there is no separate
transcribe -> chat -> TTS chain to wait on. Silero VAD still decides when the user
has finished, with the same silence wait and barge-in as bot.py.

server.py runs this instead of bot.py when BOT_MODE=realtime.
"""

import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.openai.realtime import events
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from bot import CallObserver, azure_resource_url, browser_notifier, call_settings


async def run_bot(webrtc_connection: SmallWebRTCConnection, options: dict | None = None):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    voice, prompt, silence = call_settings(options)

    llm = AzureRealtimeLLMService(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=f"{azure_resource_url().replace('https://', 'wss://')}/openai/v1/realtime",
        settings=AzureRealtimeLLMService.Settings(
            model=os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT", "gpt-realtime-2.1-mini"),
            system_instruction=prompt,
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        # Silero VAD (below) decides when the user is done, not the server.
                        turn_detection=False,
                        # Also transcribe the user, for the log and the conversation history.
                        transcription=events.InputAudioTranscription(
                            model=os.getenv("AZURE_OPENAI_STT_DEPLOYMENT", "gpt-4o-mini-transcribe"),
                        ),
                    ),
                    output=events.AudioOutput(voice=voice),
                ),
            ),
        ),
    )

    # Pipecat detects the realtime service and records the conversation from it.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                # Barge-in: as soon as Silero hears the user, interrupt the bot.
                start=[VADUserTurnStartStrategy(enable_interruptions=True)],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=silence)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        observers=[CallObserver(notify=browser_notifier(transport))],
        enable_rtvi=False,  # index.html is plain WebRTC and doesn't use the RTVI protocol
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected (realtime, voice={voice}, silence wait={silence}s)")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)  # uvicorn handles Ctrl+C
    await runner.add_workers(worker)
    await runner.run()
