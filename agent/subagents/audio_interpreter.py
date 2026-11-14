import asyncio
import sys
import time
from pathlib import Path

# Ensures 'HackPrinceton Fall 2026' is on the path if you run this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.deep_agent import AgentState, AlertEvent, DeepAgent

DISTRESS_KEYWORDS = ("help", "emergency", "fallen", "hurt", "call 911", "can't get up")


async def audio_interpreter_loop(agent: DeepAgent, raw_audio_queue: asyncio.Queue):
    """
    Consumes (transcript, confidence) tuples from the elementary audio detector.
    Respects agent.mic_enabled to prevent speaker echo from self-triggering.
    """
    while True:
        transcript, confidence = await raw_audio_queue.get()

        # 1. Strictly ignore all audio while the speaker is talking
        if not agent.mic_enabled.is_set():
            continue

        cleaned_text = transcript.strip()
        if not cleaned_text:
            continue

        # 2. During MONITORING, only escalate on distress keywords
        if agent.state == AgentState.MONITORING:
            lower_text = cleaned_text.lower()
            if any(word in lower_text for word in DISTRESS_KEYWORDS):
                await agent.event_queue.put(
                    AlertEvent(
                        source="audio",
                        trigger_reason=f"keyword:{cleaned_text}",
                        confidence=confidence,
                        # Attach recent camera frames from the shared ring-buffer if available
                        frames_jpeg=list(agent.latest_frames),
                        transcript=cleaned_text,
                        timestamp=time.time(),
                    )
                )

        # 3. During AWAITING_RESPONSE, forward ANY speech so the Deep Agent can evaluate it
        elif agent.state == AgentState.AWAITING_RESPONSE:
            await agent.event_queue.put(
                AlertEvent(
                    source="audio",
                    trigger_reason="verbal_reply",
                    confidence=confidence,
                    transcript=cleaned_text,
                    timestamp=time.time(),
                )
            )