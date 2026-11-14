import asyncio
import sys
import time
from pathlib import Path

# Ensures 'HackPrinceton Fall 2026' is on the path if you run this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.deep_agent import AgentState, AlertEvent, DeepAgent

async def audio_interpreter_loop(agent: DeepAgent, raw_audio_queue: asyncio.Queue):
    while True:
        transcript, confidence = await raw_audio_queue.get()

        # Strictly ignore all audio while the speaker is talking
        if not agent.mic_enabled.is_set():
            continue

        # During MONITORING, only escalate on distress keywords
        if agent.state == AgentState.MONITORING:
            if any(word in transcript.lower() for word in ["help", "emergency", "fallen", "hurt"]):
                await agent.event_queue.put(AlertEvent(
                    source="audio",
                    trigger_reason=f"keyword:{transcript}",
                    confidence=confidence,
                    transcript=transcript,
                    timestamp=time.time()
                ))

        # During AWAITING_RESPONSE, forward ANY speech so the Deep Agent can evaluate it
        elif agent.state == AgentState.AWAITING_RESPONSE:
            await agent.event_queue.put(AlertEvent(
                source="audio",
                trigger_reason="verbal_reply",
                confidence=confidence,
                transcript=transcript,
                timestamp=time.time()
            ))