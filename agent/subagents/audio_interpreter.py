import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# Ensures 'HackPrinceton Fall 2026' is on the path if you run this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.deep_agent import AgentState, AlertEvent, DeepAgent


# --- 1. AUDIO DATA STRUCTURE & PATTERNS ---

@dataclass
class AudioPacket:
    """
    Standardized packet from your teammate's microphone / speech-to-text module.
    Supports both transcribed speech AND acoustic sound events (e.g., loud thud/groan).
    """
    transcript: str = ""
    confidence: float = 0.90
    sound_event: Optional[str] = None    # e.g., "thud", "scream", "groan", "glass_break", "speech"
    decibels_db: Optional[float] = None  # Optional volume spike metric
    timestamp: float = field(default_factory=time.time)


# Distress keywords/phrases during normal MONITORING
DISTRESS_KEYWORDS = (
    "help",
    "emergency",
    "fallen",
    "fell",
    "hurt",
    "pain",
    "call 911",
    "can't get up",
    "cannot get up",
    "dizzy",
    "faint",
    "chest",
    "ambulance",
)

# Non-speech acoustic events that should also wake up the DeepAgent
ACOUSTIC_DISTRESS_EVENTS = ("thud", "crash", "scream", "groan", "moan", "cry", "impact")

# Words that negate a false trigger (e.g., "I don't need help")
NEGATION_PHRASES = ("don't need help", "do not need help", "no emergency", "not hurt")


def _normalize_audio_input(raw_item: Union[AudioPacket, tuple, dict, str]) -> AudioPacket:
    """
    Converts whatever format your teammate pushes into `raw_audio_queue`
    (AudioPacket, (str, float) tuple, dict, or plain str) into an AudioPacket.
    """
    if isinstance(raw_item, AudioPacket):
        return raw_item

    if isinstance(raw_item, tuple):
        text = str(raw_item[0]) if len(raw_item) > 0 else ""
        conf = float(raw_item[1]) if len(raw_item) > 1 else 0.90
        return AudioPacket(transcript=text, confidence=conf)

    if isinstance(raw_item, dict):
        return AudioPacket(
            transcript=str(raw_item.get("transcript") or raw_item.get("text") or ""),
            confidence=float(raw_item.get("confidence", 0.90)),
            sound_event=raw_item.get("sound_event") or raw_item.get("event"),
            decibels_db=raw_item.get("decibels_db") or raw_item.get("db"),
            timestamp=float(raw_item.get("timestamp", time.time())),
        )

    if isinstance(raw_item, str):
        return AudioPacket(transcript=raw_item, confidence=0.90)

    return AudioPacket()


def _normalize_text(text: str) -> str:
    """Lowercases and strips punctuation for reliable keyword matching."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(cleaned.split())


# --- 2. AUDIO INTERPRETER SUBAGENT LOOP ---

async def audio_interpreter_loop(agent: DeepAgent, raw_audio_queue: asyncio.Queue[Any]):
    """
    Consumes audio inputs from your teammate's mic worker, respects `agent.mic_enabled`
    half-duplex muting, filters for distress signals during MONITORING, and forwards
    all verbal replies during AWAITING_RESPONSE.
    """
    print("[AudioInterpreter] Online. Listening for speech transcripts and acoustic events.")

    while True:
        raw_item = await raw_audio_queue.get()

        # 1. Half-duplex lock: strictly ignore mic input while the Pi speaker is talking
        if not agent.mic_enabled.is_set():
            continue

        packet = _normalize_audio_input(raw_item)
        norm_text = _normalize_text(packet.transcript)
        sound_tag = (packet.sound_event or "").strip().lower()

        # Ignore completely empty packets or ultra-low-confidence noise
        if not norm_text and not sound_tag:
            continue
        if packet.confidence < 0.40:
            continue

        # 2. During MONITORING: Check for spoken distress keywords OR loud acoustic impacts
        if agent.state == AgentState.MONITORING:
            # Skip negated phrases like "I don't need help"
            if any(neg in norm_text for neg in NEGATION_PHRASES):
                continue

            matched_keyword = next((kw for kw in DISTRESS_KEYWORDS if kw in norm_text), None)
            matched_sound = next((ev for ev in ACOUSTIC_DISTRESS_EVENTS if ev in sound_tag), None)

            if matched_keyword or matched_sound:
                trigger_desc = (
                    f"keyword:{matched_keyword} ('{packet.transcript.strip()}')"
                    if matched_keyword
                    else f"acoustic_event:{matched_sound}"
                )
                await agent.event_queue.put(
                    AlertEvent(
                        source="audio",
                        trigger_reason=trigger_desc,
                        confidence=packet.confidence,
                        frames_jpeg=list(agent.latest_frames),
                        transcript=packet.transcript.strip() or f"[{sound_tag}]",
                        timestamp=packet.timestamp,
                    )
                )

        # 3. During AWAITING_RESPONSE: Forward ANY speech or groan so DeepAgent can evaluate it
        elif agent.state == AgentState.AWAITING_RESPONSE:
            reply_content = packet.transcript.strip() or f"[{sound_tag}]"
            await agent.event_queue.put(
                AlertEvent(
                    source="audio",
                    trigger_reason="verbal_reply",
                    confidence=packet.confidence,
                    transcript=reply_content,
                    timestamp=packet.timestamp,
                )
            )