import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import aiohttp
from google import genai
from google.genai import types
from pydantic import BaseModel


# --- 1. STATES & DATA STRUCTURES ---

class AgentState(Enum):
    MONITORING = auto()
    EVALUATING = auto()
    CHALLENGING = auto()
    AWAITING_RESPONSE = auto()
    ESCALATING = auto()


@dataclass
class AlertEvent:
    source: str               # "vision" or "audio"
    trigger_reason: str       # e.g., "horizontal_pose_3s" or "keyword:help"
    confidence: float         # Local elementary confidence (0.0 - 1.0)
    frame_jpeg: Optional[bytes] = None
    transcript: Optional[str] = None
    timestamp: float = 0.0


class EmergencyDecision(BaseModel):
    is_emergency: bool
    confidence: float
    reasoning: str


# --- 2. DEEP AGENT ORCHESTRATOR ---

class DeepAgent:
    def __init__(self, api_key: str, frontend_url: str, emergency_webhook: str):
        self.state = AgentState.MONITORING
        self.event_queue: asyncio.Queue[AlertEvent] = asyncio.Queue()

        # Synchronization flag: When cleared, Audio Subagent drops all incoming mic data
        self.mic_enabled = asyncio.Event()
        self.mic_enabled.set()

        # Official Google Gen AI SDK client
        self.gemini_client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3.8-flash"

        self.frontend_url = frontend_url
        self.emergency_webhook = emergency_webhook

        # Evidence frames and session tracking for the sanity-check frontend
        self.evidence_frames: list[bytes] = []
        self.response_timeout_sec = 8.0
        self._active_challenge_id: Optional[str] = None

    async def run(self):
        """Main decision loop for the Deep Agent."""
        print("[DeepAgent] Online (Gemini 3.8 Flash). State: MONITORING")
        while True:
            event = await self.event_queue.get()

            if self.state == AgentState.MONITORING:
                await self._handle_monitoring_event(event)

            elif self.state == AgentState.AWAITING_RESPONSE:
                await self._handle_response_event(event)

    # --- STATE HANDLERS ---

    async def _handle_monitoring_event(self, event: AlertEvent):
        print(f"[DeepAgent] Trigger received from {event.source}: {event.trigger_reason}")
        self.state = AgentState.EVALUATING
        if event.frame_jpeg:
            self.evidence_frames = [event.frame_jpeg]

        # Query Gemini for multimodal verification
        decision = await self._query_cloud_model(event)

        if decision.get("is_emergency") and decision.get("confidence", 0.0) >= 0.75:
            print(f"[DeepAgent] Gemini confirmed potential emergency: {decision.get('reasoning')}")
            await self._initiate_speaker_challenge()
        else:
            print(f"[DeepAgent] Gemini dismissed alert ({decision.get('reasoning')}). Returning to MONITORING.")
            self.evidence_frames.clear()
            self.state = AgentState.MONITORING

    async def _initiate_speaker_challenge(self):
        """Mutes mic, speaks prompt via speaker, unmutes mic, and starts timeout timer."""
        self.state = AgentState.CHALLENGING
        challenge_id = str(uuid.uuid4())
        self._active_challenge_id = challenge_id

        # 1. Turn off mic ingestion so speaker output doesn't trigger audio subagent
        self.mic_enabled.clear()
        print("[DeepAgent] Mic MUTED. Playing audio check-in...")

        # 2. Execute TTS Subagent
        await self._speak("Are you okay? Please say I am fine if you do not need help.")

        # Brief buffer so room echo fades before mic turns back on
        await asyncio.sleep(0.4)

        # 3. Turn mic back on and listen for response
        self.mic_enabled.set()
        self.state = AgentState.AWAITING_RESPONSE
        print(f"[DeepAgent] Mic UNMUTED. Waiting {self.response_timeout_sec}s for response...")

        # Launch non-blocking timeout watcher tied to this specific challenge ID
        asyncio.create_task(self._response_timeout_watcher(challenge_id))

    async def _response_timeout_watcher(self, challenge_id: str):
        """Escalates if the user fails to respond within the window."""
        await asyncio.sleep(self.response_timeout_sec)

        # Only escalate if we are STILL waiting on this exact challenge session
        if (
            self.state == AgentState.AWAITING_RESPONSE
            and self._active_challenge_id == challenge_id
        ):
            print("[DeepAgent] No response detected within window!")
            await self._escalate_emergency("Unresponsive after verbal check-in.")

    async def _handle_response_event(self, event: AlertEvent):
        """Evaluates audio updates while waiting for the person to reply."""
        if event.source == "audio" and event.transcript:
            text = event.transcript.lower()
            if any(phrase in text for phrase in ["i'm okay", "i am okay", "i'm fine", "all good"]):
                print("[DeepAgent] User confirmed they are okay. Resetting.")
                self._active_challenge_id = None
                self.evidence_frames.clear()
                self.state = AgentState.MONITORING
            else:
                print(f"[DeepAgent] Distress/unclear audio heard: '{text}'. Escalating.")
                self._active_challenge_id = None
                await self._escalate_emergency(f"Distress response: {text}")

    async def _escalate_emergency(self, reason: str):
        self.state = AgentState.ESCALATING
        print(f"🚨 [DeepAgent] ESCALATING EMERGENCY: {reason}")

        # 1. Speak confirmation to person (keep mic muted during playback)
        self.mic_enabled.clear()
        await self._speak("Contacting emergency services now. Help is on the way.")

        # 2. Send frames + telemetry to Frontend Sanity Check & trigger Phone Demo
        await asyncio.gather(
            self._push_to_frontend(reason),
            self._trigger_emergency_call(reason),
        )

        # Cooldown before resetting for next hackathon demo run
        await asyncio.sleep(10)
        self.mic_enabled.set()
        self.evidence_frames.clear()
        self.state = AgentState.MONITORING
        print("[DeepAgent] System reset for next demo.")

    # --- GEMINI REASONING & SUBAGENT HOOKS ---

    async def _query_cloud_model(self, event: AlertEvent) -> dict:
        """
        Sends frame + sensor context to Gemini using native async (client.aio)
        and structured Pydantic output.
        """
        prompt = (
            "You are an emergency triage vision and audio analyzer for an autonomous safety monitor. "
            f"Local sensor trigger: {event.trigger_reason}. "
            f"Audio transcript: '{event.transcript or 'None'}'. "
            "Analyze the attached camera frame (if provided) and the sensor trigger. "
            "Determine if a person has collapsed on the ground, appears unresponsive/injured, "
            "or is calling for help."
        )

        contents: list = [prompt]
        if event.frame_jpeg:
            contents.append(
                types.Part.from_bytes(
                    data=event.frame_jpeg,
                    mime_type="image/jpeg",
                )
            )

        try:
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EmergencyDecision,
                    temperature=0.1,
                ),
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return json.loads(response.text)

        except Exception as e:
            print(f"[DeepAgent] Gemini API Error: {e}")
            # Fail-safe fallback if hackathon Wi-Fi drops
            return {
                "is_emergency": event.confidence >= 0.85,
                "confidence": event.confidence,
                "reasoning": f"Fallback to edge confidence due to API error: {e}",
            }

    async def _speak(self, text: str):
        """Non-blocking wrapper around Raspberry Pi TTS (espeak-ng)."""
        proc = await asyncio.create_subprocess_exec(
            "espeak-ng", "-s", "150", text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    async def _push_to_frontend(self, reason: str):
        """Sends base64 frames and status to your sanity-check dashboard."""
        payload = {
            "status": "EMERGENCY_DISPATCHED",
            "reason": reason,
            "timestamp": time.time(),
            "frames_b64": [base64.b64encode(f).decode("utf-8") for f in self.evidence_frames],
        }
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(self.frontend_url, json=payload, timeout=aiohttp.ClientTimeout(total=5))
                print("[DeepAgent] Frames pushed to frontend.")
            except Exception as e:
                print(f"[DeepAgent] Frontend push failed: {e}")

    async def _trigger_emergency_call(self, reason: str):
        """Hits Twilio / webhook to dial demo phone."""
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    self.emergency_webhook,
                    json={"alert": reason},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                print("[DeepAgent] Emergency call webhook triggered.")
            except Exception as e:
                print(f"[DeepAgent] Call trigger failed: {e}")