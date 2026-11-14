import asyncio
import base64
import json
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import aiohttp
from deepagents import create_deep_agent
from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.emergency_tools import dispatch_emergency_call, lookup_emergency_contacts


# --- 1. STATES & DATA STRUCTURES ---

class AgentState(Enum):
    MONITORING = auto()
    EVALUATING = auto()
    CHALLENGING = auto()
    AWAITING_RESPONSE = auto()
    ESCALATING = auto()


@dataclass
class AlertEvent:
    source: str                                     # "vision" or "audio"
    trigger_reason: str                             # e.g., "horizontal_pose_3s" or "keyword:help"
    confidence: float                               # Local elementary confidence (0.0 - 1.0)
    frames_jpeg: list[bytes] = field(default_factory=list)  # Multi-frame sequence
    frame_jpeg: Optional[bytes] = None              # Optional single-frame fallback
    transcript: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class EmergencyDecision(BaseModel):
    is_emergency: bool
    confidence: float
    reasoning: str


# --- 2. DEEP AGENT ORCHESTRATOR ---

class DeepAgent:
    def __init__(self, api_key: str, frontend_url: str, emergency_webhook: str = ""):
        self.state = AgentState.MONITORING
        self.event_queue: asyncio.Queue[AlertEvent] = asyncio.Queue()

        # Synchronization flag: When cleared, Audio Subagent drops all incoming mic data
        self.mic_enabled = asyncio.Event()
        self.mic_enabled.set()

        # 1. Official Google Gen AI SDK client (Fast Multimodal Frame + Audio Triage)
        self.gemini_client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

        # 2. LangChain DeepAgent (Autonomous Contact Lookup & Emergency Dispatch)
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_id,
            google_api_key=api_key,
            temperature=0.1,
        )
        self.escalation_agent = create_deep_agent(
            model=self.llm,
            tools=[lookup_emergency_contacts, dispatch_emergency_call],
            system_prompt=(
                "You are an autonomous emergency response coordinator. When an unresponsive fall "
                "or acute distress event occurs, ALWAYS: "
                "1) Call `lookup_emergency_contacts` to retrieve the resident's primary contact and medical notes. "
                "2) Call `dispatch_emergency_call` using the highest-priority contact's phone number, "
                "concisely summarizing the visual/audio findings and critical medical notes."
            ),
        )

        self.frontend_url = frontend_url
        self.emergency_webhook = emergency_webhook

        # Rolling ring-buffer of recent camera frames (populated by Vision Subagent)
        # Allows audio-only triggers ("Help!") to still attach visual context!
        self.latest_frames: deque[bytes] = deque(maxlen=4)

        # Evidence frames and session tracking for the sanity-check frontend
        self.evidence_frames: list[bytes] = []
        self.response_timeout_sec = 8.0
        self._active_challenge_id: Optional[str] = None
        self._last_triage_reasoning: str = ""

    def _clear_stale_events(self):
        """Drains any stale events that queued up while muted or evaluating."""
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def run(self):
        """Main decision loop for the Deep Agent."""
        print(f"[DeepAgent] Online ({self.model_id}). State: MONITORING")
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

        # Gather sequential frames from the event, single frame, or shared camera ring-buffer
        if event.frames_jpeg:
            self.evidence_frames = list(event.frames_jpeg)
        elif event.frame_jpeg:
            self.evidence_frames = [event.frame_jpeg]
        elif self.latest_frames:
            self.evidence_frames = list(self.latest_frames)

        # Query Gemini for multimodal verification across all collected frames
        decision = await self._query_cloud_model(event, self.evidence_frames)
        self._last_triage_reasoning = decision.get("reasoning", "Anomaly detected.")

        if decision.get("is_emergency") and decision.get("confidence", 0.0) >= 0.75:
            print(f"[DeepAgent] Gemini confirmed potential emergency: {self._last_triage_reasoning}")
            await self._initiate_speaker_challenge()
        else:
            print(f"[DeepAgent] Gemini dismissed alert ({self._last_triage_reasoning}). Returning to MONITORING.")
            self.evidence_frames.clear()
            self._clear_stale_events()
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

        # 3. Flush any stale queue items, turn mic back on, and listen for response
        self._clear_stale_events()
        self.mic_enabled.set()
        self.state = AgentState.AWAITING_RESPONSE
        print(f"[DeepAgent] Mic UNMUTED. Waiting {self.response_timeout_sec}s for response...")

        # Launch non-blocking timeout watcher tied to this specific challenge ID
        asyncio.create_task(self._response_timeout_watcher(challenge_id))

    async def _response_timeout_watcher(self, challenge_id: str):
        """Escalates if the user fails to respond within the window."""
        await asyncio.sleep(self.response_timeout_sec)

        if (
            self.state == AgentState.AWAITING_RESPONSE
            and self._active_challenge_id == challenge_id
        ):
            print("[DeepAgent] No response detected within window!")
            self._active_challenge_id = None
            await self._escalate_emergency(
                f"Unresponsive after verbal check-in. Triage findings: {self._last_triage_reasoning}"
            )

    async def _handle_response_event(self, event: AlertEvent):
        """Evaluates audio updates while waiting for the person to reply."""
        if event.source == "audio" and event.transcript:
            text = event.transcript.lower()
            if any(phrase in text for phrase in ["i'm okay", "i am okay", "i'm fine", "i am fine", "all good"]):
                print("[DeepAgent] User confirmed they are okay. Resetting.")
                self._active_challenge_id = None
                self.evidence_frames.clear()
                self._clear_stale_events()
                self.state = AgentState.MONITORING
            else:
                print(f"[DeepAgent] Distress/unclear audio heard: '{text}'. Escalating.")
                self._active_challenge_id = None
                await self._escalate_emergency(
                    f"Distress response ('{text}'). Triage findings: {self._last_triage_reasoning}"
                )

    async def _escalate_emergency(self, reason: str):
        self.state = AgentState.ESCALATING
        print(f"🚨 [DeepAgent] ESCALATING EMERGENCY: {reason}")

        # 1. Speak confirmation to person (keep mic muted during playback)
        self.mic_enabled.clear()
        await self._speak("Contacting emergency services now. Help is on the way.")

        # 2. Concurrently push frames to sanity-check frontend & run LangChain DeepAgent tool dispatch
        await asyncio.gather(
            self._push_to_frontend(reason),
            self._run_escalation_agent(reason),
        )

        # Cooldown before resetting for next hackathon demo run
        await asyncio.sleep(10)
        self._clear_stale_events()
        self.mic_enabled.set()
        self.evidence_frames.clear()
        self.state = AgentState.MONITORING
        print("[DeepAgent] System reset for next demo.")

    # --- GEMINI REASONING & SUBAGENT HOOKS ---

    async def _query_cloud_model(self, event: AlertEvent, frames: list[bytes]) -> dict:
        """
        Sends sequential frames + sensor context to Gemini using native async (client.aio)
        and structured Pydantic output.
        """
        prompt = (
            "You are an emergency triage vision and audio analyzer for an autonomous safety monitor. "
            f"Local sensor trigger: {event.trigger_reason}. "
            f"Audio transcript: '{event.transcript or 'None'}'. "
            f"Attached are {len(frames)} sequential camera frame(s) captured leading up to the alert. "
            "Analyze the visual posture progression and audio context. "
            "Determine if a person has collapsed on the ground, appears unresponsive/injured, "
            "or is calling for help."
        )

        contents: list = [prompt]
        for frame_bytes in frames:
            contents.append(
                types.Part.from_bytes(
                    data=frame_bytes,
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
            return {
                "is_emergency": event.confidence >= 0.85,
                "confidence": event.confidence,
                "reasoning": f"Fallback to edge confidence due to API error: {e}",
            }

    async def _run_escalation_agent(self, reason: str):
        """Invokes the LangChain DeepAgent to look up emergency contacts and place the call."""
        try:
            result = await self.escalation_agent.ainvoke({
                "messages": [
                    HumanMessage(
                        content=(
                            f"EMERGENCY CONFIRMED. Details: {reason}. "
                            "Look up the resident's emergency contacts and medical notes, "
                            "then dispatch the emergency call immediately."
                        )
                    )
                ]
            })
            final_msg = result["messages"][-1].content
            print(f"[DeepAgent] Escalation Agent Complete: {final_msg}")
        except Exception as e:
            print(f"[DeepAgent] Escalation Agent failed ({e}), falling back to direct webhook.")
            if self.emergency_webhook:
                await self._trigger_emergency_call(reason)

    async def _speak(self, text: str):
        """Non-blocking wrapper around Raspberry Pi TTS (espeak-ng) with Windows fallback."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "espeak-ng", "-s", "150", text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except FileNotFoundError:
            # Fallback when testing on Windows laptop without espeak-ng installed
            print(f"🔊 [TTS SPEAKER]: {text}")
            await asyncio.sleep(2.0)

    async def _push_to_frontend(self, reason: str):
        """Sends base64 frames and status to your sanity-check dashboard."""
        if not self.frontend_url:
            return
        payload = {
            "status": "EMERGENCY_DISPATCHED",
            "reason": reason,
            "timestamp": time.time(),
            "frames_b64": [base64.b64encode(f).decode("utf-8") for f in self.evidence_frames],
        }
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    self.frontend_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                print("[DeepAgent] Frames pushed to frontend.")
            except Exception as e:
                print(f"[DeepAgent] Frontend push failed: {e}")

    async def _trigger_emergency_call(self, reason: str):
        """Optional direct webhook fallback if emergency_webhook URL is provided."""
        if not self.emergency_webhook:
            return
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