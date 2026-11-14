import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Safety Lock: Ensure live Twilio / Roboflow keys are cleared during tests
for env_var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "ROBOFLOW_API_KEY"):
    os.environ.pop(env_var, None)

from agent.deep_agent import AgentState, DeepAgent
from agent.subagents.audio_interpreter import audio_interpreter_loop
from agent.subagents.vision_interpreter import (
    VisionFramePacket,
    VisionMemoryConfig,
    vision_interpreter_loop,
)

DUMMY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
    b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
)


class StateTracker:
    """Helper that records every AgentState transition in order."""
    def __init__(self, agent: DeepAgent):
        self.agent = agent
        self.history: list[AgentState] = [agent.state]
        self._running = True

    async def watch(self):
        while self._running:
            if self.agent.state != self.history[-1]:
                self.history.append(self.agent.state)
                print(f"   🔄 [State Transition] -> {self.agent.state.name}")
            await asyncio.sleep(0.01)

    def stop(self):
        self._running = False


class TestFSMStages(unittest.IsolatedAsyncioTestCase):

    # =========================================================================
    # TEST 1: MONITORING Only -> Check a few frames & mic -> Standby Exit
    # =========================================================================
    async def test_1_monitoring_to_standby(self):
        print("\n" + "=" * 70)
        print("🧪 TEST 1: MONITORING -> Normal Frames & Audio -> Exit to Standby")
        print("=" * 70)

        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="")
        raw_vision_queue: asyncio.Queue = asyncio.Queue()
        raw_audio_queue: asyncio.Queue = asyncio.Queue()

        tracker = StateTracker(agent)
        tracker_task = asyncio.create_task(tracker.watch())
        agent_task = asyncio.create_task(agent.run())
        vision_task = asyncio.create_task(vision_interpreter_loop(agent, raw_vision_queue))
        audio_task = asyncio.create_task(audio_interpreter_loop(agent, raw_audio_queue))

        # 1. Feed 10 normal frames (person standing/walking, is_fallen=False)
        t0 = time.time()
        print("[Test 1] Feeding 10 normal camera frames (is_fallen=False)...")
        for i in range(10):
            await raw_vision_queue.put(
                VisionFramePacket(
                    source_frame_jpeg=DUMMY_JPEG,
                    skeleton_frame_jpeg=DUMMY_JPEG,
                    is_fallen=False,
                    risk_score=0.10,
                    timestamp=t0 + (i * 0.2),
                )
            )

        # 2. Feed normal non-distress audio via microphone
        print("[Test 1] Feeding normal mic audio: 'Just making some tea in the kitchen'")
        await raw_audio_queue.put(("Just making some tea in the kitchen", 0.94))

        # 3. Continue monitoring for a short while
        await asyncio.sleep(0.5)

        # Verify agent stayed in MONITORING the entire time and ring-buffer updated
        self.assertEqual(agent.state, AgentState.MONITORING)
        self.assertEqual(tracker.history, [AgentState.MONITORING])
        self.assertGreater(len(agent.latest_frames), 0)
        print(f"[Test 1] Verified state history: {[s.name for s in tracker.history]}")

        # 4. Exit monitoring loops cleanly to simulate Standby Mode
        print("[Test 1] No anomalies found. Exiting loops to enter STANDBY state...")
        tracker.stop()
        for task in (agent_task, vision_task, audio_task, tracker_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        print("✅ TEST 1 PASSED: Device monitored cam/mic and entered standby cleanly.")

    # =========================================================================
    # TEST 2: MONITORING -> EVALUATING -> Dismissed (Does NOT enter CHALLENGING)
    # =========================================================================
    async def test_2_evaluating_dismissed_before_challenging(self):
        print("\n" + "=" * 70)
        print("🧪 TEST 2: MONITORING -> EVALUATING -> Dismissed (Stops before CHALLENGING)")
        print("=" * 70)

        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="")

        # Mock `_query_cloud_model` so it pauses briefly in EVALUATING, then dismisses the alert
        async def mock_dismiss_eval(event, frames):
            await asyncio.sleep(0.1)  # Hold in EVALUATING briefly so tracker records it
            return {
                "is_emergency": False,
                "confidence": 0.25,
                "reasoning": "Person is doing floor stretches on a yoga mat, not a fall.",
            }

        agent._query_cloud_model = mock_dismiss_eval  # type: ignore

        raw_vision_queue: asyncio.Queue = asyncio.Queue()
        tracker = StateTracker(agent)
        tracker_task = asyncio.create_task(tracker.watch())
        agent_task = asyncio.create_task(agent.run())
        vision_task = asyncio.create_task(
            vision_interpreter_loop(
                agent,
                raw_vision_queue,
                config=VisionMemoryConfig(
                    max_frames=100,
                    min_frames_before_eval=50,
                    fall_frame_threshold=75,
                    keyframes_to_send=5,
                ),
            )
        )

        # Feed 75 fallen frames so VisionInterpreter triggers an AlertEvent
        t0 = time.time()
        print("[Test 2] Feeding 75 DOWN frames to trigger EVALUATING state...")
        for i in range(75):
            await raw_vision_queue.put(
                VisionFramePacket(
                    source_frame_jpeg=DUMMY_JPEG,
                    skeleton_frame_jpeg=DUMMY_JPEG,
                    is_fallen=True,
                    risk_score=0.88,
                    timestamp=t0 + (i * 0.2),
                )
            )

        # Wait for evaluation to complete
        await asyncio.sleep(0.4)

        # Verify state went MONITORING -> EVALUATING -> MONITORING (and NEVER hit CHALLENGING)
        expected_path = [AgentState.MONITORING, AgentState.EVALUATING, AgentState.MONITORING]
        self.assertEqual(tracker.history, expected_path)
        self.assertNotIn(AgentState.CHALLENGING, tracker.history)
        self.assertEqual(len(agent.source_evidence_frames), 0, "Evidence frames must be cleared on dismissal.")
        print(f"[Test 2] Verified state history: {[s.name for s in tracker.history]}")

        tracker.stop()
        for task in (agent_task, vision_task, tracker_task):
            task.cancel()
        print("✅ TEST 2 PASSED: Evaluated 5 keyframes, dismissed false alarm, skipped CHALLENGING.")

    # =========================================================================
    # TEST 3: Full Path -> Victim Speaks Outside Safe Keywords -> ESCALATING
    # =========================================================================
    async def test_3_full_pipeline_non_keyword_speech_to_escalation(self):
        print("\n" + "=" * 70)
        print("🧪 TEST 3: MONITORING -> EVALUATING -> CHALLENGING -> AWAITING_RESPONSE -> ESCALATING")
        print("=" * 70)

        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="")
        agent.response_timeout_sec = 8.0

        # 1. Mock `_query_cloud_model` to confirm an emergency during EVALUATING
        async def mock_confirm_eval(event, frames):
            await asyncio.sleep(0.1)  # Hold briefly so EVALUATING state is captured
            return {
                "is_emergency": True,
                "confidence": 0.96,
                "reasoning": "Roboflow & Gemini confirmed supine motionless fall on floor.",
            }

        agent._query_cloud_model = mock_confirm_eval  # type: ignore

        # 2. Speed up TTS speaker playback & escalation agent for the test
        async def fast_speak(text: str):
            print(f"   🔊 [Speaker TTS]: '{text}'")
            await asyncio.sleep(0.1)

        agent._speak = fast_speak  # type: ignore
        agent._run_escalation_agent = AsyncMock()

        raw_vision_queue: asyncio.Queue = asyncio.Queue()
        raw_audio_queue: asyncio.Queue = asyncio.Queue()

        tracker = StateTracker(agent)
        tracker_task = asyncio.create_task(tracker.watch())
        agent_task = asyncio.create_task(agent.run())
        vision_task = asyncio.create_task(vision_interpreter_loop(agent, raw_vision_queue))
        audio_task = asyncio.create_task(audio_interpreter_loop(agent, raw_audio_queue))

        # Step A: Trigger Vision Alert (75 DOWN frames)
        t0 = time.time()
        print("[Test 3] Feeding 75 DOWN frames to trigger Vision -> DeepAgent...")
        for i in range(75):
            await raw_vision_queue.put(
                VisionFramePacket(
                    source_frame_jpeg=DUMMY_JPEG,
                    skeleton_frame_jpeg=DUMMY_JPEG,
                    is_fallen=True,
                    risk_score=0.95,
                    timestamp=t0 + (i * 0.2),
                )
            )

        # Wait until agent passes EVALUATING and CHALLENGING and reaches AWAITING_RESPONSE
        while agent.state != AgentState.AWAITING_RESPONSE:
            await asyncio.sleep(0.05)

        print("[Test 3] Agent is now in AWAITING_RESPONSE. Victim speaks outside safe keywords...")
        # Safe keywords are: ["i'm okay", "i am okay", "i'm fine", "i am fine", "all good"]
        # Victim says something confused/groaning outside those keywords:
        await raw_audio_queue.put(("Ugh... what happened? My head is spinning", 0.91))

        # Wait until agent transitions to ESCALATING
        while agent.state != AgentState.ESCALATING:
            await asyncio.sleep(0.05)

        # Verify the exact 5-stage progression occurred in order!
        expected_progression = [
            AgentState.MONITORING,
            AgentState.EVALUATING,
            AgentState.CHALLENGING,
            AgentState.AWAITING_RESPONSE,
            AgentState.ESCALATING,
        ]
        self.assertEqual(tracker.history[:5], expected_progression)
        print(f"[Test 3] Verified 5-Stage Progression: {[s.name for s in tracker.history[:5]]}")

        # Wait for `_run_escalation_agent` to be called during ESCALATING
        await asyncio.sleep(0.2)
        agent._run_escalation_agent.assert_called_once()
        call_arg = agent._run_escalation_agent.call_args[0][0]
        self.assertIn("what happened? my head is spinning", call_arg)
        print(f"[Test 3] Verified Escalation Dispatch Reason: \"{call_arg}\"")

        tracker.stop()
        for task in (agent_task, vision_task, audio_task, tracker_task):
            task.cancel()
        print("✅ TEST 3 PASSED: Non-keyword reply during AWAITING_RESPONSE immediately escalated!")


if __name__ == "__main__":
    unittest.main(verbosity=2)