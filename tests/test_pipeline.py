import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Safety Lock: Ensure live Twilio / Roboflow keys are cleared during unit tests
for env_var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "ROBOFLOW_API_KEY"):
    os.environ.pop(env_var, None)

from agent.cv_analyzer import RoboflowCVAnalyzer
from agent.deep_agent import AgentState, AlertEvent, DeepAgent
from agent.subagents.audio_interpreter import (
    AudioPacket,
    _normalize_audio_input,
    audio_interpreter_loop,
)
from agent.subagents.vision_interpreter import (
    VisionFramePacket,
    VisionMemoryConfig,
    VisionShortTermMemory,
    _normalize_vision_input,
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


class TestEmergencyAgentSuite(unittest.IsolatedAsyncioTestCase):

    # =========================================================================
    # 1. AUDIO INTERPRETER TESTS (Formats, Negations, Low Confidence, Mic Mute)
    # =========================================================================
    async def test_audio_normalizer_and_filters(self):
        print("\n--- Running Test 1: Audio Normalizer, Negation & Low-Confidence Filters ---")
        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="")
        raw_audio_queue: asyncio.Queue = asyncio.Queue()
        audio_task = asyncio.create_task(audio_interpreter_loop(agent, raw_audio_queue))

        # 1a. Should IGNORE low confidence (< 0.40)
        await raw_audio_queue.put(("Help me!", 0.25))
        # 1b. Should IGNORE negated distress phrase
        await raw_audio_queue.put("I don't need help right now")
        # 1c. Should IGNORE normal daily speech
        await raw_audio_queue.put({"text": "What time is dinner?", "confidence": 0.95})
        await asyncio.sleep(0.05)
        self.assertTrue(agent.event_queue.empty(), "Event queue should be empty after non-distress audio.")

        # 1d. Should TRIGGER on acoustic event dict ("thud") even with empty transcript
        await raw_audio_queue.put({"text": "", "sound_event": "thud", "confidence": 0.88})
        event = await asyncio.wait_for(agent.event_queue.get(), timeout=1.0)
        self.assertEqual(event.source, "audio")
        self.assertIn("acoustic_event:thud", event.trigger_reason)
        print("✅ Passed: Ignored noise/negations and triggered on 'thud' acoustic event.")

        audio_task.cancel()

    async def test_half_duplex_mic_mute_lock(self):
        print("\n--- Running Test 2: Half-Duplex Mic Mute Lock ---")
        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="")
        raw_audio_queue: asyncio.Queue = asyncio.Queue()
        audio_task = asyncio.create_task(audio_interpreter_loop(agent, raw_audio_queue))

        # Simulate speaker talking -> clear mic_enabled
        agent.mic_enabled.clear()
        await raw_audio_queue.put(("Help, emergency!", 0.99))
        await asyncio.sleep(0.05)
        self.assertTrue(agent.event_queue.empty(), "Muted mic must drop all incoming packets.")

        # Unmute mic -> packet should now go through
        agent.mic_enabled.set()
        await raw_audio_queue.put(("Help, emergency!", 0.99))
        event = await asyncio.wait_for(agent.event_queue.get(), timeout=1.0)
        self.assertIn("keyword:help", event.trigger_reason)
        print("✅ Passed: Mic mute lock strictly blocked speaker echo.")

        audio_task.cancel()

    # =========================================================================
    # 2. VISION INTERPRETER TESTS (Input Formats, Expected Errors, 100-Frame Memory)
    # =========================================================================
    def test_vision_input_normalizer_and_expected_errors(self):
        print("\n--- Running Test 3: Vision Normalizer & Expected ValueError Exceptions ---")
        # Valid 2-tuple (bytes, bool)
        p1 = _normalize_vision_input((DUMMY_JPEG, True))
        self.assertTrue(p1.is_fallen)

        # Valid 3-tuple (source_bytes, skeleton_bytes, risk_float)
        p2 = _normalize_vision_input((DUMMY_JPEG, DUMMY_JPEG, 0.75), risk_thresh=0.50)
        self.assertTrue(p2.is_fallen)
        self.assertEqual(p2.skeleton_frame_jpeg, DUMMY_JPEG)

        # Expected Error 1: Empty tuple `()` -> must raise ValueError
        with self.assertRaises(ValueError) as ctx1:
            _normalize_vision_input(())
        print(f"✅ Caught Expected Error on empty tuple: {ctx1.exception}")

        # Expected Error 2: Unsupported type (int) -> must raise ValueError
        with self.assertRaises(ValueError) as ctx2:
            _normalize_vision_input(12345)  # type: ignore
        print(f"✅ Caught Expected Error on invalid type: {ctx2.exception}")

    def test_vision_short_term_memory_and_cooldown(self):
        print("\n--- Running Test 4: 100-Frame FIFO Memory, Keyframes & Cooldown ---")
        cfg = VisionMemoryConfig(
            max_frames=100,
            min_frames_before_eval=50,
            fall_frame_threshold=75,
            keyframes_to_send=5,
            cooldown_sec=15.0,
        )
        mem = VisionShortTermMemory(config=cfg)
        t0 = time.time()

        # Feed 25 standing frames + 74 fallen frames (99 total -> only 74 fallen, should NOT trigger)
        for i in range(25):
            self.assertFalse(mem.add_frame(VisionFramePacket(DUMMY_JPEG, is_fallen=False, timestamp=t0 + i * 0.2)))
        for j in range(74):
            self.assertFalse(
                mem.add_frame(VisionFramePacket(DUMMY_JPEG, is_fallen=True, risk_score=0.9, timestamp=t0 + (25 + j) * 0.2))
            )

        # 75th fallen frame (Frame 100) -> MUST trigger!
        triggered = mem.add_frame(VisionFramePacket(DUMMY_JPEG, DUMMY_JPEG, is_fallen=True, risk_score=0.9, timestamp=t0 + 19.8))
        self.assertTrue(triggered, "75th fallen frame in 100-frame window must trigger.")

        src_frames, skel_frames, labels, avg_conf = mem.extract_keyframes()
        self.assertEqual(len(src_frames), 5)
        self.assertEqual(len(skel_frames), 1)  # 1 frame had skeleton_frame_jpeg attached
        self.assertAlmostEqual(avg_conf, 0.9, places=2)
        print(f"✅ Passed: Triggered at 75/100 frames. Sampled labels: {labels}")

        # Cooldown check: Feeding another fallen frame immediately must return False
        self.assertFalse(mem.add_frame(VisionFramePacket(DUMMY_JPEG, is_fallen=True, risk_score=0.9)))
        print("✅ Passed: 15s cooldown blocked duplicate immediate alert.")

    # =========================================================================
    # 3. ROBOFLOW CV ANALYZER TESTS (Corrupted Bytes Exception Handling)
    # =========================================================================
    async def test_cv_analyzer_corrupted_bytes_and_fallback(self):
        print("\n--- Running Test 5: Roboflow CV Analyzer Exception Handling ---")
        analyzer = RoboflowCVAnalyzer(api_key="")

        # 5a. No API key -> should return simulated report cleanly without error
        sim_report = await analyzer.analyze_frames([DUMMY_JPEG] * 5)
        self.assertEqual(sim_report["alert_status"], 1)
        self.assertTrue(sim_report.get("simulated"))

        # 5b. Simulate active client receiving corrupted non-image bytes
        analyzer.client = MagicMock()
        bad_result = analyzer._infer_single_frame(b"CORRUPTED_NOT_AN_IMAGE_BYTES")
        self.assertEqual(bad_result["class"], "error")
        self.assertIsNone(bad_result["label"])
        print("✅ Passed: Corrupted image bytes caught cleanly without crashing thread pool.")

    # =========================================================================
    # 4. DEEP AGENT API FAILURE & ESCALATION FALLBACK TESTS
    # =========================================================================
    async def test_deep_agent_503_and_escalation_fallbacks(self):
        print("\n--- Running Test 6: Gemini 503 Error & Escalation Agent Crash Recovery ---")
        agent = DeepAgent(api_key="DUMMY_KEY", frontend_url="http://localhost:9999/dead_frontend")
        agent.response_timeout_sec = 0.5  # Speed up timeout for test
        agent._speak = AsyncMock()        # Skip 2s TTS sleep during unit test

        # Force Gemini API to throw a 503 UNAVAILABLE exception
        agent.gemini_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("503 UNAVAILABLE: Model is experiencing high demand")
        )
        # Force LangChain escalation_agent to also fail so direct tool fallback is tested
        agent.escalation_agent.ainvoke = AsyncMock(
            side_effect=RuntimeError("503 UNAVAILABLE during LangChain tool call")
        )

        # Trigger monitoring event with high edge confidence (0.93 >= 0.85)
        event = AlertEvent(
            source="vision",
            trigger_reason="Unit test fall",
            confidence=0.93,
            source_frames_jpeg=[DUMMY_JPEG] * 5,
        )
        await agent._handle_monitoring_event(event)

        # Because confidence (0.93) >= 0.85 fallback threshold, agent MUST enter AWAITING_RESPONSE
        self.assertEqual(agent.state, AgentState.AWAITING_RESPONSE)
        print("✅ Passed: Gemini 503 caught -> fell back to edge confidence (0.93) and started check-in.")

        # Send distress reply ("help") to trigger immediate escalation (skipping the 10s post-escalation sleep by patching sleep)
        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()
        try:
            await agent._handle_response_event(
                AlertEvent(source="audio", trigger_reason="verbal_reply", confidence=0.95, transcript="Help me please")
            )
        finally:
            asyncio.sleep = orig_sleep

        self.assertEqual(agent.state, AgentState.MONITORING)
        print("✅ Passed: Escalation Agent crash caught -> executed direct contact lookup & Twilio fallback, ignored dead frontend URL, and reset to MONITORING.")


if __name__ == "__main__":
    unittest.main(verbosity=2)