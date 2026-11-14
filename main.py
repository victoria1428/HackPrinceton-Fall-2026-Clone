import asyncio
import os
import time

from agent.deep_agent import AgentState, DeepAgent
from agent.subagents.audio_interpreter import audio_interpreter_loop
from agent.subagents.vision_interpreter import (
    VisionFramePacket,
    VisionMemoryConfig,
    vision_interpreter_loop,
)

# --- 1. SAFETY LOCK: DISABLE LIVE TWILIO CALLS FOR TESTING ---
for twilio_var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"):
    os.environ.pop(twilio_var, None)


def create_dummy_jpeg() -> bytes:
    """Generates a valid tiny 1x1 JPEG byte string in memory."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
    )


async def run_simulation():
    api_key = os.getenv("GEMINI_API_KEY", "TEST_MODE_NO_KEY")

    agent = DeepAgent(
        api_key=api_key,
        frontend_url="",
        emergency_webhook="",
    )
    agent.response_timeout_sec = 5.0

    dummy_frame = create_dummy_jpeg()
    for _ in range(3):
        agent.latest_frames.append(dummy_frame)

    # Queues for Audio and Vision Subagents
    raw_audio_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
    raw_vision_queue: asyncio.Queue[VisionFramePacket] = asyncio.Queue()

    # Configure 100-frame memory (~20s at 5 FPS), triggering at 75 fallen frames
    vision_config = VisionMemoryConfig(
        max_frames=100,
        min_frames_before_eval=50,
        fall_frame_threshold=75,
        keyframes_to_send=5,
    )

    # Start the DeepAgent brain, Audio Interpreter, and Vision Interpreter concurrently
    agent_task = asyncio.create_task(agent.run())
    audio_task = asyncio.create_task(audio_interpreter_loop(agent, raw_audio_queue))
    vision_task = asyncio.create_task(
        vision_interpreter_loop(agent, raw_vision_queue, config=vision_config)
    )

    await asyncio.sleep(0.5)

    # =========================================================================
    # TEST SCENARIO 1: Distress Detected -> Speaker Asks -> Person Says "I'm fine"
    # =========================================================================
    print("\n" + "=" * 65)
    print("🧪 SCENARIO 1: Person yells 'Help!' -> Speaker check-in -> Replies 'I'm fine'")
    print("=" * 65)

    print("\n[Sim-Mic] Feeding audio: 'Help, I have fallen!' (confidence=0.92)")
    await raw_audio_queue.put(("Help, I have fallen!", 0.92))

    while agent.state != AgentState.AWAITING_RESPONSE:
        await asyncio.sleep(0.2)

    await asyncio.sleep(1.5)
    print("\n[Sim-Mic] Feeding audio during speaker echo test (while unmuted): 'I am fine, all good'")
    await raw_audio_queue.put(("I am fine, all good", 0.95))

    await asyncio.sleep(1.5)
    print(f"[Test Check] Current Agent State after Scenario 1: {agent.state.name}")

    # =========================================================================
    # TEST SCENARIO 2: Distress Detected -> Speaker Asks -> Person Unresponsive
    # =========================================================================
    print("\n" + "=" * 65)
    print("🧪 SCENARIO 2: Person yells 'Emergency!' -> Stays silent -> Simulated Dispatch")
    print("=" * 65)

    print("\n[Sim-Mic] Feeding audio: 'Emergency, I am hurt!' (confidence=0.94)")
    await raw_audio_queue.put(("Emergency, I am hurt!", 0.94))

    while agent.state != AgentState.AWAITING_RESPONSE:
        await asyncio.sleep(0.2)

    print(f"[Test Info] Staying completely silent for {agent.response_timeout_sec}s to trigger escalation...")

    while agent.state != AgentState.MONITORING:
        await asyncio.sleep(0.5)

    # =========================================================================
    # TEST SCENARIO 3: 100-Frame YOLO Fall Sequence -> Keyframe Sampling -> Dispatch
    # =========================================================================
    print("\n" + "=" * 65)
    print("🧪 SCENARIO 3: 100-Frame YOLO Sliding Memory (25 standing -> 80 fallen)")
    print("=" * 65)

    sim_time = time.time()

    # 1. Feed 25 frames of a person standing/walking normally (tall bounding box: w=120, h=300)
    print("\n[Sim-YOLO] Feeding 25 frames of normal standing posture (5 seconds at 5 FPS)...")
    for i in range(25):
        await raw_vision_queue.put(
            VisionFramePacket(
                frame_jpeg=dummy_frame,
                person_detected=True,
                bbox_x=200.0 + (i * 2.0),  # Walking across frame
                bbox_y=240.0,
                bbox_w=120.0,
                bbox_h=300.0,              # Tall bounding box (standing)
                frame_width=640.0,
                frame_height=480.0,
                confidence=0.90,
                timestamp=sim_time + (i * 0.2),  # 0.2s per frame = 5 FPS
            )
        )

    # 2. Feed 80 frames of a person collapsed & motionless on the floor (wide bounding box: w=320, h=110)
    # As frames 76..105 arrive, the oldest standing frames are shifted out of the 100-frame deque!
    print("[Sim-YOLO] Feeding 80 frames of horizontal, stationary fallen posture...")
    for j in range(80):
        frame_idx = 25 + j
        await raw_vision_queue.put(
            VisionFramePacket(
                frame_jpeg=dummy_frame,
                person_detected=True,
                bbox_x=250.0,              # Stationary center X
                bbox_y=380.0,              # Low on the floor Y
                bbox_w=320.0,              # Wide bounding box (fallen)
                bbox_h=110.0,
                frame_width=640.0,
                frame_height=480.0,
                confidence=0.93,
                timestamp=sim_time + (frame_idx * 0.2),
            )
        )
        await asyncio.sleep(0.01)  # Fast-forward simulation so you don't wait 20 real seconds

    # Wait for VisionInterpreter to trigger DeepAgent -> Speaker Challenge -> AWAITING_RESPONSE
    while agent.state != AgentState.AWAITING_RESPONSE:
        await asyncio.sleep(0.2)

    print("\n[Sim-Mic] Person groans 'My hip hurts, help' during check-in...")
    await asyncio.sleep(1.0)
    await raw_audio_queue.put(("My hip hurts, help", 0.91))

    # Wait for immediate escalation + cooldown reset
    while agent.state != AgentState.MONITORING:
        await asyncio.sleep(0.5)

    print("\n✅ All 3 test scenarios (Audio Cancel, Audio Timeout, 100-Frame Vision Fall) completed!")
    agent_task.cancel()
    audio_task.cancel()
    vision_task.cancel()


if __name__ == "__main__":
    asyncio.run(run_simulation())