import asyncio
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensures 'HackPrinceton Fall 2026' is on the path if you run this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.deep_agent import AgentState, AlertEvent, DeepAgent


# --- 1. DATA STRUCTURES & MEMORY CONFIG ---

@dataclass
class VisionFramePacket:
    """
    Standardized packet sent from your teammate's YOLO detector into `raw_vision_queue`.
    Coordinates (bbox_x, bbox_y, bbox_w, bbox_h) can be normalized (0.0 - 1.0) or pixels;
    if pixels are used, pass frame_width and frame_height so movement is normalized.
    """
    frame_jpeg: bytes
    person_detected: bool
    bbox_x: float = 0.0          # Center X of person bounding box
    bbox_y: float = 0.0          # Center Y of person bounding box
    bbox_w: float = 0.0          # Width of bounding box
    bbox_h: float = 0.0          # Height of bounding box
    confidence: float = 0.0      # YOLO detection confidence (0.0 - 1.0)
    is_fallen_override: Optional[bool] = None  # If teammate's YOLO logic already computes fall=True/False
    frame_width: float = 1.0     # Set to camera width (e.g., 640) if bbox is in pixels
    frame_height: float = 1.0    # Set to camera height (e.g., 480) if bbox is in pixels
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisionMemoryConfig:
    """
    Tunable parameters for the short-term sliding-window memory.
    Default: 100 frames at ~5 FPS = ~20 seconds of consecutive footage.
    """
    max_frames: int = 100                  # Total frames held in FIFO sliding memory
    min_frames_before_eval: int = 50       # Minimum frames collected before checking threshold
    fall_frame_threshold: int = 75         # How many frames in memory must show a stationary fall
    keyframes_to_send: int = 5             # Subsampled frames sent to Gemini (e.g., t=0s, 5s, 10s, 15s, 20s)
    aspect_ratio_fall_thresh: float = 1.15 # Width / Height ratio indicating horizontal posture
    max_stationary_drift: float = 0.08     # Max normalized center-point movement (8% of screen) to count as stationary
    cooldown_sec: float = 15.0             # Cooldown after firing an alert so queue isn't spammed


@dataclass
class _EvaluatedFrame:
    """Internal entry stored inside the 100-frame deque."""
    packet: VisionFramePacket
    norm_x: float
    norm_y: float
    is_fallen_posture: bool
    is_stationary: bool


# --- 2. SHORT-TERM MEMORY & GEOMETRIC EVALUATOR ---

class VisionShortTermMemory:
    def __init__(self, config: Optional[VisionMemoryConfig] = None):
        self.config = config or VisionMemoryConfig()
        # deque(maxlen=100) automatically deletes frame[0] and shifts all frames left by 1
        # whenever a 101st frame is appended!
        self.buffer: deque[_EvaluatedFrame] = deque(maxlen=self.config.max_frames)
        self._last_alert_time: float = 0.0

    def update_memory_size(self, new_max_frames: int, new_threshold: int, new_min_frames: int = 50):
        """Allows live adjustment of the frame buffer size if FPS changes on the Pi."""
        self.config.max_frames = new_max_frames
        self.config.fall_frame_threshold = new_threshold
        self.config.min_frames_before_eval = new_min_frames
        self.buffer = deque(self.buffer, maxlen=new_max_frames)

    def add_frame(self, packet: VisionFramePacket) -> bool:
        """
        Pushes a new frame into the sliding window, evaluates its YOLO geometry + motion,
        and returns True if the 75-of-100 stationary fall threshold is reached.
        """
        norm_x = packet.bbox_x / max(packet.frame_width, 1.0)
        norm_y = packet.bbox_y / max(packet.frame_height, 1.0)
        norm_w = packet.bbox_w / max(packet.frame_width, 1.0)
        norm_h = packet.bbox_h / max(packet.frame_height, 1.0)

        # 1. Check horizontal/fallen geometry (either via teammate's override flag or aspect ratio)
        if not packet.person_detected:
            is_fallen_posture = False
        elif packet.is_fallen_override is not None:
            is_fallen_posture = packet.is_fallen_override
        else:
            aspect_ratio = norm_w / max(norm_h, 1e-5)
            # Fallen geometry: bounding box is wider than tall, OR center of mass is low in the frame
            is_fallen_posture = (aspect_ratio >= self.config.aspect_ratio_fall_thresh) or (norm_y >= 0.65 and aspect_ratio >= 0.85)

        # 2. Check if the person is stationary compared to recent detected frames in memory
        is_stationary = self._check_stationary(norm_x, norm_y) if packet.person_detected else False

        # Append to deque (automatically shifts and drops oldest frame if > max_frames)
        self.buffer.append(
            _EvaluatedFrame(
                packet=packet,
                norm_x=norm_x,
                norm_y=norm_y,
                is_fallen_posture=is_fallen_posture,
                is_stationary=is_stationary,
            )
        )

        # 3. Evaluate whether to trigger an escalation to DeepAgent
        if len(self.buffer) < self.config.min_frames_before_eval:
            return False

        if (time.time() - self._last_alert_time) < self.config.cooldown_sec:
            return False

        # Count how many frames in the current window show BOTH fallen posture AND stationary position
        fallen_stationary_count = sum(
            1 for f in self.buffer if f.is_fallen_posture and f.is_stationary
        )

        if fallen_stationary_count >= self.config.fall_frame_threshold:
            self._last_alert_time = time.time()
            return True

        return False

    def _check_stationary(self, curr_x: float, curr_y: float) -> bool:
        """
        Compares current bounding-box center against the person's position ~5 frames ago
        to verify they are not actively walking or exercising.
        """
        recent_detected = [f for f in self.buffer if f.packet.person_detected]
        if not recent_detected:
            return True  # First frame with a person

        # Compare against up to 5 frames back (~1 second ago at 5 FPS)
        ref_frame = recent_detected[-min(5, len(recent_detected))]
        distance = math.hypot(curr_x - ref_frame.norm_x, curr_y - ref_frame.norm_y)
        return distance <= self.config.max_stationary_drift

    def extract_keyframes(self) -> tuple[list[bytes], list[str], float]:
        """
        Subsamples `keyframes_to_send` evenly spaced frames across the 100-frame buffer
        (e.g., t=0s, t=5s, t=10s, t=15s, t=20s) and computes average YOLO confidence.
        """
        total = len(self.buffer)
        if total == 0:
            return [], [], 0.0

        k = min(self.config.keyframes_to_send, total)
        if k == 1:
            indices = [total - 1]
        else:
            # Evenly spaced indices from start (0) to latest (total - 1)
            indices = [round(i * (total - 1) / (k - 1)) for i in range(k)]

        t_start = self.buffer[0].packet.timestamp
        sampled_jpegs: list[bytes] = []
        time_labels: list[str] = []

        for idx in indices:
            entry = self.buffer[idx]
            sampled_jpegs.append(entry.packet.frame_jpeg)
            elapsed = max(0.0, entry.packet.timestamp - t_start)
            time_labels.append(f"t={elapsed:.1f}s")

        fallen_frames = [f.packet.confidence for f in self.buffer if f.is_fallen_posture]
        avg_conf = sum(fallen_frames) / len(fallen_frames) if fallen_frames else 0.85

        return sampled_jpegs, time_labels, avg_conf

    def clear(self):
        self.buffer.clear()


# --- 3. ASYNC SUBAGENT LOOP ---

async def vision_interpreter_loop(
    agent: DeepAgent,
    raw_vision_queue: asyncio.Queue[VisionFramePacket],
    config: Optional[VisionMemoryConfig] = None,
):
    """
    Consumes VisionFramePackets from the YOLO worker, updates the 100-frame sliding memory,
    keeps `agent.latest_frames` fresh for audio triggers, and dispatches multi-frame AlertEvents.
    """
    memory = VisionShortTermMemory(config=config)
    print(
        f"[VisionInterpreter] Online. Window={memory.config.max_frames} frames, "
        f"Trigger={memory.config.fall_frame_threshold} stationary fallen frames."
    )

    while True:
        packet = await raw_vision_queue.get()

        # Always keep DeepAgent's 4-frame ring-buffer updated (~once per second)
        # so audio-only triggers ("Help!") also have recent visual context
        if len(memory.buffer) % 5 == 0:
            agent.latest_frames.append(packet.frame_jpeg)

        # Only evaluate fall triggers while DeepAgent is in MONITORING state
        if agent.state != AgentState.MONITORING:
            continue

        should_alert = memory.add_frame(packet)

        if should_alert:
            keyframes_jpeg, time_labels, avg_conf = memory.extract_keyframes()
            fallen_count = sum(1 for f in memory.buffer if f.is_fallen_posture and f.is_stationary)
            timeline_str = ", ".join(time_labels)

            reason = (
                f"YOLO detected stationary fallen posture in {fallen_count}/{len(memory.buffer)} "
                f"consecutive frames. Keyframes sampled at [{timeline_str}]."
            )
            print(f"[VisionInterpreter] 🚨 Threshold met! Forwarding {len(keyframes_jpeg)} keyframes ({timeline_str}) to DeepAgent.")

            await agent.event_queue.put(
                AlertEvent(
                    source="vision",
                    trigger_reason=reason,
                    confidence=avg_conf,
                    frames_jpeg=keyframes_jpeg,
                    timestamp=time.time(),
                )
            )

            # Clear buffer after alerting so we start fresh once DeepAgent returns to MONITORING
            memory.clear()