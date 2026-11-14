import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# Ensures 'HackPrinceton Fall 2026' is on the path if you run this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.deep_agent import AgentState, AlertEvent, DeepAgent


@dataclass
class VisionFramePacket:
    source_frame_jpeg: bytes                     # Original camera frame the skeleton came from
    skeleton_frame_jpeg: Optional[bytes] = None  # The YOLO skeleton frame
    is_fallen: bool = False
    risk_score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def frame_jpeg(self) -> bytes:
        """Backward-compatible alias for source_frame_jpeg."""
        return self.source_frame_jpeg


@dataclass
class VisionMemoryConfig:
    max_frames: int = 100            # 100 frames (~20s at 5 FPS)
    min_frames_before_eval: int = 50 # Wait until at least 50 frames are in memory
    fall_frame_threshold: int = 75   # Trigger when 75 of 100 frames have is_fallen=True
    keyframes_to_send: int = 5       # Send 5 frames (t=0s, 5s, 10s, 15s, 20s) to Gemini
    risk_threshold: float = 0.50     # If teammate passes risk float instead of bool, >= 0.50 counts as fallen
    cooldown_sec: float = 15.0


def _normalize_vision_input(
    raw_item: Union[VisionFramePacket, tuple[Any, ...], dict[str, Any], bytes],
    risk_thresh: float = 0.50,
) -> VisionFramePacket:
    if isinstance(raw_item, VisionFramePacket):
        return raw_item

    if isinstance(raw_item, tuple):
        if len(raw_item) == 0:
            raise ValueError("Empty tuple passed to raw_vision_queue.")

        # Support 3-tuple: (source_frame_jpeg, skeleton_frame_jpeg, is_fallen_or_risk)
        if len(raw_item) >= 3 and isinstance(raw_item[1], (bytes, type(None))):
            source_jpeg = raw_item[0]
            skeleton_jpeg = raw_item[1]
            status_val = raw_item[2]
        else:
            source_jpeg = raw_item[0]
            skeleton_jpeg = None
            status_val = raw_item[1] if len(raw_item) > 1 else True

        if isinstance(status_val, bool):
            return VisionFramePacket(
                source_frame_jpeg=source_jpeg,
                skeleton_frame_jpeg=skeleton_jpeg,
                is_fallen=status_val,
                risk_score=0.9 if status_val else 0.1,
            )
        if isinstance(status_val, (int, float)):
            score = float(status_val)
            return VisionFramePacket(
                source_frame_jpeg=source_jpeg,
                skeleton_frame_jpeg=skeleton_jpeg,
                is_fallen=(score >= risk_thresh),
                risk_score=score,
            )

    if isinstance(raw_item, dict):
        source_jpeg = (
            raw_item.get("source_frame_jpeg")
            or raw_item.get("source_jpeg")
            or raw_item.get("frame_jpeg")
            or raw_item.get("jpeg")
            or raw_item.get("image")
            or b""
        )
        skeleton_jpeg = raw_item.get("skeleton_frame_jpeg") or raw_item.get("skeleton_jpeg")
        risk = float(raw_item.get("risk_score", raw_item.get("risk", raw_item.get("confidence", 0.85))))
        down = raw_item.get("is_fallen", raw_item.get("down", risk >= risk_thresh))
        return VisionFramePacket(
            source_frame_jpeg=source_jpeg,
            skeleton_frame_jpeg=skeleton_jpeg,
            is_fallen=bool(down),
            risk_score=risk,
        )

    if isinstance(raw_item, bytes):
        return VisionFramePacket(
            source_frame_jpeg=raw_item,
            is_fallen=True,
            risk_score=0.85,
        )

    raise ValueError(f"Unsupported vision queue item type: {type(raw_item)}")


class VisionShortTermMemory:
    def __init__(self, config: Optional[VisionMemoryConfig] = None):
        self.config = config or VisionMemoryConfig()
        self.buffer: deque[VisionFramePacket] = deque(maxlen=self.config.max_frames)
        self._last_alert_time: float = 0.0

    def update_memory_size(self, new_max_frames: int, new_threshold: int, new_min_frames: int = 50):
        """Adjusts the sliding window capacity on the fly."""
        self.config.max_frames = new_max_frames
        self.config.fall_frame_threshold = new_threshold
        self.config.min_frames_before_eval = new_min_frames
        self.buffer = deque(self.buffer, maxlen=new_max_frames)

    def add_frame(self, packet: VisionFramePacket) -> bool:
        # Appending to deque(maxlen=100) automatically drops frame 0 and shifts left by 1!
        self.buffer.append(packet)

        if len(self.buffer) < self.config.min_frames_before_eval:
            return False

        if (time.time() - self._last_alert_time) < self.config.cooldown_sec:
            return False

        fallen_count = sum(1 for p in self.buffer if p.is_fallen)
        if fallen_count >= self.config.fall_frame_threshold:
            self._last_alert_time = time.time()
            return True

        return False

    def extract_keyframes(self) -> tuple[list[bytes], list[bytes], list[str], float]:
        """
        Returns (sampled_source_jpegs, sampled_skeleton_jpegs, time_labels, avg_conf).
        """
        total = len(self.buffer)
        if total == 0:
            return [], [], [], 0.0

        k = min(self.config.keyframes_to_send, total)
        indices = [total - 1] if k == 1 else [round(i * (total - 1) / (k - 1)) for i in range(k)]

        t_start = self.buffer[0].timestamp
        sampled_source_jpegs: list[bytes] = []
        sampled_skeleton_jpegs: list[bytes] = []
        time_labels: list[str] = []

        for idx in indices:
            pkt = self.buffer[idx]
            sampled_source_jpegs.append(pkt.source_frame_jpeg)
            if pkt.skeleton_frame_jpeg:
                sampled_skeleton_jpegs.append(pkt.skeleton_frame_jpeg)
            elapsed = max(0.0, pkt.timestamp - t_start)
            time_labels.append(f"t={elapsed:.1f}s")

        fallen_scores = [p.risk_score for p in self.buffer if p.is_fallen]
        avg_conf = sum(fallen_scores) / len(fallen_scores) if fallen_scores else 0.85
        return sampled_source_jpegs, sampled_skeleton_jpegs, time_labels, avg_conf

    def clear(self):
        self.buffer.clear()


async def vision_interpreter_loop(
    agent: DeepAgent,
    raw_vision_queue: asyncio.Queue[Any],
    config: Optional[VisionMemoryConfig] = None,
):
    memory = VisionShortTermMemory(config=config)
    print(
        f"[VisionInterpreter] Online. Window={memory.config.max_frames} frames, "
        f"Trigger={memory.config.fall_frame_threshold} DOWN frames."
    )

    while True:
        raw_item = await raw_vision_queue.get()
        packet = _normalize_vision_input(raw_item, memory.config.risk_threshold)

        # Keep DeepAgent's 4-frame ring-buffer updated so audio triggers have visual context
        if len(memory.buffer) % 5 == 0 and packet.source_frame_jpeg:
            agent.latest_frames.append(packet.source_frame_jpeg)

        if agent.state != AgentState.MONITORING:
            continue

        if memory.add_frame(packet):
            source_keyframes, skeleton_keyframes, time_labels, avg_conf = memory.extract_keyframes()
            fallen_count = sum(1 for p in memory.buffer if p.is_fallen)
            timeline_str = ", ".join(time_labels)

            reason = (
                f"YOLO Pose marked person DOWN in {fallen_count}/{len(memory.buffer)} "
                f"frames. Keyframes sampled at [{timeline_str}]."
            )
            print(
                f"[VisionInterpreter] Threshold met! Forwarding {len(source_keyframes)} "
                f"keyframes ({timeline_str}) to DeepAgent."
            )

            # Build AlertEvent compatible with your current DeepAgent (frames_jpeg)
            # and populate skeleton_frames_jpeg if defined on AlertEvent
            alert_kwargs: dict[str, Any] = {
                "source": "vision",
                "trigger_reason": reason,
                "confidence": avg_conf,
                "frames_jpeg": source_keyframes,
                "timestamp": time.time(),
            }
            if "skeleton_frames_jpeg" in AlertEvent.__dataclass_fields__:
                alert_kwargs["skeleton_frames_jpeg"] = skeleton_keyframes

            await agent.event_queue.put(
              AlertEvent(
                  source="vision",
                  trigger_reason=reason,
                  confidence=avg_conf,
                  source_frames_jpeg=source_keyframes,
                  skeleton_frames_jpeg=skeleton_keyframes,
                  timestamp=time.time(),
              )
            )
            memory.clear()