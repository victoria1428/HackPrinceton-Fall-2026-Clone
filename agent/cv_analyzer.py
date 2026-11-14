import asyncio
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from inference_sdk import InferenceHTTPClient
from PIL import Image

# --- ROBOFLOW CONFIGURATION (From Teammate's Colab) ---
MODEL_ID = "victorias-workspace-lw5e8/daily-living-and-fall-detection-1-yolo26n-t1"
FALL_NAME = "falling_emergency_attention"
STAY_SECS = 5
SECS_BETWEEN = 1.0


class RoboflowCVAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.client: Optional[InferenceHTTPClient] = None
        if self.api_key:
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key,
            )

    def _infer_single_frame(self, frame_bytes: bytes) -> dict[str, Any]:
        """
        Evaluates a single frame (accepts either PNG or JPEG bytes in memory).
        Returns label (1=fall, 0=daily life, None=no pred), class_name, and confidence.
        """
        if not self.client:
            return {"label": None, "class": "no_api_key", "confidence": 0.0}

        try:
            # Convert raw PNG/JPEG bytes in RAM to a PIL Image so Roboflow SDK accepts it
            pil_image = Image.open(io.BytesIO(frame_bytes))
            raw_result = self.client.infer(pil_image, model_id=MODEL_ID)

            # Narrow type from `dict | list[dict]` so Pylance knows `.get()` is valid
            if isinstance(raw_result, list):
                result: dict[str, Any] = raw_result[0] if raw_result else {}
            elif isinstance(raw_result, dict):
                result = raw_result
            else:
                result = {}

            preds = result.get("predictions", [])
            if not isinstance(preds, list) or not preds:
                return {"label": None, "class": "none", "confidence": 0.0}

            best = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
            is_fall = 1 if best.get("class") == FALL_NAME else 0
            return {
                "label": is_fall,
                "class": str(best.get("class", "unknown")),
                "confidence": float(best.get("confidence", 0.0)),
                "bbox": {
                    "x": best.get("x"),
                    "y": best.get("y"),
                    "width": best.get("width"),
                    "height": best.get("height"),
                },
            }
        except Exception as e:
            print(f"[CVAnalyzer] Frame inference error: {e}")
            return {"label": None, "class": "error", "confidence": 0.0}

    def check_frames_sync(self, frames_bytes: list[bytes], max_workers: int = 5) -> dict[str, Any]:
        """
        Runs Roboflow inference in parallel across the sampled keyframes
        using your teammate's exact STAY_SECS / SECS_BETWEEN logic.
        """
        if not frames_bytes:
            return {"alert_status": 0, "falls_detected": 0, "total_frames": 0, "frame_details": []}

        # Offline simulation fallback if ROBOFLOW_API_KEY is not set yet
        if not self.client:
            print("[CVAnalyzer] No ROBOFLOW_API_KEY set -> returning simulated CV report.")
            return {
                "alert_status": 1,
                "falls_detected": len(frames_bytes),
                "total_frames": len(frames_bytes),
                "avg_confidence": 0.92,
                "model_id": MODEL_ID,
                "simulated": True,
            }

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            details = list(executor.map(self._infer_single_frame, frames_bytes))

        labels = [d["label"] for d in details]
        need = min(int(STAY_SECS / SECS_BETWEEN), len(labels))  # 5 frames (or fewer if <5 sent)
        last = labels[-need:] if need > 0 else []
        falls = sum(1 for lbl in last if lbl == 1)

        # Teammate's threshold: falls >= need - 1 (e.g., at least 4 of 5 frames show a fall)
        required_falls = max(1, need - 1)
        alert_status = 1 if (len(last) == need and falls >= required_falls) else 0

        fall_confs = [d["confidence"] for d in details if d["label"] == 1]
        avg_conf = sum(fall_confs) / len(fall_confs) if fall_confs else 0.0

        elapsed = time.time() - start_time
        print(
            f"[CVAnalyzer] Processed {len(frames_bytes)} frames in {elapsed:.2f}s | "
            f"Falls: {falls}/{need} -> Alert Status: {alert_status}"
        )

        return {
            "alert_status": alert_status,
            "falls_detected": falls,
            "frames_evaluated": need,
            "avg_confidence": round(avg_conf, 3),
            "model_id": MODEL_ID,
            "frame_details": details,
        }

    async def analyze_frames(self, frames_bytes: list[bytes]) -> dict[str, Any]:
        """Non-blocking async wrapper called by DeepAgent."""
        return await asyncio.to_thread(self.check_frames_sync, frames_bytes)