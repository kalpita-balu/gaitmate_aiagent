"""Replay IMU CSV data as a local, real-time GaitMate event stream.

The browser UI polls ``ui_event.json`` from the same project folder. This is a
development bridge for demos only; it does not transmit data to a network.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent import GaitMateAgent

PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_SIZE = 200  # 50 Hz × 4 seconds, matching the Week 1 training window.
# One synthetic CSV session contains exactly one 200-sample model window.
# Use --confirm 5 with a continuous live stream to debounce five predictions.
DEFAULT_CONFIRM_THRESHOLD = 1
INFERENCE_STRIDE = 10  # Assess five times per second at a 50 Hz sampling rate.
EVENT_FILE = PROJECT_ROOT / "ui_event.json"
LOG_FILE = PROJECT_ROOT / "escalations.log"


def write_latest_event(event: dict) -> None:
    """Atomically replace the latest event so the browser never reads a partial JSON file."""
    temporary = EVENT_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(event, indent=2), encoding="utf-8")
    temporary.replace(EVENT_FILE)


def simulate_sensor_stream(
    csv_path: str,
    delay: float = 0.02,
    session_id: str | None = None,
    confirm_threshold: int = DEFAULT_CONFIRM_THRESHOLD,
) -> None:
    """Replay one CSV stream, confirm persistent labels, and publish local UI events."""
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find CSV data at {file_path}")

    agent = GaitMateAgent()
    buffer: deque[dict[str, float]] = deque(maxlen=WINDOW_SIZE)
    confirmed_label: str | None = None
    pending_label: str | None = None
    pending_count = 0
    active_session: str | None = None
    session_sample_index = 0
    event_id = 0

    print(f"--- Starting local replay from {file_path.name} ---")
    try:
        with file_path.open(mode="r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                row_session_id = row["session_id"]
                if session_id and row_session_id != session_id:
                    continue

                if row_session_id != active_session:
                    buffer.clear()
                    confirmed_label = None
                    pending_label = None
                    pending_count = 0
                    session_sample_index = 0
                    active_session = row_session_id
                    print(f"Streaming session: {active_session}")

                sample = {name: float(row[name]) for name in ("ax", "ay", "az", "gx", "gy", "gz")}
                buffer.append(sample)
                session_sample_index += 1

                if len(buffer) == WINDOW_SIZE and session_sample_index % INFERENCE_STRIDE == 0:
                    result = agent.assess_window(pd.DataFrame(buffer))
                    current_label = str(result["label"])
                    if current_label == pending_label:
                        pending_count += 1
                    else:
                        pending_label = current_label
                        pending_count = 1

                    if pending_count >= confirm_threshold and confirmed_label != pending_label:
                        confirmed_label = pending_label
                        event_id += 1
                        event = {
                            "event_id": event_id,
                            "session_id": active_session,
                            "timestamp_ms": int(row["timestamp_ms"]),
                            "received_at": datetime.now(timezone.utc).isoformat(),
                            "label": current_label,
                            "severity": result["severity"],
                            "decision_score": result["decision_score"],
                            "sensor_status": "connected",
                        }
                        write_latest_event(event)
                        line = f"[EVENT] {json.dumps(event)}"
                        print(line)
                        with LOG_FILE.open("a", encoding="utf-8") as log_file:
                            log_file.write(line + "\n")

                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n--- Sensor replay stopped by user ---")
    finally:
        print("--- Replay finished. The latest confirmed event remains visible in the UI. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay GaitMate IMU CSV data into the local UI.")
    parser.add_argument("--session", help="Optional session ID, for example fall_001")
    parser.add_argument("--delay", type=float, default=0.02, help="Seconds between samples; 0.02 matches 50 Hz")
    parser.add_argument("--confirm", type=int, default=DEFAULT_CONFIRM_THRESHOLD, help="Matching predictions required before publishing")
    args = parser.parse_args()
    simulate_sensor_stream(
        str(PROJECT_ROOT / "lightweight_ml_model" / "data" / "raw" / "synthetic_imu_streams.csv"),
        delay=args.delay,
        session_id=args.session,
        confirm_threshold=args.confirm,
    )
