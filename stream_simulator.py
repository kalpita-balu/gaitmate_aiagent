import csv
import time
from collections import deque
from pathlib import Path

import pandas as pd

from agent import GaitMateAgent

WINDOW_SIZE = 200  # 50 Hz * 4 seconds, matches how the model was trained
CONFIRM_THRESHOLD = 5  # require 5 consecutive matching windows before escalating

def simulate_sensor_stream(csv_path, delay=0.02):
    """
    Reads raw IMU rows from a CSV and feeds them through a rolling
    window into GaitMateAgent, one sample at a time, like a live sensor.
    Alerts are debounced: a label must persist for CONFIRM_THRESHOLD
    consecutive windows before it is escalated.
    """
    file_path = Path(csv_path)
    if not file_path.exists():
        print(f"Error: Could not find file at {file_path}")
        return

    agent = GaitMateAgent()
    buffer = deque(maxlen=WINDOW_SIZE)
    confirmed_label = None
    pending_label = None
    pending_count = 0
    print(f"--- Starting sensor stream simulation from {file_path.name} ---")

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                sample = {
                    "ax": float(row["ax"]),
                    "ay": float(row["ay"]),
                    "az": float(row["az"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"]),
                }
                buffer.append(sample)
                current_timestamp = row["timestamp_ms"]
                if len(buffer) == WINDOW_SIZE:
                    window_df = pd.DataFrame(buffer)
                    result = agent.assess_window(window_df)
                    current = result["label"]

                    if current == pending_label:
                        pending_count += 1
                    else:
                        pending_label = current
                        pending_count = 1

                    if pending_count >= CONFIRM_THRESHOLD and confirmed_label != pending_label:
                        confirmed_label = pending_label
                        alert_line = f"[ALERT] t={current_timestamp}ms (packet {row_idx}): {result}"
                        print(alert_line)
                        with open("escalations.log", "a", encoding="utf-8") as log_file:
                            log_file.write(alert_line + "\n")

        
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n--- Sensor simulation stopped by user ---")


if __name__ == "__main__":
    target_csv = "lightweight_ml_model/data/raw/synthetic_imu_streams.csv"
    simulate_sensor_stream(target_csv, delay=0.02)
