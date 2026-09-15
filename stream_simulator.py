import csv
import time
from pathlib import Path

def simulate_sensor_stream(csv_path, delay=0.2):
    """
    Reads a CSV file row-by-row with a delay to mimic a live IMU data stream.
    """
    file_path = Path(csv_path)

    if not file_path.exists():
        print(f"Error: Could not find file at {file_path}")
        return

    print(f"--- Starting sensor stream simulation from {file_path.name} ---")

    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row_idx, row in enumerate(reader):
                print(f"[Packet {row_idx}] Data: {row}")
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n--- Sensor simulation stopped by user ---")

if __name__ == "__main__":
    # Points to your synthetic data file inside lightweight_ml_model
    target_csv = "lightweight_ml_model/raw_data/synthetic_imu_streams.csv"
    simulate_sensor_stream(target_csv, delay=0.2)
