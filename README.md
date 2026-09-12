# Overall Goal: 
This Agentic AI software would have the capabilities of detecting fall / anomaly detection which is used as a real-time event driven fall detection to predict the progression of limb rehabilitation upon the fall based on movement patterns and other sensing data from the ATFL, CFL, and PTFL strains. 

# Week 1 Goal: 
This starter implements the Week 1 goal in the proposal: create labeled IMU data, derive movement features, and train a compact classifier that separates rehabilitation movement from instability and falls.

# What is included
- `data/raw/imu_seed.csv`: a small hand-checkable IMU sample with normal, instability, and fall labels.
- `src/generate_synthetic_data.py`: creates a reproducible, larger synthetic data set for initial development.
- `src/features.py`: extracts window-level features from accelerometer and gyroscope readings.
- `src/train_model.py`: calibrates and evaluates a transparent threshold baseline from labelled sessions.
- `agent.py`: loads the threshold baseline for a future real-time processing loop.


# Week 2 Goal: 
Build the runtime processing loop that receives motion data, executes inference with zero latency, and decides when to escalate an anomaly


