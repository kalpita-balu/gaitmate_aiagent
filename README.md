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


## Week 2 Goal: 

Week 2 implements the runtime processing loop described above:
a simulated live sensor feed, a rolling window buffer, real-time inference
using the Week 1 threshold model, and debounced escalation of anomalies.

### What is included

- `stream_simulator.py`: reads IMU samples row-by-row from a CSV to
  simulate a live 50 Hz sensor feed, maintains a rolling 200-sample
  (4-second) window, and calls `GaitMateAgent.assess_window()` on every
  full window.
- `escalations.log`: a persisted, append-only log of every escalated
  alert (timestamp, label, severity, decision score).
- `agent.py`: fixed import and file paths so `GaitMateAgent` correctly
  loads the trained model from `lightweight_ml_model/artifacts/`.

### Design notes

- **Debouncing:** a classification must hold for `CONFIRM_THRESHOLD` (5)
  consecutive windows before it is escalated. This was added after
  observing that a single sliding window can partially overlap a short
  anomaly injection (the fall spike is only 18 samples), causing the
  raw per-window label to flicker between `normal`, `instability`, and
  `fall` as the window slides across the event. Debouncing trades a
  small amount of latency (~100ms) for a stable, meaningful alert
  stream. This may be revisited to give `fall` an immediate,
  non-debounced escalation path, matching the "immediate alert bypass"
  behavior described in the Week 2 plan.
- **Startup gap:** no classification is produced until the buffer fills
  with `WINDOW_SIZE` (200) samples, i.e. the first ~4 seconds of any
  stream produce no output.
- **Blocking loop:** the current simulator reads a CSV synchronously in
  a single thread. Connecting to a real, continuous sensor source
  (e.g. WebSocket/MQTT) will likely require converting this to an
  async or event-driven loop.

### How to run

```bash
python3 stream_simulator.py
```

## Week 3 UI prototype

`index.html` provides a local, branded patient-facing prototype with calibration, simulated sensor status, event logging, symptom check-ins, progress, care-team, and exercise flows. Values and guidance remain simulated and non-clinical.

Run the UI locally:

```bash
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000` in a browser.

### Replay bridge

`stream_replay.py` replays a labeled IMU session through `GaitMateAgent` and writes the latest confirmed event to a local `ui_event.json` file. The browser polls this file and updates the dashboard. The event file stays untracked because it is runtime output.

```bash
python3 stream_replay.py --session fall_001
```

For a future continuous stream, increase the debounce setting:

```bash
python3 stream_replay.py --confirm 5
```

