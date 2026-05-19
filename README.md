# Efficient-Computing-for-Artificial-Intelligence-

## Introduction
In these three project we explored different challenges of deploying deep learning models on resource-constrained devices.
We applied model optimization techniques such as quantization, pruning, and efficient architecture design to reduce
latency, memory usage, and energy consumption.
Finally, we implemented an IoT system using a Raspberry Pi connected to sensors for real-time data acquisition and local
inference.

## Requirements

- Python 3.11

Install Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `homework1/hygrometer.py` (and `homework3/publisher.py`) are meant to run on a Raspberry Pi with a DHT11 sensor and require GPIO access.
- `sounddevice` may require PortAudio system packages (on macOS: `brew install portaudio`).
- `torch`/`torchaudio` wheels differ by platform; on Raspberry Pi you may need to use compatible wheels for your OS/architecture.

## How to run

### Homework 1 (Raspberry Pi sensor + voice control + Redis)
Runs the hygrometer loop on the Raspberry Pi, listens for voice keywords and writes temperature/humidity into Redis TimeSeries.

```bash
source .venv/bin/activate
python homework1/hygrometer.py --host <redis-host> --port <redis-port> --user <redis-user> --password <redis-password>
```

### Homework 2 (Raspberry Pi sensor + ONNX voice trigger + Redis)
Same idea as homework 1, but uses ONNX models from `./saved_models/`.

```bash
source .venv/bin/activate
python homework2/hygrometer.py --host <redis-host> --port <redis-port> --user <redis-user> --password <redis-password>
```

### Homework 3 (Raspberry Pi sensor publisher over MQTT)
Publishes sensor readings to an MQTT broker (default in code: `broker.emqx.io:1883`) on topic `<topic-id>`.

```bash
source .venv/bin/activate
python homework3/publisher.py
```

### MSC dataset
The MSC dataset is a collection of `.wav` audio files where each clip contains a single spoken keyword. Each file is labeled by **embedding the spoken word in the filename** (the dataset loader extracts the label from the filename prefix).

Expected naming convention (used by `homework1/msc_dataset.py`):
- `<label>_<hash>_nohash_<id>.wav`
- example: `up_<hash>_nohash_<id>.wav`, `stop_<hash>_nohash_<id>.wav`

During training, these labeled audio clips are loaded with TorchAudio and mapped to integer class IDs (based on the `classes` list passed to `MSCDataset`). The resulting dataset is used to train the keyword-spotting / trigger model that is then deployed on-device (e.g., the ONNX pipeline used in homework 2).

#### Train / val / test split
Keep the dataset split into three separate folders so you can train and evaluate without leakage:

```
msc_dataset/
  msc-train/
    up_...wav
    stop_...wav
  msc-val/
    up_...wav
    stop_...wav
  msc-test/
    up_...wav
    stop_...wav
```

You then create one `MSCDataset` instance per split by passing the corresponding folder path (and the same ordered `classes` list) to the loader.

Implementation: `homework1/msc_dataset.py`.
