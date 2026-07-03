# IoT Network Intrusion Detection System

A machine learning system for real-time classification of IoT network traffic into seven categories — Benign, DDoS, DoS, Mirai, Recon, Spoofing, and Web_BruteForce — using a LightGBM model trained on the [CICIoT 2023 dataset](https://www.mdpi.com/1424-8220/23/13/5941). Deployed as a FastAPI service on Hugging Face Spaces via Docker.

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 0.78 | 0.86 | 0.82 |
| DDoS | 0.86 | 0.97 | **0.91** |
| DoS | 0.72 | 0.31 | 0.44 |
| Mirai | 1.00 | 1.00 | **1.00** |
| Recon | 0.76 | 0.70 | 0.73 |
| Spoofing | 0.89 | 0.78 | 0.83 |
| Web_BruteForce | 0.16 | 0.20 | 0.18 |
| **Macro avg** | **0.74** | **0.69** | **0.70** |

> Trained on 20% stratified sample (~1.6M flows), 3-fold CV, after per-class threshold tuning. Evaluated on 244,415 held-out test flows.

---

## Project Structure

```
iot-intrusion-detection/
├── data/
│   ├── raw/                        # Stratified_data.csv (not committed)
│   ├── processed/                  # ciciot2023_clean.parquet (not committed)
│   └── sample/                     # sample_20_percent.parquet (not committed)
├── models/
│   ├── lgbm_model.txt              # Trained LightGBM model (Git LFS)
│   ├── lgb_label_encoder.pkl       # Label encoder (Git LFS)
│   ├── scaler.pkl                  # RobustScaler (Git LFS)
│   ├── selected_features.json      # 18 selected feature names
│   └── thresholds.json             # Per-class decision thresholds
├── src/
│   ├── config.py                   # All paths, hyperparameters, constants
│   ├── preprocessing.py            # CSV → Parquet, stratified sampling
│   ├── feature_selection.py        # 3-stage feature reduction (47 → 18)
│   ├── training.py                 # LightGBM CV, evaluation, artifact saving
│   ├── threshold_tuning.py         # Per-class threshold optimisation
│   └── evaluate.py                 # Reports, confusion matrix, importances
├── api/
│   ├── main.py                     # FastAPI routes and middleware
│   ├── predictor.py                # Inference engine (loads once at startup)
│   └── schema.py                   # Pydantic request/response models
├── tests/
│   ├── conftest.py                 # Shared fixtures and mock data
│   ├── functional/                 # FR-17 to FR-22 (40 tests)
│   ├── non_functional/             # Performance, reliability, accuracy (26 tests)
│   └── integration/                # Realistic attack flow tests (27 tests)
├── capture.py                      # Live traffic → API via tshark
├── test_predictions.py             # 10 hand-crafted example predictions
├── test_real_samples.py            # Sample real parquet rows → API
├── main.py                         # Full pipeline orchestrator
├── Dockerfile                      # For Hugging Face Spaces deployment
└── pyproject.toml                  # uv dependencies
```

---

## Quickstart

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), raw dataset at `data/raw/Stratified_data.csv`

```bash
# Install dependencies
uv sync

# Run the full pipeline (preprocess → feature select → train → tune → evaluate)
uv run python main.py

# Or run individual steps
uv run python -m src.preprocessing
uv run python -m src.feature_selection
uv run python -m src.training
uv run python -m src.threshold_tuning
uv run python -m src.evaluate

# Start the API
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model status, features loaded, per-class thresholds |
| `GET` | `/info` | Full feature list, class names, model metadata |
| `POST` | `/predict` | Classify a single network flow |
| `POST` | `/predict/batch` | Classify up to 1,000 flows in one request |

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Header_Length": 40,
    "Protocol Type": 6,
    "Time_To_Live": 64,
    "syn_flag_number": 1,
    "ack_flag_number": 0,
    "ack_count": 0,
    "TCP": 1,
    "Tot sum": 48000,
    "Min": 40,
    "AVG": 40,
    "Number": 1200
  }'
```

```json
{
  "status": "ok",
  "result": {
    "prediction": "DDoS",
    "confidence": 0.934,
    "probabilities": {
      "Benign": 0.01, "DDoS": 0.934, "DoS": 0.04,
      "Mirai": 0.008, "Recon": 0.005, "Spoofing": 0.002,
      "Web_BruteForce": 0.001
    },
    "is_attack": true,
    "low_confidence": false
  }
}
```

`low_confidence` is `true` when the model confidence is below 60% — use this to route borderline predictions to human review rather than acting automatically.

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"flows": [{...}, {...}]}'
```

### JavaScript (for frontend integration)

```javascript
const response = await fetch("https://YOUR-SPACE.hf.space/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ "syn_flag_number": 1, "TCP": 1, "Number": 1200 })
});
const data = await response.json();
console.log(data.result.prediction);   // "DDoS"
console.log(data.result.is_attack);    // true
```

---

## Input Features

The model uses 18 features extracted from network flow statistics:

| Feature | Description |
|---|---|
| `Header_Length` | IP + TCP header size in bytes |
| `Protocol Type` | IP protocol number (6=TCP, 17=UDP, 1=ICMP, 47=GRE) |
| `Time_To_Live` | Average TTL across packets in the flow |
| `fin_flag_number` | Count of FIN-flagged packets |
| `syn_flag_number` | Count of SYN-flagged packets (high = possible flood) |
| `rst_flag_number` | Count of RST-flagged packets (high = scan or flood) |
| `psh_flag_number` | Count of PSH-flagged packets (data being pushed) |
| `ack_flag_number` | Count of ACK-flagged packets |
| `ack_count` | Count of bare ACK packets (no data payload) |
| `HTTP` | 1 if destination port is 80 |
| `HTTPS` | 1 if destination port is 443 |
| `TCP` | 1 if protocol is TCP |
| `UDP` | 1 if protocol is UDP |
| `ICMP` | 1 if protocol is ICMP |
| `Tot sum` | Total bytes transferred in the flow |
| `Min` | Minimum packet size in bytes |
| `AVG` | Average packet size in bytes |
| `Number` | Total packet count in the flow |

Missing features are imputed with `0.0` at inference time.

---

## How It Works

### Feature Selection

47 raw CICIoT 2023 features are reduced to 18 using a three-stage pipeline:

1. **Variance filter** — drops features with near-zero normalised variance (no signal)
2. **Correlation filter** — drops one of each pair with |r| > 0.98, keeping the feature with higher LightGBM importance
3. **Consensus ranking** — ranks remaining features by average of LightGBM split importance and permutation importance, keeps top 18

### Why LightGBM over XGBoost

- Native `class_weight='balanced'` handles the 900:1 DDoS/Web_BruteForce imbalance automatically
- Leaf-wise growth finds sharper minority class boundaries
- 3–5× faster training on large datasets with `n_jobs=-1`
- `softprob` output enables per-class threshold tuning

### Per-Class Threshold Tuning

Default `argmax(proba)` uses a 0.5 threshold for every class — a poor fit for imbalanced data. After training, a grid search finds the threshold per class that maximises binary F1:

```
Benign: 0.27   DDoS: 0.16   DoS: 0.60
Mirai:  0.86   Recon: 0.42  Spoofing: 0.62   Web_BruteForce: 0.63
```

At inference: `argmax(proba / threshold)`. This lifted DDoS recall from 0.69 → 0.97 without retraining.

---

## Training on Your Own Hardware

The default config trains on a 20% stratified sample — suitable for M2 Pro MacBook (~15 minutes). To switch:

```python
# In src/config.py

# Development (fast)
TRAIN_DATA_PATH = SAMPLE_20_PATH   # ~1.6M rows
N_CV_FOLDS = 3

# Production (full dataset — needs ~60min on M2 Pro or GPU machine)
TRAIN_DATA_PATH = PROCESSED_PATH   # 8.1M rows
N_CV_FOLDS = 5
```

---

## Testing

```bash
# Run full test suite (no model files needed — uses mocked predictor)
uv run pytest tests/ -v

# By category
uv run pytest tests/functional/ -v        # FR-17 to FR-22 (40 tests)
uv run pytest tests/non_functional/ -v    # Performance + reliability (26 tests)
uv run pytest tests/integration/ -v       # Realistic attack flows (27 tests)

# With coverage
uv run pytest tests/ --cov=api --cov-report=term-missing
```

107 test cases, 100% pass rate. No model files or network access required.

---

## Live Traffic Testing

Requires [Wireshark/tshark](https://www.wireshark.org/):

```bash
brew install wireshark

# Capture live traffic and classify flows in real time
sudo uv run python capture.py --interface en0 --duration 60

# Test with hand-crafted example flows
uv run python test_predictions.py

# Test with real rows sampled from the parquet dataset
uv run python test_real_samples.py --n 50
```

> **Note:** The model is trained on CICIoT 2023 synthetic IoT lab data. It is designed for IoT-dedicated network segments (factory floors, smart building subnets), not general laptop traffic. Real browsing traffic may be misclassified.

---

## Deployment

### Hugging Face Spaces (Docker)

```bash
# 1. Generate pip-compatible requirements
uv export --no-dev --no-hashes > requirements.txt

# 2. Stage model artifacts (previously gitignored)
git rm -r --cached models/
git add .
git commit -m "add model artifacts and deployment files"

# 3. Push to Hugging Face
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/iot-intrusion-detection
git push hf main
```

Hugging Face auto-builds the Docker image on every push. The API is live at:
```
https://YOUR_USERNAME-iot-intrusion-detection.hf.space
```

Swagger docs: `https://YOUR_USERNAME-iot-intrusion-detection.hf.space/docs`

> **Free tier note:** Spaces sleep after 48 hours of inactivity. The first request after idle takes ~30 seconds to wake.

### Local Docker

```bash
docker build -t iot-ids .
docker run -p 7860:7860 iot-ids
```

---

## Known Limitations

- **Web_BruteForce F1 = 0.18** — data starvation: only 6,552 raw samples (900× rarer than DDoS). Will improve significantly with full dataset training.
- **DoS F1 = 0.44** — feature overlap with DDoS. Both are flood-based attacks with similar flag and packet size distributions.
- **Synthetic training data** — CICIoT 2023 is lab-generated. Real IoT device traffic may have different distributions. Periodic retraining recommended.
- **No concept drift detection** — model is a static snapshot of 2023 attack patterns. New attack variants will not be recognised without retraining.

---

## Dataset

[CICIoT 2023](https://www.mdpi.com/1424-8220/23/13/5941) — Canadian Institute for Cybersecurity, 2023.

| Stat | Value |
|---|---|
| Total flows | 8,147,161 |
| Raw features | 47 |
| Classes | 7 |
| Largest class | DDoS (5,888,229 flows) |
| Smallest class | Web_BruteForce (6,552 flows) |

---

## References

1. Neto, E. et al. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environments.* Sensors, 23(13), 5941.
2. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
3. Rashid, M. et al. (2024). *A Long Short-Term Memory Based Approach for Detecting Cyber Attacks in IoT Using CIC-IoT2023 Dataset.*
