"""
test_real_samples.py
─────────────────────
Samples N real rows from each class directly out of the processed
parquet file and sends them to the API. This gives a honest picture
of model performance on actual CICIoT2023 data rather than
hand-crafted feature values.

Run:  uv run python test_real_samples.py
      uv run python test_real_samples.py --n 20 --data data/sample/sample_20_percent.parquet
"""
import argparse
import json
import sys
from collections import defaultdict

import polars as pl
import requests

API_URL    = "http://localhost:8000/predict"
FEATURES_PATH = "models/selected_features.json"

CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web_BruteForce"]

# Aliases needed because parquet column names have spaces
ALIASES = {
    "Protocol Type": "Protocol Type",
    "Tot sum":        "Tot sum",
    "Tot size":       "Tot size",
}


def load_features() -> list:
    try:
        with open(FEATURES_PATH) as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"  ⚠  {FEATURES_PATH} not found — run src/feature_selection.py first")
        sys.exit(1)


def load_samples(parquet_path: str, n_per_class: int, features: list) -> pl.DataFrame:
    print(f"Loading data from {parquet_path}…")
    df = pl.read_parquet(parquet_path)

    available = [f for f in features if f in df.columns]
    missing   = set(features) - set(available)
    if missing:
        print(f"  ⚠  Features not in parquet (will be sent as 0): {missing}")

    frames = []
    for cls in CLASSES:
        group = df.filter(pl.col("label_category") == cls)
        n     = min(n_per_class, group.height)
        if n == 0:
            print(f"  ⚠  No rows found for class: {cls}")
            continue
        sampled = group.sample(n=n, shuffle=True, seed=42)
        frames.append(sampled.select(available + ["label_category"]))

    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=42)


def row_to_payload(row: dict, features: list) -> dict:
    payload = {}
    for feat in features:
        val = row.get(feat, 0.0)
        payload[feat] = float(val) if val is not None else 0.0
    return payload


def send(payload: dict) -> dict:
    r = requests.post(API_URL, json=payload, timeout=5)
    r.raise_for_status()
    return r.json()["result"]


def run(parquet_path: str, n_per_class: int):
    # ── Health check ──────────────────────────────────────────────────────
    try:
        requests.get("http://localhost:8000/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print("ERROR: API not reachable. Start it with:")
        print("  uv run uvicorn api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    features = load_features()
    df       = load_samples(parquet_path, n_per_class, features)

    print(f"Testing {df.height} real rows ({n_per_class} per class)…\n")

    # ── Per-class trackers ────────────────────────────────────────────────
    results     = defaultdict(lambda: defaultdict(int))  # results[true][pred]
    conf_totals = defaultdict(float)
    low_conf    = defaultdict(int)
    total_cls   = defaultdict(int)
    errors      = 0

    for row in df.iter_rows(named=True):
        true_label = row["label_category"]
        payload    = row_to_payload(row, features)

        try:
            result     = send(payload)
            pred_label = result["prediction"]
            confidence = result["confidence"]

            results[true_label][pred_label] += 1
            conf_totals[true_label]         += confidence
            total_cls[true_label]           += 1
            if result["low_confidence"]:
                low_conf[true_label] += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Request error: {e}")

    # ── Print results ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  RESULTS ON REAL CICIoT2023 SAMPLES  ({n_per_class} per class)")
    print("=" * 70)

    correct_total = 0
    total_total   = 0

    for cls in CLASSES:
        n       = total_cls[cls]
        if n == 0:
            continue
        correct = results[cls][cls]
        avg_conf = conf_totals[cls] / n
        lc       = low_conf[cls]
        acc      = correct / n

        correct_total += correct
        total_total   += n

        bar     = "█" * int(acc * 20)
        pad     = "░" * (20 - len(bar))
        print(f"\n  {cls:<20}  {correct:>3}/{n}  ({acc:.0%})  {bar}{pad}  avg conf={avg_conf:.2%}  low_conf={lc}")

        # Show where misclassified rows went
        wrong = {k: v for k, v in results[cls].items() if k != cls}
        if wrong:
            ranked = sorted(wrong.items(), key=lambda x: x[1], reverse=True)
            confused = "  |  ".join(f"{k}: {v}" for k, v in ranked)
            print(f"  {'':20}  confused as → {confused}")

    overall = correct_total / total_total if total_total > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"  Overall accuracy on real samples: {correct_total}/{total_total}  ({overall:.1%})")
    print(f"  (Compare to held-out test accuracy: 85%)")
    if errors:
        print(f"  Request errors: {errors}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test API on real parquet samples")
    parser.add_argument(
        "--data", default="data/sample/sample_20_percent.parquet",
        help="Path to parquet file (default: 20%% sample)"
    )
    parser.add_argument(
        "--n", default=50, type=int,
        help="Number of rows to sample per class (default: 50)"
    )
    args = parser.parse_args()
    run(args.data, args.n)