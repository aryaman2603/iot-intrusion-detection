"""
evaluate.py
────────────
Standalone evaluation script. Loads the trained model and produces:
  1. Full classification report on held-out test set
  2. Confusion matrix (printed as text table)
  3. Top-20 SHAP feature importances (printed)
  4. Per-class threshold comparison (before vs after tuning)

Run:  python -m src.evaluate
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import (
    CLASSES,
    FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_LABELS_PATH,
    TEST_PROBA_PATH,
    THRESHOLDS_PATH,
    DEFAULT_FEATURES,
    TEST_SIZE,
)


def _apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = proba / thresholds[np.newaxis, :]
    return np.argmax(adjusted, axis=1)


def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    col_w = 14
    header = f"{'':20}" + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<20}" + "".join(f"{v:>{col_w},}" for v in row)
        print(row_str)


def run_evaluation():
    # ── Load artifacts ─────────────────────────────────────────────────────
    le      = joblib.load(LABEL_ENCODER_PATH)
    scaler  = joblib.load(SCALER_PATH)
    booster = lgb.Booster(model_file=MODEL_PATH)

    with open(FEATURES_PATH) as fh:
        features = json.load(fh)

    with open(THRESHOLDS_PATH) as fh:
        threshold_dict = json.load(fh)
    thresholds = np.array([threshold_dict[c] for c in le.classes_])

    # ── Load test data (reproduce same split) ─────────────────────────────
    proba  = np.load(TEST_PROBA_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    # ── 1. Baseline report ────────────────────────────────────────────────
    baseline = np.argmax(proba, axis=1)
    print("=" * 65)
    print("BASELINE (argmax, no threshold tuning)")
    print("=" * 65)
    print(classification_report(y_test, baseline, target_names=le.classes_))
    print(f"  Macro F1: {f1_score(y_test, baseline, average='macro'):.4f}\n")

    # ── 2. Tuned report ───────────────────────────────────────────────────
    tuned = _apply_thresholds(proba, thresholds)
    print("=" * 65)
    print("TUNED (per-class thresholds applied)")
    print("=" * 65)
    print(classification_report(y_test, tuned, target_names=le.classes_))
    print(f"  Macro F1: {f1_score(y_test, tuned, average='macro'):.4f}\n")

    # ── 3. Confusion matrix ───────────────────────────────────────────────
    print("=" * 65)
    print("CONFUSION MATRIX (tuned predictions)")
    print("=" * 65)
    print_confusion_matrix(y_test, tuned, list(le.classes_))

    # ── 4. Feature importance (LightGBM split-based) ──────────────────────
    print("\n" + "=" * 65)
    print("TOP-20 FEATURE IMPORTANCES (split count)")
    print("=" * 65)
    importances = booster.feature_importance(importance_type="split")
    feat_imp = sorted(
        zip(features, importances), key=lambda x: x[1], reverse=True
    )
    for rank, (feat, imp) in enumerate(feat_imp[:20], 1):
        bar = "█" * int(imp / max(i for _, i in feat_imp[:20]) * 30)
        print(f"  {rank:2d}. {feat:<35} {imp:>6}  {bar}")


if __name__ == "__main__":
    run_evaluation()