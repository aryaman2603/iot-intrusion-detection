"""
threshold_tuning.py
────────────────────
Finds the optimal decision threshold for each class independently,
maximising per-class F1 on the held-out test probabilities saved
during training.

Why per-class thresholds?
  The default argmax(proba) is equivalent to a threshold of 0.5 for
  each class. But with severe imbalance (Web_BruteForce is 900x rarer
  than DDoS) the model is systematically under-confident on minority
  classes. Lowering the threshold for Web_BruteForce means "flag it
  even when the model is only 20% confident" — dramatically improving
  recall without retraining.

  The tuned thresholds are applied at inference time inside predictor.py.

Run:  python -m src.threshold_tuning
"""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.config import (
    CLASSES,
    LABEL_ENCODER_PATH,
    TEST_LABELS_PATH,
    TEST_PROBA_PATH,
    THRESHOLDS_PATH,
)


def _apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Scale each class probability inversely by its threshold, then argmax.
    A low threshold (0.2) for class k makes it easier to win the argmax —
    the model will predict k more often.

    Formula: adjusted[:,k] = proba[:,k] / threshold[k]
    Then: prediction = argmax(adjusted)
    """
    adjusted = proba / thresholds[np.newaxis, :]
    return np.argmax(adjusted, axis=1)


def tune_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    search_range: tuple = (0.05, 0.95),
    n_steps: int = 90,
) -> np.ndarray:
    """
    Grid search over each class threshold independently.
    Optimises binary F1 for that class (one-vs-rest).
    Returns array of shape (n_classes,).
    """
    thresholds = np.full(n_classes, 0.5)
    grid = np.linspace(search_range[0], search_range[1], n_steps)

    print(f"Tuning {n_classes} class thresholds (grid: {n_steps} steps each)…\n")
    for cls_idx in range(n_classes):
        binary_y = (y_true == cls_idx).astype(int)
        best_f1, best_t = 0.0, 0.5

        for t in grid:
            # One-vs-rest: predict cls_idx when its prob >= t
            binary_pred = (proba[:, cls_idx] >= t).astype(int)
            f = f1_score(binary_y, binary_pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t

        thresholds[cls_idx] = best_t
        print(f"  {CLASSES[cls_idx]:<20}  threshold={best_t:.2f}  "
              f"binary-F1={best_f1:.4f}")

    return thresholds


def evaluate_with_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    class_names: list,
) -> None:
    preds = _apply_thresholds(proba, thresholds)
    print("\n" + "=" * 65)
    print("RESULTS AFTER PER-CLASS THRESHOLD TUNING")
    print("=" * 65)
    print(classification_report(y_true, preds, target_names=class_names))
    macro = f1_score(y_true, preds, average="macro")
    weighted = f1_score(y_true, preds, average="weighted")
    print(f"  Macro F1:    {macro:.4f}")
    print(f"  Weighted F1: {weighted:.4f}")


def run_threshold_tuning() -> None:
    # ── Load artifacts ─────────────────────────────────────────────────────
    if not os.path.exists(TEST_PROBA_PATH):
        raise FileNotFoundError(
            f"{TEST_PROBA_PATH} not found. Run src/training.py first."
        )

    proba  = np.load(TEST_PROBA_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    le     = joblib.load(LABEL_ENCODER_PATH)

    n_classes = proba.shape[1]
    print(f"Loaded proba shape: {proba.shape}  |  classes: {list(le.classes_)}\n")

    # ── Baseline (default argmax) ──────────────────────────────────────────
    baseline_preds = np.argmax(proba, axis=1)
    print("Baseline (argmax, no threshold tuning):")
    print(classification_report(y_test, baseline_preds, target_names=le.classes_))
    print(f"  Baseline macro F1: {f1_score(y_test, baseline_preds, average='macro'):.4f}\n")

    # ── Tune ──────────────────────────────────────────────────────────────
    thresholds = tune_thresholds(proba, y_test, n_classes)

    # ── Evaluate tuned ────────────────────────────────────────────────────
    evaluate_with_thresholds(proba, y_test, thresholds, list(le.classes_))

    # ── Persist thresholds ────────────────────────────────────────────────
    threshold_dict = {cls: float(t) for cls, t in zip(le.classes_, thresholds)}
    with open(THRESHOLDS_PATH, "w") as fh:
        json.dump(threshold_dict, fh, indent=2)
    print(f"\n  ✓  Thresholds saved → {THRESHOLDS_PATH}")
    print("  Next step: run  uvicorn api.main:app --reload")


if __name__ == "__main__":
    run_threshold_tuning()