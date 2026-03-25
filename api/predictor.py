"""
api/predictor.py
─────────────────
Inference engine. Loaded once at API startup via lifespan context.
Handles: feature ordering, missing value imputation, scaling,
         model inference, threshold application, and response formatting.
"""
import json
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np

from src.config import (
    CLASSES,
    FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLDS_PATH,
    DEFAULT_FEATURES,
)

LOW_CONFIDENCE_CUTOFF = 0.60   # flag predictions below this confidence


class Predictor:
    """
    Singleton inference engine.
    Thread-safe for read-only operations (LightGBM predict is GIL-released).
    """

    def __init__(self):
        # ── Load selected features ─────────────────────────────────────────
        try:
            with open(FEATURES_PATH) as fh:
                self.features: List[str] = json.load(fh)
        except FileNotFoundError:
            self.features = DEFAULT_FEATURES

        # ── Load model ────────────────────────────────────────────────────
        self.booster = lgb.Booster(model_file=MODEL_PATH)

        # ── Load scaler ───────────────────────────────────────────────────
        self.scaler = joblib.load(SCALER_PATH)

        # ── Load label encoder ────────────────────────────────────────────
        self.le = joblib.load(LABEL_ENCODER_PATH)
        self.class_names: List[str] = list(self.le.classes_)

        # ── Load per-class thresholds ─────────────────────────────────────
        try:
            with open(THRESHOLDS_PATH) as fh:
                threshold_dict: Dict[str, float] = json.load(fh)
            self.thresholds = np.array(
                [threshold_dict[c] for c in self.class_names]
            )
        except FileNotFoundError:
            # Fallback: equal thresholds (argmax behaviour)
            self.thresholds = np.full(len(self.class_names), 0.5)

        # Precompute alias map (parquet column → model feature index)
        self._alias = {
            "Protocol Type": "Protocol_Type",
            "Tot sum":        "Tot_sum",
            "Tot size":       "Tot_size",
        }

    # ── Public API ────────────────────────────────────────────────────────

    def predict_single(self, flow_dict: dict) -> dict:
        return self.predict_batch([flow_dict])[0]

    def predict_batch(self, flows: List[dict]) -> List[dict]:
        X = self._build_feature_matrix(flows)
        X_scaled = self.scaler.transform(X)
        proba = self.booster.predict(X_scaled)          # shape (n, n_classes)
        return [self._format_result(proba[i]) for i in range(len(flows))]

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_feature_matrix(self, flows: List[dict]) -> np.ndarray:
        """
        Extract selected features from each flow dict.
        Missing features are imputed with 0.0 (median-ish for scaled features).
        Raises ValueError if ALL selected features are missing from a row.
        """
        rows = []
        for i, flow in enumerate(flows):
            # Normalise aliases (Pydantic may serialise "Protocol Type" as "Protocol_Type")
            resolved = {}
            for k, v in flow.items():
                canonical = self._alias.get(k, k)
                resolved[canonical] = v

            row = []
            for feat in self.features:
                canonical = self._alias.get(feat, feat)
                val = resolved.get(canonical, resolved.get(feat, None))
                if val is None:
                    val = 0.0
                row.append(float(val))

            if all(v == 0.0 for v in row):
                raise ValueError(
                    f"Flow at index {i} has no recognised feature values. "
                    f"Expected features: {self.features[:5]}…"
                )
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def _format_result(self, prob_row: np.ndarray) -> dict:
        """Apply thresholds and format a single prediction."""
        # Threshold-adjusted argmax
        adjusted = prob_row / self.thresholds
        pred_idx = int(np.argmax(adjusted))
        pred_class = self.class_names[pred_idx]
        confidence = float(prob_row[pred_idx])

        prob_dict = {
            cls: round(float(p), 6)
            for cls, p in zip(self.class_names, prob_row)
        }

        return {
            "prediction":    pred_class,
            "confidence":    round(confidence, 6),
            "probabilities": prob_dict,
            "is_attack":     pred_class != "Benign",
            "low_confidence": confidence < LOW_CONFIDENCE_CUTOFF,
        }

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def info(self) -> dict:
        return {
            "model_type":    "LightGBM (GBDT, multiclass softmax)",
            "n_features":    len(self.features),
            "feature_names": self.features,
            "classes":       self.class_names,
            "thresholds":    {
                cls: round(float(t), 4)
                for cls, t in zip(self.class_names, self.thresholds)
            },
        }