"""
feature_selection.py
─────────────────────
Three-stage feature selection pipeline:

  Stage 1 — Drop near-zero variance features (no signal at all).
  Stage 2 — Drop highly correlated pairs (|r| > 0.95), keeping the one
             with higher LightGBM importance.
  Stage 3 — LightGBM importance + permutation importance consensus:
             keep features that rank in the top-N by both methods.

Output: models/selected_features.json

Run:  python -m src.feature_selection
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.config import (
    FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
    SAMPLE_PATH,
    SCALER_PATH,
    DEFAULT_FEATURES,
    LGBM_PARAMS,
    RANDOM_STATE,
)

# How many features to keep after the consensus stage
TARGET_N_FEATURES = 30

# Lower these if running on the 5% sample — the sample has less variance
# per feature so the default thresholds are too aggressive.
# Restore to 0.01 / 0.95 for the full dataset run.
VARIANCE_THRESHOLD    = 0.001   # was 0.01
CORRELATION_THRESHOLD = 0.98    # was 0.95


def load_sample(use_default_features: bool = False):
    """Load the 5% sample. If use_default_features=False, use ALL numeric cols."""
    df = pl.read_parquet(SAMPLE_PATH)
    exclude = {"label", "label_category"}
    if use_default_features:
        feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df["label_category"].to_list()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, feature_cols, le


# ── Stage 1: Near-zero variance ───────────────────────────────────────────

def drop_low_variance(X: np.ndarray, feature_names: list, threshold: float = 0.01):
    """Remove features whose variance (after scaling to [0,1]) < threshold."""
    # Normalise each feature to [0,1] before computing variance
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges == 0] = 1          # avoid divide-by-zero
    X_norm = (X - X.min(axis=0)) / ranges
    variances = X_norm.var(axis=0)

    mask = variances >= threshold
    kept = [f for f, m in zip(feature_names, mask) if m]
    dropped = [f for f, m in zip(feature_names, mask) if not m]
    print(f"  Stage 1 — variance filter: dropped {len(dropped)}, kept {len(kept)}")
    if dropped:
        print(f"    Dropped: {dropped}")
    return X[:, mask], kept


# ── Stage 2: Correlation filter ───────────────────────────────────────────

def drop_correlated(
    X: np.ndarray,
    feature_names: list,
    importance: np.ndarray,
    threshold: float = 0.95,
):
    """
    For each correlated pair (|r| > threshold), drop the feature with
    lower LightGBM importance (importance array must be aligned with feature_names).
    """
    corr = np.corrcoef(X.T)
    n = len(feature_names)
    to_drop = set()

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                # Keep the one with higher importance
                loser = j if importance[i] >= importance[j] else i
                to_drop.add(feature_names[loser])

    kept = [f for f in feature_names if f not in to_drop]
    mask = np.array([f not in to_drop for f in feature_names])
    print(
        f"  Stage 2 — correlation filter (|r|>{threshold}): "
        f"dropped {len(to_drop)}, kept {len(kept)}"
    )
    return X[:, mask], kept


# ── Stage 3: Consensus importance ────────────────────────────────────────

def consensus_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    le: LabelEncoder,
    n_keep: int = TARGET_N_FEATURES,
):
    """
    Train a small LightGBM, get split-based importance and permutation
    importance. Rank features by average rank across both methods. Keep top-N.
    """
    params = {**LGBM_PARAMS, "n_estimators": 200}  # fast pass for selection

    import pandas as pd
    clf = lgb.LGBMClassifier(**params)
    clf.fit(pd.DataFrame(X, columns=feature_names), y)

    # Split importance (built-in)
    split_imp = clf.feature_importances_

    # Permutation importance on a 20% subsample (fast)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(50_000, len(X)), replace=False)
    perm = permutation_importance(
        clf, X[idx], y[idx],
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )
    perm_imp = perm.importances_mean

    # Rank both (highest importance = rank 1)
    split_ranks = len(feature_names) - np.argsort(np.argsort(split_imp))
    perm_ranks  = len(feature_names) - np.argsort(np.argsort(perm_imp))
    avg_ranks   = (split_ranks + perm_ranks) / 2.0

    order = np.argsort(avg_ranks)[:n_keep]
    selected = [feature_names[i] for i in sorted(order)]

    print(f"\n  Stage 3 — consensus top-{n_keep} features selected:")
    for i, feat in enumerate(
        sorted(zip(avg_ranks[order], [feature_names[i] for i in order]))
    ):
        print(f"    {i+1:2d}. {feat[1]:<35}  avg_rank={feat[0]:.1f}")

    return selected, split_imp


def run_feature_selection() -> list:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading 5% sample…")
    X, y, all_features, le = load_sample(use_default_features=False)
    print(f"  {X.shape[0]:,} rows × {X.shape[1]} features")

    # Scale for variance / correlation stages
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Stage 1
    X_s1, feats_s1 = drop_low_variance(X_scaled, all_features, threshold=VARIANCE_THRESHOLD)

    # Quick LightGBM to get importances needed by stage 2
    print("\n  Fitting quick LightGBM for stage-2 importances…")
    _quick = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 100})
    _quick.fit(X_s1, y)
    imp_s1 = _quick.feature_importances_

    # Stage 2
    X_s2, feats_s2 = drop_correlated(X_s1, feats_s1, imp_s1, threshold=CORRELATION_THRESHOLD)

    # Stage 3 — realign X to post-s2 columns
    feat_idx = [all_features.index(f) for f in feats_s2]
    X_s2_orig = X[:, feat_idx]          # use unscaled for permutation importance
    X_s2_scaled = X_scaled[:, [feats_s1.index(f) for f in feats_s2]]

    selected_features, _ = consensus_top_features(X_s2_scaled, y, feats_s2, le)

    # Fit final scaler on selected features only (on the full unscaled X)
    sel_idx = [all_features.index(f) for f in selected_features]
    X_sel = X[:, sel_idx]
    final_scaler = RobustScaler()
    final_scaler.fit(X_sel)

    # Persist
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(final_scaler, SCALER_PATH)
    with open(FEATURES_PATH, "w") as fh:
        json.dump(selected_features, fh, indent=2)

    print(f"\n  ✓  Selected {len(selected_features)} features saved → {FEATURES_PATH}")
    print(f"  ✓  Scaler saved → {SCALER_PATH}")
    print(f"  ✓  Label encoder saved → {LABEL_ENCODER_PATH}")
    return selected_features


if __name__ == "__main__":
    run_feature_selection()