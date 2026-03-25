"""
training.py
────────────
Trains a LightGBM classifier on the full processed dataset using:
  - Selected features from feature_selection.py
  - RobustScaler (fitted during feature selection)
  - Stratified train/val/test split
  - 5-fold stratified CV with early stopping per fold
  - Final model retrained on full train+val set
  - Saves model, test probabilities, and test labels for threshold tuning

Why LightGBM over XGBoost for this problem?
  • Native class_weight='balanced' — no manual sample_weight arrays needed
  • Leaf-wise growth handles the large DDoS/minority imbalance better
  • 3-5x faster training on 8M rows with similar or better macro F1
  • DART boosting option reduces overfitting on minority classes
  • natively outputs calibrated probabilities with multiclass objective

Run:  python -m src.training
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.config import (
    CLASSES,
    FEATURES_PATH,
    LABEL_ENCODER_PATH,
    LGBM_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    N_CV_FOLDS,
    TRAIN_DATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_LABELS_PATH,
    TEST_PROBA_PATH,
    TEST_SIZE,
    VAL_SIZE,
    DEFAULT_FEATURES,
)


def load_features() -> list:
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as fh:
            features = json.load(fh)
        print(f"  Loaded {len(features)} selected features from {FEATURES_PATH}")
    else:
        features = DEFAULT_FEATURES
        print(
            f"  ⚠  {FEATURES_PATH} not found — using DEFAULT_FEATURES ({len(features)} cols). "
            "Run src/feature_selection.py first for optimal results."
        )
    return features


def load_data(features: list):
    print(f"Loading data from {TRAIN_DATA_PATH}…")
    df = pl.read_parquet(TRAIN_DATA_PATH)

    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        print(f"  ⚠  {len(missing)} features not in parquet: {missing}")

    X = df.select(available).to_numpy().astype(np.float32)
    y_raw = df["label_category"].to_list()
    print(f"  {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y_raw, available


def encode_and_split(X, y_raw):
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)   # fix class order
    y = le.transform(y_raw)

    # ── Carve out held-out test set FIRST ────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"  Train+val: {X_trainval.shape[0]:,}  |  "
        f"Test (held-out): {X_test.shape[0]:,}"
    )
    return X_trainval, X_test, y_trainval, y_test, le


def scale(X_trainval, X_test, scaler_path: str):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  Loaded scaler from {scaler_path}")
        X_tv_s = scaler.transform(X_trainval)
        X_te_s = scaler.transform(X_test)
    else:
        print("  Fitting new RobustScaler on train+val…")
        scaler = RobustScaler()
        X_tv_s = scaler.fit_transform(X_trainval)
        X_te_s = scaler.transform(X_test)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved → {scaler_path}")
    return X_tv_s, X_te_s, scaler


def cross_validate(X_trainval, y_trainval) -> None:
    """5-fold CV for an honest estimate — models not saved here."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    wf1_scores, mf1_scores = [], []

    print(f"\nRunning {N_CV_FOLDS}-fold stratified CV…")
    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(X_trainval, y_trainval), 1
    ):
        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        proba = clf.predict_proba(X_val)
        preds = np.argmax(proba, axis=1)

        wf1 = f1_score(y_val, preds, average="weighted")
        mf1 = f1_score(y_val, preds, average="macro")
        wf1_scores.append(wf1)
        mf1_scores.append(mf1)
        print(f"  Fold {fold}: weighted F1={wf1:.4f}  macro F1={mf1:.4f}  "
              f"(best iter={clf.best_iteration_})")

    print(
        f"\nCV summary — "
        f"weighted F1: {np.mean(wf1_scores):.4f} ± {np.std(wf1_scores):.4f}  |  "
        f"macro F1: {np.mean(mf1_scores):.4f} ± {np.std(mf1_scores):.4f}"
    )


def train_final_model(X_trainval, y_trainval):
    """
    Train final model on full train+val.
    Use a 5% internal val split purely for early stopping — not evaluation.
    """
    print("\nTraining final model on full train+val set…")
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"  Best iteration: {clf.best_iteration_}")
    return clf


def evaluate_on_test(clf, X_test, y_test, le: LabelEncoder):
    """Evaluate on the held-out test set and print honest metrics."""
    proba = clf.predict_proba(X_test)           # shape (n, n_classes)
    preds = np.argmax(proba, axis=1)

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET RESULTS  (honest — never seen during training)")
    print("=" * 65)
    print(classification_report(y_test, preds, target_names=le.classes_))

    macro  = f1_score(y_test, preds, average="macro")
    weighted = f1_score(y_test, preds, average="weighted")
    print(f"  Macro F1   : {macro:.4f}   ← headline metric")
    print(f"  Weighted F1: {weighted:.4f}")
    return proba


def save_artifacts(clf, le, features, proba, y_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    clf.booster_.save_model(MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    np.save(TEST_PROBA_PATH,  proba)
    np.save(TEST_LABELS_PATH, y_test)

    print(f"\n  Model      → {MODEL_PATH}")
    print(f"  Encoder    → {LABEL_ENCODER_PATH}")
    print(f"  Test proba → {TEST_PROBA_PATH}")
    print(f"  Test labels→ {TEST_LABELS_PATH}")
    print("\n  Next step: run  python -m src.threshold_tuning")


def train_model():
    features = load_features()
    X, y_raw, available_features = load_data(features)

    # Label-encode with fixed class order so API is deterministic
    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(y_raw)

    X_trainval, X_test, y_trainval, y_test = (
        lambda a, b, c, d: (a, b, c, d)
    )(*train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE))

    print(f"  Train+val: {X_trainval.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    X_tv_s, X_te_s, _ = scale(X_trainval, X_test, SCALER_PATH)

    cross_validate(X_tv_s, y_trainval)

    clf = train_final_model(X_tv_s, y_trainval)

    proba = evaluate_on_test(clf, X_te_s, y_test, le)

    save_artifacts(clf, le, available_features, proba, y_test)


if __name__ == "__main__":
    train_model()