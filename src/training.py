import polars as pl
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score

from src.config import (
    TRAIN_DATA_PATH, 
    MODEL_PATH, 
    LABEL_ENCODER_PATH, 
    SELECTED_FEATURES
)

def train_model():
    print("Loading Data for Training...")
    df = pl.read_parquet(TRAIN_DATA_PATH).select(SELECTED_FEATURES + ["label_category"])
    
    X = df.drop("label_category").to_pandas()
    y_raw = df["label_category"].to_list()
    
    print("Encoding Labels...")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    joblib.dump(le, LABEL_ENCODER_PATH)
    
    print("Calculating Sample Weights to fix imbalance...")
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights_dict = dict(zip(classes, weights))
    
    print("Class Weights used:")
    for cls, weight in zip(le.classes_, weights):
        print(f" - {cls}: {weight:.2f}")

    sample_weights = np.array([class_weights_dict[label] for label in y])

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        objective="multi:softmax",
        num_class=len(le.classes_), 
        n_jobs=-1,
        random_state=42
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    
    print("\nStarting 5-Fold Cross-Validation (Weighted)...")
    
    X_np = X.values
    
    fold = 1
    for train_index, val_index in skf.split(X_np, y):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]
        weights_train = sample_weights[train_index]
        
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="weighted")
        f1_scores.append(score)
        print(f" Fold {fold}: F1-Score = {score:.4f}")
        fold += 1

    print(f"\nAverage F1-Score: {np.mean(f1_scores):.4f}")
    
    print("Retraining on full dataset...")
    model.fit(X, y, sample_weight=sample_weights)
    
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    print("\nClassification Report (Training Data):")
    final_preds = model.predict(X)
    print(classification_report(y, final_preds, target_names=le.classes_))

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    train_model()

