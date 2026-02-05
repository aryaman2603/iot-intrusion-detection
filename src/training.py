import polars as pl
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import(
    TRAIN_DATA_PATH,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    SELECTED_FEATURES
)

def train_model():
    df = pl.read_parquet(TRAIN_DATA_PATH).select(SELECTED_FEATURES + ['label_category']).to_pandas()

    X = df.drop('label_category', axis=1)
    y = df['label_category']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        objective='multi:softmax',
        num_class=len(le.classes_),
        n_jobs=-1,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    print("Starting training with 5-fold stratified cross-validation...")

    X_np = X.values
    y_np = y_encoded
    
    fold = 1
    for train_index, val_index in skf.split(X_np, y_np):
        X_train_fold, X_val_fold = X_np[train_index], X_np[val_index]
        y_train_fold, y_val_fold = y_np[train_index], y_np[val_index]
        
        
        model.fit(X_train_fold, y_train_fold)
        
        
        preds = model.predict(X_val_fold)
        
        
        score = f1_score(y_val_fold, preds, average="weighted")
        f1_scores.append(score)
        print(f"   Fold {fold}: F1-Score = {score:.4f}")
        fold += 1

    print(f"\n Average F1-Score: {np.mean(f1_scores):.4f}")
    
    
    print(" Retraining on full dataset and saving...")
    model.fit(X, y_encoded)
    
    
    model.save_model(MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")
    
    
    print("\n Classification Report (on Training Data):")
    final_preds = model.predict(X)
    print(classification_report(y_encoded, final_preds, target_names=le.classes_))


if __name__ == "__main__":
    train_model()

