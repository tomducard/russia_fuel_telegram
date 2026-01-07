"""Baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform

# Import internal helpers (must be in same package or updated path)
from . import model_helpers

@dataclass
class ModelResult:
    model: Pipeline | XGBClassifier | RandomForestClassifier | LogisticRegression
    feature_cols: List[str]
    train_mse: float = 0.0
    test_mse: float = 0.0
    accuracy: float = 0.0
    report: str = ""
    roc_auc: float = 0.0
    f1_score: float = 0.0
    best_threshold: float = 0.5


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> int:
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to perform a time-based split.")
    split_idx = max(1, int(len(df) * train_ratio))
    return min(split_idx, len(df) - 1)


def get_feature_selection(df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> List[str]:
    # Dynamic feature selection: Use all available signal columns
    if feature_cols:
        return list(feature_cols)
    
    # Default strategy: grab all relevant numeric signals
    base_candidates = {
        "total_messages", "unique_messages", "keyword_mentions", 
        "price_mentions", "avg_price", "fuel_stress_index"
    }
    all_cols = set(df.columns)
    
    # dynamic group columns (share_*, count_*, sentiment_*, price_*, *_trend*, *_shock*, *_volatility*)
    dynamic_features = {
        c for c in all_cols 
        if c.startswith(("share_", "count_", "sentiment_", "price_"))
    }
    
    # Combine 
    return list((base_candidates & all_cols) | dynamic_features)


def train_baseline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
) -> ModelResult:
    """Train a Ridge regression on time-ordered data."""
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in DataFrame.")

    data = df.dropna(subset=[target_col]).copy()
    if "date" in data.columns:
        data = data.sort_values("date")

    features = get_feature_selection(data, feature_cols)
        
    if not features:
        raise ValueError("No feature columns available for training.")

    split_idx = _time_split(data, train_ratio=train_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(float)
    X_test = test_df[features].fillna(0.0)
    y_test = test_df[target_col].astype(float)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    train_mse = float(mean_squared_error(y_train, train_pred))
    test_mse = float(mean_squared_error(y_test, test_pred))

    return ModelResult(
        model=pipeline,
        feature_cols=features,
        train_mse=train_mse,
        test_mse=test_mse,
    )


def train_classifier(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
    model_type: str = "xgb",
    selection_method: str = "none" # Options: "none", "lasso", "rf"
) -> ModelResult:
    """
    Train a Classifier (XGBoost, RF, LR) with advanced pipeline:
    1. Feature Selection (Optional: Lasso or Random Forest Importance)
    2. Hyperparameter Tuning (RandomizedSearch + TimeSeriesSplit)
    3. Threshold Calibration
    """
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in DataFrame.")

    # 1. Prepare Data
    data = df.dropna(subset=[target_col]).copy()
    if "date" in data.columns:
        data = data.sort_values("date")

    features = get_feature_selection(data, feature_cols)
    if not features:
        raise ValueError("No feature columns available for training.")

    split_idx = _time_split(data, train_ratio=train_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(int)
    X_test = test_df[features].fillna(0.0)
    y_test = test_df[target_col].astype(int)

    # 2. Advanced Feature Selection
    current_features = features
    
    if selection_method == "lasso":
        print(">>> Running Lasso Feature Selection...")
        selected_feats = model_helpers.select_features_lasso(X_train, y_train, features)
        X_train = X_train[selected_feats]
        X_test = X_test[selected_feats]
        current_features = selected_feats
        
    elif selection_method == "rf":
        print(">>> Running Random Forest Feature Selection (Tree-Based)...")
        # Note: top_k defaults to 20 in helper, usage here relies on helper default
        selected_feats = model_helpers.select_features_rf_importance(X_train, y_train, features, top_k=20)
        X_train = X_train[selected_feats]
        X_test = X_test[selected_feats]
        current_features = selected_feats


    # 3. Model Configuration & Tuning
    print(f">>> Training {model_type.upper()} Model (Features: {len(current_features)})...")
    
    best_model = None
    
    if model_type == "xgb":
        xgb_params = {
            "n_estimators": randint(50, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.005, 0.2),
            "scale_pos_weight": [1, 5, 10, 15, 20], 
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": uniform(0, 0.5),
            "min_child_weight": randint(1, 10)
        }
        best_model = model_helpers.tune_hyperparameters(
            estimator=XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
            param_dist=xgb_params,
            X_train=X_train, y_train=y_train,
            n_iter=50, cv_splits=3
        )

    elif model_type == "rf":
        rf_params = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(5, 20),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced", "balanced_subsample", None]
        }
        best_model = model_helpers.tune_hyperparameters(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_dist=rf_params,
            X_train=X_train, y_train=y_train,
            n_iter=30, cv_splits=3
        )
        
    elif model_type == "lr":
        lr_params = {
            "C": loguniform(1e-4, 1e2),
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": uniform(0, 1) # Used only for elasticnet
        }
        best_model = model_helpers.tune_hyperparameters(
            estimator=LogisticRegression(random_state=42, max_iter=2000, solver="saga", class_weight='balanced'),
            param_dist=lr_params,
            X_train=X_train, y_train=y_train,
            n_iter=20, cv_splits=3
        )
    else:
         raise ValueError(f"Unsupported model_type: {model_type}")

    # 4. Threshold Calibration
    best_thresh = model_helpers.find_optimal_threshold(best_model, X_train, y_train, cv_splits=3)
    
    # 5. Final Evaluation
    if hasattr(best_model, "predict_proba"):
        y_proba_test = best_model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = best_model.decision_function(X_test) # Fallback for Linear SVM etc
        
    y_pred_test = (y_proba_test >= best_thresh).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba_test)
    except ValueError:
        auc = 0.5
        
    rep = classification_report(y_test, y_pred_test, zero_division=0)
    
    print(f"\n>>> FINAL RESULTS ({model_type.upper()})")
    print(f"Accuracy: {acc:.2%}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC AUC : {auc:.3f}")

    return ModelResult(
        model=best_model,
        feature_cols=current_features,
        accuracy=acc,
        report=rep,
        roc_auc=auc,
        f1_score=f1,
        best_threshold=best_thresh
    )


# --- LSTM Implementation (Unchanged) ---

def create_sequences(X, y, time_steps=30):
    """Convert 2D array into 3D sequences (Samples, TimeSteps, Features)."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
    time_steps: int = 30
) -> ModelResult:
    """Train an LSTM model for sequence classification."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # Prepare Data
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} missing.")
    
    data = df.dropna(subset=[target_col]).sort_values("date")
    features = get_feature_selection(data, feature_cols)
    print(f"LSTM Features ({len(features)}): {features}")
    
    # Scale Data (Crucial for LSTM)
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    target = data[target_col].astype(int) 
    
    # Create Sequences
    X_seq, y_seq = create_sequences(scaled_features, target, time_steps)
    
    # Split Time-Based
    split_idx = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"LSTM Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    
    # Compute Class Weights usually
    neg, pos = np.bincount(y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class Weights: {class_weight}")

    # Build Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        validation_split=0.2, 
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate on Test Set
    test_proba = model.predict(X_test).flatten()
    
    # Dynamic Threshold Tuning (Reuse logic)
    train_proba = model.predict(X_train).flatten()
    best_thresh = 0.5
    best_f1 = 0.0
    
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        train_pred_t = (train_proba >= thresh).astype(int)
        f1_t = f1_score(y_train, train_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
            
    print(f"LSTM Best Threshold: {best_thresh:.2f} (Train F1: {best_f1:.2f})")
    
    test_pred = (test_proba >= best_thresh).astype(int)
    
    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        roc_auc = 0.5
        
    report = classification_report(y_test, test_pred, zero_division=0)
    
    return ModelResult(
        model=model, 
        feature_cols=features,
        accuracy=accuracy,
        report=report,
        roc_auc=roc_auc,
        f1_score=f1,
        best_threshold=best_thresh
    )
