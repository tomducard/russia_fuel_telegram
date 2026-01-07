"""
Lightweight model evaluation helpers.
Ideally imported as `from src.rft import model_helpers`.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def evaluate_predictions(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Print comprehensive classification metrics and plot confusion matrix.
    """
    print(f"--- RÉSULTATS : {model_name} ---")
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy : {acc:.2%}")
    print(f"F1-Score : {f1:.3f}")
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            print(f"ROC AUC  : {auc:.3f}")
        except ValueError:
            print("ROC AUC  : N/A (One class only?)")

    print("\nRapport Détaillé :")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Matrice de Confusion ({model_name})")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
    
    return {"f1": f1, "accuracy": acc}


def select_features_lasso(X_train, y_train, feature_names, C=0.2):
    """
    Sélectionne les variables les plus importantes via une Régression Logistique L1 (Lasso).
    Retourne la liste des noms de colonnes retenues.
    """
    print(f">>> Sélection Lasso (C={C})...")
    
    # Standardisation sur une copie temporaire (Lasso en a besoin)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Lasso
    lasso = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42, class_weight='balanced')
    lasso.fit(X_scaled, y_train)
    
    # Extraction
    model_select = SelectFromModel(lasso, prefit=True)
    selected_mask = model_select.get_support()
    selected_feats = [f for f, s in zip(feature_names, selected_mask) if s]
    
    # Fallback
    if len(selected_feats) < 3:
        print("   ⚠️  Lasso trop agressif (< 3 features). On retourne tout.")
        return list(feature_names)
    
    print(f"   > {len(selected_feats)} variables retenues (sur {len(feature_names)})")
    return selected_feats


def tune_hyperparameters(estimator, param_dist, X_train, y_train, n_iter=50, cv_splits=3, scoring="average_precision"):
    """
    Étape 1 : Trouve les meilleurs hyperparamètres avec RandomizedSearch.
    Retourne le meilleur modèle (déjà entraîné sur tout X_train).
    """
    print(f"\n>>> 1. Tuning Hyperparamètres ({scoring})...")
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"   > Best Params : {search.best_params_}")
    return search.best_estimator_



def select_features_rf_importance(X_train, y_train, feature_names=None, top_k=20):
    """
    Sélectionne les variables via l'importance d'un Random Forest (capture le non-linéaire).
    Retourne la liste des noms de colonnes retenues.
    """
    # Auto-detection des noms si DataFrame
    if feature_names is None:
        if hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        else:
            raise ValueError("Vous devez fournir 'feature_names' si X_train n'est pas un DataFrame.")

    print(f">>> Sélection Random Forest (Top {top_k})...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    # Create sorted list (names, imp)
    feats_sorted = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Select top K
    selected_feats = [f[0] for f in feats_sorted[:top_k]]
    
    print(f"   > {len(selected_feats)} variables retenues (sur {len(feature_names)})")
    # Affiche le Top 5 pour info
    print(f"   > Top 5 : {[f[0] for f in feats_sorted[:5]]}")
    
    return selected_feats


def find_optimal_threshold(model, X_train, y_train, cv_splits=3):
    """
    Étape 2 : Trouve le seuil optimal via Validation Croisée manuelle (Out-of-Fold predictions).
    Retourne le seuil (float).
    """
    print("\n>>> 2. Calibration du Seuil (Validation Croisée)...")
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    y_proba_cv_list = []
    y_true_cv_list  = []

    for train_idx, val_idx in tscv.split(X_train):
        # On clone pour avoir une coquille vide avec les mêmes hyperparams
        clone_clf = clone(model)
        
        # Slicing robuste (supporte Pandas et Numpy)
        X_tr_fold = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
        y_tr_fold = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]
        
        clone_clf.fit(X_tr_fold, y_tr_fold)
        
        # Gestion des modèles qui n'ont pas predict_proba
        if hasattr(clone_clf, "predict_proba"):
            y_proba_cv_list.extend(clone_clf.predict_proba(X_val_fold)[:, 1])
        else:
            y_proba_cv_list.extend(clone_clf.predict(X_val_fold)) 
            
        y_true_cv_list.extend(y_val_fold)

    y_proba_cv = np.array(y_proba_cv_list)
    y_true_cv  = np.array(y_true_cv_list)

    best_thresh = 0.5
    best_f1_cv = 0.0
    
    # Scan large
    for thr in np.arange(0.15, 0.85, 0.01):
        s = f1_score(y_true_cv, (y_proba_cv >= thr).astype(int))
        if s > best_f1_cv:
            best_f1_cv = s
            best_thresh = thr

    print(f"   > Seuil Optimal : {best_thresh:.2f} (F1 CV : {best_f1_cv:.2f})")
    return best_thresh


def optimize_model_with_threshold(
    estimator, 
    param_dist, 
    X_train, y_train, 
    X_test, y_test, 
    n_iter=50, 
    cv_splits=3,
    scoring="average_precision"
):
    """(Legacy Wrapper) Combine tune -> calibrate -> evaluate."""
    # 1. Tune
    best_model = tune_hyperparameters(estimator, param_dist, X_train, y_train, n_iter, cv_splits, scoring)
    
    # 2. Threshold
    best_thresh = find_optimal_threshold(best_model, X_train, y_train, cv_splits)
    
    # 3. Valid
    print(f"\n>>> 3. VERDICT FINAL (Test Set)")
    if hasattr(best_model, "predict_proba"):
        y_proba_test = best_model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = best_model.decision_function(X_test)
        
    y_pred_test  = (y_proba_test >= best_thresh).astype(int)
    
    evaluate_predictions(y_test, y_pred_test, y_proba_test, model_name=f"{estimator.__class__.__name__}_Optimized")
    
    return best_model, best_thresh
