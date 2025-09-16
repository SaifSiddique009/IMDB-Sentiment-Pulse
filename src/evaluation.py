from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.utils import logger
import optuna
import numpy as np
import pandas as pd
import os

def cross_validate(model, X, y, n_splits=5, is_dl=False):
    """5-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, f1_scores = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx] if not is_dl else X[train_idx], X[val_idx] if not is_dl else X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if is_dl:
            model.fit(X_tr, y_tr, epochs=10, batch_size=16, verbose=0)  # Short training for CV
            preds = (model.predict(X_val) > 0.5).astype(int).flatten()
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds))
    return np.mean(acc_scores), np.mean(f1_scores)

def tune_model(model_name, model, X_train, y_train, n_trials=20):
    """Optuna Bayesian tuning."""
    def objective(trial):
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20)
            }
            tuned_model = RandomForestClassifier(**params, random_state=42)
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
            }
            tuned_model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 200),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
            }
            tuned_model = CatBoostClassifier(**params, random_state=42, verbose=0)
        else:
            return 0  # Skip non-tunable
        
        tuned_model.fit(X_train, y_train)
        preds = tuned_model.predict(X_train)  # Use train for quick tuning
        return f1_score(y_train, preds)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best params for {model_name}: {study.best_params}")
    return study.best_params

def evaluate_and_save(results: dict, output_dir: str = 'results/'):
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(output_dir, 'model_metrics.csv'))
    logger.info("Metrics saved.")