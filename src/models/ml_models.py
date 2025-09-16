from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_baseline_models():
    """Baseline ML models."""
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'NaiveBayes': MultinomialNB()
    }

def get_advanced_ml_models():
    """Advanced ML models for text."""
    return {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }