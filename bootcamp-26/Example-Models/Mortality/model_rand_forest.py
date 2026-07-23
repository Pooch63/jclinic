
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import pandas as pd
from typing import Dict

class MortalityRandomForest:
    def __init__(self, n_estimators: int = 200,
                max_depth: int = 10,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                random_state: int = 42):
        
        self.feature_importance = None

        self.model = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            # The model should learn that deaths are way less common than survival.
            # Fine-tuned hyperparameter, 0.2 seemed to work best
            # Lower sampling won't include enough death examples,
            # but higher sampling will significantly reduce the recall
            # because the model overestimates the number of true cases
            sampling_strategy = 0.15,
            n_jobs=-1
        )
        
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.model.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
