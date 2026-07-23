
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from typing import Dict

class EpiPenRandomForest:
    def __init__(self, n_estimators: int = 200,
                max_depth: int = 15,
                min_samples_split: int = 5,
                min_samples_leaf: int = 2,
                random_state: int = 42):
        
        self.feature_importance = None

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
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
