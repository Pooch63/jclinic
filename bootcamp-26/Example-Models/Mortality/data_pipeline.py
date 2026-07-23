
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class MortalityDataSetup: 
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
        features = pd.DataFrame()
        
        features['anaphylaxis_history'] = df['Flag_Anaphylaxis'].fillna(0)
        features['asthma'] = df['Flag_Asthma'].fillna(0)
        features['resp_distress'] = df['Flag_RespDistress'].fillna(0)
        features['status_asthmaticus'] = df['Flag_StatusAsthmaticus'].fillna(0)
        features['urticaria'] = df['Flag_Urticaria'].fillna(0)
        features['atopic_eczema'] = df['Flag_Atopic_Eczema'].fillna(0)
        
        features['age'] = df['Visit1_Age'].fillna(df['Visit1_Age'].median())
        features['sex_male'] = (df['Sex'] == 'M').astype(int)
        features['family_history'] = (df['Family_History'] == 'Y').astype(int)
        features['childhood_allergy'] = (df['Childhood_Allergy_History'] == 'Y').astype(int)
        
        #features['seen_allergist'] = df['Seen_Allergist'].fillna(0)
        features['visit_count'] = df['Visit_Count'].fillna(1)
        
        er_visits = ['Visit1_Type', 'Visit2_Type', 'Visit3_Type', 'Visit4_Type', 'Visit5_Type']
        features['er_visit_count'] = sum([(df[col] == 'ER').astype(int) for col in er_visits if col in df.columns])
        
        er_severity_cols = ['Visit1_ER_Severity', 'Visit2_ER_Severity', 'Visit3_ER_Severity']
        features['max_er_severity'] = df[er_severity_cols].max(axis=1).fillna(0)
        
        #comorbidity_flags = ['Flag_GI_Comorbidity', 'Flag_MentalHealth', 'Flag_Metabolic']
        #features['comorbidity_count'] = sum([df[col].fillna(0) for col in comorbidity_flags if col in df.columns])

        features["comorbidity_flag"] = df['Flag_GI_Comorbidity'].fillna(0)
        features["mental_health_flag"] = df['Flag_MentalHealth'].fillna(0)
        features["metabolic_flag"] = df['Flag_Metabolic'].fillna(0)
    
        features['antihistamine_use'] = df['Flag_Antihistamine'].fillna(0)
        features['leukotriene_mod'] = df['Flag_LeukotrieneMod'].fillna(0)
        features['rescue_inhaler'] = df['Flag_RescueInhaler'].fillna(0)
        
        target = pd.Series(data=df['Death'].fillna(0))


        return features, target
    
    # def scale_set(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    #     adult_mask = df['Visit1_Age'] >= 18
    #     df = df[adult_mask].copy()
        
    #     X, y = self.prepare_features(df)
    
    #     self.feature_names = X.columns.tolist()
        
    #     X_scaled = self.scaler.fit_transform(X)
        
    #     return X_scaled, y.values
    
    def scale_set(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)
    
    def train_scaler(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df)
