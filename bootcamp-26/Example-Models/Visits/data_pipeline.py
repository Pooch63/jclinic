import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class AllergyVisitDataSetup: 
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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
        
        features['seen_allergist'] = df['Seen_Allergist'].fillna(0)
        
        er_visit_cols = ['Visit1_Type', 'Visit2_Type', 'Visit3_Type', 'Visit4_Type', 'Visit5_Type']
        features['er_visit_count'] = sum([(df[col] == 'ER').astype(int) for col in er_visit_cols if col in df.columns])
        
        outpatient_visit_cols = ['Visit1_Type', 'Visit2_Type', 'Visit3_Type', 'Visit4_Type', 'Visit5_Type']
        features['outpatient_visit_count'] = sum([(df[col] == 'Outpatient').astype(int) for col in outpatient_visit_cols if col in df.columns])
        
        er_severity_cols = ['Visit1_ER_Severity', 'Visit2_ER_Severity', 'Visit3_ER_Severity']
        features['max_er_severity'] = df[er_severity_cols].max(axis=1).fillna(0)
        features['avg_er_severity'] = df[er_severity_cols].mean(axis=1).fillna(0.0)
        
        features['gi_comorbidity'] = df['Flag_GI_Comorbidity'].fillna(0)
        features['mental_health_flag'] = df['Flag_MentalHealth'].fillna(0)
        features['metabolic_flag'] = df['Flag_Metabolic'].fillna(0)
        
        features['antihistamine_use'] = df['Flag_Antihistamine'].fillna(0)
        features['leukotriene_mod'] = df['Flag_LeukotrieneMod'].fillna(0)
        features['rescue_inhaler'] = df['Flag_RescueInhaler'].fillna(0)
        features['epipen_prescribed'] = df['Flag_EpiPen'].fillna(0)
        features['biologic_use'] = df['Flag_Biologic'].fillna(0)
        
        lab_cols = ['Lab1_Flag', 'Lab2_Flag', 'Lab3_Flag', 'Lab4_Flag', 'Lab5_Flag']
        features['abnormal_labs'] = sum([(df[col] == 'Abnormal').astype(int) for col in lab_cols if col in df.columns])
        
        target = df['Visit_Count'].fillna(0)
        
        return features, target
    
    def train_scaler(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df)
        self.feature_names = df.columns.tolist()
    
    def scale_set(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)
