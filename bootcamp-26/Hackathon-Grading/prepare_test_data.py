import os
import shutil
import pandas as pd
import numpy as np

def main():
    print("Initializing secure dataset preparation...")
    
    src_data_path = '../data/data/hackathon_data_clean.csv'
    dest_data_path = './hackathon-data.csv'
    
    if not os.path.exists(src_data_path):
        print(f"Error: Source dataset '{src_data_path}' not found.")
        return
        
    shutil.copyfile(src_data_path, dest_data_path)
    print(f"Copied full dataset to '{dest_data_path}'")
    
    df = pd.read_csv(dest_data_path, low_memory=False)
    
    os.makedirs('./ground_truth', exist_ok=True)
    os.makedirs('./submissions', exist_ok=True)
    
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled.iloc[:25000].copy()
    test_df = df_shuffled.iloc[25000:40000].copy()
    
    print(f"Dataset split: Train = {len(train_df)} rows, Test = {len(test_df)} rows, Holdout = {len(df_shuffled) - 40000} rows.")
    
    target_cols = ['Flag_EpiPen', 'Visit_Count', 'Visit1_ER_Severity', 'Seen_Allergist', 'Death', 'Allergy_Flag']
    
    questions_config = {
        1: {'col': 'Flag_EpiPen', 'type': 'classification'},
        2: {'col': 'Visit_Count', 'type': 'regression'},
        3: {'col': 'Visit1_ER_Severity', 'type': 'classification'},
        4: {'col': 'Seen_Allergist', 'type': 'classification'},
        5: {'col': 'Death', 'type': 'classification'},
        6: {'col': 'Allergy_Flag', 'type': 'classification'}
    }
    
    for question, config in questions_config.items():
        col = config['col']
        q_dir = f'./question{question}'
        os.makedirs(q_dir, exist_ok=True)
        
        q_train = train_df.drop(columns=[c for c in target_cols if c != col]).rename(columns={'Patient_ID': 'id'})
        q_train_path = os.path.join(q_dir, 'train.csv')
        q_train.to_csv(q_train_path, index=False)
        print(f"Created training file: {q_train_path} ({len(q_train)} rows)")
        
        q_test = test_df.drop(columns=target_cols).rename(columns={'Patient_ID': 'id'})
        q_test_path = os.path.join(q_dir, 'test_X.csv')
        q_test.to_csv(q_test_path, index=False)
        print(f"Created testing file: {q_test_path} ({len(q_test)} rows)")
        
        q_gt = test_df[['Patient_ID', col]].rename(columns={'Patient_ID': 'id'})
        gt_path = f'./ground_truth/question{question}_true.csv'
        q_gt.to_csv(gt_path, index=False)
        print(f"Created ground truth file: {gt_path} ({len(q_gt)} rows)")


main()
