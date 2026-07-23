import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

QUESTION_CONFIG = {
    1: {
        'name': 'EpiPen Prescriptions',
        'target': 'Flag_EpiPen',
        'type': 'classification',
        'metric_name': 'Accuracy',
        'ascending': False
    },
    2: {
        'name': 'Hospital Visits',
        'target': 'Visit_Count',
        'type': 'regression',
        'metric_name': 'RMSE',
        'ascending': True
    },
    3: {
        'name': 'ER Visit Severity',
        'target': 'Visit1_ER_Severity',
        'type': 'classification',
        'metric_name': 'Accuracy',
        'ascending': False
    },
    4: {
        'name': 'Specialist Referrals',
        'target': 'Seen_Allergist',
        'type': 'classification',
        'metric_name': 'Accuracy',
        'ascending': False
    },
    5: {
        'name': 'Mortality Prediction',
        'target': 'Death',
        'type': 'classification',
        'metric_name': 'Accuracy',
        'ascending': False
    },
    6: {
        'name': 'Allergy Prediction',
        'target': 'Allergy_Flag',
        'type': 'classification',
        'metric_name': 'Accuracy',
        'ascending': False
    }
}

SUBMISSIONS_DIR = './submissions'
GROUND_TRUTH_DIR = './ground_truth'
LEADERBOARD_FILE = 'final_leaderboard.csv'

def parse_filename(filename):
    match = re.match(r'^question([1-6])_(.+)\.csv$', filename, re.IGNORECASE)
    if match:
        question_num = int(match.group(1))
        team_name = match.group(2)
        return question_num, team_name
    return None

def grade_submission(file_path, question_num):
    config = QUESTION_CONFIG[question_num]
    gt_file = os.path.join(GROUND_TRUTH_DIR, f'question{question_num}_true.csv')
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file '{gt_file}' not found.")
        
    try:
        student_df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Could not read submission CSV: {e}")
        
    try:
        true_df = pd.read_csv(gt_file)
    except Exception as e:
        raise ValueError(f"Could not read ground truth CSV: {e}")
        
    if 'id' not in student_df.columns:
        raise ValueError("Missing 'id' column in submission file.")
    if 'id' not in true_df.columns:
        raise ValueError("Missing 'id' column in ground truth file.")
        
    student_ids = set(student_df['id'])
    true_ids = set(true_df['id'])
    
    if student_ids != true_ids:
        missing_ids = true_ids - student_ids
        extra_ids = student_ids - true_ids
        err_msg = "Row IDs mismatch."
        if missing_ids:
            err_msg += f" Missing {len(missing_ids)} IDs."
        if extra_ids:
            err_msg += f" Extraneous {len(extra_ids)} IDs."
        raise ValueError(err_msg)
        
    merged = pd.merge(true_df, student_df, on='id', suffixes=('_true', '_pred'))
    
    true_cols = [c for c in true_df.columns if c != 'id']
    pred_cols = [c for c in student_df.columns if c != 'id']
    
    if len(true_cols) == 0:
        raise ValueError("Ground truth file contains only ID column and no target labels.")
    if len(pred_cols) != 1:
        raise ValueError(f"Submission file must have exactly one prediction column (found {len(pred_cols)}).")
        
    target_col = true_cols[0]
    pred_col = pred_cols[0]
    
    merged = merged.dropna(subset=[target_col])
    
    if len(merged) == 0:
        raise ValueError("No valid ground truth rows remaining after dropping NaNs.")
        
    if merged[pred_col].isnull().any():
        raise ValueError("Submission contains NaN/missing prediction values.")
        
    y_true = merged[target_col].values
    y_pred = pd.to_numeric(merged[pred_col], errors='coerce').values
    
    if np.isnan(y_pred).any():
        raise ValueError(f"Non-numeric predictions found in column '{pred_col}'.")
        
    if config['type'] == 'classification':
        try:
            score = accuracy_score(y_true, y_pred.astype(int))
        except Exception as e:
            raise ValueError(f"Error computing Accuracy: {e}")
    else:
        try:
            score = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            from sklearn.metrics import root_mean_squared_error
            score = root_mean_squared_error(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Error computing RMSE: {e}")
            
    return score

def main():
    print("Initializing hackathon grading run...")
    
    if not os.path.exists(SUBMISSIONS_DIR):
        print(f"Error: Directory '{SUBMISSIONS_DIR}' does not exist.")
        sys.exit(1)
        
    all_files = sorted(os.listdir(SUBMISSIONS_DIR))
    leaderboard_records = []
    error_records = []
    
    for filename in all_files:
        file_path = os.path.join(SUBMISSIONS_DIR, filename)
        if not os.path.isfile(file_path):
            continue
            
        parsed = parse_filename(filename)
        if not parsed:
            print(f"Warning: Skipping file '{filename}' (does not match naming pattern question[1-6]_[TeamName].csv)")
            continue
            
        question_num, team_name = parsed
        config = QUESTION_CONFIG[question_num]
        
        try:
            score = grade_submission(file_path, question_num)
            leaderboard_records.append({
                'Question': question_num,
                'Question_Name': config['name'],
                'Team': team_name,
                'Metric': config['metric_name'],
                'Score': score,
                'Status': 'Success',
                'Error_Detail': ''
            })
            print(f"Graded Question {question_num} ({config['name']}) for team {team_name} successfully. Score: {score:.5f}")
        except Exception as e:
            error_message = str(e)
            error_records.append({
                'Question': question_num,
                'Team': team_name,
                'Filename': filename,
                'Error_Detail': error_message
            })
            leaderboard_records.append({
                'Question': question_num,
                'Question_Name': config['name'],
                'Team': team_name,
                'Metric': config['metric_name'],
                'Score': np.nan,
                'Status': f"Failed: {error_message}",
                'Error_Detail': error_message
            })
            print(f"Error grading Question {question_num} for team {team_name}: {error_message}")
            
    df_all = pd.DataFrame(leaderboard_records)
    sorted_df_list = []
    
    print("\n--- Leaderboards ---")
    
    for question_num in sorted(QUESTION_CONFIG.keys()):
        config = QUESTION_CONFIG[question_num]
        print(f"\nQuestion {question_num}: {config['name']} ({config['metric_name']})")
        
        if df_all.empty:
            print("No submissions found.")
            continue
            
        question_success = df_all[(df_all['Question'] == question_num) & (df_all['Status'] == 'Success')].copy()
        
        if question_success.empty:
            print("No successful submissions.")
        else:
            question_success = question_success.sort_values(by='Score', ascending=config['ascending'])
            question_success['Rank'] = range(1, len(question_success) + 1)
            display_cols = ['Rank', 'Team', 'Score']
            print(question_success[display_cols].to_string(index=False, formatters={'Score': '{:,.5f}'.format}))
            sorted_df_list.append(question_success)
            
        question_failures = df_all[(df_all['Question'] == question_num) & (df_all['Status'] != 'Success')].copy()
        if not question_failures.empty:
            question_failures['Rank'] = np.nan
            sorted_df_list.append(question_failures)

    if error_records:
        print("\n--- Summary of Failures & Errors ---")
        df_errors = pd.DataFrame(error_records)
        print(df_errors[['Filename', 'Error_Detail']].to_string(index=False))
        
    if sorted_df_list:
        final_leaderboard = pd.concat(sorted_df_list, ignore_index=True)
        export_cols = ['Question', 'Question_Name', 'Rank', 'Team', 'Metric', 'Score', 'Status', 'Error_Detail']
        final_leaderboard = final_leaderboard[export_cols]
        final_leaderboard.to_csv(LEADERBOARD_FILE, index=False)
        print(f"\nSuccessfully saved leaderboard to '{LEADERBOARD_FILE}'")
    else:
        print("\nNo leaderboard created (no submission files were processed).")

main()
