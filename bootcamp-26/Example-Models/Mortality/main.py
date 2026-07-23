import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any
from flax import nnx
import jax

from data_pipeline import MortalityDataSetup
from model_rand_forest import MortalityRandomForest

class MortalityModelEvaluator:

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                       y_pred_proba: np.ndarray) -> Dict[str, Any]:
                       
        results = {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'auc_score': roc_auc_score(y_true, y_pred_proba),
        }
        return results

    @staticmethod
    def plot_training_history(history: Dict[str, list]):
  
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Train History", fontsize=16)

        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main(df: pd.DataFrame) -> Dict[str, Any]:

    data_setup = MortalityDataSetup()
    X, y = data_setup.prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4321, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=4321, stratify=y_train
    )

    data_setup.train_scaler(X_train)
    X_train = data_setup.scale_set(X_train)
    X_val = data_setup.scale_set(X_val)
    X_test = data_setup.scale_set(X_test)
    
    class_weights_array = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"Class weights: {class_weights}")

    deaths = 0
    for death in np.array(y_train):
        if death == 1:
            deaths += 1
    print(f"Deaths in training set: {deaths}")
    print(f"Lives in training set: {len(y_train) - deaths}")

    tree = MortalityRandomForest()
    tree.train(X_train, y_train)

    # Threshold derived from threshold brute force approach below
    preds = tree.predict(X_val, 0.5888)
    
    fp, tp, fn, tn = 0, 0, 0, 0
    
    for truth, pred in zip(y_val, preds):
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1

    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")

    # 1 = F1, 2 = F2, 0.5 = F0.5
    # Because this is predicting death, precision is much more important than
    # recall. We don't want to tell a patient they're about to die if they're not going to.
    # Therefore, we don't want F1 score, we want to weight precision higher. I've played around with
    # 0.25 < F < 1, and 0.5 works pretty well.
    beta_sq = 0.5
    
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    print(f"Precision: {tp / (tp + fp):.2f}")
    print(f"Recall: {tp / (tp + fn):.2f}")
    print(f"F1 Score: {2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))):.2f}")
    print(f"F{beta_sq} Score: {(1 + beta_sq) * (tp / (tp + fp)) * (tp / (tp + fn)) / (beta_sq * (tp / (tp + fp)) + (tp / (tp + fn))):.2f}")

    from sklearn.metrics import precision_recall_curve

    # Get probabilities on your validation/test set
    y_prob = tree.predict_proba(X_val)[:, 1]

    # Calculate precision, recall, and corresponding thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    y_prob = tree.predict_proba(X_val)[:, 1]
    y_val_np = np.array(y_val)

    sort_idx = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[sort_idx]
    y_val_sorted = y_val_np[sort_idx]

    tp_cum = np.cumsum(y_val_sorted)
    fp_cum = np.cumsum(1 - y_val_sorted)
    total_positives_val = np.sum(y_val_sorted)

    precisions_val = np.divide(
        tp_cum, tp_cum + fp_cum,
        out=np.zeros_like(tp_cum, dtype=float),
        where=(tp_cum + fp_cum) != 0
    )
    recalls_val = np.divide(
        tp_cum, total_positives_val,
        out=np.zeros_like(tp_cum, dtype=float),
        where=total_positives_val != 0
    )

    numerator = (1 + beta_sq) * precisions_val * recalls_val
    denominator = (beta_sq * precisions_val) + recalls_val
    f_beta_scores = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )

    best_idx = np.argmax(f_beta_scores)
    best_threshold = y_prob_sorted[best_idx]

    # If we want the specified F-score, we should have this threshold
    print(f"F1 Score: {2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))):.2f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Best Val F{beta_sq}-Score: {f_beta_scores[best_idx]:.4f}")

    # Don't do a neural network. There are only 360 death examples in the ENTIRE
    # dataset, so a neural network will significantly overfit.

    return

data = pd.read_csv('hackathon_data_clean.csv')
results = main(data)