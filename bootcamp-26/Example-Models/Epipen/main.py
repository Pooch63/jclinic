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

from data_pipeline import EpipenDataSetup
from model_nn import EpiPenNeuralNetwork, EpiPenTrainer

class EpiPenModelEvaluator:

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

    data_setup = EpipenDataSetup()
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

    rngs = nnx.Rngs(42)
    nn_model = EpiPenNeuralNetwork(rngs, input_dim=X_train.shape[1])
    trainer = EpiPenTrainer(nn_model)

    print("\nTraining Neural Network...")
    nn_history = trainer.train(X_train, y_train, X_val, y_val, class_weights)

    # optimize classification threshold on val split
    val_logits = nn_model(X_val).flatten()
    val_pred_proba = np.array(jax.nn.sigmoid(val_logits))
    from sklearn.metrics import f1_score
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (val_pred_proba > threshold).astype(int)
        score = f1_score(y_val, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    print(f"\nBest classification threshold found on validation set: {best_threshold:.4f} (Val F1: {best_f1:.4f})")

    evaluator = EpiPenModelEvaluator()
    nn_logits = nn_model(X_test).flatten()
    nn_pred_proba = np.array(jax.nn.sigmoid(nn_logits))
    nn_pred = (nn_pred_proba > best_threshold).astype(int)
    nn_results = evaluator.evaluate_model(
        y_test, nn_pred, nn_pred_proba,
    )

    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80)
    print(f"\nAUC Score: {nn_results['auc_score']:.4f}")
    print("\nClassification Report:")
    print(nn_results['classification_report'])
    print("\nConfusion Matrix:")
    print(nn_results['confusion_matrix'])

    evaluator.plot_training_history(nn_history)

    return {
        'model': trainer,
        'results': nn_results,
        'history': nn_history
    }

data = pd.read_csv('hackathon_data_clean.csv')
results = main(data)