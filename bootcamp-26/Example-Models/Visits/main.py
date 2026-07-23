import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Any
from flax import nnx

from data_pipeline import AllergyVisitDataSetup
from model_nn import AllergyVisitNeuralNetwork, AllergyVisitTrainer

class AllergyVisitModelEvaluator:

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:

        results = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        }
        return results

    @staticmethod
    def plot_training_history(history: Dict[str, list]):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History - Allergy Visit Prediction", fontsize=16)
        
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss (MSE)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['train_mae'], label='Training MAE', color='blue')
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='red')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['train_mse'], label='Training MSE', color='blue')
        axes[1, 0].plot(history['val_mse'], label='Validation MSE', color='red')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['train_mae'], label='Train MAE', color='blue', linestyle='-')
        axes[1, 1].plot(history['val_mae'], label='Val MAE', color='red', linestyle='-')
        axes[1, 1].set_title('MAE Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('allergy_visit_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main(df: pd.DataFrame) -> Dict[str, Any]:

    data_setup = AllergyVisitDataSetup()
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
    

    rngs = nnx.Rngs(42)
    nn_model = AllergyVisitNeuralNetwork(rngs, input_dim=X_train.shape[1])
    trainer = AllergyVisitTrainer(nn_model)

    print("\nTraining Neural Network...")
    nn_history = trainer.train(X_train, y_train, X_val, y_val)

    evaluator = AllergyVisitModelEvaluator()
    nn_pred = np.array(nn_model(X_test).flatten())
    nn_results = evaluator.evaluate_model(
        y_test, nn_pred
    )
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"  MAE: {nn_results['mae']:.4f}")
    print(f"  MSE: {nn_results['mse']:.4f}")
    print(f"  RMSE: {nn_results['rmse']:.4f}")
    print(f"  R²: {nn_results['r2']:.4f}")
    print(f"  MAPE: {nn_results['mape']:.2f}%")

    evaluator.plot_training_history(nn_history)

    return {
        'model': trainer,
        'results': nn_results,
        'history': nn_history
    }

data = pd.read_csv('hackathon_data_clean.csv')
results = main(data)