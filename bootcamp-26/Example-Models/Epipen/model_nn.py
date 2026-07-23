import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from typing import Dict, Any
import warnings



class EpiPenNeuralNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, input_dim: int = 19):
        self.dense1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.bn1 = nnx.BatchNorm(128, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.3, rngs=rngs)

        self.dense2 = nnx.Linear(128, 64, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.3, rngs=rngs)

        self.dense3 = nnx.Linear(64, 32, rngs=rngs)
        self.bn3 = nnx.BatchNorm(32, rngs=rngs)
        self.dropout3 = nnx.Dropout(0.2, rngs=rngs)

        self.output = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x, training: bool = False):
        x = self.dense1(x)
        x = self.bn1(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = self.dropout1(x, deterministic=not training)

        x = self.dense2(x)
        x = self.bn2(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = self.dropout2(x, deterministic=not training)

        x = self.dense3(x)
        x = self.bn3(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = self.dropout3(x, deterministic=not training)

        x = self.output(x)
        return x
    

class EpiPenTrainer:
    def __init__(self, model: EpiPenNeuralNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_auc': [], 'val_auc': []}

    @staticmethod
    @nnx.jit  
    def weighted_binary_crossentropy(logits, labels, class_weights):
        logits = jnp.array(logits)
        labels = jnp.array(labels)
        
        weights = jnp.where(labels == 1, class_weights[1], class_weights[0])
        
        loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        weighted_loss = loss * weights
        
        return jnp.mean(weighted_loss)
    

    @staticmethod
    @nnx.jit
    def compute_auc(logits, labels):
        probabilities = jax.nn.sigmoid(logits).flatten()
        labels = labels.flatten()
        
        sorted_indices = jnp.argsort(-probabilities)
        sorted_labels = labels[sorted_indices]
        
        n_pos = jnp.sum(sorted_labels)
        n_neg = len(sorted_labels) - n_pos
        
        pred = (n_pos == 0) | (n_neg == 0)

        def true_branch(_):
            return jnp.array(0.5)

        def false_branch(_):
            tp_cumsum = jnp.cumsum(sorted_labels)
            fp_cumsum = jnp.cumsum(1 - sorted_labels)
            
            tp_cumsum = jnp.concatenate([jnp.array([0]), tp_cumsum])
            fp_cumsum = jnp.concatenate([jnp.array([0]), fp_cumsum])
            
            tpr = tp_cumsum / jnp.maximum(n_pos, 1)
            fpr = fp_cumsum / jnp.maximum(n_neg, 1)
            
            return jnp.trapezoid(tpr, fpr)

        auc = jax.lax.cond(pred, true_branch, false_branch, operand=None)

        return auc

    @staticmethod
    @nnx.jit
    def compute_metrics(logits, labels):
        predictions = jax.nn.sigmoid(logits) > 0.5
        accuracy = jnp.mean(predictions == labels)
        auc = EpiPenTrainer.compute_auc(logits, labels)
        return {'accuracy': accuracy, 'auc': auc}
    

    @staticmethod
    @nnx.jit
    def train_step(model: EpiPenNeuralNetwork, batch_x: jnp.ndarray, batch_y: jnp.ndarray, class_weights: Dict[int, float], optimizer: nnx.Optimizer):
        def loss_fn(model: EpiPenNeuralNetwork):
            logits = model(batch_x, training=True)
            loss = EpiPenTrainer.weighted_binary_crossentropy(logits, batch_y, class_weights)
            return loss
        
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        
        logits = model(batch_x, training=True)
        metrics = EpiPenTrainer.compute_metrics(logits, batch_y)
        return loss, metrics

    @staticmethod
    @nnx.jit 
    def val_step(model, batch_x: jnp.ndarray, batch_y: jnp.ndarray, class_weights: Dict[int, float]):
        logits = model(batch_x, training=False)
        loss = EpiPenTrainer.weighted_binary_crossentropy(logits, batch_y, class_weights)
        metrics = EpiPenTrainer.compute_metrics(logits, batch_y)
        return loss, metrics
    
    def create_batches(self, X, y, batch_size=32, shuffle=True):
        n_samples = X.shape[0]
        indices = jnp.arange(n_samples)
        
        if shuffle:
            key = jax.random.PRNGKey(42)
            indices = jax.random.permutation(key, indices)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              class_weights: Dict[int, float],
              epochs: int = 10,
              batch_size: int = 32,
              early_stop: int = 3) -> Dict[str, Any]:

        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.0}
        
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train).reshape(-1, 1)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val).reshape(-1, 1)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        total_batches = (len(X_train) + batch_size - 1) // batch_size
        print(f"Starting training for {epochs} epochs...")
        print(f"Total batches per epoch: {total_batches}")
        print("-" * 80)
        
        for epoch in range(epochs):
            train_losses = []
            train_accuracies = []
            train_auc = []
            
            batch_idx = 0
            for batch_x, batch_y in self.create_batches(X_train, y_train, batch_size):
                loss, metrics = EpiPenTrainer.train_step(self.model, batch_x, batch_y, class_weights, self.optimizer)
                train_losses.append(loss)
                train_accuracies.append(metrics['accuracy'])
                train_auc.append(metrics['auc'])
                if batch_idx % 100 == 0 and batch_idx > 0:
                    current_loss = jnp.mean(jnp.array(train_losses[-100:]))
                    current_acc = jnp.mean(jnp.array(train_accuracies[-100:]))
                    print(f"  Epoch {epoch:3d}, Batch {batch_idx:4d}: "
                          f"Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")
                    
                
                batch_idx += 1
            
            val_losses = []
            val_accuracies = []
            val_auc = []
            
            for batch_x, batch_y in self.create_batches(X_val, y_val, batch_size, shuffle=False):
                loss, metrics = EpiPenTrainer.val_step(self.model, batch_x, batch_y, class_weights)
                val_losses.append(loss)
                val_accuracies.append(metrics['accuracy'])
                val_auc.append(metrics['auc'])
            
            epoch_train_loss = jnp.mean(jnp.array(train_losses))
            epoch_train_acc = jnp.mean(jnp.array(train_accuracies))
            epoch_val_loss = jnp.mean(jnp.array(val_losses))
            epoch_val_acc = jnp.mean(jnp.array(val_accuracies))
            epoch_train_auc = jnp.mean(jnp.array(train_auc))
            epoch_val_auc = jnp.mean(jnp.array(val_auc))

            
            self.history['train_loss'].append(float(epoch_train_loss))
            self.history['val_loss'].append(float(epoch_val_loss))
            self.history['train_acc'].append(float(epoch_train_acc))
            self.history['val_acc'].append(float(epoch_val_acc))
            self.history['train_auc'].append(float(epoch_train_auc))
            self.history['val_auc'].append(float(epoch_val_auc))
            
            print(f"EPOCH {epoch:3d} SUMMARY:")
            print(f"  Train - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
            print(f"  Val   - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
            print(f"  Train AUC: {self.history['train_auc'][-1]:.4f}, Val AUC: {self.history['val_auc'][-1]:.4f}")
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                early_stop_counter = 0
                self.best_model_state = nnx.state(self.model)
                print(f"  *** New best val loss: {best_val_loss:.4f} ***")
            else:
                early_stop_counter += 1
                print(f"  early_stop: {early_stop_counter}/{early_stop}")
                if early_stop_counter >= early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    nnx.update(self.model, self.best_model_state)
                    break
            
            print("-" * 80)
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.history
    