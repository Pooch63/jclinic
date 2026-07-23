import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import warnings

class AllergyVisitNeuralNetwork(nnx.Module):
    
    def __init__(self, rngs: nnx.Rngs, input_dim: int = 24):
        self.dense1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.bn1 = nnx.BatchNorm(128, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.2, rngs=rngs)

        self.dense2 = nnx.Linear(128, 256, rngs=rngs)
        self.bn2 = nnx.BatchNorm(256, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.3, rngs=rngs)

        self.dense3 = nnx.Linear(256, 128, rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        self.dropout3 = nnx.Dropout(0.2, rngs=rngs)

        self.dense4 = nnx.Linear(128, 64, rngs=rngs)
        self.bn4 = nnx.BatchNorm(64, rngs=rngs)
        self.dropout4 = nnx.Dropout(0.1, rngs=rngs)

        self.output = nnx.Linear(64, 1, rngs=rngs)

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

        x = self.dense4(x)
        x = self.bn4(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = self.dropout4(x, deterministic=not training)

        x = self.output(x)
        x = jax.nn.softplus(x)
        return x


class AllergyVisitTrainer:
    
    def __init__(self, model: AllergyVisitNeuralNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'train_mse': [], 'val_mse': []}

    @staticmethod
    @nnx.jit
    def mse_loss(predictions, targets):
        """Mean squared error loss"""
        predictions = jnp.array(predictions)
        targets = jnp.array(targets)
        return jnp.mean((predictions - targets) ** 2)

    @staticmethod
    @nnx.jit
    def compute_metrics(predictions, targets):
        predictions = jnp.array(predictions).flatten()
        targets = jnp.array(targets).flatten()
        
        mae = jnp.mean(jnp.abs(predictions - targets))
        mse = jnp.mean((predictions - targets) ** 2)
        rmse = jnp.sqrt(mse)
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse}

    @staticmethod
    @nnx.jit
    def train_step(model: AllergyVisitNeuralNetwork, batch_x: jnp.ndarray, batch_y: jnp.ndarray, optimizer: nnx.Optimizer):
        def loss_fn(model: AllergyVisitNeuralNetwork):
            predictions = model(batch_x, training=True)
            loss = AllergyVisitTrainer.mse_loss(predictions, batch_y)
            return loss
        
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        
        predictions = model(batch_x, training=True)
        metrics = AllergyVisitTrainer.compute_metrics(predictions, batch_y)
        return loss, metrics

    @staticmethod
    @nnx.jit
    def val_step(model, batch_x: jnp.ndarray, batch_y: jnp.ndarray):
        predictions = model(batch_x, training=False)
        loss = AllergyVisitTrainer.mse_loss(predictions, batch_y)
        metrics = AllergyVisitTrainer.compute_metrics(predictions, batch_y)
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
              epochs: int = 10,
              batch_size: int = 32,
              early_stop: int = 3) -> Dict[str, Any]:

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
            train_maes = []
            train_mses = []
            
            batch_idx = 0
            for batch_x, batch_y in self.create_batches(X_train, y_train, batch_size):
                loss, metrics = AllergyVisitTrainer.train_step(self.model, batch_x, batch_y, self.optimizer)
                train_losses.append(loss)
                train_maes.append(metrics['mae'])
                train_mses.append(metrics['mse'])
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    current_loss = jnp.mean(jnp.array(train_losses[-50:]))
                    current_mae = jnp.mean(jnp.array(train_maes[-50:]))
                    print(f"  Epoch {epoch:3d}, Batch {batch_idx:4d}: "
                          f"Loss: {current_loss:.4f}, MAE: {current_mae:.4f}")
                
                batch_idx += 1
            
            val_losses = []
            val_maes = []
            val_mses = []
            
            for batch_x, batch_y in self.create_batches(X_val, y_val, batch_size, shuffle=False):
                loss, metrics = AllergyVisitTrainer.val_step(self.model, batch_x, batch_y)
                val_losses.append(loss)
                val_maes.append(metrics['mae'])
                val_mses.append(metrics['mse'])
            
            epoch_train_loss = jnp.mean(jnp.array(train_losses))
            epoch_train_mae = jnp.mean(jnp.array(train_maes))
            epoch_train_mse = jnp.mean(jnp.array(train_mses))
            epoch_val_loss = jnp.mean(jnp.array(val_losses))
            epoch_val_mae = jnp.mean(jnp.array(val_maes))
            epoch_val_mse = jnp.mean(jnp.array(val_mses))
            
            self.history['train_loss'].append(float(epoch_train_loss))
            self.history['val_loss'].append(float(epoch_val_loss))
            self.history['train_mae'].append(float(epoch_train_mae))
            self.history['val_mae'].append(float(epoch_val_mae))
            self.history['train_mse'].append(float(epoch_train_mse))
            self.history['val_mse'].append(float(epoch_val_mse))
            
            print(f"EPOCH {epoch:3d} SUMMARY:")
            print(f"  Train - Loss: {epoch_train_loss:.4f}, MAE: {epoch_train_mae:.4f}, MSE: {epoch_train_mse:.4f}")
            print(f"  Val   - Loss: {epoch_val_loss:.4f}, MAE: {epoch_val_mae:.4f}, MSE: {epoch_val_mse:.4f}")
            
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
