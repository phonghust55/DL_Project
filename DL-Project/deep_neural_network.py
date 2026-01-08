"""
Deep Neural Network using TensorFlow/Keras for UNSW-NB15 Intrusion Detection.

This module implements a true deep learning architecture with:
- Multiple hidden layers (5+ layers)
- Batch Normalization for stable training
- Dropout for regularization
- Learning rate scheduling
- Early stopping
"""

import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_deep_neural_network(
    input_dim: int,
    hidden_layers: Tuple[int, ...] = (512, 256, 128, 64, 32),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> keras.Model:
    """
    Create a Deep Neural Network for binary classification.
    
    Architecture:
    - Input layer
    - Multiple Dense layers with BatchNorm and Dropout
    - Output layer with sigmoid activation
    
    Args:
        input_dim: Number of input features
        hidden_layers: Tuple of neurons for each hidden layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization strength
        activation: Activation function for hidden layers
        output_activation: Activation for output layer
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(name="DeepNeuralNetwork")
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Hidden layers with BatchNorm and Dropout
    for i, units in enumerate(hidden_layers):
        model.add(
            layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer="he_normal",
                name=f"dense_{i+1}",
            )
        )
        model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        model.add(layers.Activation(activation, name=f"activation_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))
    
    # Output layer
    model.add(layers.Dense(1, activation=output_activation, name="output"))
    
    return model


def create_residual_dnn(
    input_dim: int,
    block_sizes: Tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Create a Deep Neural Network with Residual (Skip) Connections.
    
    This architecture helps with gradient flow in deeper networks.
    
    Args:
        input_dim: Number of input features
        block_sizes: Tuple of neurons for each residual block
        dropout_rate: Dropout rate
        l2_reg: L2 regularization
    
    Returns:
        Compiled Keras model with residual connections
    """
    inputs = layers.Input(shape=(input_dim,), name="input")
    
    # Initial projection to first block size
    x = layers.Dense(
        block_sizes[0],
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer="he_normal",
        name="initial_projection",
    )(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.Activation("relu", name="initial_activation")(x)
    
    # Residual blocks
    for i, units in enumerate(block_sizes):
        # Skip connection
        shortcut = x
        
        # First dense layer in block
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer="he_normal",
            name=f"block_{i+1}_dense_1",
        )(x)
        x = layers.BatchNormalization(name=f"block_{i+1}_bn_1")(x)
        x = layers.Activation("relu", name=f"block_{i+1}_act_1")(x)
        x = layers.Dropout(dropout_rate, name=f"block_{i+1}_dropout_1")(x)
        
        # Second dense layer in block
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer="he_normal",
            name=f"block_{i+1}_dense_2",
        )(x)
        x = layers.BatchNormalization(name=f"block_{i+1}_bn_2")(x)
        
        # Project shortcut if dimensions don't match
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"block_{i+1}_shortcut",
            )(shortcut)
        
        # Add skip connection
        x = layers.Add(name=f"block_{i+1}_add")([x, shortcut])
        x = layers.Activation("relu", name=f"block_{i+1}_act_2")(x)
        x = layers.Dropout(dropout_rate, name=f"block_{i+1}_dropout_2")(x)
    
    # Output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResidualDNN")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
    
    Returns:
        Compiled model
    """
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    
    return model


def get_callbacks(
    model_name: str = "deep_nn",
    patience_early_stop: int = 10,
    patience_lr: int = 5,
    min_lr: float = 1e-7,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    use_tensorboard: bool = True,
) -> list:
    """
    Get training callbacks for deep learning.
    
    Args:
        model_name: Name for saving checkpoints
        patience_early_stop: Patience for early stopping
        patience_lr: Patience for learning rate reduction
        min_lr: Minimum learning rate
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        use_tensorboard: Whether to use TensorBoard logging
    
    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            TensorBoard(
                log_dir=os.path.join(log_dir, model_name),
                histogram_freq=1,
                write_graph=True,
            )
        )
    
    return callbacks


def lr_schedule(epoch: int, lr: float) -> float:
    """
    Custom learning rate schedule with warmup and decay.
    
    Args:
        epoch: Current epoch
        lr: Current learning rate
    
    Returns:
        New learning rate
    """
    warmup_epochs = 5
    if epoch < warmup_epochs:
        # Linear warmup
        return lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        decay_epochs = 100
        progress = (epoch - warmup_epochs) / decay_epochs
        return lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))


class DNNClassifier:
    """
    Wrapper class for Deep Neural Network that provides sklearn-like interface.
    
    This allows the DNN to work with sklearn's evaluation utilities.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_layers: Tuple[int, ...] = (512, 256, 128, 64, 32),
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        use_residual: bool = False,
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_residual = use_residual
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.history = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the model to training data."""
        set_seed(self.random_state)
        
        # Determine input dimension
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Create model
        if self.use_residual:
            self.model = create_residual_dnn(
                input_dim=self.input_dim,
                block_sizes=self.hidden_layers[:3],  # Use first 3 for residual blocks
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
            )
        else:
            self.model = create_deep_neural_network(
                input_dim=self.input_dim,
                hidden_layers=self.hidden_layers,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
            )
        
        # Compile model
        compile_model(self.model, learning_rate=self.learning_rate)
        
        # Print model summary
        if self.verbose:
            self.model.summary()
        
        # Get callbacks
        callbacks = get_callbacks(
            model_name="deep_nn",
            patience_early_stop=10,
            patience_lr=5,
            use_tensorboard=False,  # Disable for simplicity
        )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Handle class imbalance with class weights
        class_counts = np.bincount(y.astype(int))
        total = len(y)
        class_weight = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1]),
        }
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
        )
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        proba_pos = self.model.predict(X, verbose=0).flatten()
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
    
    def evaluate(self, X, y):
        """Evaluate model on test data."""
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "DNNClassifier":
        """Load model from file."""
        instance = cls()
        instance.model = keras.models.load_model(filepath)
        return instance


# Hyperparameter configurations for experiments
def get_model_configs() -> dict:
    """
    Get different model configurations for experimentation.
    
    Returns:
        Dictionary of model configurations
    """
    return {
        "DNN_Small": {
            "hidden_layers": (256, 128, 64),
            "dropout_rate": 0.2,
            "l2_reg": 1e-4,
        },
        "DNN_Medium": {
            "hidden_layers": (512, 256, 128, 64, 32),
            "dropout_rate": 0.3,
            "l2_reg": 1e-4,
        },
        "DNN_Large": {
            "hidden_layers": (1024, 512, 256, 128, 64, 32),
            "dropout_rate": 0.4,
            "l2_reg": 1e-5,
        },
        "DNN_Residual": {
            "hidden_layers": (256, 128, 64),
            "dropout_rate": 0.3,
            "l2_reg": 1e-4,
            "use_residual": True,
        },
    }

