#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble prediction system for ATFNet.
Combines multiple models for improved prediction accuracy.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger('atfnet.models.ensemble')


class BasePredictor:
    """Base class for all predictors in the ensemble."""
    
    def __init__(self, name: str):
        """
        Initialize the base predictor.
        
        Args:
            name: Predictor name
        """
        self.name = name
        self.model = None
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the predictor.
        
        Args:
            X: Input features
            y: Target values
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def save(self, path: str) -> None:
        """
        Save the predictor.
        
        Args:
            path: Save path
        """
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path: str) -> None:
        """
        Load the predictor.
        
        Args:
            path: Load path
        """
        raise NotImplementedError("Subclasses must implement load method")


class LSTMPredictor(BasePredictor):
    """LSTM-based predictor for time series forecasting."""
    
    def __init__(
        self, 
        name: str = "lstm_predictor",
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM predictor.
        
        Args:
            name: Predictor name
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__(name)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.input_shape = None
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model.
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
        """
        self.input_shape = input_shape
        
        inputs = Input(shape=input_shape)
        
        if self.bidirectional:
            x = Bidirectional(LSTM(
                self.units, 
                return_sequences=False,
                dropout=self.dropout
            ))(inputs)
        else:
            x = LSTM(
                self.units, 
                return_sequences=False,
                dropout=self.dropout
            )(inputs)
        
        outputs = Dense(1)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            X: Input features (shape: [n_samples, sequence_length, n_features])
            y: Target values
            validation_split: Validation split ratio
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            callbacks: Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X).flatten()
    
    def save(self, path: str) -> None:
        """
        Save the LSTM model.
        
        Args:
            path: Save path
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the LSTM model.
        
        Args:
            path: Load path
        """
        self.model = load_model(path)
        self.is_trained = True


class GRUPredictor(BasePredictor):
    """GRU-based predictor for time series forecasting."""
    
    def __init__(
        self, 
        name: str = "gru_predictor",
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = True
    ):
        """
        Initialize the GRU predictor.
        
        Args:
            name: Predictor name
            units: Number of GRU units
            dropout: Dropout rate
            learning_rate: Learning rate
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__(name)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.input_shape = None
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the GRU model.
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
        """
        self.input_shape = input_shape
        
        inputs = Input(shape=input_shape)
        
        if self.bidirectional:
            x = Bidirectional(GRU(
                self.units, 
                return_sequences=False,
                dropout=self.dropout
            ))(inputs)
        else:
            x = GRU(
                self.units, 
                return_sequences=False,
                dropout=self.dropout
            )(inputs)
        
        outputs = Dense(1)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> Dict[str, List[float]]:
        """
        Train the GRU model.
        
        Args:
            X: Input features (shape: [n_samples, sequence_length, n_features])
            y: Target values
            validation_split: Validation split ratio
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            callbacks: Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the GRU model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X).flatten()
    
    def save(self, path: str) -> None:
        """
        Save the GRU model.
        
        Args:
            path: Save path
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the GRU model.
        
        Args:
            path: Load path
        """
        self.model = load_model(path)
        self.is_trained = True


class RandomForestPredictor(BasePredictor):
    """Random Forest based predictor for time series forecasting."""
    
    def __init__(
        self, 
        name: str = "rf_predictor",
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5
    ):
        """
        Initialize the Random Forest predictor.
        
        Args:
            name: Predictor name
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            y: Target values
        """
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the Random Forest model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the Random Forest model.
        
        Args:
            path: Save path
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """
        Load the Random Forest model.
        
        Args:
            path: Load path
        """
        self.model = joblib.load(path)
        self.is_trained = True


class GradientBoostingPredictor(BasePredictor):
    """Gradient Boosting based predictor for time series forecasting."""
    
    def __init__(
        self, 
        name: str = "gb_predictor",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8
    ):
        """
        Initialize the Gradient Boosting predictor.
        
        Args:
            name: Predictor name
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gradient Boosting model.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            y: Target values
        """
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the Gradient Boosting model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the Gradient Boosting model.
        
        Args:
            path: Save path
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """
        Load the Gradient Boosting model.
        
        Args:
            path: Load path
        """
        self.model = joblib.load(path)
        self.is_trained = True


class ElasticNetPredictor(BasePredictor):
    """ElasticNet based predictor for time series forecasting."""
    
    def __init__(
        self, 
        name: str = "elasticnet_predictor",
        alpha: float = 1.0,
        l1_ratio: float = 0.5
    ):
        """
        Initialize the ElasticNet predictor.
        
        Args:
            name: Predictor name
            alpha: Regularization strength
            l1_ratio: L1 ratio (0 for Ridge, 1 for Lasso)
        """
        super().__init__(name)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ElasticNet model.
        
        Args:
            X: Input features (shape: [n_samples, n_features])
            y: Target values
        """
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ElasticNet model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Reshape X if it's 3D (from RNN input)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the ElasticNet model.
        
        Args:
            path: Save path
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """
        Load the ElasticNet model.
        
        Args:
            path: Load path
        """
        self.model = joblib.load(path)
        self.is_trained = True


class EnsemblePredictor:
    """Ensemble predictor that combines multiple models for improved prediction accuracy."""
    
    def __init__(
        self, 
        base_predictors: Optional[List[BasePredictor]] = None,
        meta_predictor: Optional[BasePredictor] = None,
        ensemble_method: str = 'stacking'
    ):
        """
        Initialize the ensemble predictor.
        
        Args:
            base_predictors: List of base predictors
            meta_predictor: Meta predictor for stacking
            ensemble_method: Ensemble method ('stacking', 'averaging', 'weighted')
        """
        self.base_predictors = base_predictors or []
        self.meta_predictor = meta_predictor
        self.ensemble_method = ensemble_method
        self.weights = None
        self.is_trained = False
    
    def add_predictor(self, predictor: BasePredictor) -> None:
        """
        Add a predictor to the ensemble.
        
        Args:
            predictor: Predictor to add
        """
        self.base_predictors.append(predictor)
    
    def set_meta_predictor(self, predictor: BasePredictor) -> None:
        """
        Set the meta predictor for stacking.
        
        Args:
            predictor: Meta predictor
        """
        self.meta_predictor = predictor
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the ensemble.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Validation split ratio
            epochs: Number of epochs (for neural network models)
            batch_size: Batch size (for neural network models)
            verbose: Verbosity level
            
        Returns:
            Training results
        """
        if not self.base_predictors:
            raise ValueError("No base predictors in ensemble")
        
        # Split data for stacking
        if self.ensemble_method == 'stacking' and validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Train base predictors
        base_results = {}
        for predictor in self.base_predictors:
            logger.info(f"Training base predictor: {predictor.name}")
            
            if isinstance(predictor, (LSTMPredictor, GRUPredictor)):
                # Neural network predictor
                history = predictor.train(
                    X_train, y_train,
                    validation_split=validation_split if X_val is None else 0,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
                base_results[predictor.name] = {
                    'loss': history['loss'][-1],
                    'val_loss': history['val_loss'][-1] if 'val_loss' in history else None
                }
            else:
                # Traditional ML predictor
                predictor.train(X_train, y_train)
                base_results[predictor.name] = {'trained': True}
        
        # For stacking, train meta predictor on base predictions
        if self.ensemble_method == 'stacking' and self.meta_predictor is not None and X_val is not None:
            logger.info(f"Training meta predictor: {self.meta_predictor.name}")
            
            # Get base predictions on validation set
            meta_X = np.column_stack([
                predictor.predict(X_val) for predictor in self.base_predictors
            ])
            
            # Train meta predictor
            self.meta_predictor.train(meta_X, y_val)
            
            # Retrain base predictors on full dataset
            for predictor in self.base_predictors:
                if isinstance(predictor, (LSTMPredictor, GRUPredictor)):
                    predictor.train(X, y, validation_split=0, epochs=epochs, batch_size=batch_size, verbose=verbose)
                else:
                    predictor.train(X, y)
        
        # For weighted ensemble, calculate weights based on validation performance
        elif self.ensemble_method == 'weighted' and X_val is not None:
            # Get base predictions on validation set
            val_predictions = np.column_stack([
                predictor.predict(X_val) for predictor in self.base_predictors
            ])
            
            # Calculate MSE for each predictor
            mse = np.mean((val_predictions - y_val.reshape(-1, 1)) ** 2, axis=0)
            
            # Calculate weights as inverse of MSE (better models get higher weights)
            weights = 1 / (mse + 1e-10)
            self.weights = weights / np.sum(weights)
            
            logger.info(f"Ensemble weights: {self.weights}")
        
        self.is_trained = True
        return base_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Get base predictions
        base_predictions = np.column_stack([
            predictor.predict(X) for predictor in self.base_predictors
        ])
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'stacking' and self.meta_predictor is not None:
            # Use meta predictor to combine base predictions
            return self.meta_predictor.predict(base_predictions)
        
        elif self.ensemble_method == 'weighted' and self.weights is not None:
            # Use weighted average of base predictions
            return np.sum(base_predictions * self.weights, axis=1)
        
        else:
            # Use simple average of base predictions
            return np.mean(base_predictions, axis=1)
    
    def save(self, directory: str) -> None:
        """
        Save the ensemble.
        
        Args:
            directory: Save directory
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save base predictors
        for i, predictor in enumerate(self.base_predictors):
            predictor.save(os.path.join(directory, f"{predictor.name}.model"))
        
        # Save meta predictor if it exists
        if self.meta_predictor is not None:
            self.meta_predictor.save(os.path.join(directory, f"{self.meta_predictor.name}.model"))
        
        # Save ensemble configuration
        config = {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'base_predictors': [p.name for p in self.base_predictors],
            'meta_predictor': self.meta_predictor.name if self.meta_predictor else None
        }
        
        import json
        with open(os.path.join(directory, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f)
    
    def load(self, directory: str) -> None:
        """
        Load the ensemble.
        
        Args:
            directory: Load directory
        """
        # Load ensemble configuration
        import json
        with open(os.path.join(directory, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)
        
        self.ensemble_method = config['ensemble_method']
        self.weights = np.array(config['weights']) if config['weights'] else None
        
        # Load base predictors
        self.base_predictors = []
        for name in config['base_predictors']:
            if 'lstm' in name.lower():
                predictor = LSTMPredictor(name=name)
            elif 'gru' in name.lower():
                predictor = GRUPredictor(name=name)
            elif 'rf' in name.lower():
                predictor = RandomForestPredictor(name=name)
            elif 'gb' in name.lower():
                predictor = GradientBoostingPredictor(name=name)
            elif 'elasticnet' in name.lower():
                predictor = ElasticNetPredictor(name=name)
            else:
                raise ValueError(f"Unknown predictor type: {name}")
            
            predictor.load(os.path.join(directory, f"{name}.model"))
            self.base_predictors.append(predictor)
        
        # Load meta predictor if it exists
        if config['meta_predictor']:
            name = config['meta_predictor']
            if 'lstm' in name.lower():
                self.meta_predictor = LSTMPredictor(name=name)
            elif 'gru' in name.lower():
                self.meta_predictor = GRUPredictor(name=name)
            elif 'rf' in name.lower():
                self.meta_predictor = RandomForestPredictor(name=name)
            elif 'gb' in name.lower():
                self.meta_predictor = GradientBoostingPredictor(name=name)
            elif 'elasticnet' in name.lower():
                self.meta_predictor = ElasticNetPredictor(name=name)
            else:
                raise ValueError(f"Unknown predictor type: {name}")
            
            self.meta_predictor.load(os.path.join(directory, f"{name}.model"))
        
        self.is_trained = True


def create_default_ensemble() -> EnsemblePredictor:
    """
    Create a default ensemble predictor.
    
    Returns:
        Default ensemble predictor
    """
    # Create base predictors
    lstm = LSTMPredictor(name="lstm_predictor", units=64, dropout=0.2)
    gru = GRUPredictor(name="gru_predictor", units=64, dropout=0.2)
    rf = RandomForestPredictor(name="rf_predictor", n_estimators=100)
    gb = GradientBoostingPredictor(name="gb_predictor", n_estimators=100)
    
    # Create meta predictor
    meta = ElasticNetPredictor(name="elasticnet_meta", alpha=0.5, l1_ratio=0.5)
    
    # Create ensemble
    ensemble = EnsemblePredictor(
        base_predictors=[lstm, gru, rf, gb],
        meta_predictor=meta,
        ensemble_method='stacking'
    )
    
    return ensemble
