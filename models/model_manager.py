import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger('atfnet.model_manager')


class ModelManager:
    """Model management class for the ATFNet Python implementation."""
    
    def __init__(self, config):
        """
        Initialize the model manager.
        
        Args:
            config (dict): Model manager configuration
        """
        self.config = config
        self.models_dir = config.get('models_dir', 'models')
        
        # Create directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
    
    def build_lstm_model(self, input_shape, output_shape=1, model_config=None):
        """
        Build an LSTM model for time series prediction.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, features)
            output_shape (int, optional): Output shape. Defaults to 1.
            model_config (dict, optional): Model configuration. Defaults to None.
            
        Returns:
            tf.keras.Model: LSTM model
        """
        if model_config is None:
            model_config = self.config.get('lstm', {})
        
        # Get model parameters
        units = model_config.get('units', [64, 32])
        dropout = model_config.get('dropout', 0.2)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        # Build model
        model = Sequential()
        
        # Add LSTM layers
        model.add(LSTM(units[0], return_sequences=len(units) > 1, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        for i in range(1, len(units)):
            model.add(LSTM(units[i], return_sequences=i < len(units) - 1))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Add output layer
        model.add(Dense(output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_gru_model(self, input_shape, output_shape=1, model_config=None):
        """
        Build a GRU model for time series prediction.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, features)
            output_shape (int, optional): Output shape. Defaults to 1.
            model_config (dict, optional): Model configuration. Defaults to None.
            
        Returns:
            tf.keras.Model: GRU model
        """
        if model_config is None:
            model_config = self.config.get('gru', {})
        
        # Get model parameters
        units = model_config.get('units', [64, 32])
        dropout = model_config.get('dropout', 0.2)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        # Build model
        model = Sequential()
        
        # Add GRU layers
        model.add(GRU(units[0], return_sequences=len(units) > 1, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        for i in range(1, len(units)):
            model.add(GRU(units[i], return_sequences=i < len(units) - 1))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Add output layer
        model.add(Dense(output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_hybrid_model(self, input_shape, output_shape=1, model_config=None):
        """
        Build a hybrid model combining LSTM and GRU for time series prediction.
        
        Args:
            input_shape (tuple): Input shape (sequence_length, features)
            output_shape (int, optional): Output shape. Defaults to 1.
            model_config (dict, optional): Model configuration. Defaults to None.
            
        Returns:
            tf.keras.Model: Hybrid model
        """
        if model_config is None:
            model_config = self.config.get('hybrid', {})
        
        # Get model parameters
        lstm_units = model_config.get('lstm_units', [64])
        gru_units = model_config.get('gru_units', [64])
        dense_units = model_config.get('dense_units', [32])
        dropout = model_config.get('dropout', 0.2)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        # Build model
        input_layer = Input(shape=input_shape)
        
        # LSTM branch
        lstm = LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1)(input_layer)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(dropout)(lstm)
        
        for i in range(1, len(lstm_units)):
            lstm = LSTM(lstm_units[i], return_sequences=i < len(lstm_units) - 1)(lstm)
            lstm = BatchNormalization()(lstm)
            lstm = Dropout(dropout)(lstm)
        
        # GRU branch
        gru = GRU(gru_units[0], return_sequences=len(gru_units) > 1)(input_layer)
        gru = BatchNormalization()(gru)
        gru = Dropout(dropout)(gru)
        
        for i in range(1, len(gru_units)):
            gru = GRU(gru_units[i], return_sequences=i < len(gru_units) - 1)(gru)
            gru = BatchNormalization()(gru)
            gru = Dropout(dropout)(gru)
        
        # Combine branches
        combined = Concatenate()([lstm, gru])
        
        # Dense layers
        for units in dense_units:
            combined = Dense(units, activation='relu')(combined)
            combined = BatchNormalization()(combined)
            combined = Dropout(dropout)(combined)
        
        # Output layer
        output_layer = Dense(output_shape)(combined)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model_type, X_train, y_train, model_name=None, **kwargs):
        """
        Train a model.
        
        Args:
            model_type (str): Model type ('lstm', 'gru', 'hybrid')
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            model_name (str, optional): Model name. Defaults to None.
            **kwargs: Additional arguments for model training
            
        Returns:
            tuple: Trained model, training history
        """
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Get input shape
        input_shape = X_train.shape[1:]
        
        # Build model
        if model_type == 'lstm':
            model = self.build_lstm_model(input_shape, **kwargs)
        elif model_type == 'gru':
            model = self.build_gru_model(input_shape, **kwargs)
        elif model_type == 'hybrid':
            model = self.build_hybrid_model(input_shape, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Generate model name if not provided
        if model_name is None:
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1)
        )
        
        # Save model and history
        model.save(os.path.join(model_dir, 'model.h5'))
        
        with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        
        # Save model metadata
        metadata = {
            'model_type': model_type,
            'input_shape': input_shape,
            'output_shape': y_train.shape[1] if len(y_train.shape) > 1 else 1,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': kwargs
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Store model in memory
        self.models[model_name] = model
        
        logger.info(f"Model {model_name} trained and saved to {model_dir}")
        
        return model, history
    
    def load_model(self, model_name):
        """
        Load a model from disk.
        
        Args:
            model_name (str): Model name
            
        Returns:
            tf.keras.Model: Loaded model
        """
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} not found")
            return None
        
        model_path = os.path.join(model_dir, 'model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return None
        
        try:
            model = load_model(model_path)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def predict(self, model_name, X):
        """
        Make predictions with a model.
        
        Args:
            model_name (str): Model name
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        # Load model if not in memory
        if model_name not in self.models:
            model = self.load_model(model_name)
            if model is None:
                return None
        else:
            model = self.models[model_name]
        
        # Make predictions
        predictions = model.predict(X)
        
        return predictions
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a model.
        
        Args:
            model_name (str): Model name
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        # Load model if not in memory
        if model_name not in self.models:
            model = self.load_model(model_name)
            if model is None:
                return None
        else:
            model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-output case
            direction_correct = np.sum(np.sign(y_test) == np.sign(y_pred)) / np.prod(y_test.shape)
        else:
            # Single output case
            direction_correct = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        # Store metrics
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'direction_accuracy': float(direction_correct)
        }
        
        # Save metrics
        model_dir = os.path.join(self.models_dir, model_name)
        
        with open(os.path.join(model_dir, 'evaluation.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model {model_name} evaluated: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.6f}, Dir Acc={direction_correct:.2f}")
        
        return metrics
    
    def get_model_list(self):
        """
        Get list of available models.
        
        Returns:
            list: List of model names
        """
        return [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
    
    def get_model_metadata(self, model_name):
        """
        Get model metadata.
        
        Args:
            model_name (str): Model name
            
        Returns:
            dict: Model metadata
        """
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} not found")
            return None
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file {metadata_path} not found")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata for model {model_name}: {e}")
            return None
    
    def delete_model(self, model_name):
        """
        Delete a model.
        
        Args:
            model_name (str): Model name
            
        Returns:
            bool: True if model was deleted successfully, False otherwise
        """
        import shutil
        
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} not found")
            return False
        
        try:
            # Remove from memory
            if model_name in self.models:
                del self.models[model_name]
            
            # Remove from disk
            shutil.rmtree(model_dir)
            
            logger.info(f"Model {model_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
