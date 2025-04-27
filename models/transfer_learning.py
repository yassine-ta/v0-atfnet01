#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transfer Learning module for ATFNet.
Implements methods for transferring knowledge between different markets and timeframes.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Try to import tensorflow, with fallback to PyTorch
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Input
    BACKEND = 'tensorflow'
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        BACKEND = 'pytorch'
    except ImportError:
        raise ImportError("Either TensorFlow or PyTorch is required for the Transfer Learning module")

logger = logging.getLogger('atfnet.models.transfer_learning')


class TransferLearning:
    """
    Transfer Learning for ATFNet.
    Implements methods for transferring knowledge between different markets and timeframes.
    """
    
    def __init__(self, source_model: Any = None, config: Dict[str, Any] = None, 
                output_dir: str = 'transfer_learning'):
        """
        Initialize the transfer learning module.
        
        Args:
            source_model: Source model to transfer from
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.source_model = source_model
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.transfer_method = self.config.get('transfer_method', 'fine_tuning')
        self.freeze_layers = self.config.get('freeze_layers', 'partial')
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 50)
        
        # Initialize state variables
        self.target_model = None
        self.transfer_history = []
        self.performance_metrics = {}
        
        # Determine model type
        self.model_type = self._determine_model_type()
        
        logger.info(f"Transfer Learning initialized for model type: {self.model_type}")
    
    def _determine_model_type(self) -> str:
        """
        Determine the type of model.
        
        Returns:
            Model type string
        """
        if self.source_model is None:
            return "unknown"
        
        # Check for TensorFlow/Keras model
        if hasattr(self.source_model, 'predict') and hasattr(self.source_model, 'fit'):
            try:
                import tensorflow as tf
                if isinstance(self.source_model, tf.keras.Model):
                    return "tensorflow"
            except ImportError:
                pass
        
        # Check for PyTorch model
        if hasattr(self.source_model, 'forward') and hasattr(self.source_model, 'parameters'):
            try:
                import torch.nn as nn
                if isinstance(self.source_model, nn.Module):
                    return "pytorch"
            except ImportError:
                pass
        
        return "unknown"
    
    def set_source_model(self, model: Any) -> None:
        """
        Set source model to transfer from.
        
        Args:
            model: Source model
        """
        self.source_model = model
        self.model_type = self._determine_model_type()
        logger.info(f"Source model set with type: {self.model_type}")
    
    def transfer_knowledge(self, X_target: np.ndarray, y_target: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None,
                          transfer_method: str = None,
                          freeze_layers: str = None) -> Any:
        """
        Transfer knowledge from source model to target domain.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            transfer_method: Transfer method (default: from config)
            freeze_layers: Freeze strategy (default: from config)
            
        Returns:
            Target model
        """
        if self.source_model is None:
            raise ValueError("Source model not set. Call set_source_model first.")
        
        # Set default parameters
        if transfer_method is None:
            transfer_method = self.transfer_method
        
        if freeze_layers is None:
            freeze_layers = self.freeze_layers
        
        # Perform transfer based on method
        if transfer_method == 'fine_tuning':
            self.target_model = self._fine_tuning(X_target, y_target, X_val, y_val, freeze_layers)
        elif transfer_method == 'feature_extraction':
            self.target_model = self._feature_extraction(X_target, y_target, X_val, y_val)
        elif transfer_method == 'domain_adaptation':
            self.target_model = self._domain_adaptation(X_target, y_target, X_val, y_val)
        else:
            raise ValueError(f"Unknown transfer method: {transfer_method}")
        
        # Evaluate transfer performance
        self._evaluate_transfer(X_target, y_target, X_val, y_val)
        
        # Store transfer details
        self.transfer_history.append({
            'timestamp': datetime.now(),
            'method': transfer_method,
            'freeze_layers': freeze_layers,
            'target_shape': X_target.shape,
            'performance': self.performance_metrics
        })
        
        logger.info(f"Knowledge transferred using {transfer_method} method")
        
        return self.target_model
    
    def _fine_tuning(self, X_target: np.ndarray, y_target: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    freeze_layers: str) -> Any:
        """
        Transfer knowledge using fine-tuning.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            freeze_layers: Freeze strategy
            
        Returns:
            Target model
        """
        if self.model_type == 'tensorflow':
            return self._fine_tuning_tensorflow(X_target, y_target, X_val, y_val, freeze_layers)
        elif self.model_type == 'pytorch':
            return self._fine_tuning_pytorch(X_target, y_target, X_val, y_val, freeze_layers)
        else:
            raise ValueError(f"Fine-tuning not supported for model type: {self.model_type}")
    
    def _fine_tuning_tensorflow(self, X_target: np.ndarray, y_target: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              freeze_layers: str) -> Any:
        """
        Fine-tuning for TensorFlow models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            freeze_layers: Freeze strategy
            
        Returns:
            Target model
        """
        # Clone source model
        target_model = tf.keras.models.clone_model(self.source_model)
        target_model.set_weights(self.source_model.get_weights())
        
        # Freeze layers based on strategy
        if freeze_layers == 'all':
            # Freeze all layers except the last one
            for layer in target_model.layers[:-1]:
                layer.trainable = False
        elif freeze_layers == 'partial':
            # Freeze first half of the layers
            n_layers = len(target_model.layers)
            for i, layer in enumerate(target_model.layers):
                if i < n_layers // 2:
                    layer.trainable = False
        elif freeze_layers == 'none':
            # Don't freeze any layers
            pass
        else:
            raise ValueError(f"Unknown freeze strategy: {freeze_layers}")
        
        # Compile model
        target_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        history = target_model.fit(
            X_target, y_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0
        )
        
        # Store training history
        self.train_history = history.history
        
        return target_model
    
    def _fine_tuning_pytorch(self, X_target: np.ndarray, y_target: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           freeze_layers: str) -> Any:
        """
        Fine-tuning for PyTorch models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            freeze_layers: Freeze strategy
            
        Returns:
            Target model
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Clone source model
        target_model = type(self.source_model)(*self.source_model.init_args)
        target_model.load_state_dict(self.source_model.state_dict())
        
        # Freeze layers based on strategy
        if freeze_layers == 'all':
            # Freeze all layers except the last one
            for name, param in target_model.named_parameters():
                if 'output' not in name and 'fc' not in name:
                    param.requires_grad = False
        elif freeze_layers == 'partial':
            # Freeze first half of the layers
            freeze_count = sum(1 for _ in target_model.parameters()) // 2
            for i, param in enumerate(target_model.parameters()):
                if i < freeze_count:
                    param.requires_grad = False
        elif freeze_layers == 'none':
            # Don't freeze any layers
            pass
        else:
            raise ValueError(f"Unknown freeze strategy: {freeze_layers}")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_target)
        y_tensor = torch.FloatTensor(y_target)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(target_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        train_history = {
            'loss': [],
            'val_loss': []
        }
        
        # Train model
        target_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = target_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record training loss
            train_history['loss'].append(epoch_loss / len(dataloader))
            
            # Validate
            if val_dataloader is not None:
                val_loss = 0.0
                target_model.eval()
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_dataloader:
                        val_outputs = target_model(X_val_batch)
                        val_loss += criterion(val_outputs, y_val_batch).item()
                
                train_history['val_loss'].append(val_loss / len(val_dataloader))
                target_model.train()
        
        # Store training history
        self.train_history = train_history
        
        return target_model
    
    def _feature_extraction(self, X_target: np.ndarray, y_target: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Transfer knowledge using feature extraction.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        if self.model_type == 'tensorflow':
            return self._feature_extraction_tensorflow(X_target, y_target, X_val, y_val)
        elif self.model_type == 'pytorch':
            return self._feature_extraction_pytorch(X_target, y_target, X_val, y_val)
        else:
            raise ValueError(f"Feature extraction not supported for model type: {self.model_type}")
    
    def _feature_extraction_tensorflow(self, X_target: np.ndarray, y_target: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Feature extraction for TensorFlow models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        # Create feature extractor
        feature_extractor = Model(
            inputs=self.source_model.inputs,
            outputs=self.source_model.layers[-2].output
        )
        
        # Extract features
        features_target = feature_extractor.predict(X_target)
        
        # Extract validation features
        features_val = None
        if X_val is not None:
            features_val = feature_extractor.predict(X_val)
        
        # Create new model
        input_layer = Input(shape=(features_target.shape[1],))
        output_layer = Dense(1)(input_layer)
        
        target_model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        target_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Prepare validation data
        validation_data = None
        if features_val is not None and y_val is not None:
            validation_data = (features_val, y_val)
        
        # Train model
        history = target_model.fit(
            features_target, y_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0
        )
        
        # Store training history
        self.train_history = history.history
        
        # Create combined model
        combined_inputs = self.source_model.inputs
        features = feature_extractor(combined_inputs)
        outputs = target_model(features)
        
        combined_model = Model(inputs=combined_inputs, outputs=outputs)
        
        return combined_model
    
    def _feature_extraction_pytorch(self, X_target: np.ndarray, y_target: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Feature extraction for PyTorch models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create feature extractor
        class FeatureExtractor(nn.Module):
            def __init__(self, source_model):
                super(FeatureExtractor, self).__init__()
                # Assuming the source model has a structure where we can access layers
                # This is a simplified example and may need to be adapted
                self.features = nn.Sequential(*list(source_model.children())[:-1])
            
            def forward(self, x):
                return self.features(x)
        
        feature_extractor = FeatureExtractor(self.source_model)
        
        # Extract features
        feature_extractor.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_target)
            features_target = feature_extractor(X_tensor).cpu().numpy()
            
            if X_val is not None:
                X_val_tensor = torch.FloatTensor(X_val)
                features_val = feature_extractor(X_val_tensor).cpu().numpy()
            else:
                features_val = None
        
        # Create new model
        class TargetModel(nn.Module):
            def __init__(self, input_size):
                super(TargetModel, self).__init__()
                self.fc = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        target_model = TargetModel(features_target.shape[1])
        
        # Prepare data
        features_tensor = torch.FloatTensor(features_target)
        y_tensor = torch.FloatTensor(y_target)
        dataset = TensorDataset(features_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data
        val_dataloader = None
        if features_val is not None and y_val is not None:
            features_val_tensor = torch.FloatTensor(features_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(features_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(target_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        train_history = {
            'loss': [],
            'val_loss': []
        }
        
        # Train model
        target_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = target_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record training loss
            train_history['loss'].append(epoch_loss / len(dataloader))
            
            # Validate
            if val_dataloader is not None:
                val_loss = 0.0
                target_model.eval()
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_dataloader:
                        val_outputs = target_model(X_val_batch)
                        val_loss += criterion(val_outputs, y_val_batch).item()
                
                train_history['val_loss'].append(val_loss / len(val_dataloader))
                target_model.train()
        
        # Store training history
        self.train_history = train_history
        
        # Create combined model
        class CombinedModel(nn.Module):
            def __init__(self, feature_extractor, target_model):
                super(CombinedModel, self).__init__()
                self.feature_extractor = feature_extractor
                self.target_model = target_model
            
            def forward(self, x):
                features = self.feature_extractor(x)
                return self.target_model(features)
        
        combined_model = CombinedModel(feature_extractor, target_model)
        
        return combined_model
    
    def _domain_adaptation(self, X_target: np.ndarray, y_target: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Transfer knowledge using domain adaptation.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        if self.model_type == 'tensorflow':
            return self._domain_adaptation_tensorflow(X_target, y_target, X_val, y_val)
        elif self.model_type == 'pytorch':
            return self._domain_adaptation_pytorch(X_target, y_target, X_val, y_val)
        else:
            raise ValueError(f"Domain adaptation not supported for model type: {self.model_type}")
    
    def _domain_adaptation_tensorflow(self, X_target: np.ndarray, y_target: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Domain adaptation for TensorFlow models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        # Create domain adaptation model
        # This is a simplified implementation of domain-adversarial neural networks (DANN)
        
        # Feature extractor (shared between domains)
        feature_extractor = tf.keras.models.Model(
            inputs=self.source_model.inputs,
            outputs=self.source_model.layers[-2].output
        )
        
        # Task classifier (predicts target variable)
        task_input = tf.keras.layers.Input(shape=(feature_extractor.output.shape[-1],))
        task_output = tf.keras.layers.Dense(1)(task_input)
        task_classifier = tf.keras.models.Model(inputs=task_input, outputs=task_output)
        
        # Combined model
        input_layer = self.source_model.inputs
        features = feature_extractor(input_layer)
        predictions = task_classifier(features)
        
        target_model = tf.keras.models.Model(inputs=input_layer, outputs=predictions)
        
        # Compile model
        target_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        history = target_model.fit(
            X_target, y_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0
        )
        
        # Store training history
        self.train_history = history.history
        
        return target_model
    
    def _domain_adaptation_pytorch(self, X_target: np.ndarray, y_target: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Domain adaptation for PyTorch models.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Target model
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create feature extractor
        class FeatureExtractor(nn.Module):
            def __init__(self, source_model):
                super(FeatureExtractor, self).__init__()
                # Assuming the source model has a structure where we can access layers
                self.features = nn.Sequential(*list(source_model.children())[:-1])
            
            def forward(self, x):
                return self.features(x)
        
        feature_extractor = FeatureExtractor(self.source_model)
        
        # Create task classifier
        class TaskClassifier(nn.Module):
            def __init__(self, input_size):
                super(TaskClassifier, self).__init__()
                self.fc = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        # Get feature size
        with torch.no_grad():
            X_sample = torch.FloatTensor(X_target[:1])
            feature_size = feature_extractor(X_sample).shape[1]
        
        task_classifier = TaskClassifier(feature_size)
        
        # Create combined model
        class CombinedModel(nn.Module):
            def __init__(self, feature_extractor, task_classifier):
                super(CombinedModel, self).__init__()
                self.feature_extractor = feature_extractor
                self.task_classifier = task_classifier
            
            def forward(self, x):
                features = self.feature_extractor(x)
                return self.task_classifier(features)
        
        target_model = CombinedModel(feature_extractor, task_classifier)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_target)
        y_tensor = torch.FloatTensor(y_target)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(target_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        train_history = {
            'loss': [],
            'val_loss': []
        }
        
        # Train model
        target_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = target_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record training loss
            train_history['loss'].append(epoch_loss / len(dataloader))
            
            # Validate
            if val_dataloader is not None:
                val_loss = 0.0
                target_model.eval()
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_dataloader:
                        val_outputs = target_model(X_val_batch)
                        val_loss += criterion(val_outputs, y_val_batch).item()
                
                train_history['val_loss'].append(val_loss / len(val_dataloader))
                target_model.train()
        
        # Store training history
        self.train_history = train_history
        
        return target_model
    
    def _evaluate_transfer(self, X_target: np.ndarray, y_target: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Evaluate transfer performance.
        
        Args:
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            X_val: Validation feature matrix
            y_val: Validation target variable
        """
        if self.target_model is None:
            logger.warning("Target model not available. Run transfer_knowledge first.")
            return
        
        # Evaluate on target domain
        if self.model_type == 'tensorflow':
            y_pred_target = self.target_model.predict(X_target).flatten()
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.target_model.predict(X_val).flatten()
        else:  # PyTorch
            import torch
            self.target_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_target)
                y_pred_target = self.target_model(X_tensor).cpu().numpy().flatten()
                
                if X_val is not None and y_val is not None:
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_pred_val = self.target_model(X_val_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        mse_target = mean_squared_error(y_target, y_pred_target)
        r2_target = r2_score(y_target, y_pred_target)
        
        metrics = {
            'target_mse': mse_target,
            'target_r2': r2_target
        }
        
        if X_val is not None and y_val is not None:
            mse_val = mean_squared_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)
            
            metrics['val_mse'] = mse_val
            metrics['val_r2'] = r2_val
        
        # Store metrics
        self.performance_metrics = metrics
        
        logger.info(f"Transfer performance: Target MSE = {mse_target:.6f}, Target R² = {r2_target:.6f}")
        
        if 'val_mse' in metrics:
            logger.info(f"Validation: MSE = {metrics['val_mse']:.6f}, R² = {metrics['val_r2']:.6f}")
    
    def plot_training_history(self, save_plot: bool = True,
                             filename: str = 'transfer_learning_history.png') -> None:
        """
        Plot training history.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not hasattr(self, 'train_history') or not self.train_history:
            logger.warning("No training history to plot. Run transfer_knowledge first.")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot training loss
            plt.plot(self.train_history['loss'], 'b-', label='Training Loss')
            
            # Plot validation loss if available
            if 'val_loss' in self.train_history and self.train_history['val_loss']:
                plt.plot(self.train_history['val_loss'], 'r-', label='Validation Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Transfer Learning History')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Training history plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            raise
    
    def plot_performance_comparison(self, X_source: np.ndarray, y_source: np.ndarray,
                                  X_target: np.ndarray, y_target: np.ndarray,
                                  save_plot: bool = True,
                                  filename: str = 'performance_comparison.png') -> None:
        """
        Plot performance comparison between source and target models.
        
        Args:
            X_source: Source domain feature matrix
            y_source: Source domain target variable
            X_target: Target domain feature matrix
            y_target: Target domain target variable
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.source_model is None or self.target_model is None:
            logger.warning("Source or target model not available. Run transfer_knowledge first.")
            return
        
        try:
            # Get predictions
            if self.model_type == 'tensorflow':
                y_pred_source = self.source_model.predict(X_source).flatten()
                y_pred_target_source = self.source_model.predict(X_target).flatten()
                y_pred_target = self.target_model.predict(X_target).flatten()
            else:  # PyTorch
                import torch
                self.source_model.eval()
                self.target_model.eval()
                with torch.no_grad():
                    X_source_tensor = torch.FloatTensor(X_source)
                    y_pred_source = self.source_model(X_source_tensor).cpu().numpy().flatten()
                    
                    X_target_tensor = torch.FloatTensor(X_target)
                    y_pred_target_source = self.source_model(X_target_tensor).cpu().numpy().flatten()
                    y_pred_target = self.target_model(X_target_tensor).cpu().numpy().flatten()
            
            # Calculate metrics
            source_mse = mean_squared_error(y_source, y_pred_source)
            source_r2 = r2_score(y_source, y_pred_source)
            
            target_source_mse = mean_squared_error(y_target, y_pred_target_source)
            target_source_r2 = r2_score(y_target, y_pred_target_source)
            
            target_mse = mean_squared_error(y_target, y_pred_target)
            target_r2 = r2_score(y_target, y_pred_target)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot MSE comparison
            models = ['Source on Source', 'Source on Target', 'Target on Target']
            mse_values = [source_mse, target_source_mse, target_mse]
            
            ax1.bar(models, mse_values, color=['blue', 'red', 'green'])
            ax1.set_ylabel('Mean Squared Error')
            ax1.set_title('MSE Comparison')
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Plot R² comparison
            r2_values = [source_r2, target_source_r2, target_r2]
            
            ax2.bar(models, r2_values, color=['blue', 'red', 'green'])
            ax2.set_ylabel('R² Score')
            ax2.set_title('R² Comparison')
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add values on bars
            for i, ax in enumerate([ax1, ax2]):
                values = mse_values if i == 0 else r2_values
                for j, v in enumerate(values):
                    ax.text(j, v, f'{v:.4f}', ha='center', va='bottom')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Performance comparison plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance comparison: {e}")
            raise
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save target model to file.
        
        Args:
            filepath: Path to save model (default: output_dir/target_model)
            
        Returns:
            Path to saved model
        """
        if self.target_model is None:
            logger.warning("Target model not available. Run transfer_knowledge first.")
            return None
        
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'target_model')
        
        if self.model_type == 'tensorflow':
            self.target_model.save(f"{filepath}.h5")
            logger.info(f"Target model saved to {filepath}.h5")
            return f"{filepath}.h5"
        else:  # PyTorch
            import torch
            torch.save(self.target_model.state_dict(), f"{filepath}.pt")
            logger.info(f"Target model saved to {filepath}.pt")
            return f"{filepath}.pt"
    
    def load_model(self, filepath: str) -> None:
        """
        Load target model from file.
        
        Args:
            filepath: Path to model file
        """
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return
        
        if self.model_type == 'tensorflow':
            import tensorflow as tf
            self.target_model = tf.keras.models.load_model(filepath)
            logger.info(f"Target model loaded from {filepath}")
        else:  # PyTorch
            import torch
            if self.target_model is None:
                logger.warning("Target model not initialized. Cannot load weights.")
                return
            
            self.target_model.load_state_dict(torch.load(filepath))
            logger.info(f"Target model loaded from {filepath}")
