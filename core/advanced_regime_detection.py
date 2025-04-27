#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced market regime detection for ATFNet.
Implements unsupervised learning for regime identification.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger('atfnet.core.regime_detection')


class UnsupervisedRegimeDetector:
    """
    Unsupervised market regime detector using clustering algorithms.
    """
    
    def __init__(
        self, 
        n_regimes: int = 4,
        method: str = 'gmm',
        lookback_period: int = 100,
        feature_set: str = 'comprehensive'
    ):
        """
        Initialize the unsupervised regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            method: Clustering method ('kmeans', 'gmm', 'dbscan')
            lookback_period: Lookback period for regime detection
            feature_set: Feature set to use ('basic', 'comprehensive')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.lookback_period = lookback_period
        self.feature_set = feature_set
        
        # Initialize model
        self._init_model()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize PCA
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        
        # Regime characteristics
        self.regime_characteristics = {}
        
        # Current regime
        self.current_regime = None
    
    def _init_model(self) -> None:
        """Initialize clustering model based on method."""
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=42,
                covariance_type='full',
                n_init=10
            )
        elif self.method == 'dbscan':
            # DBSCAN doesn't need n_regimes, it determines automatically
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection.
        
        Args:
            df: Price data with indicators
            
        Returns:
            Feature matrix
        """
        # Basic features
        features = []
        
        # Volatility features
        if 'atr_14' in df.columns:
            features.append(df['atr_14'] / df['close'])  # Normalized ATR
        
        if 'close' in df.columns and len(df) > 1:
            # Log returns
            log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            features.append(log_returns)
            
            # Volatility (rolling std of returns)
            vol = log_returns.rolling(20).std().fillna(0)
            features.append(vol)
        
        # Trend features
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            # EMA ratio
            features.append(df['ema_20'] / df['ema_50'] - 1)
        
        if 'trend_strength' in df.columns:
            features.append(df['trend_strength'])
        
        # Comprehensive feature set
        if self.feature_set == 'comprehensive':
            # Momentum features
            if 'rsi_14' in df.columns:
                features.append((df['rsi_14'] - 50) / 50)  # Normalized RSI
            
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                features.append(df['macd'] - df['macd_signal'])  # MACD histogram
            
            # Mean reversion features
            if 'close' in df.columns and 'ema_20' in df.columns:
                features.append((df['close'] / df['ema_20'] - 1) * 100)  # % distance from EMA
            
            # Frequency domain features
            for col in ['spectral_entropy', 'cyclicality_score', 'freq_trend_strength']:
                if col in df.columns:
                    features.append(df[col])
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        return feature_matrix
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the regime detection model.
        
        Args:
            df: Price data with indicators
            
        Returns:
            Fitting results
        """
        # Extract features
        features = self._extract_features(df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        reduced_features = self.pca.fit_transform(scaled_features)
        
        logger.info(f"PCA reduced features from {features.shape[1]} to {reduced_features.shape[1]} dimensions")
        
        # Fit model
        if self.method in ['kmeans', 'gmm']:
            self.model.fit(reduced_features)
            
            # Get regime labels
            if self.method == 'kmeans':
                labels = self.model.labels_
            else:  # GMM
                labels = self.model.predict(reduced_features)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(reduced_features, labels)
            else:
                silhouette = 0
                
            logger.info(f"Fitted {self.method} with {self.n_regimes} regimes, silhouette score: {silhouette:.4f}")
            
        else:  # DBSCAN
            labels = self.model.fit_predict(reduced_features)
            
            # DBSCAN may return -1 for noise points
            n_clusters = len(np.unique(labels[labels >= 0]))
            noise_points = np.sum(labels == -1)
            
            logger.info(f"Fitted DBSCAN with {n_clusters} regimes, {noise_points} noise points")
            
            # If DBSCAN found too few or too many clusters, fall back to GMM
            if n_clusters < 2 or n_clusters > 10:
                logger.warning(f"DBSCAN found {n_clusters} regimes, falling back to GMM")
                self.method = 'gmm'
                self._init_model()
                self.model.fit(reduced_features)
                labels = self.model.predict(reduced_features)
        
        # Analyze regime characteristics
        self._analyze_regimes(df, labels)
        
        # Return results
        return {
            'n_regimes': len(np.unique(labels[labels >= 0])),
            'regime_distribution': np.bincount(labels[labels >= 0]),
            'regime_characteristics': self.regime_characteristics
        }
    
    def _analyze_regimes(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        """
        Analyze characteristics of detected regimes.
        
        Args:
            df: Price data with indicators
            labels: Regime labels
        """
        # Ensure df and labels have the same length
        if len(df) != len(labels):
            logger.warning(f"DataFrame length ({len(df)}) doesn't match labels length ({len(labels)})")
            labels = labels[:len(df)]
        
        # Add labels to dataframe
        df_with_labels = df.copy()
        df_with_labels['regime'] = labels
        
        # Analyze each regime
        unique_regimes = np.unique(labels)
        unique_regimes = unique_regimes[unique_regimes >= 0]  # Filter out noise points
        
        self.regime_characteristics = {}
        
        for regime in unique_regimes:
            regime_data = df_with_labels[df_with_labels['regime'] == regime]
            
            # Skip if too few data points
            if len(regime_data) < 10:
                continue
            
            # Calculate regime characteristics
            characteristics = {}
            
            # Volatility
            if 'atr_14' in regime_data.columns and 'close' in regime_data.columns:
                characteristics['volatility'] = (regime_data['atr_14'] / regime_data['close']).mean()
            
            # Returns
            if 'close' in regime_data.columns:
                returns = regime_data['close'].pct_change().dropna()
                characteristics['mean_return'] = returns.mean()
                characteristics['return_volatility'] = returns.std()
                characteristics['sharpe'] = characteristics['mean_return'] / characteristics['return_volatility'] if characteristics['return_volatility'] > 0 else 0
            
            # Trend
            if 'trend_strength' in regime_data.columns:
                characteristics['trend_strength'] = regime_data['trend_strength'].mean()
            
            # Frequency domain
            for col in ['spectral_entropy', 'cyclicality_score', 'freq_trend_strength']:
                if col in regime_data.columns:
                    characteristics[col] = regime_data[col].mean()
            
            # Regime type classification
            if 'volatility' in characteristics and 'trend_strength' in characteristics:
                if characteristics['volatility'] > 0.015:  # High volatility
                    if characteristics['trend_strength'] > 0.6:
                        characteristics['type'] = 'Volatile Trend'
                    else:
                        characteristics['type'] = 'Choppy'
                else:  # Low volatility
                    if characteristics['trend_strength'] > 0.6:
                        characteristics['type'] = 'Stable Trend'
                    else:
                        characteristics['type'] = 'Range Bound'
            
            self.regime_characteristics[int(regime)] = characteristics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Args:
            df: Price data with indicators
            
        Returns:
            Regime labels
        """
        # Extract features
        features = self._extract_features(df)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Apply PCA
        reduced_features = self.pca.transform(scaled_features)
        
        # Predict regimes
        if self.method == 'kmeans':
            labels = self.model.predict(reduced_features)
        elif self.method == 'gmm':
            labels = self.model.predict(reduced_features)
        else:  # DBSCAN
            labels = self.model.fit_predict(reduced_features)
        
        return labels
    
    def detect_regime(self, df: pd.DataFrame) -> int:
        """
        Detect current market regime.
        
        Args:
            df: Recent price data with indicators
            
        Returns:
            Current regime
        """
        # Ensure enough data
        if len(df) < self.lookback_period:
            logger.warning(f"Not enough data for regime detection. Need {self.lookback_period} 
            logger.warning(f"Not enough data for regime detection. Need {self.lookback_period} points, got {len(df)}")
            
            # Use last regime if not enough data
            if self.current_regime is not None:
                return self.current_regime
            else:
                return 0  # Default regime
        
        # Use recent data for regime detection
        recent_data = df.iloc[-self.lookback_period:]
        
        # Predict regime
        labels = self.predict(recent_data)
        
        # Get most common regime in recent data
        if self.method == 'dbscan' and -1 in labels:
            # Filter out noise points
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                most_common = np.bincount(valid_labels).argmax()
            else:
                most_common = 0  # Default regime if all points are noise
        else:
            most_common = np.bincount(labels).argmax()
        
        self.current_regime = int(most_common)
        
        # Log regime characteristics
        if self.current_regime in self.regime_characteristics:
            regime_type = self.regime_characteristics[self.current_regime].get('type', 'Unknown')
            logger.info(f"Detected regime {self.current_regime} ({regime_type})")
        else:
            logger.info(f"Detected regime {self.current_regime} (characteristics unknown)")
        
        return self.current_regime
    
    def get_regime_characteristics(self, regime: Optional[int] = None) -> Dict[str, Any]:
        """
        Get characteristics of a specific regime.
        
        Args:
            regime: Regime to get characteristics for (default: current regime)
            
        Returns:
            Regime characteristics
        """
        if regime is None:
            regime = self.current_regime
        
        if regime is None:
            logger.warning("No regime detected yet")
            return {}
        
        if regime in self.regime_characteristics:
            return self.regime_characteristics[regime]
        else:
            logger.warning(f"No characteristics available for regime {regime}")
            return {}
    
    def save(self, path: str) -> None:
        """
        Save the regime detector.
        
        Args:
            path: Save path
        """
        import joblib
        
        # Create a dictionary with all necessary objects
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_regimes': self.n_regimes,
            'method': self.method,
            'lookback_period': self.lookback_period,
            'feature_set': self.feature_set,
            'regime_characteristics': self.regime_characteristics,
            'current_regime': self.current_regime
        }
        
        # Save to file
        joblib.dump(save_dict, path)
        
        logger.info(f"Saved regime detector to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the regime detector.
        
        Args:
            path: Load path
        """
        import joblib
        
        # Load from file
        save_dict = joblib.load(path)
        
        # Restore all objects
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.pca = save_dict['pca']
        self.n_regimes = save_dict['n_regimes']
        self.method = save_dict['method']
        self.lookback_period = save_dict['lookback_period']
        self.feature_set = save_dict['feature_set']
        self.regime_characteristics = save_dict['regime_characteristics']
        self.current_regime = save_dict['current_regime']
        
        logger.info(f"Loaded regime detector from {path}")


class RegimeTransitionPredictor:
    """
    Predicts transitions between market regimes.
    """
    
    def __init__(
        self, 
        lookback_window: int = 50,
        prediction_horizon: int = 10,
        transition_threshold: float = 0.6
    ):
        """
        Initialize the regime transition predictor.
        
        Args:
            lookback_window: Lookback window for transition prediction
            prediction_horizon: Prediction horizon
            transition_threshold: Probability threshold for transition prediction
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.transition_threshold = transition_threshold
        
        # Initialize model
        self.model = None
        
        # Regime history
        self.regime_history = []
        
        # Transition matrix
        self.transition_matrix = None
    
    def update_regime_history(self, regime: int) -> None:
        """
        Update regime history.
        
        Args:
            regime: Current regime
        """
        self.regime_history.append(regime)
        
        # Keep history limited to lookback window
        if len(self.regime_history) > self.lookback_window:
            self.regime_history = self.regime_history[-self.lookback_window:]
    
    def calculate_transition_matrix(self) -> np.ndarray:
        """
        Calculate regime transition matrix.
        
        Returns:
            Transition matrix
        """
        if len(self.regime_history) < 10:
            logger.warning("Not enough regime history for transition matrix calculation")
            return None
        
        # Get unique regimes
        unique_regimes = np.unique(self.regime_history)
        n_regimes = len(unique_regimes)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i]
            to_regime = self.regime_history[i + 1]
            
            # Get indices in unique_regimes
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)
        
        self.transition_matrix = transition_matrix
        return transition_matrix
    
    def predict_transition(self, current_regime: int) -> Tuple[bool, Optional[int], float]:
        """
        Predict regime transition.
        
        Args:
            current_regime: Current regime
            
        Returns:
            Tuple of (transition predicted, predicted regime, probability)
        """
        # Update regime history
        self.update_regime_history(current_regime)
        
        # Calculate transition matrix if needed
        if self.transition_matrix is None:
            self.calculate_transition_matrix()
        
        if self.transition_matrix is None:
            return False, None, 0.0
        
        # Get unique regimes
        unique_regimes = np.unique(self.regime_history)
        
        # Get index of current regime
        try:
            current_idx = np.where(unique_regimes == current_regime)[0][0]
        except IndexError:
            logger.warning(f"Current regime {current_regime} not in history")
            return False, None, 0.0
        
        # Get transition probabilities from current regime
        transition_probs = self.transition_matrix[current_idx]
        
        # Find most likely transition
        if np.max(transition_probs) > self.transition_threshold and np.argmax(transition_probs) != current_idx:
            next_idx = np.argmax(transition_probs)
            next_regime = unique_regimes[next_idx]
            probability = transition_probs[next_idx]
            
            return True, int(next_regime), float(probability)
        else:
            return False, None, 0.0
    
    def get_transition_matrix_df(self) -> pd.DataFrame:
        """
        Get transition matrix as a DataFrame.
        
        Returns:
            Transition matrix DataFrame
        """
        if self.transition_matrix is None:
            self.calculate_transition_matrix()
        
        if self.transition_matrix is None:
            return pd.DataFrame()
        
        # Get unique regimes
        unique_regimes = np.unique(self.regime_history)
        
        # Create DataFrame
        df = pd.DataFrame(
            self.transition_matrix,
            index=[f"From Regime {r}" for r in unique_regimes],
            columns=[f"To Regime {r}" for r in unique_regimes]
        )
        
        return df


class AdvancedMarketRegimeDetector:
    """
    Advanced market regime detector that combines unsupervised learning with transition prediction.
    """
    
    def __init__(
        self, 
        n_regimes: int = 4,
        method: str = 'gmm',
        lookback_period: int = 100,
        feature_set: str = 'comprehensive',
        prediction_horizon: int = 10,
        transition_threshold: float = 0.6
    ):
        """
        Initialize the advanced market regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            method: Clustering method ('kmeans', 'gmm', 'dbscan')
            lookback_period: Lookback period for regime detection
            feature_set: Feature set to use ('basic', 'comprehensive')
            prediction_horizon: Prediction horizon for transition prediction
            transition_threshold: Probability threshold for transition prediction
        """
        # Initialize regime detector
        self.regime_detector = UnsupervisedRegimeDetector(
            n_regimes=n_regimes,
            method=method,
            lookback_period=lookback_period,
            feature_set=feature_set
        )
        
        # Initialize transition predictor
        self.transition_predictor = RegimeTransitionPredictor(
            lookback_window=lookback_period,
            prediction_horizon=prediction_horizon,
            transition_threshold=transition_threshold
        )
        
        # Current regime
        self.current_regime = None
        
        # Predicted transition
        self.predicted_transition = None
        self.transition_probability = 0.0
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the regime detection model.
        
        Args:
            df: Price data with indicators
            
        Returns:
            Fitting results
        """
        return self.regime_detector.fit(df)
    
    def detect_regime(self, df: pd.DataFrame) -> int:
        """
        Detect current market regime and predict transitions.
        
        Args:
            df: Recent price data with indicators
            
        Returns:
            Current regime
        """
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(df)
        
        # Predict transition
        transition_predicted, next_regime, probability = self.transition_predictor.predict_transition(self.current_regime)
        
        if transition_predicted:
            self.predicted_transition = next_regime
            self.transition_probability = probability
            
            # Get regime characteristics
            current_characteristics = self.regime_detector.get_regime_characteristics(self.current_regime)
            next_characteristics = self.regime_detector.get_regime_characteristics(next_regime)
            
            current_type = current_characteristics.get('type', 'Unknown')
            next_type = next_characteristics.get('type', 'Unknown')
            
            logger.info(f"Predicted transition from regime {self.current_regime} ({current_type}) to {next_regime} ({next_type}) with probability {probability:.2f}")
        else:
            self.predicted_transition = None
            self.transition_probability = 0.0
        
        return self.current_regime
    
    def get_regime_characteristics(self, regime: Optional[int] = None) -> Dict[str, Any]:
        """
        Get characteristics of a specific regime.
        
        Args:
            regime: Regime to get characteristics for (default: current regime)
            
        Returns:
            Regime characteristics
        """
        return self.regime_detector.get_regime_characteristics(regime)
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get transition matrix as a DataFrame.
        
        Returns:
            Transition matrix DataFrame
        """
        return self.transition_predictor.get_transition_matrix_df()
    
    def optimize_for_regime(self, regime: Optional[int] = None) -> Dict[str, Any]:
        """
        Get optimal parameters for a specific regime.
        
        Args:
            regime: Regime to optimize for (default: current regime)
            
        Returns:
            Optimal parameters
        """
        if regime is None:
            regime = self.current_regime
        
        if regime is None:
            logger.warning("No regime detected yet")
            return {}
        
        # Get regime characteristics
        characteristics = self.get_regime_characteristics(regime)
        
        if not characteristics:
            logger.warning(f"No characteristics available for regime {regime}")
            return {}
        
        # Define optimal parameters based on regime type
        regime_type = characteristics.get('type', 'Unknown')
        
        if regime_type == 'Stable Trend':
            # Optimize for trend following
            return {
                'use_trend_filter': True,
                'trend_strength_threshold': 0.5,
                'atr_multiplier': 1.5,
                'rr_ratio': 2.5,
                'time_domain_weight': 0.7,
                'freq_domain_weight': 0.3
            }
        
        elif regime_type == 'Volatile Trend':
            # Optimize for volatile trend
            return {
                'use_trend_filter': True,
                'trend_strength_threshold': 0.7,
                'atr_multiplier': 2.0,
                'rr_ratio': 2.0,
                'time_domain_weight': 0.6,
                'freq_domain_weight': 0.4
            }
        
        elif regime_type == 'Range Bound':
            # Optimize for range trading
            return {
                'use_trend_filter': False,
                'trend_strength_threshold': 0.3,
                'atr_multiplier': 1.0,
                'rr_ratio': 1.5,
                'time_domain_weight': 0.4,
                'freq_domain_weight': 0.6
            }
        
        elif regime_type == 'Choppy':
            # Optimize for choppy market (reduce trading)
            return {
                'use_trend_filter': True,
                'trend_strength_threshold': 0.8,
                'atr_multiplier': 2.5,
                'rr_ratio': 3.0,
                'time_domain_weight': 0.3,
                'freq_domain_weight': 0.7
            }
        
        else:
            # Default parameters
            return {
                'use_trend_filter': True,
                'trend_strength_threshold': 0.6,
                'atr_multiplier': 1.5,
                'rr_ratio': 2.0,
                'time_domain_weight': 0.5,
                'freq_domain_weight': 0.5
            }
    
    def save(self, path: str) -> None:
        """
        Save the advanced market regime detector.
        
        Args:
            path: Save path
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save regime detector
        self.regime_detector.save(f"{path}_detector")
        
        # Save transition predictor
        import joblib
        joblib.dump(self.transition_predictor, f"{path}_transition")
        
        logger.info(f"Saved advanced market regime detector to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the advanced market regime detector.
        
        Args:
            path: Load path
        """
        # Load regime detector
        self.regime_detector.load(f"{path}_detector")
        
        # Load transition predictor
        import joblib
        self.transition_predictor = joblib.load(f"{path}_transition")
        
        logger.info(f"Loaded advanced market regime detector from {path}")
