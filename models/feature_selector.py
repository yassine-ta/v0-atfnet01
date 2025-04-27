#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive feature selection for ATFNet.
Implements feature importance analysis and dynamic feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger('atfnet.models.feature_selector')


class FeatureSelector:
    """
    Feature selector for time series data.
    """
    
    def __init__(
        self, 
        method: str = 'random_forest',
        n_features: Optional[int] = None,
        threshold: float = 0.01,
        cv_folds: int = 5
    ):
        """
        Initialize the feature selector.
        
        Args:
            method: Feature selection method ('random_forest', 'gradient_boosting', 'mutual_info', 'f_regression', 'lasso', 'pca')
            n_features: Number of features to select (if None, use threshold)
            threshold: Importance threshold for feature selection
            cv_folds: Number of cross-validation folds for LassoCV
        """
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.cv_folds = cv_folds
        
        # Initialize model
        self._init_model()
        
        # Feature importance
        self.feature_importances_ = None
        self.selected_features_ = None
        self.feature_names_ = None
        
        # Scaler
        self.scaler = StandardScaler()
    
    def _init_model(self) -> None:
        """Initialize feature selection model."""
        if self.method == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif self.method == 'mutual_info':
            self.model = SelectKBest(
                mutual_info_regression,
                k='all' if self.n_features is None else self.n_features
            )
        elif self.method == 'f_regression':
            self.model = SelectKBest(
                f_regression,
                k='all' if self.n_features is None else self.n_features
            )
        elif self.method == 'lasso':
            if self.n_features is None:
                self.model = LassoCV(
                    cv=self.cv_folds,
                    random_state=42
                )
            else:
                self.model = Lasso(
                    alpha=0.01,
                    random_state=42
                )
        elif self.method == 'pca':
            self.model = PCA(
                n_components=self.n_features if self.n_features is not None else 0.95
            )
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            
        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature names
        if feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            if len(feature_names) != X.shape[1]:
                raise ValueError(f"Length of feature_names ({len(feature_names)}) does not match X.shape[1] ({X.shape[1]})")
            self.feature_names_ = feature_names
        
        # Fit model
        if self.method in ['random_forest', 'gradient_boosting', 'lasso']:
            self.model.fit(X_scaled, y)
            
            if self.method in ['random_forest', 'gradient_boosting']:
                self.feature_importances_ = self.model.feature_importances_
            else:  # lasso
                self.feature_importances_ = np.abs(self.model.coef_)
            
            # Select features based on importance
            if self.n_features is not None:
                # Select top n_features
                indices = np.argsort(self.feature_importances_)[::-1][:self.n_features]
            else:
                # Select features above threshold
                indices = np.where(self.feature_importances_ > self.threshold)[0]
            
            self.selected_features_ = [self.feature_names_[i] for i in indices]
            
        elif self.method in ['mutual_info', 'f_regression']:
            self.model.fit(X_scaled, y)
            self.feature_importances_ = self.model.scores_
            
            # Select features based on importance
            if self.n_features is not None:
                # SelectKBest already selects top n_features
                indices = self.model.get_support(indices=True)
            else:
                # Select features above threshold
                indices = np.where(self.feature_importances_ > self.threshold)[0]
            
            self.selected_features_ = [self.feature_names_[i] for i in indices]
            
        elif self.method == 'pca':
            self.model.fit(X_scaled)
            # For PCA, feature_importances_ are the explained variance ratios
            self.feature_importances_ = self.model.explained_variance_ratio_
            
            # For PCA, we don't select original features but components
            self.selected_features_ = [f"PC{i+1}" for i in range(len(self.feature_importances_))]
        
        logger.info(f"Selected {len(self.selected_features_)} features using {self.method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.method in ['random_forest', 'gradient_boosting', 'lasso']:
            if self.n_features is not None:
                # Select top n_features
                indices = np.argsort(self.feature_importances_)[::-1][:self.n_features]
            else:
                # Select features above threshold
                indices = np.where(self.feature_importances_ > self.threshold)[0]
            
            return X_scaled[:, indices]
            
        elif self.method in ['mutual_info', 'f_regression']:
            return self.model.transform(X_scaled)
            
        elif self.method == 'pca':
            return self.model.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary of feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature selector not fitted yet")
        
        return {name: importance for name, importance in zip(self.feature_names_, self.feature_importances_)}
    
    def get_selected_features(self) -> List[str]:
        """
        Get selected features.
        
        Returns:
            List of selected feature names
        """
        if self.selected_features_ is None:
            raise ValueError("Feature selector not fitted yet")
        
        return self.selected_features_
    
    def save(self, path: str) -> None:
        """
        Save the feature selector.
        
        Args:
            path: Save path
        """
        joblib.dump(self, path)
        logger.info(f"Saved feature selector to {path}")
    
    @staticmethod
    def load(path: str) -> 'FeatureSelector':
        """
        Load a feature selector.
        
        Args:
            path: Load path
            
        Returns:
            Loaded feature selector
        """
        model = joblib.load(path)
        logger.info(f"Loaded feature selector from {path}")
        return model


class AdaptiveFeatureSelector:
    """
    Adaptive feature selector that selects different features for different market regimes.
    """
    
    def __init__(
        self, 
        method: str = 'random_forest',
        n_features: Optional[Dict[str, int]] = None,
        threshold: float = 0.01
    ):
        """
        Initialize the adaptive feature selector.
        
        Args:
            method: Feature selection method
            n_features: Dictionary mapping regime types to number of features
            threshold: Importance threshold for feature selection
        """
        self.method = method
        self.n_features = n_features or {
            'Stable Trend': 10,
            'Volatile Trend': 15,
            'Range Bound': 12,
            'Choppy': 20,
            'default': 15
        }
        self.threshold = threshold
        
        # Feature selectors for each regime
        self.selectors = {}
        
        # Current regime
        self.current_regime = None
        
        # Feature names
        self.feature_names = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        regimes: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'AdaptiveFeatureSelector':
        """
        Fit feature selectors for each regime.
        
        Args:
            X: Feature matrix
            y: Target values
            regimes: Regime labels
            feature_names: Feature names
            
        Returns:
            Self
        """
        # Store feature names
        self.feature_names = feature_names
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        
        # Fit a selector for each regime
        for regime in unique_regimes:
            # Skip noise points
            if regime < 0:
                continue
            
            # Get regime type
            if isinstance(regime, (int, np.integer)):
                regime_type = f"Regime_{regime}"
            else:
                regime_type = str(regime)
            
            # Get data for this regime
            mask = (regimes == regime)
            if np.sum(mask) < 10:
                logger.warning(f"Not enough data for regime {regime} (n={np.sum(mask)})")
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            
            # Get number of features for this regime
            n_feat = self.n_features.get(regime_type, self.n_features.get('default'))
            
            # Create and fit selector
            selector = FeatureSelector(
                method=self.method,
                n_features=n_feat,
                threshold=self.threshold
            )
            
            selector.fit(X_regime, y_regime, feature_names)
            
            # Store selector
            self.selectors[regime_type] = selector
            
            logger.info(f"Fitted feature selector for {regime_type} with {len(selector.get_selected_features())} features")
        
        # Create default selector
        default_n_features = self.n_features.get('default')
        default_selector = FeatureSelector(
            method=self.method,
            n_features=default_n_features,
            threshold=self.threshold
        )
        
        default_selector.fit(X, y, feature_names)
        self.selectors['default'] = default_selector
        
        return self
    
    def set_regime(self, regime: Union[int, str]) -> None:
        """
        Set current regime.
        
        Args:
            regime: Current regime
        """
        if isinstance(regime, (int, np.integer)):
            regime_type = f"Regime_{regime}"
        else:
            regime_type = str(regime)
        
        if regime_type in self.selectors:
            self.current_regime = regime_type
        else:
            self.current_regime = 'default'
            logger.warning(f"No feature selector for regime {regime}, using default")
    
    def transform(self, X: np.ndarray, regime: Optional[Union[int, str]] = None) -> np.ndarray:
        """
        Transform data using selected features for current regime.
        
        Args:
            X: Feature matrix
            regime: Regime to use (if None, use current regime)
            
        Returns:
            Transformed feature matrix
        """
        if regime is not None:
            self.set_regime(regime)
        
        if self.current_regime is None:
            self.current_regime = 'default'
            logger.warning("No regime set, using default")
        
        return self.selectors[self.current_regime].transform(X)
    
    def get_selected_features(self, regime: Optional[Union[int, str]] = None) -> List[str]:
        """
        Get selected features for a regime.
        
        Args:
            regime: Regime to get features for (if None, use current regime)
            
        Returns:
            List of selected feature names
        """
        if regime is not None:
            self.set_regime(regime)
        
        if self.current_regime is None:
            self.current_regime = 'default'
            logger.warning("No regime set, using default")
        
        return self.selectors[self.current_regime].get_selected_features()
    
    def get_feature_importances(self, regime: Optional[Union[int, str]] = None) -> Dict[str, float]:
        """
        Get feature importances for a regime.
        
        Args:
            regime: Regime to get importances for (if None, use current regime)
            
        Returns:
            Dictionary of feature importances
        """
        if regime is not None:
            self.set_regime(regime)
        
        if self.current_regime is None:
            self.current_regime = 'default'
            logger.warning("No regime set, using default")
        
        return self.selectors[self.current_regime].get_feature_importances()
    
    def save(self, path: str) -> None:
        """
        Save the adaptive feature selector.
        
        Args:
            path: Save path
        """
        joblib.dump(self, path)
        logger.info(f"Saved adaptive feature selector to {path}")
    
    @staticmethod
    def load(path: str) -> 'AdaptiveFeatureSelector':
        """
        Load an adaptive feature selector.
        
        Args:
            path: Load path
            
        Returns:
            Loaded adaptive feature selector
        """
        model = joblib.load(path)
        logger.info(f"Loaded adaptive feature selector from {path}")
        return model


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance across different market regimes.
    """
    
    def __init__(self, feature_selector: AdaptiveFeatureSelector):
        """
        Initialize the feature importance analyzer.
        
        Args:
            feature_selector: Adaptive feature selector
        """
        self.feature_selector = feature_selector
    
    def get_regime_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importances for all regimes.
        
        Returns:
            Dictionary mapping regimes to feature importances
        """
        result = {}
        
        for regime in self.feature_selector.selectors:
            result[regime] = self.feature_selector.get_feature_importances(regime)
        
        return result
    
    def get_top_features(self, n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top n features for each regime.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary mapping regimes to top features
        """
        result = {}
        
        for regime in self.feature_selector.selectors:
            importances = self.feature_selector.get_feature_importances(regime)
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            result[regime] = sorted_features[:n]
        
        return result
    
    def get_feature_importance_across_regimes(self, feature: str) -> Dict[str, float]:
        """
        Get importance of a specific feature across all regimes.
        
        Args:
            feature: Feature name
            
        Returns:
            Dictionary mapping regimes to feature importance
        """
        result = {}
        
        for regime in self.feature_selector.selectors:
            importances = self.feature_selector.get_feature_importances(regime)
            if feature in importances:
                result[regime] = importances[feature]
            else:
                result[regime] = 0.0
        
        return result
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get summary of feature importances across all regimes.
        
        Returns:
            DataFrame with feature importances
        """
        # Get all feature names
        all_features = set()
        for regime in self.feature_selector.selectors:
            importances = self.feature_selector.get_feature_importances(regime)
            all_features.update(importances.keys())
        
        # Create DataFrame
        data = []
        for feature in all_features:
            row = {'Feature': feature}
            for regime in self.feature_selector.selectors:
                importances = self.feature_selector.get_feature_importances(regime)
                row[regime] = importances.get(feature, 0.0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by average importance
        regime_cols = [col for col in df.columns if col != 'Feature']
        df['Average'] = df[regime_cols].mean(axis=1)
        df = df.sort_values('Average', ascending=False)
        
        return df
