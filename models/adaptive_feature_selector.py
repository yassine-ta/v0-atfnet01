#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Feature Selection module for ATFNet.
Dynamically selects the most relevant features based on market conditions.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import joblib
from datetime import datetime

logger = logging.getLogger('atfnet.models.feature_selector')


class AdaptiveFeatureSelector:
    """
    Adaptive feature selection for ATFNet.
    Dynamically selects the most relevant features based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'feature_selection'):
        """
        Initialize the adaptive feature selector.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.min_features = self.config.get('min_features', 5)
        self.max_features = self.config.get('max_features', 20)
        self.selection_threshold = self.config.get('selection_threshold', 0.01)
        self.update_frequency = self.config.get('update_frequency', 24)  # hours
        self.feature_stability = self.config.get('feature_stability', 0.7)  # 0-1 scale
        
        # Initialize state variables
        self.feature_importances = {}
        self.selected_features = []
        self.feature_history = []
        self.last_update_time = None
        self.scaler = StandardScaler()
        
        # Initialize selection methods
        self.selection_methods = {
            'mutual_info': self._mutual_info_selection,
            'f_regression': self._f_regression_selection,
            'random_forest': self._random_forest_selection,
            'gradient_boosting': self._gradient_boosting_selection,
            'lasso': self._lasso_selection,
            'correlation': self._correlation_selection
        }
        
        # Default selection method
        self.primary_method = self.config.get('primary_method', 'random_forest')
        self.secondary_methods = self.config.get('secondary_methods', 
                                               ['mutual_info', 'correlation'])
        
        # Market regime adaptation
        self.regime_feature_counts = {
            'trending': self.config.get('trending_features', int(self.max_features * 0.8)),
            'ranging': self.config.get('ranging_features', int(self.max_features * 0.6)),
            'volatile': self.config.get('volatile_features', self.max_features),
            'choppy': self.config.get('choppy_features', int(self.max_features * 0.5))
        }
        
        logger.info(f"Adaptive Feature Selector initialized with {self.primary_method} as primary method")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       feature_names: List[str] = None,
                       market_regime: str = None,
                       force_update: bool = False) -> List[str]:
        """
        Select the most relevant features based on current data.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features (uses X.columns if None)
            market_regime: Current market regime
            force_update: Force update regardless of time since last update
            
        Returns:
            List of selected feature names
        """
        # Use column names if feature_names not provided
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Check if update is needed
        current_time = datetime.now()
        
        if not force_update and self.last_update_time is not None:
            hours_since_update = (current_time - self.last_update_time).total_seconds() / 3600
            if hours_since_update < self.update_frequency and self.selected_features:
                logger.info(f"Using cached feature selection (last updated {hours_since_update:.1f} hours ago)")
                return self.selected_features
        
        # Determine number of features to select based on market regime
        if market_regime and market_regime in self.regime_feature_counts:
            k_features = self.regime_feature_counts[market_regime]
        else:
            k_features = self.max_features
        
        # Ensure k_features is within bounds
        k_features = max(self.min_features, min(k_features, X.shape[1], self.max_features))
        
        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply primary selection method
            if self.primary_method in self.selection_methods:
                primary_importances = self.selection_methods[self.primary_method](X_scaled, y, feature_names)
            else:
                logger.warning(f"Unknown primary method: {self.primary_method}, using random_forest")
                primary_importances = self._random_forest_selection(X_scaled, y, feature_names)
            
            # Apply secondary selection methods
            all_importances = [primary_importances]
            
            for method in self.secondary_methods:
                if method in self.selection_methods:
                    secondary_importances = self.selection_methods[method](X_scaled, y, feature_names)
                    all_importances.append(secondary_importances)
            
            # Combine importances (weighted average)
            combined_importances = {}
            weights = [1.0] + [0.5] * len(self.secondary_methods)  # Primary method has more weight
            
            for feature in feature_names:
                combined_importances[feature] = sum(
                    weights[i] * importances.get(feature, 0) 
                    for i, importances in enumerate(all_importances)
                ) / sum(weights)
            
            # Sort features by importance
            sorted_features = sorted(combined_importances.items(), key=lambda x: x[1], reverse=True)
            
            # Select top k features that meet threshold
            selected = []
            for feature, importance in sorted_features:
                if importance >= self.selection_threshold and len(selected) < k_features:
                    selected.append(feature)
            
            # Ensure minimum number of features
            if len(selected) < self.min_features:
                remaining = [f for f, _ in sorted_features if f not in selected]
                selected.extend(remaining[:self.min_features - len(selected)])
            
            # Apply feature stability (keep some previously selected features for stability)
            if self.selected_features and self.feature_stability > 0:
                # Determine how many previous features to keep
                keep_count = int(k_features * self.feature_stability)
                
                # Get previous features that are still important
                previous_features = [f for f in self.selected_features 
                                    if f in feature_names and f not in selected]
                
                # Sort previous features by importance
                previous_features = sorted(
                    previous_features, 
                    key=lambda f: combined_importances.get(f, 0), 
                    reverse=True
                )
                
                # Keep top previous features
                keep_features = previous_features[:keep_count]
                
                # Remove least important current features to make room
                if keep_features:
                    selected = selected[:(k_features - len(keep_features))]
                    selected.extend(keep_features)
            
            # Update state
            self.feature_importances = combined_importances
            self.selected_features = selected
            self.last_update_time = current_time
            
            # Record feature history
            self.feature_history.append({
                'timestamp': current_time,
                'selected_features': selected.copy(),
                'market_regime': market_regime,
                'feature_count': len(selected)
            })
            
            # Log selected features
            logger.info(f"Selected {len(selected)} features for regime '{market_regime}': {selected[:5]}...")
            
            return selected
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            if self.selected_features:
                logger.info("Using previously selected features")
                return self.selected_features
            else:
                logger.info(f"Using top {self.min_features} features")
                return feature_names[:self.min_features]
    
    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Normalize scores
        if np.sum(mi_scores) > 0:
            mi_scores = mi_scores / np.sum(mi_scores)
        
        # Create importance dictionary
        importances = {feature: score for feature, score in zip(feature_names, mi_scores)}
        
        return importances
    
    def _f_regression_selection(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using F-regression.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Calculate F-scores
        f_scores, _ = f_regression(X, y)
        
        # Handle NaN values
        f_scores = np.nan_to_num(f_scores)
        
        # Normalize scores
        if np.sum(f_scores) > 0:
            f_scores = f_scores / np.sum(f_scores)
        
        # Create importance dictionary
        importances = {feature: score for feature, score in zip(feature_names, f_scores)}
        
        return importances
    
    def _random_forest_selection(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using Random Forest importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create importance dictionary
        importance_dict = {feature: importance for feature, importance in zip(feature_names, importances)}
        
        return importance_dict
    
    def _gradient_boosting_selection(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using Gradient Boosting importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Train Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X, y)
        
        # Get feature importances
        importances = gb.feature_importances_
        
        # Create importance dictionary
        importance_dict = {feature: importance for feature, importance in zip(feature_names, importances)}
        
        return importance_dict
    
    def _lasso_selection(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using Lasso coefficients.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Train Lasso
        lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        
        # Get coefficients
        coefficients = np.abs(lasso.coef_)
        
        # Normalize coefficients
        if np.sum(coefficients) > 0:
            coefficients = coefficients / np.sum(coefficients)
        
        # Create importance dictionary
        importance_dict = {feature: coef for feature, coef in zip(feature_names, coefficients)}
        
        return importance_dict
    
    def _correlation_selection(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, float]:
        """
        Select features using correlation with target.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importances
        """
        # Calculate correlation for each feature
        correlations = {}
        
        for i, feature in enumerate(feature_names):
            try:
                # Calculate Pearson correlation
                corr, _ = pearsonr(X[:, i], y)
                # Use absolute correlation as importance
                correlations[feature] = abs(corr)
            except:
                correlations[feature] = 0.0
        
        # Handle NaN values
        correlations = {f: (0.0 if np.isnan(v) else v) for f, v in correlations.items()}
        
        # Normalize correlations
        total = sum(correlations.values())
        if total > 0:
            correlations = {f: v / total for f, v in correlations.items()}
        
        return correlations
    
    def evaluate_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                  test_size: float = 0.3,
                                  random_state: int = 42) -> Dict[str, Any]:
        """
        Evaluate the performance of feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Select features
        selected_features = self.select_features(X_train, y_train, feature_names, force_update=True)
        
        # Train model with all features
        rf_all = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_all.fit(X_train, y_train)
        y_pred_all = rf_all.predict(X_test)
        mse_all = mean_squared_error(y_test, y_pred_all)
        r2_all = r2_score(y_test, y_pred_all)
        
        # Train model with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        rf_selected = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_selected.fit(X_train_selected, y_train)
        y_pred_selected = rf_selected.predict(X_test_selected)
        mse_selected = mean_squared_error(y_test, y_pred_selected)
        r2_selected = r2_score(y_test, y_pred_selected)
        
        # Calculate improvement
        mse_improvement = (mse_all - mse_selected) / mse_all * 100
        r2_improvement = (r2_selected - r2_all) / max(0.001, abs(r2_all)) * 100
        
        # Prepare results
        results = {
            'selected_features': selected_features,
            'feature_count': len(selected_features),
            'mse_all': mse_all,
            'r2_all': r2_all,
            'mse_selected': mse_selected,
            'r2_selected': r2_selected,
            'mse_improvement': mse_improvement,
            'r2_improvement': r2_improvement
        }
        
        logger.info(f"Feature selection evaluation: {len(selected_features)} features, "
                   f"MSE improvement: {mse_improvement:.2f}%, R² improvement: {r2_improvement:.2f}%")
        
        return results
    
    def plot_feature_importance(self, top_n: int = 20, save_plot: bool = True,
                               filename: str = 'feature_importance.png') -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.feature_importances:
            logger.warning("No feature importances to plot")
            return
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        top_features = sorted_features[:top_n]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bar chart
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Reverse order for better visualization
        features.reverse()
        importances.reverse()
        
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {len(top_features)} Feature Importances')
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {self.output_dir}/{filename}")
        
        plt.close()
    
    def plot_feature_stability(self, save_plot: bool = True,
                              filename: str = 'feature_stability.png') -> None:
        """
        Plot feature stability over time.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.feature_history:
            logger.warning("No feature history to plot")
            return
        
        # Get all unique features
        all_features = set()
        for entry in self.feature_history:
            all_features.update(entry['selected_features'])
        
        # Count feature occurrences
        feature_counts = {feature: 0 for feature in all_features}
        
        for entry in self.feature_history:
            for feature in entry['selected_features']:
                feature_counts[feature] += 1
        
        # Calculate stability score (0-100%)
        stability_scores = {feature: count / len(self.feature_history) * 100 
                          for feature, count in feature_counts.items()}
        
        # Sort by stability
        sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 20 features
        top_features = sorted_features[:20]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bar chart
        features = [f[0] for f in top_features]
        scores = [f[1] for f in top_features]
        
        # Reverse order for better visualization
        features.reverse()
        scores.reverse()
        
        plt.barh(features, scores, color='lightgreen')
        plt.xlabel('Stability Score (%)')
        plt.ylabel('Feature')
        plt.title('Feature Selection Stability')
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Feature stability plot saved to {self.output_dir}/{filename}")
        
        plt.close()
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save feature selector state.
        
        Args:
            filepath: Path to save model (default: output_dir/feature_selector.pkl)
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'feature_selector.pkl')
        
        # Prepare state to save
        state = {
            'feature_importances': self.feature_importances,
            'selected_features': self.selected_features,
            'last_update_time': self.last_update_time,
            'scaler': self.scaler,
            'config': self.config
        }
        
        # Save state
        joblib.dump(state, filepath)
        logger.info(f"Feature selector saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load feature selector state.
        
        Args:
            filepath: Path to load model from (default: output_dir/feature_selector.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'feature_selector.pkl')
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return
        
        # Load state
        state = joblib.load(filepath)
        
        # Restore state
        self.feature_importances = state.get('feature_importances', {})
        self.selected_features = state.get('selected_features', [])
        self.last_update_time = state.get('last_update_time')
        self.scaler = state.get('scaler', StandardScaler())
        
        # Update config with saved values
        if 'config' in state:
            for key, value in state['config'].items():
                if key not in self.config:
                    self.config[key] = value
        
        logger.info(f"Feature selector loaded from {filepath}")
