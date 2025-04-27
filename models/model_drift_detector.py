#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Drift Detection module for ATFNet.
Implements methods to detect when models are no longer performing well.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from scipy import stats
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('atfnet.models.drift_detector')


class ModelDriftDetector:
    """
    Model drift detection for ATFNet.
    Implements methods to detect when models are no longer performing well.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'model_drift'):
        """
        Initialize the model drift detector.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.window_size = self.config.get('window_size', 100)
        self.alert_threshold = self.config.get('alert_threshold', 2.0)  # Standard deviations
        self.critical_threshold = self.config.get('critical_threshold', 3.0)  # Standard deviations
        self.min_samples = self.config.get('min_samples', 30)
        self.metrics = self.config.get('metrics', ['mse', 'mae', 'r2'])
        
        # Initialize state variables
        self.reference_distribution = {}
        self.current_window = {}
        self.drift_scores = {}
        self.drift_history = []
        self.alerts = []
        
        # Initialize statistical tests
        self.statistical_tests = {
            'ks_test': self._ks_test,
            'ad_test': self._ad_test,
            'wasserstein': self._wasserstein_distance
        }
        
        # Default test
        self.primary_test = self.config.get('primary_test', 'ks_test')
        
        logger.info(f"Model Drift Detector initialized with {self.primary_test} test")
    
    def set_reference_distribution(self, predictions: np.ndarray, targets: np.ndarray,
                                  feature_matrix: np.ndarray = None) -> Dict[str, Any]:
        """
        Set reference distribution for drift detection.
        
        Args:
            predictions: Model predictions
            targets: Actual target values
            feature_matrix: Feature matrix (optional)
            
        Returns:
            Reference distribution statistics
        """
        try:
            # Calculate performance metrics
            metrics = self._calculate_metrics(predictions, targets)
            
            # Calculate prediction statistics
            pred_stats = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'median': np.median(predictions),
                'skew': stats.skew(predictions),
                'kurtosis': stats.kurtosis(predictions)
            }
            
            # Calculate error statistics
            errors = predictions - targets
            error_stats = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'median': np.median(errors),
                'skew': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors)
            }
            
            # Store reference distributions
            self.reference_distribution = {
                'predictions': predictions.copy(),
                'targets': targets.copy(),
                'errors': errors.copy(),
                'metrics': metrics,
                'pred_stats': pred_stats,
                'error_stats': error_stats,
                'timestamp': datetime.now()
            }
            
            # Store feature statistics if provided
            if feature_matrix is not None:
                feature_stats = {
                    'mean': np.mean(feature_matrix, axis=0),
                    'std': np.std(feature_matrix, axis=0),
                    'min': np.min(feature_matrix, axis=0),
                    'max': np.max(feature_matrix, axis=0)
                }
                self.reference_distribution['feature_stats'] = feature_stats
            
            # Initialize current window
            self.current_window = {
                'predictions': [],
                'targets': [],
                'errors': [],
                'metrics': {},
                'feature_values': [] if feature_matrix is not None else None
            }
            
            logger.info(f"Reference distribution set with {len(predictions)} samples")
            
            return self.reference_distribution
            
        except Exception as e:
            logger.error(f"Error setting reference distribution: {e}")
            raise
    
    def update_window(self, predictions: np.ndarray, targets: np.ndarray,
                     feature_values: np.ndarray = None) -> None:
        """
        Update current window with new predictions and targets.
        
        Args:
            predictions: New model predictions
            targets: New actual target values
            feature_values: New feature values (optional)
        """
        try:
            # Ensure arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            if not isinstance(targets, np.ndarray):
                targets = np.array(targets)
            
            # Add to current window
            self.current_window['predictions'].extend(predictions)
            self.current_window['targets'].extend(targets)
            self.current_window['errors'].extend(predictions - targets)
            
            # Add feature values if provided
            if feature_values is not None and self.current_window['feature_values'] is not None:
                if not isinstance(feature_values, np.ndarray):
                    feature_values = np.array(feature_values)
                
                self.current_window['feature_values'].extend(feature_values)
            
            # Trim window to size
            if len(self.current_window['predictions']) > self.window_size:
                self.current_window['predictions'] = self.current_window['predictions'][-self.window_size:]
                self.current_window['targets'] = self.current_window['targets'][-self.window_size:]
                self.current_window['errors'] = self.current_window['errors'][-self.window_size:]
                
                if self.current_window['feature_values'] is not None:
                    self.current_window['feature_values'] = self.current_window['feature_values'][-self.window_size:]
            
            # Update metrics
            if len(self.current_window['predictions']) >= self.min_samples:
                self.current_window['metrics'] = self._calculate_metrics(
                    np.array(self.current_window['predictions']),
                    np.array(self.current_window['targets'])
                )
            
        except Exception as e:
            logger.error(f"Error updating window: {e}")
            raise
    
    def detect_drift(self, predictions: np.ndarray = None, targets: np.ndarray = None,
                    feature_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Detect model drift.
        
        Args:
            predictions: New predictions (optional)
            targets: New targets (optional)
            feature_values: New feature values (optional)
            
        Returns:
            Drift detection results
        """
        # Update window if new data provided
        if predictions is not None and targets is not None:
            self.update_window(predictions, targets, feature_values)
        
        # Check if we have enough data
        if len(self.current_window['predictions']) < self.min_samples:
            logger.warning(f"Not enough samples for drift detection. Need {self.min_samples}, have {len(self.current_window['predictions'])}")
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Check if reference distribution is set
        if not self.reference_distribution:
            logger.warning("Reference distribution not set")
            return {'drift_detected': False, 'reason': 'no_reference'}
        
        try:
            # Calculate drift scores
            drift_scores = {}
            
            # Performance metric drift
            metric_drift = {}
            for metric in self.metrics:
                if metric in self.current_window['metrics'] and metric in self.reference_distribution['metrics']:
                    ref_value = self.reference_distribution['metrics'][metric]
                    current_value = self.current_window['metrics'][metric]
                    
                    if metric == 'r2':  # For R², lower is worse
                        change = ref_value - current_value
                    else:  # For error metrics, higher is worse
                        change = current_value - ref_value
                    
                    # Calculate relative change
                    if abs(ref_value) > 1e-10:
                        relative_change = change / abs(ref_value)
                    else:
                        relative_change = change
                    
                    metric_drift[metric] = {
                        'reference': ref_value,
                        'current': current_value,
                        'absolute_change': change,
                        'relative_change': relative_change
                    }
            
            drift_scores['metric_drift'] = metric_drift
            
            # Distribution drift
            distribution_drift = {}
            
            # Prediction distribution drift
            pred_test_result = self.statistical_tests[self.primary_test](
                self.reference_distribution['predictions'],
                np.array(self.current_window['predictions'])
            )
            distribution_drift['predictions'] = pred_test_result
            
            # Error distribution drift
            error_test_result = self.statistical_tests[self.primary_test](
                self.reference_distribution['errors'],
                np.array(self.current_window['errors'])
            )
            distribution_drift['errors'] = error_test_result
            
            drift_scores['distribution_drift'] = distribution_drift
            
            # Feature drift
            if (self.current_window['feature_values'] is not None and 
                'feature_stats' in self.reference_distribution):
                
                feature_drift = self._detect_feature_drift(
                    np.array(self.current_window['feature_values'])
                )
                drift_scores['feature_drift'] = feature_drift
            
            # Store drift scores
            self.drift_scores = drift_scores
            
            # Determine if drift is detected
            drift_detected = False
            drift_level = 'none'
            drift_reasons = []
            
            # Check metric drift
            for metric, values in metric_drift.items():
                if abs(values['relative_change']) > self.critical_threshold:
                    drift_detected = True
                    drift_level = 'critical'
                    drift_reasons.append(f"{metric}_drift_critical")
                elif abs(values['relative_change']) > self.alert_threshold:
                    drift_detected = True
                    if drift_level != 'critical':
                        drift_level = 'warning'
                    drift_reasons.append(f"{metric}_drift_warning")
            
            # Check distribution drift
            if distribution_drift['predictions']['p_value'] < 0.01:
                drift_detected = True
                drift_level = 'critical'
                drift_reasons.append("prediction_distribution_drift_critical")
            elif distribution_drift['predictions']['p_value'] < 0.05:
                drift_detected = True
                if drift_level != 'critical':
                    drift_level = 'warning'
                drift_reasons.append("prediction_distribution_drift_warning")
            
            if distribution_drift['errors']['p_value'] < 0.01:
                drift_detected = True
                drift_level = 'critical'
                drift_reasons.append("error_distribution_drift_critical")
            elif distribution_drift['errors']['p_value'] < 0.05:
                drift_detected = True
                if drift_level != 'critical':
                    drift_level = 'warning'
                drift_reasons.append("error_distribution_drift_warning")
            
            # Check feature drift if available
            if 'feature_drift' in drift_scores:
                feature_drift = drift_scores['feature_drift']
                if feature_drift['critical_features']:
                    drift_detected = True
                    drift_level = 'critical'
                    drift_reasons.append("feature_drift_critical")
                elif feature_drift['warning_features']:
                    drift_detected = True
                    if drift_level != 'critical':
                        drift_level = 'warning'
                    drift_reasons.append("feature_drift_warning")
            
            # Prepare result
            result = {
                'drift_detected': drift_detected,
                'drift_level': drift_level,
                'drift_reasons': drift_reasons,
                'drift_scores': drift_scores,
                'timestamp': datetime.now()
            }
            
            # Add to history
            self.drift_history.append(result)
            
            # Add alert if drift detected
            if drift_detected:
                self.alerts.append({
                    'level': drift_level,
                    'reasons': drift_reasons,
                    'timestamp': datetime.now()
                })
                
                logger.warning(f"Model drift detected: {drift_level} level due to {', '.join(drift_reasons)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            raise
    
    def _detect_feature_drift(self, current_features: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in feature distributions.
        
        Args:
            current_features: Current feature values
            
        Returns:
            Feature drift results
        """
        feature_stats = self.reference_distribution['feature_stats']
        
        # Calculate current feature statistics
        current_mean = np.mean(current_features, axis=0)
        current_std = np.std(current_features, axis=0)
        
        # Calculate z-scores for mean shift
        mean_shift = (current_mean - feature_stats['mean']) / (feature_stats['std'] + 1e-10)
        
        # Calculate relative change in standard deviation
        std_change = (current_std - feature_stats['std']) / (feature_stats['std'] + 1e-10)
        
        # Identify drifting features
        warning_features = []
        critical_features = []
        
        for i in range(len(mean_shift)):
            if abs(mean_shift[i]) > self.critical_threshold or abs(std_change[i]) > self.critical_threshold:
                critical_features.append(i)
            elif abs(mean_shift[i]) > self.alert_threshold or abs(std_change[i]) > self.alert_threshold:
                warning_features.append(i)
        
        return {
            'mean_shift': mean_shift,
            'std_change': std_change,
            'warning_features': warning_features,
            'critical_features': critical_features
        }
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            predictions: Model predictions
            targets: Actual target values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if 'mse' in self.metrics:
            metrics['mse'] = mean_squared_error(targets, predictions)
        
        if 'rmse' in self.metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        
        if 'mae' in self.metrics:
            metrics['mae'] = mean_absolute_error(targets, predictions)
        
        if 'r2' in self.metrics:
            metrics['r2'] = r2_score(targets, predictions)
        
        if 'mape' in self.metrics:
            # Avoid division by zero
            mask = targets != 0
            if np.sum(mask) > 0:
                metrics['mape'] = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
            else:
                metrics['mape'] = np.nan
        
        return metrics
    
    def _ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Test results
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'threshold': 0.05
        }
    
    def _ad_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Perform Anderson-Darling test for distribution comparison.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Test results
        """
        # Combine samples
        combined = np.concatenate([reference, current])
        n1, n2 = len(reference), len(current)
        n = n1 + n2
        
        # Rank the combined sample
        ranks = stats.rankdata(combined)
        
        # Split ranks back to original samples
        ranks1 = ranks[:n1]
        ranks2 = ranks[n1:]
        
        # Calculate A-D statistic
        a_sq = (1.0 / (n1 * n2)) * sum(((ranks1[i] - (i + 1))**2) / ((i + 1) * (n - i)) for i in range(n1))
        
        # Approximate p-value
        a_sq_star = a_sq * (1.0 + 0.75/n + 2.25/n**2)
        
        if a_sq_star <= 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14 * a_sq_star - 223.73 * a_sq_star**2)
        elif a_sq_star <= 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796 * a_sq_star - 59.938 * a_sq_star**2)
        elif a_sq_star <= 0.6:
            p_value = np.exp(0.9177 - 4.279 * a_sq_star - 1.38 * a_sq_star**2)
        else:
            p_value = np.exp(1.2937 - 5.709 * a_sq_star + 0.0186 * a_sq_star**2)
        
        return {
            'statistic': a_sq,
            'p_value': p_value,
            'threshold': 0.05
        }
    
    def _wasserstein_distance(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Calculate Wasserstein distance between distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            Distance results
        """
        # Standardize distributions for fair comparison
        scaler = StandardScaler()
        reference_std = scaler.fit_transform(reference.reshape(-1, 1)).flatten()
        current_std = scaler.transform(current.reshape(-1, 1)).flatten()
        
        # Calculate Wasserstein distance
        distance = stats.wasserstein_distance(reference_std, current_std)
        
        # Convert to p-value-like metric (heuristic)
        # Smaller distances are less significant (higher p-value)
        p_value = np.exp(-5 * distance)
        
        return {
            'statistic': distance,
            'p_value': p_value,
            'threshold': 0.05
        }
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on drift detection.
        
        Returns:
            List of recommendations
        """
        if not self.drift_scores:
            return ["No drift analysis available. Run detect_drift first."]
        
        recommendations = []
        
        # Check if drift is detected
        if not self.drift_history or not self.drift_history[-1]['drift_detected']:
            recommendations.append("No significant drift detected. Model is performing as expected.")
            return recommendations
        
        # Get latest drift result
        latest_drift = self.drift_history[-1]
        
        # Add general recommendation based on drift level
        if latest_drift['drift_level'] == 'critical':
            recommendations.append("CRITICAL: Model performance has significantly degraded. "
                                 "Consider retraining the model immediately.")
        else:
            recommendations.append("WARNING: Model performance is showing signs of degradation. "
                                 "Monitor closely and prepare for potential retraining.")
        
        # Add specific recommendations based on drift reasons
        for reason in latest_drift['drift_reasons']:
            if 'metric_drift' in reason:
                metric = reason.split('_')[0]
                recommendations.append(f"Performance metric '{metric}' has drifted. "
                                     f"Review recent predictions and consider model tuning.")
            
            if 'prediction_distribution' in reason:
                recommendations.append("The distribution of model predictions has changed. "
                                     "This may indicate a shift in the underlying data patterns.")
            
            if 'error_distribution' in reason:
                recommendations.append("The distribution of prediction errors has changed. "
                                     "The model may be systematically over or under-predicting.")
            
            if 'feature_drift' in reason:
                if 'feature_drift' in self.drift_scores:
                    feature_drift = self.drift_scores['feature_drift']
                    if feature_drift['critical_features']:
                        recommendations.append(f"Critical drift detected in {len(feature_drift['critical_features'])} features. "
                                             f"Review feature engineering and data quality.")
                    elif feature_drift['warning_features']:
                        recommendations.append(f"Warning: {len(feature_drift['warning_features'])} features show signs of drift. "
                                             f"Monitor these features closely.")
        
        # Add data collection recommendation
        recommendations.append("Collect additional training data that reflects the current data distribution.")
        
        # Add model adaptation recommendation
        recommendations.append("Consider implementing an adaptive learning approach to continuously update the model.")
        
        return recommendations
    
    def plot_drift_metrics(self, save_plot: bool = True,
                          filename: str = 'drift_metrics.png') -> None:
        """
        Plot drift metrics over time.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.drift_history:
            logger.warning("No drift history to plot")
            return
        
        try:
            # Extract data
            timestamps = [entry['timestamp'] for entry in self.drift_history]
            
            # Extract metrics
            metrics_data = {}
            for metric in self.metrics:
                metrics_data[metric] = []
            
            for entry in self.drift_history:
                for metric in self.metrics:
                    if metric in entry['drift_scores']['metric_drift']:
                        metrics_data[metric].append(entry['drift_scores']['metric_drift'][metric]['relative_change'])
                    else:
                        metrics_data[metric].append(np.nan)
            
            # Extract distribution p-values
            p_values = {
                'predictions': [],
                'errors': []
            }
            
            for entry in self.drift_history:
                if 'distribution_drift' in entry['drift_scores']:
                    p_values['predictions'].append(entry['drift_scores']['distribution_drift']['predictions']['p_value'])
                    p_values['errors'].append(entry['drift_scores']['distribution_drift']['errors']['p_value'])
                else:
                    p_values['predictions'].append(np.nan)
                    p_values['errors'].append(np.nan)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot metric drift
            for metric, values in metrics_data.items():
                ax1.plot(timestamps, values, 'o-', label=metric)
            
            # Add threshold lines
            ax1.axhline(y=self.alert_threshold, color='orange', linestyle='--', alpha=0.7, label='Alert Threshold')
            ax1.axhline(y=self.critical_threshold, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
            ax1.axhline(y=-self.alert_threshold, color='orange', linestyle='--', alpha=0.7)
            ax1.axhline(y=-self.critical_threshold, color='red', linestyle='--', alpha=0.7)
            
            ax1.set_ylabel('Relative Change')
            ax1.set_title('Performance Metric Drift')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot distribution p-values
            ax2.plot(timestamps, p_values['predictions'], 'o-', label='Predictions')
            ax2.plot(timestamps, p_values['errors'], 'o-', label='Errors')
            
            # Add threshold line
            ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Significance Threshold')
            ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
            
            ax2.set_ylabel('P-Value')
            ax2.set_xlabel('Time')
            ax2.set_title('Distribution Drift')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Drift metrics plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting drift metrics: {e}")
            raise
    
    def plot_feature_drift(self, save_plot: bool = True,
                          filename: str = 'feature_drift.png') -> None:
        """
        Plot feature drift.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.drift_history or 'feature_drift' not in self.drift_scores:
            logger.warning("No feature drift data to plot")
            return
        
        try:
            # Get feature drift data
            feature_drift = self.drift_scores['feature_drift']
            mean_shift = feature_drift['mean_shift']
            std_change = feature_drift['std_change']
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot mean shift
            x = np.arange(len(mean_shift))
            ax1.bar(x, mean_shift, alpha=0.7)
            
            # Add threshold lines
            ax1.axhline(y=self.alert_threshold, color='orange', linestyle='--', alpha=0.7, label='Alert Threshold')
            ax1.axhline(y=self.critical_threshold, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
            ax1.axhline(y=-self.alert_threshold, color='orange', linestyle='--', alpha=0.7)
            ax1.axhline(y=-self.critical_threshold, color='red', linestyle='--', alpha=0.7)
            
            ax1.set_ylabel('Z-Score')
            ax1.set_title('Feature Mean Shift')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'F{i}' for i in x], rotation=90)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot std change
            ax2.bar(x, std_change, alpha=0.7)
            
            # Add threshold lines
            ax2.axhline(y=self.alert_threshold, color='orange', linestyle='--', alpha=0.7, label='Alert Threshold')
            ax2.axhline(y=self.critical_threshold, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
            ax2.axhline(y=-self.alert_threshold, color='orange', linestyle='--', alpha=0.7)
            ax2.axhline(y=-self.critical_threshold, color='red', linestyle='--', alpha=0.7)
            
            ax2.set_ylabel('Relative Change')
            ax2.set_xlabel('Feature')
            ax2.set_title('Feature Std Change')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'F{i}' for i in x], rotation=90)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight drifting features
            for i in feature_drift['critical_features']:
                ax1.bar(i, mean_shift[i], color='red', alpha=0.7)
                ax2.bar(i, std_change[i], color='red', alpha=0.7)
            
            for i in feature_drift['warning_features']:
                if i not in feature_drift['critical_features']:
                    ax1.bar(i, mean_shift[i], color='orange', alpha=0.7)
                    ax2.bar(i, std_change[i], color='orange', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Feature drift plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature drift: {e}")
            raise
    
    def save_state(self, filepath: str = None) -> str:
        """
        Save detector state.
        
        Args:
            filepath: Path to save state (default: output_dir/drift_detector.pkl)
            
        Returns:
            Path to saved state
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'drift_detector.pkl')
        
        # Prepare state to save
        state = {
            'reference_distribution': self.reference_distribution,
            'current_window': self.current_window,
            'drift_scores': self.drift_scores,
            'drift_history': self.drift_history,
            'alerts': self.alerts,
            'config': self.config
        }
        
        # Save state
        joblib.dump(state, filepath)
        logger.info(f"Drift detector state saved to {filepath}")
        
        return filepath
    
    def load_state(self, filepath: str = None) -> None:
        """
        Load detector state.
        
        Args:
            filepath: Path to load state from (default: output_dir/drift_detector.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'drift_detector.pkl')
        
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found")
            return
        
        # Load state
        state = joblib.load(filepath)
        
        # Restore state
        self.reference_distribution = state.get('reference_distribution', {})
        self.current_window = state.get('current_window', {})
        self.drift_scores = state.get('drift_scores', {})
        self.drift_history = state.get('drift_history', [])
        self.alerts = state.get('alerts', [])
        
        # Update config with saved values
        if 'config' in state:
            for key, value in state['config'].items():
                if key not in self.config:
                    self.config[key] = value
        
        logger.info(f"Drift detector state loaded from {filepath}")
