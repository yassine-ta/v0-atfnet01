#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explainable AI module for ATFNet.
Implements methods to explain model decisions and predictions.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import joblib
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import shap

logger = logging.getLogger('atfnet.models.explainable_ai')


class ExplainableAI:
    """
    Explainable AI for ATFNet.
    Implements methods to explain model decisions and predictions.
    """
    
    def __init__(self, model: Any = None, config: Dict[str, Any] = None, 
                output_dir: str = 'explainable_ai'):
        """
        Initialize the explainable AI module.
        
        Args:
            model: Model to explain
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.model = model
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize state variables
        self.feature_importance = None
        self.shap_values = None
        self.explanation_history = []
        
        # Determine model type
        self.model_type = self._determine_model_type()
        
        logger.info(f"Explainable AI initialized for model type: {self.model_type}")
    
    def _determine_model_type(self) -> str:
        """
        Determine the type of model.
        
        Returns:
            Model type string
        """
        if self.model is None:
            return "unknown"
        
        # Check for TensorFlow/Keras model
        if hasattr(self.model, 'predict') and hasattr(self.model, 'fit'):
            try:
                import tensorflow as tf
                if isinstance(self.model, tf.keras.Model):
                    return "tensorflow"
            except ImportError:
                pass
        
        # Check for PyTorch model
        if hasattr(self.model, 'forward') and hasattr(self.model, 'parameters'):
            try:
                import torch.nn as nn
                if isinstance(self.model, nn.Module):
                    return "pytorch"
            except ImportError:
                pass
        
        # Check for scikit-learn model
        if hasattr(self.model, 'predict') and hasattr(self.model, 'fit'):
            return "sklearn"
        
        # Check for XGBoost model
        if hasattr(self.model, 'predict') and hasattr(self.model, 'get_booster'):
            return "xgboost"
        
        # Check for LightGBM model
        if hasattr(self.model, 'predict') and hasattr(self.model, 'booster_'):
            return "lightgbm"
        
        return "unknown"
    
    def set_model(self, model: Any) -> None:
        """
        Set model to explain.
        
        Args:
            model: Model to explain
        """
        self.model = model
        self.model_type = self._determine_model_type()
        logger.info(f"Model set with type: {self.model_type}")
    
    def compute_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str] = None,
                                  method: str = 'permutation',
                                  n_repeats: int = 10) -> Dict[str, float]:
        """
        Compute feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            method: Method to compute importance ('permutation', 'shap', or 'built-in')
            n_repeats: Number of repeats for permutation importance
            
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
        
        # Set default feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Compute importance based on method
        if method == 'permutation':
            importance = self._compute_permutation_importance(X, y, n_repeats)
        elif method == 'shap':
            importance = self._compute_shap_importance(X, feature_names)
        elif method == 'built-in':
            importance = self._compute_builtin_importance(feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create importance dictionary
        importance_dict = {feature: importance[i] for i, feature in enumerate(feature_names)}
        
        # Store feature importance
        self.feature_importance = importance_dict
        
        logger.info(f"Feature importance computed using {method} method")
        
        return importance_dict
    
    def _compute_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                       n_repeats: int = 10) -> np.ndarray:
        """
        Compute permutation feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_repeats: Number of repeats
            
        Returns:
            Array of feature importances
        """
        try:
            # Compute permutation importance
            result = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            # Return mean importance
            return result.importances_mean
            
        except Exception as e:
            logger.error(f"Error computing permutation importance: {e}")
            
            # Fallback to manual computation
            logger.info("Falling back to manual permutation importance calculation")
            
            # Get baseline score
            baseline_score = self._get_model_score(X, y)
            
            # Initialize importance array
            importance = np.zeros(X.shape[1])
            
            # Compute importance for each feature
            for i in range(X.shape[1]):
                # Create copy of data
                X_permuted = X.copy()
                
                # Permute feature
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Compute score
                permuted_score = self._get_model_score(X_permuted, y)
                
                # Calculate importance
                importance[i] = baseline_score - permuted_score
            
            return importance
    
    def _get_model_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Get model score.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Model score
        """
        if self.model_type in ['tensorflow', 'pytorch']:
            # For neural networks, use MSE
            y_pred = self._model_predict(X)
            return -np.mean((y_pred - y) ** 2)  # Negative MSE
        else:
            # For other models, use built-in score method
            return self.model.score(X, y)
    
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get model predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Model predictions
        """
        if self.model_type == 'tensorflow':
            return self.model.predict(X).flatten()
        elif self.model_type == 'pytorch':
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).cpu().numpy().flatten()
        else:
            return self.model.predict(X)
    
    def _compute_shap_importance(self, X: np.ndarray, 
                                feature_names: List[str]) -> np.ndarray:
        """
        Compute SHAP feature importance.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Array of feature importances
        """
        try:
            # Create explainer based on model type
            if self.model_type == 'tensorflow':
                explainer = shap.DeepExplainer(self.model, X[:100])
            elif self.model_type == 'pytorch':
                import torch
                self.model.eval()
                X_tensor = torch.FloatTensor(X[:100])
                explainer = shap.DeepExplainer(self.model, X_tensor)
            elif self.model_type in ['xgboost', 'lightgbm']:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X[:100])
            
            # Compute SHAP values
            if self.model_type == 'pytorch':
                import torch
                X_tensor = torch.FloatTensor(X)
                shap_values = explainer.shap_values(X_tensor)
            else:
                shap_values = explainer.shap_values(X)
            
            # Store SHAP values
            self.shap_values = shap_values
            self.shap_explainer = explainer
            self.shap_data = X
            self.shap_feature_names = feature_names
            
            # Compute importance
            if isinstance(shap_values, list):
                # For multi-output models
                importance = np.abs(np.array(shap_values)).mean(axis=1).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error computing SHAP importance: {e}")
            
            # Fallback to permutation importance
            logger.info("Falling back to permutation importance")
            
            # Generate dummy target for permutation importance
            y_dummy = self._model_predict(X)
            
            return self._compute_permutation_importance(X, y_dummy)
    
    def _compute_builtin_importance(self, feature_names: List[str]) -> np.ndarray:
        """
        Compute built-in feature importance.
        
        Args:
            feature_names: Names of features
            
        Returns:
            Array of feature importances
        """
        try:
            if self.model_type in ['xgboost', 'lightgbm']:
                # Get importance from booster
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'feature_importances_'):
                # Get importance from model
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Get importance from coefficients
                importance = np.abs(self.model.coef_)
                if importance.ndim > 1:
                    importance = importance.mean(axis=0)
            else:
                raise ValueError("Model does not have built-in feature importance")
            
            return importance
            
        except Exception as e:
            logger.error(f"Error computing built-in importance: {e}")
            
            # Return uniform importance
            return np.ones(len(feature_names)) / len(feature_names)
    
    def explain_prediction(self, X: np.ndarray, feature_names: List[str] = None,
                          method: str = 'shap') -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            X: Feature matrix (single sample or batch)
            feature_names: Names of features
            method: Method to explain prediction ('shap' or 'lime')
            
        Returns:
            Dictionary with explanation
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Set default feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Get prediction
        prediction = self._model_predict(X)
        
        # Explain based on method
        if method == 'shap':
            explanation = self._explain_with_shap(X, feature_names)
        elif method == 'lime':
            explanation = self._explain_with_lime(X, feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add prediction to explanation
        explanation['prediction'] = prediction
        
        # Store explanation
        self.explanation_history.append({
            'timestamp': datetime.now(),
            'method': method,
            'prediction': prediction,
            'explanation': explanation
        })
        
        logger.info(f"Prediction explained using {method} method")
        
        return explanation
    
    def _explain_with_shap(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Explain prediction with SHAP.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with SHAP explanation
        """
        try:
            # Create explainer if not already created
            if not hasattr(self, 'shap_explainer'):
                if self.model_type == 'tensorflow':
                    explainer = shap.DeepExplainer(self.model, X[:1])
                elif self.model_type == 'pytorch':
                    import torch
                    self.model.eval()
                    X_tensor = torch.FloatTensor(X[:1])
                    explainer = shap.DeepExplainer(self.model, X_tensor)
                elif self.model_type in ['xgboost', 'lightgbm']:
                    explainer = shap.TreeExplainer(self.model)
                else:
                    explainer = shap.KernelExplainer(self.model.predict, X[:1])
                
                self.shap_explainer = explainer
            
            # Compute SHAP values
            if self.model_type == 'pytorch':
                import torch
                X_tensor = torch.FloatTensor(X)
                shap_values = self.shap_explainer.shap_values(X_tensor)
            else:
                shap_values = self.shap_explainer.shap_values(X)
            
            # Process SHAP values
            if isinstance(shap_values, list):
                # For multi-output models
                shap_values = np.array(shap_values).mean(axis=0)
            
            # Create explanation
            explanation = {
                'method': 'shap',
                'shap_values': shap_values,
                'feature_names': feature_names,
                'base_value': getattr(self.shap_explainer, 'expected_value', 0)
            }
            
            # Add feature contributions
            contributions = {}
            for i, feature in enumerate(feature_names):
                contributions[feature] = shap_values[0, i]
            
            explanation['contributions'] = contributions
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining with SHAP: {e}")
            
            # Fallback to simple feature importance
            logger.info("Falling back to simple feature importance")
            
            # Compute prediction
            prediction = self._model_predict(X)[0]
            
            # Create explanation
            explanation = {
                'method': 'feature_importance',
                'feature_names': feature_names
            }
            
            # Add feature contributions
            if self.feature_importance is not None:
                contributions = {}
                for feature, importance in self.feature_importance 
                contributions = {}
                for feature, importance in self.feature_importance.items():
                    contributions[feature] = importance * prediction
                
                explanation['contributions'] = contributions
            
            return explanation
    
    def _explain_with_lime(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Explain prediction with LIME.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with LIME explanation
        """
        try:
            # Import LIME
            from lime import lime_tabular
            
            # Create explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X,
                feature_names=feature_names,
                mode='regression' if self.model_type != 'classification' else 'classification',
                random_state=42
            )
            
            # Define prediction function
            def predict_fn(x):
                return self._model_predict(x)
            
            # Explain prediction
            explanation = explainer.explain_instance(
                data_row=X[0],
                predict_fn=predict_fn,
                num_features=len(feature_names)
            )
            
            # Extract explanation
            lime_explanation = {
                'method': 'lime',
                'feature_names': feature_names,
                'intercept': explanation.intercept,
                'local_exp': explanation.local_exp
            }
            
            # Add feature contributions
            contributions = {}
            for feature_id, weight in explanation.local_exp[1]:
                feature = feature_names[feature_id]
                contributions[feature] = weight
            
            lime_explanation['contributions'] = contributions
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error explaining with LIME: {e}")
            
            # Fallback to simple feature importance
            logger.info("Falling back to simple feature importance")
            
            # Compute prediction
            prediction = self._model_predict(X)[0]
            
            # Create explanation
            explanation = {
                'method': 'feature_importance',
                'feature_names': feature_names
            }
            
            # Add feature contributions
            if self.feature_importance is not None:
                contributions = {}
                for feature, importance in self.feature_importance.items():
                    contributions[feature] = importance * prediction
                
                explanation['contributions'] = contributions
            
            return explanation
    
    def plot_feature_importance(self, top_n: int = 20, save_plot: bool = True,
                               filename: str = 'feature_importance.png') -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.feature_importance is None:
            logger.warning("No feature importance to plot. Run compute_feature_importance first.")
            return
        
        try:
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
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
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def plot_shap_summary(self, save_plot: bool = True,
                         filename: str = 'shap_summary.png') -> None:
        """
        Plot SHAP summary.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not hasattr(self, 'shap_values') or self.shap_values is None:
            logger.warning("No SHAP values to plot. Run explain_prediction with method='shap' first.")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot SHAP summary
            if isinstance(self.shap_values, list):
                # For multi-output models
                shap.summary_plot(self.shap_values[0], self.shap_data, 
                                 feature_names=self.shap_feature_names, show=False)
            else:
                shap.summary_plot(self.shap_values, self.shap_data, 
                                 feature_names=self.shap_feature_names, show=False)
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"SHAP summary plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {e}")
            raise
    
    def plot_shap_dependence(self, feature: str, interaction_feature: str = None,
                            save_plot: bool = True,
                            filename: str = 'shap_dependence.png') -> None:
        """
        Plot SHAP dependence plot.
        
        Args:
            feature: Feature to plot
            interaction_feature: Interaction feature
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not hasattr(self, 'shap_values') or self.shap_values is None:
            logger.warning("No SHAP values to plot. Run explain_prediction with method='shap' first.")
            return
        
        try:
            # Get feature index
            feature_idx = self.shap_feature_names.index(feature)
            
            # Get interaction feature index
            interaction_idx = None
            if interaction_feature is not None:
                interaction_idx = self.shap_feature_names.index(interaction_feature)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot SHAP dependence
            if isinstance(self.shap_values, list):
                # For multi-output models
                shap.dependence_plot(feature_idx, self.shap_values[0], self.shap_data, 
                                    interaction_index=interaction_idx,
                                    feature_names=self.shap_feature_names, show=False)
            else:
                shap.dependence_plot(feature_idx, self.shap_values, self.shap_data, 
                                    interaction_index=interaction_idx,
                                    feature_names=self.shap_feature_names, show=False)
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"SHAP dependence plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP dependence: {e}")
            raise
    
    def plot_explanation(self, explanation: Dict[str, Any] = None,
                        save_plot: bool = True,
                        filename: str = 'explanation.png') -> None:
        """
        Plot explanation.
        
        Args:
            explanation: Explanation to plot (default: latest explanation)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if explanation is None:
            if not self.explanation_history:
                logger.warning("No explanation to plot. Run explain_prediction first.")
                return
            explanation = self.explanation_history[-1]['explanation']
        
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Get contributions
            contributions = explanation.get('contributions', {})
            
            # Sort contributions
            sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Select top 10 contributions
            top_contributions = sorted_contributions[:10]
            
            # Extract features and values
            features = [c[0] for c in top_contributions]
            values = [c[1] for c in top_contributions]
            
            # Reverse order for better visualization
            features.reverse()
            values.reverse()
            
            # Create color map
            colors = ['red' if v < 0 else 'green' for v in values]
            
            # Plot horizontal bar chart
            plt.barh(features, values, color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Contribution')
            plt.ylabel('Feature')
            plt.title(f"Prediction: {explanation.get('prediction', 'N/A')}")
            plt.grid(True, linestyle='--', alpha=0.7, axis='x')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Explanation plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting explanation: {e}")
            raise
    
    def generate_explanation_report(self, explanation: Dict[str, Any] = None,
                                   save_report: bool = True,
                                   filename: str = 'explanation_report.txt') -> str:
        """
        Generate explanation report.
        
        Args:
            explanation: Explanation to report (default: latest explanation)
            save_report: Whether to save the report
            filename: Filename for saved report
            
        Returns:
            Report text
        """
        if explanation is None:
            if not self.explanation_history:
                logger.warning("No explanation to report. Run explain_prediction first.")
                return "No explanation available."
            explanation = self.explanation_history[-1]['explanation']
        
        try:
            # Create report
            report = []
            report.append("=" * 50)
            report.append("PREDICTION EXPLANATION REPORT")
            report.append("=" * 50)
            report.append("")
            
            # Add prediction
            report.append(f"Prediction: {explanation.get('prediction', 'N/A')}")
            report.append("")
            
            # Add method
            report.append(f"Explanation method: {explanation.get('method', 'N/A')}")
            report.append("")
            
            # Add contributions
            contributions = explanation.get('contributions', {})
            if contributions:
                report.append("Feature contributions:")
                report.append("-" * 30)
                
                # Sort contributions
                sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for feature, value in sorted_contributions:
                    direction = "positive" if value >= 0 else "negative"
                    report.append(f"{feature}: {value:.6f} ({direction})")
                
                report.append("")
            
            # Add base value for SHAP
            if explanation.get('method') == 'shap':
                report.append(f"Base value: {explanation.get('base_value', 'N/A')}")
                report.append("")
            
            # Add intercept for LIME
            if explanation.get('method') == 'lime':
                report.append(f"Intercept: {explanation.get('intercept', 'N/A')}")
                report.append("")
            
            # Add timestamp
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Join report
            report_text = "\n".join(report)
            
            # Save report
            if save_report:
                with open(os.path.join(self.output_dir, filename), 'w') as f:
                    f.write(report_text)
                logger.info(f"Explanation report saved to {self.output_dir}/{filename}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return f"Error generating report: {e}"
