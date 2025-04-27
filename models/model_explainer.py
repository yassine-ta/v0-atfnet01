#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model explainability module for ATFNet.
Provides tools to interpret and visualize model decisions using SHAP and LIME.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import tensorflow as tf
from tensorflow.keras.models import Model
import shap
from lime import lime_tabular
from sklearn.pipeline import Pipeline

logger = logging.getLogger('atfnet.models.explainer')


class ModelExplainer:
    """
    Provides explainability tools for machine learning models in ATFNet.
    Implements SHAP and LIME techniques for model interpretation.
    """
    
    def __init__(self, output_dir: str = 'explanations'):
        """
        Initialize the model explainer.
        
        Args:
            output_dir: Directory to save explanation outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
    
    def setup_shap_explainer(self, model: Union[Model, Pipeline], model_type: str = 'keras'):
        """
        Set up SHAP explainer for a model.
        
        Args:
            model: Model to explain (Keras model or sklearn Pipeline)
            model_type: Type of model ('keras', 'tree', 'linear', etc.)
        """
        try:
            if model_type == 'keras':
                # For deep learning models
                self.shap_explainer = shap.DeepExplainer(model, None)
            elif model_type == 'tree':
                # For tree-based models
                self.shap_explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                # For linear models
                self.shap_explainer = shap.LinearExplainer(model, None)
            else:
                # Default to KernelExplainer for black-box models
                self.shap_explainer = shap.KernelExplainer(model.predict, None)
                
            logger.info(f"SHAP explainer set up for {model_type} model")
        except Exception as e:
            logger.error(f"Failed to set up SHAP explainer: {e}")
            raise
    
    def setup_lime_explainer(self, training_data: np.ndarray, feature_names: List[str], 
                             categorical_features: List[int] = None):
        """
        Set up LIME explainer for tabular data.
        
        Args:
            training_data: Representative dataset for LIME
            feature_names: Names of features
            categorical_features: Indices of categorical features
        """
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                categorical_features=categorical_features,
                mode='regression'
            )
            logger.info("LIME explainer set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up LIME explainer: {e}")
            raise
    
    def explain_with_shap(self, X: np.ndarray, feature_names: List[str], 
                          sample_indices: List[int] = None, background_data: np.ndarray = None,
                          save_plot: bool = True, filename_prefix: str = 'shap_explanation'):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X: Input data to explain
            feature_names: Names of features
            sample_indices: Indices of samples to explain (defaults to all)
            background_data: Background data for SHAP (for DeepExplainer)
            save_plot: Whether to save plots to disk
            filename_prefix: Prefix for saved files
            
        Returns:
            Dict containing SHAP values and summary information
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not set up. Call setup_shap_explainer first.")
        
        try:
            # Set background data for DeepExplainer if needed
            if isinstance(self.shap_explainer, shap.DeepExplainer) and background_data is not None:
                self.shap_explainer.explainer.data = background_data
            
            # Select samples to explain
            if sample_indices is not None:
                X_explain = X[sample_indices]
            else:
                X_explain = X
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_summary.png"), 
                           bbox_inches='tight', dpi=300)
                logger.info(f"SHAP summary plot saved to {self.output_dir}/{filename_prefix}_summary.png")
            
            plt.close()
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type='bar', show=False)
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_importance.png"), 
                           bbox_inches='tight', dpi=300)
                logger.info(f"SHAP importance plot saved to {self.output_dir}/{filename_prefix}_importance.png")
            
            plt.close()
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                # For multi-output models
                importance = {
                    feature_names[i]: np.mean(np.abs(np.array(shap_values)[:, :, i])) 
                    for i in range(len(feature_names))
                }
            else:
                # For single-output models
                importance = {
                    feature_names[i]: np.mean(np.abs(shap_values[:, i])) 
                    for i in range(len(feature_names))
                }
            
            # Sort features by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'shap_values': shap_values,
                'feature_importance': sorted_importance
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            raise
    
    def explain_with_lime(self, X: np.ndarray, predict_fn: Callable, 
                          sample_index: int, num_features: int = 10,
                          save_plot: bool = True, filename_prefix: str = 'lime_explanation'):
        """
        Generate LIME explanation for a single prediction.
        
        Args:
            X: Input data
            predict_fn: Prediction function (model.predict)
            sample_index: Index of sample to explain
            num_features: Number of features to include in explanation
            save_plot: Whether to save plot to disk
            filename_prefix: Prefix for saved files
            
        Returns:
            Dict containing LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not set up. Call setup_lime_explainer first.")
        
        try:
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                X[sample_index], 
                predict_fn,
                num_features=num_features
            )
            
            # Get feature weights
            feature_weights = explanation.as_list()
            
            # Plot explanation
            if save_plot:
                fig = explanation.as_pyplot_figure(label=0)
                fig.savefig(os.path.join(self.output_dir, f"{filename_prefix}_sample_{sample_index}.png"), 
                           bbox_inches='tight', dpi=300)
                plt.close(fig)
                logger.info(f"LIME explanation saved to {self.output_dir}/{filename_prefix}_sample_{sample_index}.png")
            
            return {
                'feature_weights': feature_weights,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            raise
    
    def generate_feature_importance_report(self, importance_dict: Dict[str, float], 
                                          method: str = 'shap',
                                          save_csv: bool = True,
                                          filename: str = 'feature_importance.csv'):
        """
        Generate a feature importance report.
        
        Args:
            importance_dict: Dictionary of feature importances
            method: Method used to generate importances ('shap' or 'lime')
            save_csv: Whether to save as CSV
            filename: Filename for CSV
            
        Returns:
            pd.DataFrame: Feature importance report
        """
        try:
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Add method column
            importance_df['Method'] = method
            
            # Save to CSV
            if save_csv:
                csv_path = os.path.join(self.output_dir, filename)
                importance_df.to_csv(csv_path, index=False)
                logger.info(f"Feature importance report saved to {csv_path}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error generating feature importance report: {e}")
            raise
    
    def compare_explanations(self, shap_importance: Dict[str, float], 
                            lime_importance: Dict[str, float],
                            save_plot: bool = True,
                            filename: str = 'explanation_comparison.png'):
        """
        Compare SHAP and LIME explanations.
        
        Args:
            shap_importance: SHAP feature importance dictionary
            lime_importance: LIME feature importance dictionary
            save_plot: Whether to save plot
            filename: Filename for plot
            
        Returns:
            pd.DataFrame: Comparison DataFrame
        """
        try:
            # Create DataFrames
            shap_df = pd.DataFrame({
                'Feature': list(shap_importance.keys()),
                'Importance': list(shap_importance.values()),
                'Method': 'SHAP'
            })
            
            # Convert LIME weights to absolute values for comparison
            lime_importance_abs = {k: abs(v) for k, v in lime_importance.items()}
            
            lime_df = pd.DataFrame({
                'Feature': list(lime_importance_abs.keys()),
                'Importance': list(lime_importance_abs.values()),
                'Method': 'LIME'
            })
            
            # Combine DataFrames
            combined_df = pd.concat([shap_df, lime_df])
            
            # Normalize importances for each method
            for method in ['SHAP', 'LIME']:
                max_importance = combined_df[combined_df['Method'] == method]['Importance'].max()
                combined_df.loc[combined_df['Method'] == method, 'Importance'] /= max_importance
            
            # Get common features
            common_features = set(shap_importance.keys()).intersection(set(lime_importance.keys()))
            comparison_df = combined_df[combined_df['Feature'].isin(common_features)]
            
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            # Pivot for plotting
            pivot_df = comparison_df.pivot(index='Feature', columns='Method', values='Importance')
            pivot_df = pivot_df.sort_values('SHAP', ascending=False)
            
            # Plot
            pivot_df.plot(kind='bar', figsize=(12, 8))
            plt.title('Feature Importance Comparison: SHAP vs LIME')
            plt.ylabel('Normalized Importance')
            plt.xlabel('Feature')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Explanation comparison saved to {self.output_dir}/{filename}")
            
            plt.close()
            
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error comparing explanations: {e}")
            raise
    
    def explain_prediction_sequence(self, model: Model, X_sequence: np.ndarray, 
                                   feature_names: List[str], time_steps: List[int] = None,
                                   save_plot: bool = True, filename_prefix: str = 'sequence_explanation'):
        """
        Explain predictions for a sequence of inputs (for time series models).
        
        Args:
            model: Model to explain
            X_sequence: Sequence input data (3D: samples, time steps, features)
            feature_names: Names of features
            time_steps: Time steps to explain (defaults to all)
            save_plot: Whether to save plots
            filename_prefix: Prefix for saved files
            
        Returns:
            Dict containing explanations for each time step
        """
        if time_steps is None:
            time_steps = list(range(X_sequence.shape[1]))
        
        try:
            # Create time step models
            time_step_models = {}
            explanations = {}
            
            for t in time_steps:
                # Create a model that outputs the contribution of time step t
                time_step_input = tf.keras.Input(shape=X_sequence.shape[1:])
                
                # Create a mask that selects only the current time step
                mask = np.zeros(X_sequence.shape[1])
                mask[t] = 1
                
                # Apply mask and connect to original model
                masked_input = tf.keras.layers.Lambda(
                    lambda x: x * mask.reshape(1, -1, 1)
                )(time_step_input)
                
                output = model(masked_input)
                time_step_model = tf.keras.Model(inputs=time_step_input, outputs=output)
                time_step_models[t] = time_step_model
                
                # Set up SHAP explainer for this time step
                self.setup_shap_explainer(time_step_model, 'keras')
                
                # Generate explanation
                explanation = self.explain_with_shap(
                    X_sequence, 
                    feature_names,
                    save_plot=save_plot,
                    filename_prefix=f"{filename_prefix}_timestep_{t}"
                )
                
                explanations[t] = explanation
            
            # Create temporal importance plot
            self._plot_temporal_importance(explanations, time_steps, feature_names, 
                                          save_plot, f"{filename_prefix}_temporal")
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining prediction sequence: {e}")
            raise
    
    def _plot_temporal_importance(self, explanations: Dict[int, Dict], 
                                 time_steps: List[int], feature_names: List[str],
                                 save_plot: bool = True, filename_prefix: str = 'temporal'):
        """
        Plot feature importance over time.
        
        Args:
            explanations: Dictionary of explanations by time step
            time_steps: Time steps to include
            feature_names: Feature names
            save_plot: Whether to save plot
            filename_prefix: Prefix for saved files
        """
        try:
            # Extract importance for each feature at each time step
            temporal_importance = {}
            
            for feature in feature_names:
                temporal_importance[feature] = []
                
                for t in time_steps:
                    if feature in explanations[t]['feature_importance']:
                        temporal_importance[feature].append(
                            explanations[t]['feature_importance'][feature]
                        )
                    else:
                        temporal_importance[feature].append(0)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot top N features
            top_n = 5
            top_features = sorted(
                feature_names, 
                key=lambda f: np.mean(temporal_importance[f]), 
                reverse=True
            )[:top_n]
            
            for feature in top_features:
                plt.plot(time_steps, temporal_importance[feature], label=feature, marker='o')
            
            plt.title('Feature Importance Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('SHAP Importance')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_importance.png"), 
                           bbox_inches='tight', dpi=300)
                logger.info(f"Temporal importance plot saved to {self.output_dir}/{filename_prefix}_importance.png")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting temporal importance: {e}")
            raise
