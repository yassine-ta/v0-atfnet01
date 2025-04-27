#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration manager for ATFNet.
This module provides centralized configuration management for the ATFNet system.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union
import copy

logger = logging.getLogger('atfnet.config_manager')


class ConfigManager:
    """
    Centralized configuration manager for ATFNet.
    
    This class provides a centralized way to manage configuration settings
    for the ATFNet system, including loading, saving, and merging configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration file. If None, default config will be used.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.params_config: Dict[str, Any] = {}
        
        # Load configuration
        if config_path is not None:
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Load parameters configuration
        self.load_parameters_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
            ValueError: If the configuration file format is not supported
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file {config_path} not found")
            raise FileNotFoundError
        
        try:
            # Determine file type
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                raise ValueError(f"Unsupported configuration file format: {config_path}")
            
            logger.info(f"Configuration loaded from {config_path}")
            return self.config
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def load_parameters_config(self, params_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load parameters configuration from file.
        
        Args:
            params_path: Path to parameters configuration file. If None, default path will be used.
            
        Returns:
            Parameters configuration dictionary
        """
        try:
            # Determine parameters path
            if params_path is None:
                # Try to find parameters config in the same directory as the main config
                if self.config_path is not None:
                    config_dir = os.path.dirname(self.config_path)
                    params_path = os.path.join(config_dir, 'atfnet_parameters.yaml')
                else:
                    # Use default path
                    params_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'config',
                        'atfnet_parameters.yaml'
                    )
            
            # Check if parameters file exists
            if not os.path.exists(params_path):
                logger.warning(f"Parameters configuration file {params_path} not found")
                return {}
            
            # Load parameters configuration
            with open(params_path, 'r') as f:
                self.params_config = yaml.safe_load(f)
            
            # Merge parameters into main config
            self._merge_configs(self.config, self.params_config)
            
            logger.info(f"Parameters configuration loaded from {params_path}")
            return self.params_config
        
        except Exception as e:
            logger.error(f"Failed to load parameters configuration: {e}")
            return {}
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file. If None, the original path will be used.
            
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            # Determine save path
            save_path = config_path if config_path is not None else self.config_path
            
            if save_path is None:
                logger.error("No configuration path specified")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Determine file type
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif save_path.endswith('.json'):
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            else:
                logger.error(f"Unsupported configuration file format: {save_path}")
                return False
            
            logger.info(f"Configuration saved to {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def save_parameters_config(self, params_path: Optional[str] = None) -> bool:
        """
        Save parameters configuration to file.
        
        Args:
            params_path: Path to save parameters configuration file. If None, the original path will be used.
            
        Returns:
            True if parameters configuration was saved successfully, False otherwise
        """
        try:
            # Determine save path
            if params_path is None:
                if self.config_path is not None:
                    config_dir = os.path.dirname(self.config_path)
                    params_path = os.path.join(config_dir, 'atfnet_parameters.yaml')
                else:
                    params_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'config',
                        'atfnet_parameters.yaml'
                    )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            
            # Save parameters configuration
            with open(params_path, 'w') as f:
                yaml.dump(self.params_config, f, default_flow_style=False)
            
            logger.info(f"Parameters configuration saved to {params_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save parameters configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def get_params_config(self) -> Dict[str, Any]:
        """
        Get the current parameters configuration.
        
        Returns:
            Current parameters configuration dictionary
        """
        return copy.deepcopy(self.params_config)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary containing configuration updates
        """
        self._update_nested_dict(self.config, updates)
    
    def update_params_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the parameters configuration with new values.
        
        Args:
            updates: Dictionary containing parameters configuration updates
        """
        self._update_nested_dict(self.params_config, updates)
        
        # Update main config with new parameters
        self._merge_configs(self.config, self.params_config)
    
    def get_param(self, param_path: str, default: Any = None) -> Any:
        """
        Get a parameter value from the configuration.
        
        Args:
            param_path: Path to the parameter (e.g., 'atfnet.sequence_length')
            default: Default value to return if parameter is not found
            
        Returns:
            Parameter value or default if not found
        """
        # Split path into components
        components = param_path.split('.')
        
        # Start with the main config
        current = self.config
        
        # Traverse the path
        for component in components:
            if isinstance(current, dict) and component in current:
                current = current[component]
            else:
                return default
        
        return current
    
    def _merge_configs(self, config: Dict[str, Any], params_config: Dict[str, Any]) -> None:
        """
        Merge parameters config into main config.
        
        Args:
            config: Main configuration dictionary
            params_config: Parameters configuration dictionary
        """
        # Merge strategy parameters
        if 'strategy' in config and 'strategy' in params_config:
            config['strategy'].update(params_config['strategy'])
        
        # Merge attention parameters
        if 'attention' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['attention_threshold'] = params_config['attention']['attention_threshold']
            config['strategy']['learning_rate'] = params_config['attention']['learning_rate']
        
        # Merge risk management parameters
        if 'risk_management' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['base_risk_per_trade'] = params_config['risk_management']['base_risk_per_trade']
            config['strategy']['max_risk_per_trade'] = params_config['risk_management']['max_risk_per_trade']
            config['strategy']['enable_dynamic_risk'] = params_config['risk_management']['enable_dynamic_risk']
        
        # Merge position management parameters
        if 'position_management' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['max_positions'] = params_config['position_management']['max_positions']
        
        # Merge technical indicators parameters
        if 'indicators' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['atr_multiplier'] = params_config['indicators']['atr_multiplier']
            config['strategy']['use_trend_filter'] = params_config['indicators']['use_trend_filter']
        
        # Merge ATFNet core parameters
        if 'atfnet' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['sequence_length'] = params_config['atfnet']['sequence_length']
            config['strategy']['prediction_length'] = params_config['atfnet']['prediction_length']
            config['strategy']['prediction_threshold'] = params_config['atfnet']['prediction_threshold']
        
        # Merge regime detection parameters
        if 'regime_detection' in params_config:
            if 'strategy' not in config:
                config['strategy'] = {}
            config['strategy']['enable_regime_detection'] = params_config['regime_detection']['enable_regime_detection']
            config['strategy']['trend_strength_threshold'] = params_config['regime_detection']['trend_strength_threshold']
            config['strategy']['volatility_threshold'] = params_config['regime_detection']['volatility_threshold']
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        logger.warning("Using default configuration")
        
        return {
            'mt5': {
                'terminal_path': '',
                'login': 0,
                'password': '',
                'server': '',
                'timeout': 60000,
                'timezone': 'UTC',
                'deviation': 10,
                'magic': 123456
            },
            'data': {
                'data_dir': 'data',
                'cache_dir': 'data/cache',
                'feature_dir': 'data/features'
            },
            'models': {
                'models_dir': 'models',
                'lstm': {
                    'units': [64, 32],
                    'dropout': 0.2,
                    'learning_rate': 0.001
                },
                'gru': {
                    'units': [64, 32],
                    'dropout': 0.2,
                    'learning_rate': 0.001
                },
                'hybrid': {
                    'lstm_units': [64],
                    'gru_units': [64],
                    'dense_units': [32],
                    'dropout': 0.2,
                    'learning_rate': 0.001
                }
            },
            'strategy': {
                'sequence_length': 60,
                'prediction_threshold': 0.0015,
                'attention_threshold': 0.6,
                'learning_rate': 0.01,
                'max_positions': 3,
                'risk_per_trade': 0.01,
                'atr_multiplier': 1.0,
                'use_trailing_stop': True,
                'trailing_stop_multiplier': 2.0,
                'check_interval': 60,
                'initial_balance': 10000,
                'target_column': 'returns',
                'target_shift': -1,
                'train_regime_classifier': True,
                'model_type': 'lstm',
                'epochs': 100,
                'batch_size': 32
            }
        }
