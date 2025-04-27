#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration loader utility for the ATFNet Python implementation.
"""

import os
import yaml
import json
import logging

logger = logging.getLogger('atfnet.config_loader')


class ConfigLoader:
    """Configuration loader class for the ATFNet Python implementation."""
    
    def __init__(self, config_path):
        """
        Initialize the configuration loader.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
    
    def load_config(self):
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file {self.config_path} not found")
            return self._get_default_config()
        
        try:
            # Determine file type
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {self.config_path}")
                return self._get_default_config()
            
            # Load parameters config if it exists
            params_path = os.path.join(os.path.dirname(self.config_path), 'atfnet_parameters.yaml')
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        params_config = yaml.safe_load(f)
                    
                    # Merge parameters into config
                    self._merge_configs(config, params_config)
                    logger.info(f"Parameters loaded from {params_path}")
                except Exception as e:
                    logger.error(f"Failed to load parameters: {e}")
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _merge_configs(self, config, params_config):
        """
        Merge parameters config into main config.
        
        Args:
            config (dict): Main configuration dictionary
            params_config (dict): Parameters configuration dictionary
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
    
    def _get_default_config(self):
        """
        Get default configuration.
        
        Returns:
            dict: Default configuration dictionary
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
    
    def save_config(self, config):
        """
        Save configuration to file.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            bool: True if configuration was saved successfully, False otherwise
        """
        try:
            # Determine file type
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            else:
                logger.error(f"Unsupported configuration file format: {self.config_path}")
                return False
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
