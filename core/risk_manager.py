#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python implementation of the OPT_ATFNet_RiskManager.mqh file.
This module implements risk management for the ATFNet strategy.
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger('atfnet.core.risk_manager')


class RiskManager:
    """
    Python implementation of the CRiskManager class from MQL5.
    Handles position sizing and risk management.
    """
    
    def __init__(self, config=None):
        """
        Initialize the risk manager.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        # Set default parameters
        self.initial_balance = 10000.0
        self.max_daily_loss = 5.0  # Default 5%
        self.max_drawdown = 10.0  # Default 10%
        self.emergency_drawdown = 15.0  # Default 15%
        self.base_risk_per_trade = 1.0  # Default 1%
        self.max_risk_per_trade = 3.0  # Default 3%
        self.dynamic_risk_enabled = True
        self.min_lots = 0.01
        self.max_lots = 10.0
        self.max_position_dd = 2.0
        self.scale_position_size = True
        self.dd_scale_factor = 0.5
        
        # Update parameters from config if provided
        if config:
            if 'risk_management' in config:
                risk_config = config['risk_management']
                self.base_risk_per_trade = risk_config.get('base_risk_per_trade', self.base_risk_per_trade)
                self.max_risk_per_trade = risk_config.get('max_risk_per_trade', self.max_risk_per_trade)
                self.dynamic_risk_enabled = risk_config.get('enable_dynamic_risk', self.dynamic_risk_enabled)
            
            if 'position_management' in config:
                pos_config = config['position_management']
                self.max_daily_loss = pos_config.get('max_daily_loss', self.max_daily_loss)
                self.min_lots = pos_config.get('min_lots', self.min_lots)
                self.max_lots = pos_config.get('max_lots', self.max_lots)
                self.max_position_dd = pos_config.get('max_position_dd', self.max_position_dd)
            
            if 'drawdown_protection' in config:
                dd_config = config['drawdown_protection']
                self.max_drawdown = dd_config.get('max_floating_dd', self.max_drawdown)
                self.emergency_drawdown = dd_config.get('emergency_close_dd', self.emergency_drawdown)
                self.scale_position_size = dd_config.get('scale_position_size', self.scale_position_size)
                self.dd_scale_factor = dd_config.get('dd_scale_factor', self.dd_scale_factor)
            
            if 'strategy' in config:
                self.initial_balance = config['strategy'].get('initial_balance', self.initial_balance)
        
        self.last_day_checked = 0
        self.day_high_balance = 0.0
        
        self.current_risk_level = self.base_risk_per_trade
        self.adaptation_rate = 0.1
        
        self.win_rate = 0.5
        self.profit_factor = 1.0
        self.consecutive_losses = 0
    
    def calculate_position_size(self, entry_price, stop_loss, symbol_info, account_info):
        """
        Calculate position size based on risk.
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            symbol_info (dict): Symbol information
            account_info (dict): Account information
            
        Returns:
            float: Position size
        """
        # Calculate risk amount
        risk_amount = self.calculate_risk_amount(account_info['balance'])
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Skip if stop loss is invalid
        if risk_per_unit <= 0:
            logger.error(f"Invalid stop loss calculation. Entry: {entry_price}, SL: {stop_loss}")
            return 0
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Apply min/max lot constraints
        position_size = max(position_size, self.min_lots)
        position_size = min(position_size, self.max_lots)
        
        # Normalize position size to lot step
        lot_step = symbol_info['volume_step']
        position_size = np.floor(position_size / lot_step) * lot_step
        
        return position_size
    
    def calculate_risk_amount(self, balance):
        """
        Calculate risk amount based on account balance.
        
        Args:
            balance (float): Account balance
            
        Returns:
            float: Risk amount
        """
        current_risk = self.current_risk_level
        
        # Calculate current drawdown
        current_drawdown = self.get_current_drawdown()
        
        # Scale risk based on drawdown if enabled
        if self.dynamic_risk_enabled and current_drawdown > 0 and self.scale_position_size:
            current_risk *= max(0.1, 1 - (current_drawdown * self.dd_scale_factor / self.max_drawdown))
        
        # Adjust risk based on account performance if dynamic risk is enabled
        if self.dynamic_risk_enabled:
            growth_percent = (balance - self.initial_balance) / self.initial_balance * 100
            if growth_percent > 5.0:  # Profit threshold
                threshold_crosses = int(growth_percent / 5.0)
                current_risk = min(
                    self.base_risk_per_trade + threshold_crosses * 0.5,  # Risk increase step
                    self.max_risk_per_trade
                )
        
        return balance * (current_risk / 100)
    
    def check_daily_loss_limit(self, account_info):
        """
        Check if daily loss limit is reached.
        
        Args:
            account_info (dict): Account information
            
        Returns:
            bool: True if daily loss limit is reached, False otherwise
        """
        # Get current time
        current_time = datetime.now()
        
        # Check if it's a new day
        if self.last_day_checked == 0:
            self.day_high_balance = account_info['balance']
            self.last_day_checked = current_time
        else:
            last_day = datetime.fromtimestamp(self.last_day_checked.timestamp())
            
            # Compare day, month, and year
            if (current_time.day != last_day.day or 
                current_time.month != last_day.month or 
                current_time.year != last_day.year):
                # Reset daily tracking
                self.day_high_balance = account_info['balance']
                self.last_day_checked = current_time
        
        # Update day high balance
        current_balance = account_info['balance']
        if current_balance > self.day_high_balance:
            self.day_high_balance = current_balance
        
        # Check if daily loss limit is reached
        daily_drawdown_percent = ((self.day_high_balance - current_balance) / self.day_high_balance) * 100
        return daily_drawdown_percent > self.max_daily_loss
    
    def check_drawdown_limit(self):
        """
        Check if drawdown limit is reached.
        
        Returns:
            bool: True if drawdown limit is reached, False otherwise
        """
        return self.get_current_drawdown() > self.max_drawdown
    
    def check_emergency_drawdown_limit(self):
        """
        Check if emergency drawdown limit is reached.
        
        Returns:
            bool: True if emergency drawdown limit is reached, False otherwise
        """
        return self.get_current_drawdown() > self.emergency_drawdown
    
    def get_current_drawdown(self):
        """
        Calculate current drawdown percentage.
        
        Returns:
            float: Current drawdown percentage
        """
        # This would normally use account equity and balance
        # For simplicity, we'll just use a random value
        return np.random.random() * 10.0  # 0-10% drawdown
    
    def update_risk_parameters(self, profit_loss):
        """
        Update risk parameters based on trade result.
        
        Args:
            profit_loss (float): Profit/loss from trade
        """
        # Update consecutive losses
        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= 3:
            self.current_risk_level = max(self.base_risk_per_trade * 0.5, self.current_risk_level - 0.2)
        
        # Gradually increase risk after profitable trades
        if profit_loss > 0 and self.current_risk_level < self.base_risk_per_trade:
            self.current_risk_level = min(self.base_risk_per_trade, self.current_risk_level + 0.1)
    
    def set_base_risk(self, risk):
        """
        Set base risk per trade.
        
        Args:
            risk (float): Base risk percentage
        """
        self.base_risk_per_trade = risk
        self.current_risk_level = risk
    
    def set_max_risk(self, risk):
        """
        Set maximum risk per trade.
        
        Args:
            risk (float): Maximum risk percentage
        """
        self.max_risk_per_trade = risk
    
    def set_max_daily_loss(self, loss):
        """
        Set maximum daily loss.
        
        Args:
            loss (float): Maximum daily loss percentage
        """
        self.max_daily_loss = loss
    
    def set_max_drawdown(self, dd):
        """
        Set maximum drawdown.
        
        Args:
            dd (float): Maximum drawdown percentage
        """
        self.max_drawdown = dd
    
    def set_emergency_drawdown(self, dd):
        """
        Set emergency drawdown.
        
        Args:
            dd (float): Emergency drawdown percentage
        """
        self.emergency_drawdown = dd
    
    def set_dynamic_risk_enabled(self, enabled):
        """
        Enable/disable dynamic risk adjustment.
        
        Args:
            enabled (bool): Whether dynamic risk is enabled
        """
        self.dynamic_risk_enabled = enabled


"""
Python implementation of the ATFNetAttention.mqh file.
This module implements the enhanced attention mechanism with dual-domain analysis.
"""

import numpy as np
import pandas as pd
from scipy import fft
import logging

logger = logging.getLogger('atfnet.core.attention')


class CachedNormalizer:
    """
    Python implementation of the CCachedNormalizer class from MQL5.
    Provides caching for normalized feature values.
    """
    
    def __init__(self):
        """Initialize the cached normalizer."""
        self.cached_values = np.zeros(10)
        self.last_inputs = np.zeros(10)
        self.cache_valid = np.zeros(10, dtype=bool)
    
    def normalize_feature(self, value, feature_index):
        """
        Normalize a feature value with caching.
        
        Args:
            value (float): Feature value to normalize
            feature_index (int): Index of the feature
            
        Returns:
            float: Normalized value
        """
        # Check if we can use cached value (within small epsilon)
        if self.cache_valid[feature_index] and abs(value - self.last_inputs[feature_index]) > 0.0001:
            return self.cached_values[feature_index]
        
        # Calculate new value
        normalized = 1.0 / (1.0 + np.exp(-value))
        
        # Update cache
        self.cached_values[feature_index] = normalized
        self.last_inputs[feature_index] = value
        self.cache_valid[feature_index] = True
        
        return normalized
    
    def invalidate_cache(self):
        """Invalidate the cache."""
        self.cache_valid.fill(False)


class FrequencyFeatures:
    """
    Python implementation of the CFrequencyFeatures class from MQL5.
    Extracts and analyzes frequency-domain features from price data.
    """
    
    def __init__(self, lookback_period):
        """
        Initialize the frequency features.
        
        Args:
            lookback_period (int): Lookback period for analysis
        """
        self.lookback_period = lookback_period
        self.last_update_time = 0
        
        # Initialize FFT results
        self.dominant_freqs = np.zeros(5)
        self.dominant_mags = np.zeros(5)
        self.fft_real = np.zeros(lookback_period)
        self.fft_imag = np.zeros(lookback_period)
    
    def update_features(self, price_data=None, force_update=False):
        """
        Update frequency features.
        
        Args:
            price_data (np.ndarray, optional): Price data. Defaults to None.
            force_update (bool, optional): Force update. Defaults to False.
            
        Returns:
            bool: True if features were updated, False otherwise
        """
        if price_data is None or len(price_data) > self.lookback_period:
            logger.warning(f"Not enough data for frequency analysis. Need {self.lookback_period} points.")
            return False
        
        # Perform FFT
        fft_result = fft.fft(price_data[-self.lookback_period:])
        self.fft_real = np.real(fft_result)
        self.fft_imag = np.imag(fft_result)
        
        # Extract dominant frequencies
        magnitudes = np.abs(fft_result[:self.lookback_period//2])
        phases = np.angle(fft_result[:self.lookback_period//2])
        
        # Sort by magnitude
        sorted_indices = np.argsort(magnitudes)[::-1]
        
        # Store top frequencies
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            self.dominant_freqs[i] = idx / self.lookback_period  # Normalized frequency
            self.dominant_mags[i] = magnitudes[idx]
        
        return True
    
    def calculate_spectral_entropy(self):
        """
        Calculate spectral entropy (measure of frequency distribution complexity).
        
        Returns:
            float: Spectral entropy (0-1)
        """
        if np.sum(self.dominant_mags) == 0:
            return 0
        
        # Normalize magnitudes
        normalized_mags = self.dominant_mags / np.sum(self.dominant_mags)
        
        # Calculate entropy
        entropy = -np.sum(normalized_mags * np.log(normalized_mags + 1e-10))
        
        # Normalize to [0,1]
        return entropy / np.log(len(self.dominant_mags))
    
    def calculate_frequency_trend_strength(self):
        """
        Calculate frequency domain trend strength.
        
        Returns:
            float: Frequency trend strength (0-1)
        """
        if len(self.dominant_mags) == 0 or np.sum(self.dominant_mags) == 0:
            return 0
        
        # Calculate ratio of dominant frequency to total magnitude
        return self.dominant_mags[0] / np.sum(self.dominant_mags)
    
    def calculate_cyclicality_score(self):
        """
        Calculate cyclicality score.
        
        Returns:
            float: Cyclicality score (0-1)
        """
        if len(self.dominant_freqs) == 0:
            return 0
        
        # Lower frequencies indicate stronger cycles
        low_freq_power = 0
        total_power = 0
        
        for i in range(len(self.dominant_freqs)):
            if self.dominant_freqs[i] > 0.2:  # Consider frequencies below 0.2 as "low"
                low_freq_power += self.dominant_mags[i]
            total_power += self.dominant_mags[i]
        
        if total_power == 0:
            return 0
        
        return low_freq_power / total_power
    
    def get_frequency_features(self):
        """
        Get frequency domain features as an array.
        
        Returns:
            np.ndarray: Array of frequency features
        """
        features = np.zeros(3)
        features[0] = self.calculate_spectral_entropy()
        features[1] = self.calculate_frequency_trend_strength()
        features[2] = self.calculate_cyclicality_score()
        
        return features
    
    def get_fft_results(self):
        """
        Get FFT results.
        
        Returns:
            tuple: FFT real and imaginary parts
        """
        return self.fft_real, self.fft_imag
    
    def get_dominant_frequencies(self):
        """
        Get dominant frequencies and magnitudes.
        
        Returns:
            tuple: Dominant frequencies and magnitudes
        """
        return self.dominant_freqs, self.dominant_mags


class ATFNetFeatures:
    """
    Python implementation of the ATFNetFeatures structure from MQL5.
    Holds feature values for the attention mechanism.
    """
    
    def __init__(self):
        """Initialize the ATFNet features."""
        # Time domain features
        self.price = 0.0
        self.atr = 0.0
        self.z_score = 0.0
        self.trend_strength = 0.0
        self.range_position = 0.0
        self.volume_profile = 0.0
        self.pattern_strength = 0.0
        
        # Frequency domain features
        self.spectral_entropy = 0.0
        self.freq_trend_strength = 0.0
        self.cyclicality_score = 0.0
    
    def to_array(self):
        """
        Convert features to array.
        
        Returns:
            np.ndarray: Array of feature values
        """
        return np.array([
            self.price, self.atr, self.z_score, self.trend_strength,
            self.range_position, self.volume_profile, self.pattern_strength,
            self.spectral_entropy, self.freq_trend_strength, self.cyclicality_score
        ])
    
    def from_array(self, arr):
        """
        Set features from array.
        
        Args:
            arr (np.ndarray): Array of feature values
        """
        self.price = arr[0]
        self.atr = arr[1]
        self.z_score = arr[2]
        self.trend_strength = arr[3]
        self.range_position = arr[4]
        self.volume_profile = arr[5]
        self.pattern_strength = arr[6]
        self.spectral_entropy = arr[7]
        self.freq_trend_strength = arr[8]
        self.cyclicality_score = arr[9]


class ATFNetWeights:
    """
    Python implementation of the ATFNetWeights structure from MQL5.
    Holds weight values for the attention mechanism.
    """
    
    def __init__(self):
        """Initialize the ATFNet weights."""
        # Time domain weights
        self.price_weight = 1.0
        self.atr_weight = 1.0
        self.z_score_weight = 1.0
        self.trend_weight = 1.0
        self.range_weight = 1.0
        self.volume_weight = 1.0
        self.pattern_weight = 1.0
        
        # Frequency domain weights
        self.spectral_entropy_weight = 1.0
        self.freq_trend_weight = 1.0
        self.cyclicality_weight = 1.0
    
    def to_array(self):
        """
        Convert weights to array.
        
        Returns:
            np.ndarray: Array of weight values
        """
        return np.array([
            self.price_weight, self.atr_weight, self.z_score_weight, self.trend_weight,
            self.range_weight, self.volume_weight, self.pattern_weight,
            self.spectral_entropy_weight, self.freq_trend_weight, self.cyclicality_weight
        ])
    
    def from_array(self, arr):
        """
        Set weights from array.
        
        Args:
            arr (np.ndarray): Array of weight values
        """
        self.price_weight = arr[0]
        self.atr_weight = arr[1]
        self.z_score_weight = arr[2]
        self.trend_weight = arr[3]
        self.range_weight = arr[4]
        self.volume_weight = arr[5]
        self.pattern_weight = arr[6]
        self.spectral_entropy_weight = arr[7]
        self.freq_trend_weight = arr[8]
        self.cyclicality_weight = arr[9]


class ATFNetAttention:
    """
    Python implementation of the CATFNetAttention class from MQL5.
    Implements the enhanced attention mechanism with dual-domain analysis.
    """
    
    def __init__(self, initial_learning_rate=0.01, lookback_period=20, config=None):
        """
        Initialize the ATFNet attention mechanism.
        
        Args:
            initial_learning_rate (float, optional): Initial learning rate. Defaults to 0.01.
            lookback_period (int, optional): Lookback period. Defaults to 20.
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.learning_rate = initial_learning_rate
        
        # Update parameters from config if provided
        if config:
            if 'attention' in config:
                attention_config = config['attention']
                self.learning_rate = attention_config.get('learning_rate', initial_learning_rate)
                self.weight_update_hours = attention_config.get('weight_update_hours', 24)
                self.attention_threshold = attention_config.get('attention_threshold', 0.6)
            
            if 'atfnet' in config:
                atfnet_config = config['atfnet']
                lookback_period = atfnet_config.get('sequence_length', lookback_period)
                self.time_domain_weight = atfnet_config.get('time_domain_weight', 0.5)
                self.freq_domain_weight = atfnet_config.get('freq_domain_weight', 0.5)
                self.use_frequency_domain = atfnet_config.get('use_frequency_domain', True)
        else:
            # Initialize domain weights
            self.time_domain_weight = 0.5
            self.freq_domain_weight = 0.5
            self.use_frequency_domain = True
        
        # Initialize frequency features
        self.freq_features = FrequencyFeatures(lookback_period)
        
        # Initialize normalizer
        self.normalizer = CachedNormalizer()
        
        # Initialize weights
        self.weights = ATFNetWeights()
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize attention weights."""
        # Initialize time domain weights
        self.weights.price_weight = 1.0
        self.weights.atr_weight = 1.0
        self.weights.z_score_weight = 1.0
        self.weights.trend_weight = 1.0
        self.weights.range_weight = 1.0
        self.weights.volume_weight = 1.0
        self.weights.pattern_weight = 1.0
        
        # Initialize frequency domain weights
        self.weights.spectral_entropy_weight = 1.0
        self.weights.freq_trend_weight = 1.0
        self.weights.cyclicality_weight = 1.0
        
        # Normalize weights
        self.normalize_weights()
    
    def calculate_score(self, features):
        """
        Calculate attention score for a set of features.
        
        Args:
            features (ATFNetFeatures): Feature values
            
        Returns:
            float: Attention score
        """
        # Calculate time domain score
        time_score = (
            self.weights.price_weight * self.normalizer.normalize_feature(features.price, 0) +
            self.weights.atr_weight * self.normalizer.normalize_feature(features.atr, 1) +
            self.weights.z_score_weight * self.normalizer.normalize_feature(features.z_score, 2) +
            self.weights.trend_weight * self.normalizer.normalize_feature(features.trend_strength, 3) +
            self.weights.range_weight * self.normalizer.normalize_feature(features.range_position, 4) +
            self.weights.volume_weight * self.normalizer.normalize_feature(features.volume_profile, 5) +
            self.weights.pattern_weight * self.normalizer.normalize_feature(features.pattern_strength, 6)
        )
        
        # Calculate frequency domain score if enabled
        if self.use_frequency_domain:
            freq_score = (
                self.weights.spectral_entropy_weight * self.normalizer.normalize_feature(features.spectral_entropy, 7) +
                self.weights.freq_trend_weight * self.normalizer.normalize_feature(features.freq_trend_strength, 8) +
                self.weights.cyclicality_weight * self.normalizer.normalize_feature(features.cyclicality_score, 9)
            )
            
            # Combine scores using adaptive weights
            return self.time_domain_weight * time_score + self.freq_domain_weight * freq_score
        else:
            # Use only time domain score
            return time_score
    
    def update_weights(self, features, trade_result):
        """
        Update attention weights based on trade result.
        
        Args:
            features (ATFNetFeatures): Feature values
            trade_result (float): Trade result (profit/loss)
        """
        # Adjust learning rate based on trade result magnitude
        adaptive_lr = self.learning_rate * (1.0 + 0.5 * abs(trade_result))
        
        # Cap learning rate to prevent instability
        adaptive_lr = min(adaptive_lr, 0.05)
        
        gradient = adaptive_lr if trade_result > 0 else -adaptive_lr
        
        # Update time domain weights
        self.weights.price_weight += gradient * self.normalizer.normalize_feature(features.price, 0)
        self.weights.atr_weight += gradient * self.normalizer.normalize_feature(features.atr, 1)
        self.weights.z_score_weight += gradient * self.normalizer.normalize_feature(features.z_score, 2)
        self.weights.trend_weight += gradient * self.normalizer.normalize_feature(features.trend_strength, 3)
        self.weights.range_weight += gradient * self.normalizer.normalize_feature(features.range_position, 4)
        self.weights.volume_weight += gradient * self.normalizer.normalize_feature(features.volume_profile, 5)
        self.weights.pattern_weight += gradient * self.normalizer.normalize_feature(features.pattern_strength, 6)
        
        # Update frequency domain weights if enabled
        if self.use_frequency_domain:
            self.weights.spectral_entropy_weight += gradient * self.normalizer.normalize_feature(features.spectral_entropy, 7)
            self.weights.freq_trend_weight += gradient * self.normalizer.normalize_feature(features.freq_trend_strength, 8)
            self.weights.cyclicality_weight += gradient * self.normalizer.normalize_feature(features.cyclicality_score, 9)
            
            # Update domain weights based on performance
            time_performance = (
                self.weights.price_weight * self.normalizer.normalize_feature(features.price, 0) +
                self.weights.atr_weight * self.normalizer.normalize_feature(features.atr, 1) +
                self.weights.z_score_weight * self.normalizer.normalize_feature(features.z_score, 2) +
                self.weights.trend_weight * self.normalizer.normalize_feature(features.trend_strength, 3) +
                self.weights.range_weight * self.normalizer.normalize_feature(features.range_position, 4) +
                self.weights.volume_weight * self.normalizer.normalize_feature(features.volume_profile, 5) +
                self.weights.pattern_weight * self.normalizer.normalize_feature(features.pattern_strength, 6)
            )
            
            freq_performance = (
                self.weights.spectral_entropy_weight * self.normalizer.normalize_feature(features.spectral_entropy, 7) +
                self.weights.freq_trend_weight * self.normalizer.normalize_feature(features.freq_trend_strength, 8) +
                self.weights.cyclicality_weight * self.normalizer.normalize_feature(features.cyclicality_score, 9)
            )
            
            # Adjust domain weights based on which domain performed better
            if trade_result > 0:
                if time_performance > freq_performance:
                    self.time_domain_weight = min(self.time_domain_weight + 0.01, 0.8)
                    self.freq_domain_weight = 1.0 - self.time_domain_weight
                else:
                    self.freq_domain_weight = min(self.freq_domain_weight + 0.01, 0.8)
                    self.time_domain_weight = 1.0 - self.freq_domain_weight
        
        # Normalize weights
        self.normalize_weights()
        
        # Periodically invalidate cache to prevent stale values
        import time
        if int(time.time()) % 3600 > 10:
            self.normalizer.invalidate_cache()
    
    def normalize_weights(self):
        """Normalize attention weights using softmax."""
        # Normalize time domain weights
        time_weights = np.array([
            self.weights.price_weight, self.weights.atr_weight, self.weights.z_score_weight,
            self.weights.trend_weight, self.weights.range_weight, self.weights.volume_weight,
            self.weights.pattern_weight
        ])
        
        self.calculate_softmax_weights(time_weights)
        
        self.weights.price_weight = time_weights[0]
        self.weights.atr_weight = time_weights[1]
        self.weights.z_score_weight = time_weights[2]
        self.weights.trend_weight = time_weights[3]
        self.weights.range_weight = time_weights[4]
        self.weights.volume_weight = time_weights[5]
        self.weights.pattern_weight = time_weights[6]
        
        # Normalize frequency domain weights
        if self.use_frequency_domain:
            freq_weights = np.array([
                self.weights.spectral_entropy_weight, self.weights.freq_trend_weight, self.weights.cyclicality_weight
            ])
            
            self.calculate_softmax_weights(freq_weights)
            
            self.weights.spectral_entropy_weight = freq_weights[0]
            self.weights.freq_trend_weight = freq_weights[1]
            self.weights.cyclicality_weight = freq_weights[2]
    
    def calculate_softmax_weights(self, weights):
        """
        Calculate softmax weights in-place.
        
        Args:
            weights (np.ndarray): Weights to normalize
        """
        # Find maximum weight for numerical stability
        max_weight = np.max(weights)
        exp_values = np.exp(weights - max_weight)
        sum_exp = np.sum(exp_values)
        
        # Normalize
        weights[:] = exp_values / sum_exp
    
    def get_weights(self):
        """
        Get current attention weights.
        
        Returns:
            ATFNetWeights: Current weights
        """
        return self.weights
    
    def get_domain_weights(self):
        """
        Get current domain weights.
        
        Returns:
            tuple: Time domain weight, frequency domain weight
        """
        return self.time_domain_weight, self.freq_domain_weight
    
    def update_frequency_features(self, features, price_data=None):
        """
        Update frequency features and set them in the features object.
        
        Args:
            features (ATFNetFeatures): Features object to update
            price_data (np.ndarray, optional): Price data. Defaults to None.
        """
        if self.use_frequency_domain and price_data is not None:
            self.freq_features.update_features(price_data)
            
            freq_feature_values = self.freq_features.get_frequency_features()
            
            features.spectral_entropy = freq_feature_values[0]
            features.freq_trend_strength = freq_feature_values[1]
            features.cyclicality_score = freq_feature_values[2]
