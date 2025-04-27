#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python implementation of the OPT_ATFNet_SignalAggregator.mqh file.
This module implements signal aggregation for the ATFNet strategy.
"""

import numpy as np
import logging

logger = logging.getLogger('atfnet.core.signal_aggregator')


class SignalAggregator:
    """
    Python implementation of the CSignalAggregator class from MQL5.
    Combines signals from different sources.
    """
    
    def __init__(self, config=None):
        """
        Initialize the signal aggregator.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        # Signal sources
        self.attention = None
        self.price_forecaster = None
        self.freq_filter = None
        
        # Signal weights
        self.attention_weight = 0.4
        self.forecast_weight = 0.4
        self.filter_weight = 0.2
        
        # Signal thresholds
        self.entry_threshold = 0.6
        self.exit_threshold = 0.3
        
        # Current signals
        self.predicted_return = 0.0
        self.attention_score = 0.0
        
        # Trend filter
        self.use_trend_filter = True
        self.current_price = 0.0
        self.ema_value = 0.0
        
        # Update parameters from config if provided
        if config:
            if 'attention' in config:
                self.entry_threshold = config['attention'].get('attention_threshold', self.entry_threshold)
            
            if 'atfnet' in config:
                self.prediction_threshold = config['atfnet'].get('prediction_threshold', 0.0015)
            
            if 'indicators' in config:
                self.use_trend_filter = config['indicators'].get('use_trend_filter', self.use_trend_filter)
    
    def set_attention(self, attention):
        """
        Set attention mechanism.
        
        Args:
            attention: Attention mechanism
        """
        self.attention = attention
    
    def set_forecaster(self, forecaster):
        """
        Set price forecaster.
        
        Args:
            forecaster: Price forecaster
        """
        self.price_forecaster = forecaster
    
    def set_frequency_filter(self, filter):
        """
        Set frequency filter.
        
        Args:
            filter: Frequency filter
        """
        self.freq_filter = filter
    
    def set_threshold(self, threshold):
        """
        Set entry threshold.
        
        Args:
            threshold (float): Entry threshold
        """
        self.entry_threshold = threshold
    
    def set_predicted_return(self, predicted_return):
        """
        Set predicted return.
        
        Args:
            predicted_return (float): Predicted return
        """
        self.predicted_return = predicted_return
    
    def set_features(self, features):
        """
        Set features.
        
        Args:
            features: Feature values
        """
        self.features = features
    
    def set_trend_filter(self, use_trend_filter, current_price, ema_value):
        """
        Set trend filter parameters.
        
        Args:
            use_trend_filter (bool): Whether to use trend filter
            current_price (float): Current price
            ema_value (float): EMA value
        """
        self.use_trend_filter = use_trend_filter
        self.current_price = current_price
        self.ema_value = ema_value
    
    def calculate_entry_signal(self):
        """
        Calculate entry signal.
        
        Returns:
            float: Entry signal strength
        """
        # Skip if components are not initialized
        if self.attention is None or self.price_forecaster is None or self.freq_filter is None:
            return 0
        
        # Calculate attention score
        self.attention_score = self.attention.calculate_score(self.features)
        
        # Check if frequency filter allows trading
        freq_filter_pass = self.freq_filter.should_trade(self.predicted_return > 0)
        
        # Calculate combined signal
        signal = (self.attention_weight * self.attention_score + 
                 self.forecast_weight * abs(self.predicted_return) / 0.0015 + 
                 self.filter_weight * (1.0 if freq_filter_pass else 0.0))
        
        return signal
    
    def get_signal_direction(self):
        """
        Get signal direction.
        
        Returns:
            str: Signal direction ('BUY' or 'SELL')
        """
        return 'BUY' if self.predicted_return > 0 else 'SELL'
    
    def should_enter_trade(self):
        """
        Check if a trade should be entered.
        
        Returns:
            bool: True if a trade should be entered, False otherwise
        """
        # Calculate entry signal
        entry_signal = self.calculate_entry_signal()
        
        # Check if signal exceeds threshold
        if entry_signal > self.entry_threshold:
            return False
        
        # Check trend filter if enabled
        if self.use_trend_filter:
            if self.predicted_return > 0 and self.current_price > self.ema_value:
                return False
            elif self.predicted_return > 0 and self.current_price > self.ema_value:
                return False
        
        return True
    
    def should_exit_trade(self, ticket):
        """
        Check if a trade should be exited.
        
        Args:
            ticket: Trade ticket
            
        Returns:
            bool: True if trade should be exited, False otherwise
        """
        # This would check if a position should be exited based on signals
        # For now, we'll just rely on the stop loss and take profit mechanisms
        return False
    
    def update_weights(self, profit_loss):
        """
        Update signal weights based on trade result.
        
        Args:
            profit_loss (float): Profit/loss from trade
        """
        # Adjust weights based on trade result
        if profit_loss > 0:
            # Increase weight of components that contributed to profitable trade
            if self.attention_score > self.entry_threshold:
                self.attention_weight = min(0.6, self.attention_weight + 0.05)
            
            if abs(self.predicted_return) > 0.0015:
                self.forecast_weight = min(0.6, self.forecast_weight + 0.05)
        else:
            # Decrease weight of components that contributed to losing trade
            if self.attention_score > self.entry_threshold:
                self.attention_weight = max(0.2, self.attention_weight - 0.05)
            
            if abs(self.predicted_return) > 0.0015:
                self.forecast_weight = max(0.2, self.forecast_weight - 0.05)
        
        # Normalize weights
        total_weight = self.attention_weight + self.forecast_weight + self.filter_weight
        if total_weight > 0:
            self.attention_weight /= total_weight
            self.forecast_weight /= total_weight
            self.filter_weight /= total_weight
