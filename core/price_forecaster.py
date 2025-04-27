#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python implementation of the PriceForecaster.mqh file.
This module implements price forecasting for the ATFNet strategy.
"""

import numpy as np
from scipy import fft
import logging

from core.atfnet_attention import FrequencyFeatures

logger = logging.getLogger('atfnet.core.forecaster')


class AdaptiveForecastCombiner:
    """
    Python implementation of the CAdaptiveForecastCombiner class from MQL5.
    Combines forecasts from different methods with adaptive weights.
    """
    
    def __init__(self, initial_time_weight=0.5, adaptation_rate=0.05):
        """
        Initialize the adaptive forecast combiner.
        
        Args:
            initial_time_weight (float, optional): Initial weight for time domain. Defaults to 0.5.
            adaptation_rate (float, optional): Adaptation rate. Defaults to 0.05.
        """
        self.time_weight = initial_time_weight
        self.freq_weight = 1.0 - initial_time_weight
        self.adaptation_rate = adaptation_rate
        self.last_time_error = 1.0
        self.last_freq_error = 1.0
    
    def combine_forecasts(self, time_forecast, freq_forecast):
        """
        Combine forecasts from different methods.
        
        Args:
            time_forecast (float): Time domain forecast
            freq_forecast (float): Frequency domain forecast
            
        Returns:
            float: Combined forecast
        """
        return self.time_weight * time_forecast + self.freq_weight * freq_forecast
    
    def update_weights(self, time_forecast, freq_forecast, actual_value):
        """
        Update weights based on forecast errors.
        
        Args:
            time_forecast (float): Time domain forecast
            freq_forecast (float): Frequency domain forecast
            actual_value (float): Actual value
        """
        # Calculate forecast errors
        time_error = abs(time_forecast - actual_value)
        freq_error = abs(freq_forecast - actual_value)
        
        # Update error history with exponential smoothing
        self.last_time_error = 0.7 * self.last_time_error + 0.3 * time_error
        self.last_freq_error = 0.7 * self.last_freq_error + 0.3 * freq_error
        
        # Calculate total error
        total_error = self.last_time_error + self.last_freq_error
        
        if total_error > 0:
            # Inverse error weighting
            new_time_weight = self.last_freq_error / total_error
            new_freq_weight = self.last_time_error / total_error
            
            # Apply adaptation rate
            self.time_weight = self.time_weight * (1 - self.adaptation_rate) + new_time_weight * self.adaptation_rate
            self.freq_weight = 1.0 - self.time_weight
    
    def get_weights(self):
        """
        Get current weights.
        
        Returns:
            tuple: Time weight, frequency weight
        """
        return self.time_weight, self.freq_weight


class PriceForecaster:
    """
    Python implementation of the CPriceForecaster class from MQL5.
    Forecasts future prices using multiple methods.
    """
    
    def __init__(self, forecast_horizon, lookback_period):
        """
        Initialize the price forecaster.
        
        Args:
            forecast_horizon (int): Forecast horizon
            lookback_period (int): Lookback period
        """
        self.forecast_horizon = forecast_horizon
        self.freq_features = FrequencyFeatures(lookback_period)
        self.combiner = AdaptiveForecastCombiner(0.5, 0.05)
        
        # RNN parameters
        self.rnn_weights = np.ones(10) * 0.1
        self.rnn_hidden_state = 0.0
        self.rnn_output_weight = 0.5
        self.rnn_learning_rate = 0.01
        self.last_rnn_prediction = 0.0
    
    def forecast_price(self, prices=None):
        """
        Forecast future price.
        
        Args:
            prices (np.ndarray, optional): Price data. Defaults to None.
            
        Returns:
            float: Forecasted price
        """
        if prices is None or len(prices) < 30:
            logger.warning("Not enough price data for forecasting")
            return 0
        
        # Time domain forecast (linear regression)
        time_forecast = self.forecast_time_method(prices)
        
        # Frequency domain forecast
        freq_forecast = self.forecast_frequency_method(prices)
        
        # RNN forecast
        rnn_forecast = self.forecast_rnn_method(prices)
        
        # Combine forecasts adaptively
        combined_base = self.combiner.combine_forecasts(time_forecast, freq_forecast)
        return (combined_base + rnn_forecast) / 2.0
    
    def forecast_time_method(self, prices):
        """
        Forecast using time domain method (linear regression).
        
        Args:
            prices (np.ndarray): Price data
            
        Returns:
            float: Forecasted price
        """
        n = len(prices)
        x = np.arange(n)
        
        # Calculate linear regression
        sum_x = np.sum(x)
        sum_y = np.sum(prices)
        sum_xy = np.sum(x * prices)
        sum_x2 = np.sum(x * x)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Forecast future price
        return intercept + slope * (n + self.forecast_horizon)
    
    def forecast_frequency_method(self, prices):
        """
        Forecast using frequency domain method.
        
        Args:
            prices (np.ndarray): Price data
            
        Returns:
            float: Forecasted price
        """
        if not self.freq_features.update_features(prices):
            return prices[0]  # Return current price if update fails
        
        # Get frequency features
        features = self.freq_features.get_frequency_features()
        
        spectral_entropy = features[0]
        trend_strength = features[1]
        cyclicality = features[2]
        
        # Get FFT results
        fft_real, fft_imag = self.freq_features.get_fft_results()
        
        # Get dominant frequencies and magnitudes
        freqs, mags = self.freq_features.get_dominant_frequencies()
        
        # Reconstruct signal with dominant frequencies
        forecast = prices[0]  # Start with current price
        
        for i in range(min(3, len(freqs))):
            # Add contribution of each dominant frequency
            freq = freqs[i]
            mag = mags[i]
            
            # Calculate phase (approximate)
            freq_index = int(freq * len(fft_real))
            if freq_index >= len(fft_real):
                freq_index = len(fft_real) - 1
            
            phase = np.arctan2(fft_imag[freq_index], fft_real[freq_index])
            
            forecast += mag * np.cos(2 * np.pi * freq * self.forecast_horizon + phase)
        
        return forecast
    
    def forecast_rnn_method(self, prices):
        """
        Forecast using RNN method.
        
        Args:
            prices (np.ndarray): Price data
            
        Returns:
            float: Forecasted price
        """
        n = len(prices)
        if n < 10:
            return prices[0]
        
        # Normalize prices
        temp = np.zeros(10)
        for i in range(10):
            temp[i] = (prices[n - 10 + i] - prices[n - 10]) / prices[n - 10]
        
        # Update hidden state
        new_hidden = 0.0
        for i in range(10):
            new_hidden += self.rnn_weights[i] * temp[i]
        
        new_hidden = np.tanh(new_hidden)
        self.rnn_hidden_state = 0.7 * self.rnn_hidden_state + 0.3 * new_hidden
        
        # Make prediction
        self.last_rnn_prediction = prices[n - 1] * (1.0 + self.rnn_output_weight * self.rnn_hidden_state)
        return self.last_rnn_prediction
    
    def update_combiner(self, actual_price):
        """
        Update forecast combiner with actual price outcome.
        
        Args:
            actual_price (float): Actual price
        """
        # This would normally use historical data to calculate what would have been forecasted
        # For simplicity, we'll just use random values
        old_time_forecast = actual_price * (1.0 + np.random.normal(0, 0.01))
        old_freq_forecast = actual_price * (1.0 + np.random.normal(0, 0.01))
        old_rnn_forecast = self.last_rnn_prediction
        
        # Update RNN weights
        rnn_error = actual_price - old_rnn_forecast
        gradient = rnn_error * self.rnn_learning_rate
        
        self.rnn_output_weight += gradient * self.rnn_hidden_state
        self.rnn_output_weight = max(-1.0, min(1.0, self.rnn_output_weight))
        
        for i in range(10):
            input_val = np.random.normal(0, 0.01)  # Placeholder for actual input values
            self.rnn_weights[i] += gradient * input_val
            self.rnn_weights[i] = max(-0.5, min(0.5, self.rnn_weights[i]))
        
        # Update weights based on actual outcome
        self.combiner.update_weights(old_time_forecast, old_freq_forecast, actual_price)
    
    def get_combiner_weights(self):
        """
        Get combiner weights.
        
        Returns:
            tuple: Time weight, frequency weight
        """
        return self.combiner.get_weights()
    
    def is_forecast_bullish(self):
        """
        Check if forecast is bullish.
        
        Returns:
            bool: True if forecast is bullish, False otherwise
        """
        # This would normally use the current price and forecast
        # For simplicity, we'll just use a random value
        return np.random.random() > 0.5


class FrequencyFilter:
    """
    Python implementation of the CFrequencyFilter class from MQL5.
    Filters trades based on frequency domain analysis.
    """
    
    def __init__(self, lookback_period, entropy_threshold=0.6, cyclicality_threshold=0.5):
        """
        Initialize the frequency filter.
        
        Args:
            lookback_period (int): Lookback period
            entropy_threshold (float, optional): Entropy threshold. Defaults to 0.6.
            cyclicality_threshold (float, optional): Cyclicality threshold. Defaults to 0.5.
        """
        self.freq_features = FrequencyFeatures(lookback_period)
        self.entropy_threshold = entropy_threshold
        self.cyclicality_threshold = cyclicality_threshold
    
    def should_trade(self, is_long, price_data=None):
        """
        Check if a trade should be taken based on frequency analysis.
        
        Args:
            is_long (bool): Whether the trade is long
            price_data (np.ndarray, optional): Price data. Defaults to None.
            
        Returns:
            bool: True if trade should be taken, False otherwise
        """
        if not self.freq_features.update_features(price_data):
            return True  # Default to allowing trades if we can't update features
        
        features = self.freq_features.get_frequency_features()
        
        entropy = features[0]
        trend_strength = features[1]
        cyclicality = features[2]
        
        # High entropy (complex, noisy market) - reduce trading
        if entropy > self.entropy_threshold:
            return False
        
        # For trending markets (low cyclicality), allow trend trades
        if cyclicality < self.cyclicality_threshold:
            # In trending markets, only allow trades in the direction of the trend
            trend = self.calculate_trend_direction()
            return (is_long and trend > 0) or (not is_long and trend < 0)
        
        # For cyclical markets, allow counter-trend trades
        return True
    
    def calculate_trend_direction(self):
        """
        Calculate trend direction.
        
        Returns:
            float: Trend direction (positive for uptrend, negative for downtrend)
        """
        # This would normally use EMA values
        # For simplicity, we'll just use a random value
        return np.random.normal(0, 1)
