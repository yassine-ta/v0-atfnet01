#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python implementation of the OPT_ATFNet_MarketRegime.mqh file.
This module implements market regime detection for the ATFNet strategy.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('atfnet.core.market_regime')


class MarketRegimeDetector:
    """
    Python implementation of the CMarketRegimeDetector class from MQL5.
    Detects market regimes based on price data.
    """
    
    # Market regime enum
    REGIME_TRENDING = 0
    REGIME_RANGING = 1
    REGIME_VOLATILE = 2
    REGIME_CHOPPY = 3
    
    def __init__(self, config=None):
        """
        Initialize the market regime detector.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.current_regime = self.REGIME_RANGING
        
        # Set default parameters
        self.lookback_bars = 100
        self.trend_threshold = 0.6
        self.volatility_threshold = 1.5
        self.enable_regime_detection = True
        
        # Update parameters from config if provided
        if config and 'regime_detection' in config:
            regime_config = config['regime_detection']
            self.lookback_bars = regime_config.get('regime_lookback_bars', self.lookback_bars)
            self.trend_threshold = regime_config.get('trend_strength_threshold', self.trend_threshold)
            self.volatility_threshold = regime_config.get('volatility_threshold', self.volatility_threshold)
            self.enable_regime_detection = regime_config.get('enable_regime_detection', self.enable_regime_detection)
        
        # Cached values
        self.trend_strength = 0.0
        self.volatility = 0.0
        self.cyclicality = 0.0
    
    def detect_regime(self, price_data=None):
        """
        Detect current market regime.
        
        Args:
            price_data (pd.DataFrame, optional): Price data. Defaults to None.
            
        Returns:
            int: Market regime
        """
        if not self.enable_regime_detection:
            return self.REGIME_RANGING
            
        if price_data is None or len(price_data) < self.lookback_bars:
            logger.warning(f"Not enough data for regime detection. Need {self.lookback_bars} bars.")
            return self.current_regime
        
        # Calculate trend strength
        self.trend_strength = self.calculate_trend_strength(price_data)
        
        # Calculate volatility
        self.volatility = self.calculate_volatility(price_data)
        
        # Calculate cyclicality
        self.cyclicality = self.calculate_cyclicality(price_data)
        
        # Determine regime
        if self.trend_strength > self.trend_threshold:
            if self.volatility > self.volatility_threshold:
                self.current_regime = self.REGIME_VOLATILE
            else:
                self.current_regime = self.REGIME_TRENDING
        else:
            if self.volatility > self.volatility_threshold:
                self.current_regime = self.REGIME_CHOPPY
            else:
                self.current_regime = self.REGIME_RANGING
        
        return self.current_regime
    
    def calculate_trend_strength(self, prices):
        """
        Calculate trend strength (0-1).
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            float: Trend strength
        """
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close'].values
        
        n = len(prices)
        if n < 2:
            return 0
        
        # Linear regression
        x = np.arange(n)
        
        # Calculate sums
        sum_x = np.sum(x)
        sum_y = np.sum(prices)
        sum_xy = np.sum(x * prices)
        sum_x2 = np.sum(x * x)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        mean_y = sum_y / n
        total_ss = np.sum((prices - mean_y) ** 2)
        
        predicted = intercept + slope * x
        residual_ss = np.sum((prices - predicted) ** 2)
        
        r_squared = 1 - (residual_ss / total_ss) if total_ss > 0 else 0
        
        return r_squared
    
    def calculate_volatility(self, prices):
        """
        Calculate volatility.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            float: Volatility
        """
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close'].values
        
        n = len(prices)
        if n < 2:
            return 0
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate standard deviation of returns
        return np.std(returns)
    
    def calculate_cyclicality(self, prices):
        """
        Calculate cyclicality.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            float: Cyclicality
        """
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close'].values
        
        n = len(prices)
        if n < 4:
            return 0
        
        # Count direction changes
        direction_changes = 0
        current_direction = 0
        
        for i in range(1, n):
            new_direction = 1 if prices[i] > prices[i-1] else -1
            
            if current_direction != 0 and new_direction != current_direction:
                direction_changes += 1
            
            current_direction = new_direction
        
        # Calculate cyclicality as ratio of direction changes to total possible changes
        return direction_changes / (n - 2) if n > 2 else 0
    
    def optimize_for_regime(self, regime):
        """
        Optimize strategy parameters for current regime.
        
        Args:
            regime (int): Market regime
        """
        if regime == self.REGIME_TRENDING:
            # Optimize for trending markets
            # - Increase trend following weight
            # - Reduce mean reversion weight
            # - Increase position holding time
            logger.info("Optimizing for trending market")
        
        elif regime == self.REGIME_RANGING:
            # Optimize for ranging markets
            # - Increase mean reversion weight
            # - Reduce trend following weight
            # - Decrease position holding time
            logger.info("Optimizing for ranging market")
        
        elif regime == self.REGIME_VOLATILE:
            # Optimize for volatile markets
            # - Widen stop losses
            # - Reduce position sizes
            # - Increase profit targets
            logger.info("Optimizing for volatile market")
        
        elif regime == self.REGIME_CHOPPY:
            # Optimize for choppy markets
            # - Reduce overall exposure
            # - Increase signal thresholds
            # - Consider pausing trading
            logger.info("Optimizing for choppy market")
    
    def get_current_regime(self):
        """
        Get current market regime.
        
        Returns:
            int: Current market regime
        """
        return self.current_regime
    
    def get_trend_strength(self):
        """
        Get trend strength.
        
        Returns:
            float: Trend strength
        """
        return self.trend_strength
    
    def get_volatility(self):
        """
        Get volatility.
        
        Returns:
            float: Volatility
        """
        return self.volatility
    
    def get_cyclicality(self):
        """
        Get cyclicality.
        
        Returns:
            float: Cyclicality
        """
        return self.cyclicality
