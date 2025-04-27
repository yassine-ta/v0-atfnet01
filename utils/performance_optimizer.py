#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance optimization utilities for ATFNet.
Implements vectorized operations, caching, and parallel processing.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import lru_cache
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger('atfnet.utils.performance')


class FeatureCache:
    """
    Cache for feature calculations to avoid redundant computations.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the feature cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if key in self.cache:
            self.timestamps[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate
        }


class VectorizedIndicators:
    """
    Vectorized implementation of technical indicators for improved performance.
    """
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price data
            period: SMA period
            
        Returns:
            SMA values
        """
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            EMA values
        """
        alpha = 2 / (period + 1)
        alpha_rev = 1 - alpha
        
        # Calculate using vectorized operations
        n = len(data)
        pows = alpha_rev ** (np.arange(n))
        scale_arr = 1 / pows
        weights = alpha * alpha_rev ** (np.arange(n))
        
        # Calculate EMA
        ema_result = np.zeros_like(data)
        ema_result[0] = data[0]
        
        for i in range(1, n):
            ema_result[i] = alpha * data[i] + alpha_rev * ema_result[i-1]
        
        return ema_result
    
    @staticmethod
    def rsi(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: Price data
            period: RSI period
            
        Returns:
            RSI values
        """
        # Calculate price changes
        delta = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(delta)
        avg_loss = np.zeros_like(delta)
        
        # First average
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])
        
        # Subsequent averages
        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RS and RSI
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad beginning with NaN
        result = np.full(len(data), np.nan)
        result[period:] = rsi[period-1:]
        
        return result
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        # Calculate true range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR
        atr_result = np.zeros_like(close)
        atr_result[0] = tr[0]
        
        for i in range(1, len(close)):
            atr_result[i] = ((period - 1) * atr_result[i-1] + tr[i]) / period
        
        return atr_result
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        # Calculate middle band (SMA)
        middle_band = np.zeros_like(data)
        for i in range(period-1, len(data)):
            middle_band[i] = np.mean(data[i-period+1:i+1])
        
        # Calculate standard deviation
        std = np.zeros_like(data)
        for i in range(period-1, len(data)):
            std[i] = np.std(data[i-period+1:i+1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + std_dev * std
        lower_band = middle_band - std_dev * std
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD.
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        # Calculate fast and slow EMAs
        fast_ema = VectorizedIndicators.ema(data, fast_period)
        slow_ema = VectorizedIndicators.ema(data, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = VectorizedIndicators.ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


class ParallelBacktester:
    """
    Parallel backtester for improved performance.
    """
    
    def __init__(self, n_processes: Optional[int] = None):
        """
        Initialize the parallel backtester.
        
        Args:
            n_processes: Number of processes to use (default: CPU count)
        """
        self.n_processes = n_processes or mp.cpu_count()
    
    def run_parallel_backtest(
        self, 
        backtest_func: Callable, 
        parameter_sets: List[Dict[str, Any]],
        shared_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run backtests in parallel.
        
        Args:
            backtest_func: Backtest function
            parameter_sets: List of parameter sets
            shared_data: Shared data for all backtests
            
        Returns:
            List of backtest results
        """
        # Create a pool of workers
        pool = mp.Pool(processes=self.n_processes)
        
        # Prepare arguments
        args = [(backtest_func, params, shared_data) for params in parameter_sets]
        
        # Run backtests in parallel
        results = pool.map(self._run_single_backtest, args)
        
        # Close the pool
        pool.close()
        pool.join()
        
        return results
    
    @staticmethod
    def _run_single_backtest(args: Tuple) -> Dict[str, Any]:
        """
        Run a single backtest.
        
        Args:
            args: Tuple of (backtest_func, params, shared_data)
            
        Returns:
            Backtest results
        """
        backtest_func, params, shared_data = args
        
        try:
            if shared_data:
                return backtest_func(params=params, **shared_data)
            else:
                return backtest_func(params=params)
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e), 'params': params}


class PerformanceMonitor:
    """
    Performance monitoring for ATFNet.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.timers = {}
        self.counters = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """
        Start a timer.
        
        Args:
            name: Timer name
        """
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time
        """
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' not started")
            return 0
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.timers:
            self.timers[name] = []
        
        self.timers[name].append(elapsed)
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a counter.
        
        Args:
            name: Counter name
            value: Value to increment by
        """
        if name not in self.counters:
            self.counters[name] = 0
        
        self.counters[name] += value
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a timer.
        
        Args:
            name: Timer name
            
        Returns:
            Timer statistics
        """
        if name not in self.timers or not self.timers[name]:
            return {'count': 0, 'total': 0, 'mean': 0, 'min': 0, 'max': 0}
        
        times = self.timers[name]
        
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'min': min(times),
            'max': max(times)
        }
    
    def get_counter_value(self, name: str) -> int:
        """
        Get counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value
        """
        return self.counters.get(name, 0)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all statistics.
        
        Returns:
            All statistics
        """
        stats = {
            'timers': {name: self.get_timer_stats(name) for name in self.timers},
            'counters': self.counters.copy()
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.timers.clear()
        self.counters.clear()
        self.start_times.clear()


# Create global instances
feature_cache = FeatureCache()
performance_monitor = PerformanceMonitor()


# Decorator for caching function results
def cached_result(max_size: int = 128):
    """
    Decorator for caching function results.
    
    Args:
        max_size: Maximum cache size
        
    Returns:
        Decorated function
    """
    def decorator(func):
        cache = lru_cache(maxsize=max_size)(func)
        
        def wrapper(*args, **kwargs):
            result = cache(*args, **kwargs)
            return result
        
        wrapper.cache_info = cache.cache_info
        wrapper.cache_clear = cache.cache_clear
        
        return wrapper
    
    return decorator


# Decorator for timing function execution
def timed_execution(name: Optional[str] = None):
    """
    Decorator for timing function execution.
    
    Args:
        name: Timer name (default: function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            performance_monitor.start_timer(timer_name)
            result = func(*args, **kwargs)
            performance_monitor.stop_timer(timer_name)
            return result
        
        return wrapper
    
    return decorator
