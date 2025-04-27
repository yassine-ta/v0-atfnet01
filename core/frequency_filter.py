#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frequency Filter Module for ATFNet.

This module provides functionality for filtering price data in the frequency domain,
allowing for noise reduction and trend isolation.
"""

import numpy as np
from scipy import signal
import pandas as pd
import pywt


class FrequencyFilter:
    """
    A class for filtering financial time series data in the frequency domain.
    
    This class implements various filtering techniques including:
    - Fourier Transform based filtering
    - Wavelet Transform based filtering
    - Butterworth filters
    - Savitzky-Golay filters
    """
    
    def __init__(self, lookback_period=None, *args, **kwargs):
        """
        lookback_period: number of bars to look back when filtering frequency
        """
        # store lookback (default to 14 if not provided)
        self.lookback_period = lookback_period or 14

        self.config = kwargs.get('config', {})
        self.filter_type = self.config.get('filter_type', 'butterworth')
        self.cutoff_frequency = self.config.get('cutoff_frequency', 0.1)
        self.order = self.config.get('order', 4)
        self.window_size = self.config.get('window_size', 11)
        self.polynomial_order = self.config.get('polynomial_order', 3)
        self.wavelet = self.config.get('wavelet', 'db4')
        self.level = self.config.get('level', 3)
    
    def apply_filter(self, data, filter_type=None):
        """
        Apply the specified filter to the input data.
        
        Args:
            data (numpy.ndarray): Input time series data
            filter_type (str, optional): Type of filter to apply. If None, uses the default filter_type.
                Options: 'butterworth', 'savgol', 'fft', 'wavelet'
        
        Returns:
            numpy.ndarray: Filtered data
        """
        filter_type = filter_type or self.filter_type
        
        if filter_type == 'butterworth':
            return self._apply_butterworth(data)
        elif filter_type == 'savgol':
            return self._apply_savgol(data)
        elif filter_type == 'fft':
            return self._apply_fft_filter(data)
        elif filter_type == 'wavelet':
            return self._apply_wavelet_filter(data)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def _apply_butterworth(self, data):
        """
        Apply a Butterworth low-pass filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
        
        Returns:
            numpy.ndarray: Filtered data
        """
        nyquist = 0.5
        normal_cutoff = self.cutoff_frequency / nyquist
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)
    
    def _apply_savgol(self, data):
        """
        Apply a Savitzky-Golay filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
        
        Returns:
            numpy.ndarray: Filtered data
        """
        return signal.savgol_filter(data, self.window_size, self.polynomial_order)
    
    def _apply_fft_filter(self, data):
        """
        Apply a Fast Fourier Transform based filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
        
        Returns:
            numpy.ndarray: Filtered data
        """
        # Compute the FFT
        fft_data = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data))
        
        # Create a mask for frequencies to keep
        mask = frequencies <= self.cutoff_frequency
        
        # Apply the mask
        fft_data_filtered = fft_data * mask
        
        # Inverse FFT to get back to time domain
        return np.fft.irfft(fft_data_filtered, len(data))
    
    def _apply_wavelet_filter(self, data):
        """
        Apply a Wavelet Transform based filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
        
        Returns:
            numpy.ndarray: Filtered data
        """
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Threshold the detail coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i])/2, mode='soft')
        
        # Reconstruct the signal
        return pywt.waverec(coeffs, self.wavelet)
    
    def bandpass_filter(self, data, lowcut, highcut, fs=1.0):
        """
        Apply a bandpass filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
            lowcut (float): Low cutoff frequency
            highcut (float): High cutoff frequency
            fs (float): Sampling frequency
        
        Returns:
            numpy.ndarray: Filtered data
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(self.order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def adaptive_filter(self, data, window_size=None):
        """
        Apply an adaptive filter that adjusts parameters based on data characteristics.
        
        Args:
            data (numpy.ndarray): Input time series data
            window_size (int, optional): Size of the rolling window for volatility calculation
        
        Returns:
            numpy.ndarray: Filtered data
        """
        window_size = window_size or min(50, len(data) // 4)
        
        # Convert to pandas Series for rolling calculations
        series = pd.Series(data)
        
        # Calculate rolling volatility
        volatility = series.rolling(window=window_size).std().fillna(method='bfill')
        
        # Normalize volatility to range [0, 1]
        norm_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)
        
        # Adjust cutoff frequency based on volatility
        # Higher volatility -> higher cutoff (less filtering)
        adaptive_cutoff = self.cutoff_frequency * (1 + norm_vol)
        
        # Apply filter with varying cutoff frequency
        filtered_data = np.zeros_like(data)
        for i in range(len(data)):
            cutoff = adaptive_cutoff.iloc[i]
            nyquist = 0.5
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
            
            # Apply filter to a window around the current point
            window_start = max(0, i - window_size)
            window_end = min(len(data), i + window_size + 1)
            window_data = data[window_start:window_end]
            
            filtered_window = signal.filtfilt(b, a, window_data)
            filtered_data[i] = filtered_window[i - window_start]
        
        return filtered_data
    
    def kalman_filter(self, data, process_variance=1e-5, measurement_variance=1e-3):
        """
        Apply a Kalman filter to the data.
        
        Args:
            data (numpy.ndarray): Input time series data
            process_variance (float): Process variance parameter
            measurement_variance (float): Measurement variance parameter
        
        Returns:
            numpy.ndarray: Filtered data
        """
        # Initialize Kalman filter parameters
        n = len(data)
        filtered_data = np.zeros(n)
        prediction = data[0]
        prediction_variance = 1.0
        
        # Apply Kalman filter
        for i in range(n):
            # Prediction update
            prediction_variance += process_variance
            
            # Measurement update
            kalman_gain = prediction_variance / (prediction_variance + measurement_variance)
            prediction = prediction + kalman_gain * (data[i] - prediction)
            prediction_variance = (1 - kalman_gain) * prediction_variance
            
            filtered_data[i] = prediction
        
        return filtered_data
