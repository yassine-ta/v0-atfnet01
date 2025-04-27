#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced frequency domain analysis for ATFNet.
Implements wavelet transforms and enhanced spectral analysis techniques.
"""

import numpy as np
import pandas as pd
import pywt
from scipy import signal, fft
import logging
from typing import Tuple, List, Dict, Optional, Union, Any

logger = logging.getLogger('atfnet.core.advanced_frequency')


class WaveletAnalyzer:
    """
    Implements wavelet-based time-frequency analysis for financial time series.
    Provides multi-resolution analysis capabilities beyond standard FFT.
    """
    
    def __init__(self, wavelet: str = 'db8', max_level: int = 5):
        """
        Initialize the wavelet analyzer.
        
        Args:
            wavelet: Wavelet type to use (default: 'db8')
            max_level: Maximum decomposition level (default: 5)
        """
        self.wavelet = wavelet
        self.max_level = max_level
        self.coeffs = None
        self.features = {}
    
    def decompose(self, data: np.ndarray) -> List:
        """
        Perform wavelet decomposition on price data.
        
        Args:
            data: Price time series data
            
        Returns:
            List of wavelet coefficients
        """
        # Ensure data length is sufficient
        if len(data) < 2**self.max_level:
            logger.warning(f"Data length ({len(data)}) is too short for wavelet decomposition at level {self.max_level}")
            self.max_level = int(np.log2(len(data))) - 1
            
        # Perform wavelet decomposition
        self.coeffs = pywt.wavedec(data, self.wavelet, level=self.max_level)
        return self.coeffs
    
    def reconstruct(self, modified_coeffs: Optional[List] = None) -> np.ndarray:
        """
        Reconstruct time series from wavelet coefficients.
        
        Args:
            modified_coeffs: Modified coefficients (default: None, uses stored coeffs)
            
        Returns:
            Reconstructed time series
        """
        if modified_coeffs is None:
            if self.coeffs is None:
                raise ValueError("No coefficients available for reconstruction")
            modified_coeffs = self.coeffs
            
        return pywt.waverec(modified_coeffs, self.wavelet)
    
    def denoise(self, data: np.ndarray, threshold_method: str = 'soft', threshold_level: float = 0.3) -> np.ndarray:
        """
        Denoise time series using wavelet thresholding.
        
        Args:
            data: Price time series data
            threshold_method: Thresholding method ('soft' or 'hard')
            threshold_level: Threshold level as a fraction of max coefficient
            
        Returns:
            Denoised time series
        """
        # Decompose
        coeffs = self.decompose(data)
        
        # Apply thresholding to detail coefficients
        modified_coeffs = [coeffs[0]]  # Keep approximation coefficients
        
        for i in range(1, len(coeffs)):
            # Calculate threshold
            detail = coeffs[i]
            threshold = threshold_level * np.max(np.abs(detail))
            
            # Apply threshold
            if threshold_method == 'soft':
                modified_detail = pywt.threshold(detail, threshold, mode='soft')
            else:
                modified_detail = pywt.threshold(detail, threshold, mode='hard')
                
            modified_coeffs.append(modified_detail)
        
        # Reconstruct
        return self.reconstruct(modified_coeffs)
    
    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features from time series.
        
        Args:
            data: Price time series data
            
        Returns:
            Dictionary of wavelet features
        """
        # Decompose if not already done
        if self.coeffs is None:
            self.decompose(data)
        
        # Calculate energy distribution across scales
        total_energy = 0
        energy_by_level = []
        
        for i, coeff in enumerate(self.coeffs):
            energy = np.sum(coeff**2)
            energy_by_level.append(energy)
            total_energy += energy
        
        # Normalize energies
        if total_energy > 0:
            energy_by_level = [e / total_energy for e in energy_by_level]
        
        # Calculate features
        self.features = {
            'wavelet_entropy': -np.sum([e * np.log(e + 1e-10) for e in energy_by_level]),
            'detail_energy_ratio': np.sum(energy_by_level[1:]) / (energy_by_level[0] + 1e-10),
            'scale_ratio': energy_by_level[1] / (np.sum(energy_by_level[2:]) + 1e-10) if len(energy_by_level) > 2 else 0,
            'trend_strength': energy_by_level[0],
            'noise_level': np.mean(energy_by_level[-2:]) if len(energy_by_level) > 2 else 0
        }
        
        return self.features
    
    def forecast(self, data: np.ndarray, horizon: int = 10) -> np.ndarray:
        """
        Forecast future values using wavelet decomposition.
        
        Args:
            data: Price time series data
            horizon: Forecast horizon
            
        Returns:
            Forecasted values
        """
        # Decompose
        self.decompose(data)
        
        # Forecast each component separately
        forecasts = []
        
        for i, coeff in enumerate(self.coeffs):
            # Use AR model for each component
            if len(coeff) > 10:  # Need sufficient data for AR model
                ar_order = min(5, len(coeff) // 10)
                ar_model = signal.AR(coeff)
                ar_params = ar_model.fit(ar_order).prediction_x
                
                # Forecast component
                component_forecast = np.zeros(horizon)
                last_values = coeff[-ar_order:]
                
                for j in range(horizon):
                    # AR prediction
                    component_forecast[j] = np.sum(ar_params * np.flip(last_values))
                    # Update last values
                    last_values = np.append(last_values[1:], component_forecast[j])
                
                forecasts.append(component_forecast)
            else:
                # Use simple mean for short components
                forecasts.append(np.ones(horizon) * np.mean(coeff))
        
        # Reconstruct forecast
        # Need to pad forecasts to match coefficient lengths
        padded_forecasts = []
        for i, (coeff, forecast) in enumerate(zip(self.coeffs, forecasts)):
            if i == 0:  # Approximation coefficient
                padded = np.append(coeff, forecast)
            else:  # Detail coefficients
                # Calculate padding based on reconstruction requirements
                padding_length = len(coeff) + horizon
                padded = np.zeros(padding_length)
                padded[:len(coeff)] = coeff
                # No need to add forecast for detail coefficients
            
            padded_forecasts.append(padded)
        
        # Reconstruct with padded coefficients
        reconstructed = pywt.waverec(padded_forecasts, self.wavelet)
        
        # Return only the forecasted part
        return reconstructed[-horizon:]


class EnhancedSpectralAnalyzer:
    """
    Enhanced spectral analysis for financial time series.
    Combines FFT with advanced spectral estimation techniques.
    """
    
    def __init__(self, max_freq: float = 0.5, method: str = 'welch'):
        """
        Initialize the spectral analyzer.
        
        Args:
            max_freq: Maximum frequency to analyze (default: 0.5)
            method: Spectral estimation method (default: 'welch')
        """
        self.max_freq = max_freq
        self.method = method
        self.psd = None
        self.freqs = None
        self.dominant_cycles = None
    
    def estimate_spectrum(self, data: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate power spectral density of time series.
        
        Args:
            data: Price time series data
            fs: Sampling frequency (default: 1.0)
            
        Returns:
            Tuple of (frequencies, power spectral density)
        """
        # Detrend data
        detrended_data = signal.detrend(data)
        
        # Apply window
        windowed_data = detrended_data * signal.windows.hann(len(detrended_data))
        
        if self.method == 'welch':
            # Welch's method for PSD estimation
            self.freqs, self.psd = signal.welch(
                windowed_data, 
                fs=fs, 
                nperseg=min(256, len(data)//2),
                scaling='spectrum'
            )
        elif self.method == 'periodogram':
            # Periodogram
            self.freqs, self.psd = signal.periodogram(
                windowed_data,
                fs=fs,
                scaling='spectrum'
            )
        elif self.method == 'lombscargle':
            # Lomb-Scargle periodogram (good for unevenly sampled data)
            self.freqs = np.linspace(0.01, self.max_freq, 100)
            self.psd = signal.lombscargle(
                np.arange(len(data)),
                windowed_data,
                self.freqs * 2 * np.pi
            )
        else:
            # Default to FFT
            n = len(data)
            fft_result = fft.fft(windowed_data)
            self.psd = np.abs(fft_result[:n//2])**2 / n
            self.freqs = fft.fftfreq(n, d=1/fs)[:n//2]
        
        return self.freqs, self.psd
    
    def find_dominant_cycles(self, n_peaks: int = 3) -> List[Dict[str, float]]:
        """
        Find dominant cycles in the spectrum.
        
        Args:
            n_peaks: Number of dominant peaks to find
            
        Returns:
            List of dictionaries with cycle information
        """
        if self.psd is None or self.freqs is None:
            raise ValueError("Spectrum not estimated yet")
        
        # Find peaks in PSD
        peak_indices, _ = signal.find_peaks(self.psd, height=np.mean(self.psd))
        
        # Sort peaks by power
        sorted_peaks = sorted(peak_indices, key=lambda i: self.psd[i], reverse=True)
        
        # Get top n_peaks
        top_peaks = sorted_peaks[:min(n_peaks, len(sorted_peaks))]
        
        # Create cycle information
        self.dominant_cycles = []
        for idx in top_peaks:
            # Skip DC component (zero frequency)
            if self.freqs[idx] < 0.01:
                continue
                
            cycle = {
                'frequency': self.freqs[idx],
                'period': 1 / self.freqs[idx] if self.freqs[idx] > 0 else float('inf'),
                'power': self.psd[idx],
                'relative_power': self.psd[idx] / np.sum(self.psd)
            }
            self.dominant_cycles.append(cycle)
        
        return self.dominant_cycles
    
    def calculate_spectral_features(self) -> Dict[str, float]:
        """
        Calculate spectral features from PSD.
        
        Returns:
            Dictionary of spectral features
        """
        if self.psd is None or self.freqs is None:
            raise ValueError("Spectrum not estimated yet")
        
        # Calculate total power
        total_power = np.sum(self.psd)
        
        # Calculate spectral centroid
        centroid = np.sum(self.freqs * self.psd) / total_power if total_power > 0 else 0
        
        # Calculate spectral spread
        spread = np.sqrt(np.sum(((self.freqs - centroid) ** 2) * self.psd) / total_power) if total_power > 0 else 0
        
        # Calculate spectral skewness
        skewness = np.sum(((self.freqs - centroid) ** 3) * self.psd) / (total_power * (spread ** 3)) if spread > 0 else 0
        
        # Calculate spectral kurtosis
        kurtosis = np.sum(((self.freqs - centroid) ** 4) * self.psd) / (total_power * (spread ** 4)) if spread > 0 else 0
        
        # Calculate spectral entropy
        normalized_psd = self.psd / total_power if total_power > 0 else np.zeros_like(self.psd)
        entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-10))
        
        # Calculate spectral flatness
        geometric_mean = np.exp(np.mean(np.log(self.psd + 1e-10)))
        arithmetic_mean = np.mean(self.psd)
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Calculate low/high frequency ratio
        low_freq_idx = np.where(self.freqs <= 0.1)[0]
        high_freq_idx = np.where(self.freqs > 0.1)[0]
        
        low_power = np.sum(self.psd[low_freq_idx])
        high_power = np.sum(self.psd[high_freq_idx])
        
        lf_hf_ratio = low_power / high_power if high_power > 0 else float('inf')
        
        return {
            'spectral_centroid': centroid,
            'spectral_spread': spread,
            'spectral_skewness': skewness,
            'spectral_kurtosis': kurtosis,
            'spectral_entropy': entropy,
            'spectral_flatness': flatness,
            'lf_hf_ratio': lf_hf_ratio
        }
    
    def filter_components(self, data: np.ndarray, low_cutoff: float = 0.0, high_cutoff: float = 0.1) -> np.ndarray:
        """
        Filter time series to specific frequency band.
        
        Args:
            data: Price time series data
            low_cutoff: Low cutoff frequency
            high_cutoff: High cutoff frequency
            
        Returns:
            Filtered time series
        """
        # Design filter
        nyquist = 0.5
        b, a = signal.butter(
            4, 
            [low_cutoff / nyquist, high_cutoff / nyquist], 
            btype='band'
        )
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data


class AdvancedFrequencyAnalyzer:
    """
    Combined frequency domain analyzer that integrates wavelet and spectral analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced frequency analyzer.
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            'wavelet_type': 'db8',
            'max_wavelet_level': 5,
            'spectral_method': 'welch',
            'max_frequency': 0.5,
            'n_dominant_cycles': 3,
            'denoise_threshold': 0.3
        }
        
        # Update with provided config
        if config is not None:
            self.config.update(config)
        
        # Initialize analyzers
        self.wavelet_analyzer = WaveletAnalyzer(
            wavelet=self.config['wavelet_type'],
            max_level=self.config['max_wavelet_level']
        )
        
        self.spectral_analyzer = EnhancedSpectralAnalyzer(
            max_freq=self.config['max_frequency'],
            method=self.config['spectral_method']
        )
        
        # Feature cache
        self.features = {}
    
    def analyze(self, data: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive frequency domain analysis.
        
        Args:
            data: Price time series data
            
        Returns:
            Dictionary of frequency domain features
        """
        # Ensure data is numpy array
        data = np.asarray(data)
        
        # Denoise data
        denoised_data = self.wavelet_analyzer.denoise(
            data, 
            threshold_level=self.config['denoise_threshold']
        )
        
        # Extract wavelet features
        wavelet_features = self.wavelet_analyzer.extract_features(data)
        
        # Estimate spectrum
        self.spectral_analyzer.estimate_spectrum(denoised_data)
        
        # Find dominant cycles
        dominant_cycles = self.spectral_analyzer.find_dominant_cycles(
            n_peaks=self.config['n_dominant_cycles']
        )
        
        # Calculate spectral features
        spectral_features = self.spectral_analyzer.calculate_spectral_features()
        
        # Combine features
        self.features = {**wavelet_features, **spectral_features}
        
        # Add dominant cycle information
        if dominant_cycles:
            self.features['dominant_cycle_period'] = dominant_cycles[0]['period']
            self.features['dominant_cycle_power'] = dominant_cycles[0]['relative_power']
        
        return self.features
    
    def forecast(self, data: np.ndarray, horizon: int = 10) -> np.ndarray:
        """
        Forecast future values using combined frequency domain methods.
        
        Args:
            data: Price time series data
            horizon: Forecast horizon
            
        Returns:
            Forecasted values
        """
        # Get wavelet forecast
        wavelet_forecast = self.wavelet_analyzer.forecast(data, horizon)
        
        # For now, just return the wavelet forecast
        # In a more advanced implementation, we could combine multiple forecasting methods
        return wavelet_forecast
    
    def get_cyclical_component(self, data: np.ndarray) -> np.ndarray:
        """
        Extract cyclical component from time series.
        
        Args:
            data: Price time series data
            
        Returns:
            Cyclical component of time series
        """
        # Estimate spectrum
        self.spectral_analyzer.estimate_spectrum(data)
        
        # Find dominant cycles
        dominant_cycles = self.spectral_analyzer.find_dominant_cycles()
        
        if not dominant_cycles:
            return np.zeros_like(data)
        
        # Get dominant cycle frequency
        dominant_freq = dominant_cycles[0]['frequency']
        
        # Filter around dominant frequency
        bandwidth = dominant_freq * 0.3  # 30% bandwidth
        low_cutoff = max(0.001, dominant_freq - bandwidth)
        high_cutoff = min(0.5, dominant_freq + bandwidth)
        
        # Apply bandpass filter
        cyclical_component = self.spectral_analyzer.filter_components(
            data, 
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff
        )
        
        return cyclical_component
    
    def get_trend_component(self, data: np.ndarray) -> np.ndarray:
        """
        Extract trend component from time series.
        
        Args:
            data: Price time series data
            
        Returns:
            Trend component of time series
        """
        # Decompose with wavelet
        coeffs = self.wavelet_analyzer.decompose(data)
        
        # Keep only approximation coefficient (trend)
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        
        # Reconstruct
        trend_component = self.wavelet_analyzer.reconstruct(trend_coeffs)
        
        # Ensure same length as input
        if len(trend_component) != len(data):
            trend_component = trend_component[:len(data)]
        
        return trend_component
    
    def get_noise_component(self, data: np.ndarray) -> np.ndarray:
        """
        Extract noise component from time series.
        
        Args:
            data: Price time series data
            
        Returns:
            Noise component of time series
        """
        # Decompose with wavelet
        coeffs = self.wavelet_analyzer.decompose(data)
        
        # Keep only highest frequency detail coefficient (noise)
        noise_coeffs = [np.zeros_like(coeffs[0])]
        for i in range(1, len(coeffs)):
            if i == len(coeffs) - 1:  # Highest frequency detail
                noise_coeffs.append(coeffs[i])
            else:
                noise_coeffs.append(np.zeros_like(coeffs[i]))
        
        # Reconstruct
        noise_component = self.wavelet_analyzer.reconstruct(noise_coeffs)
        
        # Ensure same length as input
        if len(noise_component) != len(data):
            noise_component = noise_component[:len(data)]
        
        return noise_component
