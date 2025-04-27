#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wavelet Analysis module for ATFNet.
Implements wavelet-based multi-scale analysis for time series decomposition.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import pywt
from scipy import signal
from datetime import datetime

logger = logging.getLogger('atfnet.core.wavelet')


class WaveletAnalyzer:
    """
    Wavelet analysis for ATFNet.
    Implements wavelet-based multi-scale analysis for time series decomposition.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'wavelet_analysis'):
        """
        Initialize the wavelet analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.wavelet = self.config.get('wavelet', 'db8')  # Default wavelet
        self.max_level = self.config.get('max_level', 5)  # Maximum decomposition level
        self.mode = self.config.get('mode', 'symmetric')  # Border extension mode
        
        # Initialize state variables
        self.coefficients = None
        self.reconstructed = None
        self.features = {}
        
        logger.info(f"Wavelet Analyzer initialized with {self.wavelet} wavelet")
    
    def decompose(self, time_series: np.ndarray, levels: int = None) -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition of time series.
        
        Args:
            time_series: Input time series
            levels: Number of decomposition levels (default: max_level)
            
        Returns:
            Dictionary of wavelet coefficients
        """
        # Ensure time_series is a numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Set default levels
        if levels is None:
            levels = self.max_level
        
        # Ensure levels is valid
        max_possible_level = pywt.dwt_max_level(len(time_series), pywt.Wavelet(self.wavelet).dec_len)
        levels = min(levels, max_possible_level)
        
        try:
            # Perform multilevel wavelet decomposition
            coeffs = pywt.wavedec(time_series, self.wavelet, mode=self.mode, level=levels)
            
            # Store coefficients
            self.coefficients = coeffs
            
            # Create dictionary of coefficients
            coeff_dict = {'approximation': coeffs[0]}
            for i in range(1, len(coeffs)):
                coeff_dict[f'detail_{i}'] = coeffs[i]
            
            logger.info(f"Wavelet decomposition completed with {levels} levels")
            
            return coeff_dict
            
        except Exception as e:
            logger.error(f"Error in wavelet decomposition: {e}")
            raise
    
    def reconstruct(self, coefficients: List[np.ndarray] = None, 
                   levels: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Reconstruct time series from wavelet coefficients.
        
        Args:
            coefficients: Wavelet coefficients (default: self.coefficients)
            levels: Levels to include in reconstruction (default: all)
            
        Returns:
            Dictionary of reconstructed signals
        """
        if coefficients is None:
            if self.coefficients is None:
                raise ValueError("No coefficients available. Run decompose first.")
            coefficients = self.coefficients
        
        try:
            # Get total number of levels
            total_levels = len(coefficients) - 1
            
            # Set default levels
            if levels is None:
                levels = list(range(total_levels + 1))
            
            # Create reconstruction dictionary
            reconstructed = {}
            
            # Reconstruct approximation
            if 0 in levels:
                approx_coeffs = [coefficients[0]] + [None] * total_levels
                reconstructed['approximation'] = pywt.waverec(approx_coeffs, self.wavelet, mode=self.mode)
            
            # Reconstruct details
            for i in range(1, total_levels + 1):
                if i in levels:
                    detail_coeffs = [None] * total_levels + [None]
                    detail_coeffs[i] = coefficients[i]
                    reconstructed[f'detail_{i}'] = pywt.waverec(detail_coeffs, self.wavelet, mode=self.mode)
            
            # Reconstruct full signal
            reconstructed['full'] = pywt.waverec(coefficients, self.wavelet, mode=self.mode)
            
            # Store reconstructed signals
            self.reconstructed = reconstructed
            
            logger.info(f"Wavelet reconstruction completed for levels {levels}")
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error in wavelet reconstruction: {e}")
            raise
    
    def extract_features(self, time_series: np.ndarray = None, 
                        coefficients: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """
        Extract features from wavelet decomposition.
        
        Args:
            time_series: Original time series (optional)
            coefficients: Wavelet coefficients (default: from decompose)
            
        Returns:
            Dictionary of wavelet-based features
        """
        if coefficients is None:
            if self.coefficients is None:
                raise ValueError("No coefficients available. Run decompose first.")
            
            # Convert list to dictionary
            coefficients = {'approximation': self.coefficients[0]}
            for i in range(1, len(self.coefficients)):
                coefficients[f'detail_{i}'] = self.coefficients[i]
        
        try:
            # Initialize features dictionary
            features = {}
            
            # Extract energy features
            energy = {}
            total_energy = 0
            
            for name, coeff in coefficients.items():
                energy[name] = np.sum(coeff**2)
                total_energy += energy[name]
            
            # Calculate relative energy
            for name in energy:
                features[f'energy_{name}'] = energy[name]
                features[f'relative_energy_{name}'] = energy[name] / total_energy if total_energy > 0 else 0
            
            # Extract statistical features for each component
            for name, coeff in coefficients.items():
                features[f'mean_{name}'] = np.mean(coeff)
                features[f'std_{name}'] = np.std(coeff)
                features[f'skew_{name}'] = self._skewness(coeff)
                features[f'kurtosis_{name}'] = self._kurtosis(coeff)
                features[f'max_{name}'] = np.max(coeff)
                features[f'min_{name}'] = np.min(coeff)
                features[f'range_{name}'] = np.max(coeff) - np.min(coeff)
            
            # Extract entropy features
            for name, coeff in coefficients.items():
                features[f'entropy_{name}'] = self._entropy(coeff)
            
            # Extract trend strength if original time series is provided
            if time_series is not None:
                if isinstance(time_series, pd.Series):
                    time_series = time_series.values
                
                # Calculate trend strength using approximation component
                if 'approximation' in coefficients:
                    approx = coefficients['approximation']
                    # Ensure same length
                    min_len = min(len(time_series), len(approx))
                    trend_strength = 1 - np.var(time_series[:min_len] - approx[:min_len]) / np.var(time_series[:min_len])
                    features['trend_strength'] = max(0, trend_strength)  # Ensure non-negative
            
            # Store features
            self.features = features
            
            logger.info(f"Extracted {len(features)} wavelet features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting wavelet features: {e}")
            raise
    
    def continuous_wavelet_transform(self, time_series: np.ndarray, 
                                    scales: np.ndarray = None,
                                    wavelet: str = 'cmor') -> Dict[str, Any]:
        """
        Perform continuous wavelet transform.
        
        Args:
            time_series: Input time series
            scales: Scales for CWT (default: automatic)
            wavelet: Wavelet to use for CWT
            
        Returns:
            Dictionary with CWT results
        """
        # Ensure time_series is a numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Set default scales
        if scales is None:
            scales = np.arange(1, min(128, len(time_series) // 2))
        
        try:
            # Perform CWT
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)
            
            # Calculate scalogram (power)
            scalogram = np.abs(coefficients)**2
            
            # Calculate global wavelet spectrum
            global_wavelet_spectrum = np.mean(scalogram, axis=1)
            
            # Calculate scale-averaged wavelet power
            scale_averaged_power = np.mean(scalogram, axis=0)
            
            # Prepare results
            results = {
                'coefficients': coefficients,
                'frequencies': frequencies,
                'scales': scales,
                'scalogram': scalogram,
                'global_wavelet_spectrum': global_wavelet_spectrum,
                'scale_averaged_power': scale_averaged_power
            }
            
            logger.info(f"Continuous wavelet transform completed with {len(scales)} scales")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in continuous wavelet transform: {e}")
            raise
    
    def detect_cycles(self, time_series: np.ndarray, 
                     min_period: int = 2, 
                     max_period: int = None) -> Dict[str, Any]:
        """
        Detect cycles in time series using wavelet analysis.
        
        Args:
            time_series: Input time series
            min_period: Minimum period to detect
            max_period: Maximum period to detect (default: len(time_series)/2)
            
        Returns:
            Dictionary with cycle detection results
        """
        # Ensure time_series is a numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Set default max_period
        if max_period is None:
            max_period = len(time_series) // 2
        
        try:
            # Define scales based on periods
            scales = np.arange(min_period, min(max_period, len(time_series) // 2))
            
            # Perform CWT
            cwt_result = self.continuous_wavelet_transform(time_series, scales)
            
            # Find dominant periods
            global_power = cwt_result['global_wavelet_spectrum']
            dominant_indices = signal.find_peaks(global_power)[0]
            
            # Sort by power
            dominant_indices = sorted(dominant_indices, key=lambda i: global_power[i], reverse=True)
            
            # Extract dominant periods
            dominant_periods = []
            for idx in dominant_indices:
                if idx < len(scales):
                    dominant_periods.append({
                        'period': scales[idx],
                        'power': global_power[idx]
                    })
            
            # Calculate significance
            mean_power = np.mean(global_power)
            std_power = np.std(global_power)
            
            for period in dominant_periods:
                period['significance'] = (period['power'] - mean_power) / std_power
            
            # Filter by significance
            significant_periods = [p for p in dominant_periods if p['significance'] > 2.0]
            
            # Prepare results
            results = {
                'dominant_periods': dominant_periods[:5],  # Top 5
                'significant_periods': significant_periods,
                'global_power': global_power,
                'scales': scales
            }
            
            logger.info(f"Detected {len(significant_periods)} significant cycles")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cycle detection: {e}")
            raise
    
    def denoise(self, time_series: np.ndarray, 
               threshold_method: str = 'soft',
               threshold_level: str = 'universal',
               wavelet: str = None,
               level: int = None) -> np.ndarray:
        """
        Denoise time series using wavelet thresholding.
        
        Args:
            time_series: Input time series
            threshold_method: Thresholding method ('soft' or 'hard')
            threshold_level: Threshold level ('universal' or 'bayes')
            wavelet: Wavelet to use (default: self.wavelet)
            level: Decomposition level (default: self.max_level)
            
        Returns:
            Denoised time series
        """
        # Ensure time_series is a numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Set defaults
        if wavelet is None:
            wavelet = self.wavelet
        
        if level is None:
            level = self.max_level
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(time_series, wavelet, mode=self.mode, level=level)
            
            # Calculate threshold
            if threshold_level == 'universal':
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(time_series)))
            elif threshold_level == 'bayes':
                # Simplified BayesShrink
                threshold = []
                for i in range(1, len(coeffs)):
                    sigma = np.median(np.abs(coeffs[i])) / 0.6745
                    threshold.append(sigma * np.sqrt(2 * np.log(len(coeffs[i]))))
            else:
                # Default to universal
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(time_series)))
            
            # Apply thresholding
            if isinstance(threshold, list):
                # Level-dependent thresholding
                new_coeffs = [coeffs[0]]  # Keep approximation coefficients
                for i in range(1, len(coeffs)):
                    if threshold_method == 'soft':
                        new_coeffs.append(pywt.threshold(coeffs[i], threshold[i-1], mode='soft'))
                    else:
                        new_coeffs.append(pywt.threshold(coeffs[i], threshold[i-1], mode='hard'))
            else:
                # Global thresholding
                new_coeffs = [coeffs[0]]  # Keep approximation coefficients
                for i in range(1, len(coeffs)):
                    if threshold_method == 'soft':
                        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
                    else:
                        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
            
            # Reconstruct signal
            denoised = pywt.waverec(new_coeffs, wavelet, mode=self.mode)
            
            # Ensure same length as original
            denoised = denoised[:len(time_series)]
            
            logger.info(f"Denoised time series using {threshold_method} thresholding")
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error in wavelet denoising: {e}")
            raise
    
    def trend_cycle_decomposition(self, time_series: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, cycle, and noise components.
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with trend, cycle, and noise components
        """
        # Ensure time_series is a numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        try:
            # Perform wavelet decomposition
            self.decompose(time_series)
            
            # Reconstruct components
            reconstructed = self.reconstruct()
            
            # Extract trend (approximation)
            trend = reconstructed['approximation']
            
            # Extract cycles (mid-level details)
            cycle = np.zeros_like(time_series)
            for i in range(1, len(self.coefficients) - 2):
                if f'detail_{i}' in reconstructed:
                    cycle += reconstructed[f'detail_{i}'][:len(time_series)]
            
            # Extract noise (high-frequency details)
            noise = np.zeros_like(time_series)
            for i in range(max(1, len(self.coefficients) - 2), len(self.coefficients)):
                if f'detail_{i}' in reconstructed:
                    noise += reconstructed[f'detail_{i}'][:len(time_series)]
            
            # Ensure all components have the same length
            min_len = min(len(trend), len(cycle), len(noise), len(time_series))
            trend = trend[:min_len]
            cycle = cycle[:min_len]
            noise = noise[:min_len]
            
            # Prepare results
            results = {
                'trend': trend,
                'cycle': cycle,
                'noise': noise,
                'original': time_series[:min_len]
            }
            
            logger.info("Trend-cycle decomposition completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in trend-cycle decomposition: {e}")
            raise
    
    def plot_decomposition(self, time_series: np.ndarray = None, 
                          save_plot: bool = True,
                          filename: str = 'wavelet_decomposition.png') -> None:
        """
        Plot wavelet decomposition.
        
        Args:
            time_series: Original time series (optional)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.coefficients is None:
            logger.warning("No decomposition to plot. Run decompose first.")
            return
        
        if self.reconstructed is None:
            # Reconstruct components
            self.reconstruct()
        
        try:
            # Get number of components
            n_components = len(self.reconstructed)
            
            # Create figure
            fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components), sharex=True)
            
            # Ensure axes is a list
            if n_components == 1:
                axes = [axes]
            
            # Plot original time series if provided
            if time_series is not None:
                if isinstance(time_series, pd.Series):
                    time_series = time_series.values
                
                axes[0].plot(time_series, 'k-', label='Original')
                axes[0].set_title('Original Time Series',  'k-', label='Original')
                axes[0].set_title('Original Time Series')
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Plot components
            i = 1
            for name, signal in self.reconstructed.items():
                if i < len(axes):
                    axes[i].plot(signal[:len(time_series) if time_series is not None else len(signal)], 
                               'b-', label=name)
                    axes[i].set_title(f'Component: {name}')
                    axes[i].legend()
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                    i += 1
            
            # Set common labels
            plt.xlabel('Time')
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Decomposition plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting decomposition: {e}")
            raise
    
    def plot_scalogram(self, cwt_result: Dict[str, Any] = None, 
                      time_series: np.ndarray = None,
                      save_plot: bool = True,
                      filename: str = 'wavelet_scalogram.png') -> None:
        """
        Plot wavelet scalogram.
        
        Args:
            cwt_result: CWT result from continuous_wavelet_transform
            time_series: Original time series (optional)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if cwt_result is None:
            if time_series is None:
                logger.warning("No CWT result or time series provided")
                return
            
            # Perform CWT
            cwt_result = self.continuous_wavelet_transform(time_series)
        
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                          gridspec_kw={'height_ratios': [1, 3]})
            
            # Plot original time series if provided
            if time_series is not None:
                if isinstance(time_series, pd.Series):
                    time_series = time_series.values
                
                ax1.plot(time_series, 'k-')
                ax1.set_title('Original Time Series')
                ax1.grid(True, linestyle='--', alpha=0.7)
            else:
                ax1.set_visible(False)
            
            # Plot scalogram
            scales = cwt_result['scales']
            scalogram = cwt_result['scalogram']
            
            # Convert scales to periods
            periods = scales
            
            # Create time array
            times = np.arange(scalogram.shape[1])
            
            # Plot scalogram as image
            im = ax2.pcolormesh(times, periods, scalogram, cmap='jet', shading='auto')
            ax2.set_yscale('log')
            ax2.set_ylabel('Period')
            ax2.set_xlabel('Time')
            ax2.set_title('Wavelet Scalogram')
            ax2.invert_yaxis()  # Invert y-axis to have smaller periods at the top
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Power')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Scalogram plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting scalogram: {e}")
            raise
    
    def plot_global_wavelet_spectrum(self, cwt_result: Dict[str, Any] = None,
                                    time_series: np.ndarray = None,
                                    save_plot: bool = True,
                                    filename: str = 'global_wavelet_spectrum.png') -> None:
        """
        Plot global wavelet spectrum.
        
        Args:
            cwt_result: CWT result from continuous_wavelet_transform
            time_series: Original time series (optional)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if cwt_result is None:
            if time_series is None:
                logger.warning("No CWT result or time series provided")
                return
            
            # Perform CWT
            cwt_result = self.continuous_wavelet_transform(time_series)
        
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot global wavelet spectrum
            scales = cwt_result['scales']
            global_spectrum = cwt_result['global_wavelet_spectrum']
            
            # Convert scales to periods
            periods = scales
            
            plt.plot(periods, global_spectrum, 'b-')
            plt.xscale('log')
            plt.xlabel('Period')
            plt.ylabel('Power')
            plt.title('Global Wavelet Spectrum')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Find and mark peaks
            peak_indices = signal.find_peaks(global_spectrum)[0]
            for idx in peak_indices:
                if idx < len(periods):
                    plt.plot(periods[idx], global_spectrum[idx], 'ro')
                    plt.text(periods[idx], global_spectrum[idx], f' {periods[idx]:.1f}', 
                           verticalalignment='bottom')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Global wavelet spectrum plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting global wavelet spectrum: {e}")
            raise
    
    def _skewness(self, x: np.ndarray) -> float:
        """
        Calculate skewness of a distribution.
        
        Args:
            x: Input array
            
        Returns:
            Skewness value
        """
        n = len(x)
        if n < 3:
            return 0.0
        
        m3 = np.sum((x - np.mean(x))**3) / n
        m2 = np.sum((x - np.mean(x))**2) / n
        
        if m2 == 0:
            return 0.0
        
        return m3 / (m2**1.5)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """
        Calculate kurtosis of a distribution.
        
        Args:
            x: Input array
            
        Returns:
            Kurtosis value
        """
        n = len(x)
        if n < 4:
            return 0.0
        
        m4 = np.sum((x - np.mean(x))**4) / n
        m2 = np.sum((x - np.mean(x))**2) / n
        
        if m2 == 0:
            return 0.0
        
        return m4 / (m2**2) - 3  # Excess kurtosis
    
    def _entropy(self, x: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a distribution.
        
        Args:
            x: Input array
            
        Returns:
            Entropy value
        """
        # Normalize and bin the data
        x = x - np.min(x)
        if np.max(x) > 0:
            x = x / np.max(x)
        
        # Use histogram to estimate probability distribution
        hist, _ = np.histogram(x, bins=min(100, len(x)//10), density=True)
        
        # Calculate entropy
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
