#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multifractal analysis module for ATFNet.
Implements multifractal detrended fluctuation analysis (MFDFA) for market regime detection.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import signal
from scipy.stats import linregress

logger = logging.getLogger('atfnet.core.multifractal')


class MultifractalAnalyzer:
    """
    Implements multifractal analysis techniques for financial time series.
    Uses multifractal detrended fluctuation analysis (MFDFA) to detect market regimes.
    """
    
    def __init__(self, output_dir: str = 'multifractal_analysis'):
        """
        Initialize the multifractal analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameters
        self.scales = np.logspace(1, 3, 20).astype(int)  # Scales for analysis
        self.q_list = np.linspace(-5, 5, 21)  # List of q values
        
        # Results storage
        self.hurst_exponent = None
        self.multifractal_spectrum = None
        self.singularity_spectrum = None
        self.regime_history = []
    
    def compute_mfdfa(self, time_series: np.ndarray, scales: np.ndarray = None, 
                     q_list: np.ndarray = None, order: int = 1) -> Dict[str, Any]:
        """
        Compute Multifractal Detrended Fluctuation Analysis.
        
        Args:
            time_series: Input time series
            scales: Scales for analysis (window sizes)
            q_list: List of q values for multifractal analysis
            order: Order of polynomial fit for detrending
            
        Returns:
            Dict containing MFDFA results
        """
        if scales is None:
            scales = self.scales
        
        if q_list is None:
            q_list = self.q_list
        
        try:
            # Convert to numpy array if needed
            if isinstance(time_series, pd.Series):
                time_series = time_series.values
            
            # Ensure minimum length
            if len(time_series) < max(scales) * 2:
                raise ValueError(f"Time series too short for selected scales. Need at least {max(scales) * 2} points.")
            
            # Calculate profile (cumulative sum of deviations from mean)
            profile = np.cumsum(time_series - np.mean(time_series))
            
            # Initialize fluctuation function
            F_q = np.zeros((len(scales), len(q_list)))
            
            # Calculate fluctuation function for each scale
            for i, scale in enumerate(scales):
                # Number of non-overlapping segments
                n_segments = int(len(profile) / scale)
                
                # Skip if too few segments
                if n_segments < 4:
                    continue
                
                # Initialize fluctuation for this scale
                fluctuation = np.zeros(2 * n_segments)
                
                # Forward direction
                for j in range(n_segments):
                    segment = profile[j * scale:(j + 1) * scale]
                    # Fit polynomial and calculate residuals
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, order)
                    fit = np.polyval(coeffs, x)
                    # Calculate variance of residuals
                    fluctuation[j] = np.sqrt(np.mean((segment - fit) ** 2))
                
                # Backward direction
                for j in range(n_segments):
                    segment = profile[-(j + 1) * scale:-j * scale if j > 0 else None]
                    # Fit polynomial and calculate residuals
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, order)
                    fit = np.polyval(coeffs, x)
                    # Calculate variance of residuals
                    fluctuation[n_segments + j] = np.sqrt(np.mean((segment - fit) ** 2))
                
                # Calculate q-order fluctuation function
                for k, q in enumerate(q_list):
                    if q == 0:
                        # Special case for q=0
                        F_q[i, k] = np.exp(0.5 * np.mean(np.log(fluctuation ** 2)))
                    else:
                        F_q[i, k] = np.mean(fluctuation ** q) ** (1 / q)
            
            # Calculate scaling exponents (Hurst exponents)
            H_q = np.zeros(len(q_list))
            R_squared = np.zeros(len(q_list))
            
            for k, q in enumerate(q_list):
                # Linear regression in log-log space
                valid_idx = F_q[:, k] > 0
                if np.sum(valid_idx) > 2:  # Need at least 3 points for regression
                    log_scales = np.log(scales[valid_idx])
                    log_F_q = np.log(F_q[valid_idx, k])
                    slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_F_q)
                    H_q[k] = slope
                    R_squared[k] = r_value ** 2
            
            # Calculate mass exponent tau(q)
            tau_q = q_list * H_q - 1
            
            # Calculate singularity spectrum f(alpha)
            alpha = np.gradient(tau_q, q_list)  # Singularity strength
            f_alpha = q_list * alpha - tau_q    # Singularity spectrum
            
            # Store results
            self.hurst_exponent = H_q[np.where(q_list == 2)[0][0]]  # H(2) is the standard Hurst exponent
            self.multifractal_spectrum = {'q': q_list, 'H_q': H_q, 'tau_q': tau_q, 'R_squared': R_squared}
            self.singularity_spectrum = {'alpha': alpha, 'f_alpha': f_alpha}
            
            # Calculate multifractality strength
            multifractality = np.max(H_q) - np.min(H_q)
            
            # Calculate spectrum width
            spectrum_width = np.max(alpha) - np.min(alpha)
            
            results = {
                'hurst_exponent': self.hurst_exponent,
                'multifractality': multifractality,
                'spectrum_width': spectrum_width,
                'multifractal_spectrum': self.multifractal_spectrum,
                'singularity_spectrum': self.singularity_spectrum,
                'scales': scales,
                'q_list': q_list
            }
            
            logger.info(f"MFDFA computed: Hurst={self.hurst_exponent:.3f}, Multifractality={multifractality:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing MFDFA: {e}")
            raise
    
    def plot_results(self, save_plots: bool = True, filename_prefix: str = 'mfdfa'):
        """
        Plot MFDFA results.
        
        Args:
            save_plots: Whether to save plots
            filename_prefix: Prefix for saved files
        """
        if self.multifractal_spectrum is None or self.singularity_spectrum is None:
            raise ValueError("No MFDFA results available. Run compute_mfdfa first.")
        
        try:
            # Plot generalized Hurst exponents
            plt.figure(figsize=(10, 6))
            plt.plot(self.multifractal_spectrum['q'], self.multifractal_spectrum['H_q'], 'o-')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Walk (H=0.5)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('q')
            plt.ylabel('H(q)')
            plt.title('Generalized Hurst Exponents')
            plt.legend()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_hurst.png"), 
                           bbox_inches='tight', dpi=300)
            
            plt.close()
            
            # Plot mass exponent
            plt.figure(figsize=(10, 6))
            plt.plot(self.multifractal_spectrum['q'], self.multifractal_spectrum['tau_q'], 'o-')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('q')
            plt.ylabel('τ(q)')
            plt.title('Mass Exponent')
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_tau.png"), 
                           bbox_inches='tight', dpi=300)
            
            plt.close()
            
            # Plot singularity spectrum
            plt.figure(figsize=(10, 6))
            plt.plot(self.singularity_spectrum['alpha'], self.singularity_spectrum['f_alpha'], 'o-')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('α')
            plt.ylabel('f(α)')
            plt.title('Multifractal Singularity Spectrum')
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_spectrum.png"), 
                           bbox_inches='tight', dpi=300)
            
            plt.close()
            
            logger.info(f"MFDFA plots saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting MFDFA results: {e}")
            raise
    
    def detect_regime(self, time_series: np.ndarray, window_size: int = 500, 
                     overlap: int = 100, threshold_persistent: float = 0.65,
                     threshold_antipersistent: float = 0.45,
                     threshold_multifractal: float = 0.5) -> Dict[str, Any]:
        """
        Detect market regime based on multifractal properties.
        
        Args:
            time_series: Input time series
            window_size: Size of sliding window
            overlap: Overlap between consecutive windows
            threshold_persistent: Threshold for persistent regime
            threshold_antipersistent: Threshold for anti-persistent regime
            threshold_multifractal: Threshold for multifractal strength
            
        Returns:
            Dict containing regime detection results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(time_series, pd.Series):
                time_series = time_series.values
            
            # Ensure minimum length
            if len(time_series) < window_size:
                raise ValueError(f"Time series too short for window size {window_size}")
            
            # Initialize results
            step_size = window_size - overlap
            n_windows = max(1, int((len(time_series) - window_size) / step_size) + 1)
            
            hurst_values = np.zeros(n_windows)
            multifractality_values = np.zeros(n_windows)
            regimes = np.zeros(n_windows, dtype=int)
            timestamps = []
            
            # Define regime codes
            REGIME_RANDOM_WALK = 0
            REGIME_PERSISTENT = 1
            REGIME_ANTIPERSISTENT = 2
            REGIME_MULTIFRACTAL = 3
            
            # Process each window
            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = min(start_idx + window_size, len(time_series))
                
                if end_idx - start_idx < window_size / 2:
                    # Skip if window is too small
                    continue
                
                # Extract window
                window = time_series[start_idx:end_idx]
                
                # Compute MFDFA for this window
                results = self.compute_mfdfa(window)
                
                # Store results
                hurst_values[i] = results['hurst_exponent']
                multifractality_values[i] = results['multifractality']
                
                # Determine regime
                if multifractality_values[i] > threshold_multifractal:
                    regimes[i] = REGIME_MULTIFRACTAL
                elif hurst_values[i] > threshold_persistent:
                    regimes[i] = REGIME_PERSISTENT
                elif hurst_values[i] < threshold_antipersistent:
                    regimes[i] = REGIME_ANTIPERSISTENT
                else:
                    regimes[i] = REGIME_RANDOM_WALK
                
                # Store timestamp for this window
                if isinstance(time_series, pd.Series) and hasattr(time_series, 'index'):
                    timestamps.append(time_series.index[start_idx])
                else:
                    timestamps.append(start_idx)
            
            # Store regime history
            self.regime_history.append({
                'timestamps': timestamps,
                'hurst_values': hurst_values,
                'multifractality_values': multifractality_values,
                'regimes': regimes
            })
            
            # Determine current regime (most recent window)
            current_regime = regimes[-1]
            
            # Map regime code to description
            regime_descriptions = {
                REGIME_RANDOM_WALK: "Random Walk (Efficient Market)",
                REGIME_PERSISTENT: "Persistent (Trending Market)",
                REGIME_ANTIPERSISTENT: "Anti-Persistent (Mean-Reverting Market)",
                REGIME_MULTIFRACTAL: "Multifractal (Complex, Volatile Market)"
            }
            
            current_regime_desc = regime_descriptions.get(current_regime, "Unknown")
            
            # Calculate regime statistics
            regime_stats = {
                'regime_counts': {desc: np.sum(regimes == code) for code, desc in regime_descriptions.items()},
                'regime_percentages': {desc: np.sum(regimes == code) / len(regimes) * 100 
                                      for code, desc in regime_descriptions.items()},
                'mean_hurst': np.mean(hurst_values),
                'mean_multifractality': np.mean(multifractality_values)
            }
            
            # Prepare results
            results = {
                'current_regime_code': current_regime,
                'current_regime': current_regime_desc,
                'hurst_exponent': hurst_values[-1],
                'multifractality': multifractality_values[-1],
                'regime_history': {
                    'timestamps': timestamps,
                    'hurst_values': hurst_values.tolist(),
                    'multifractality_values': multifractality_values.tolist(),
                    'regimes': regimes.tolist()
                },
                'regime_stats': regime_stats
            }
            
            logger.info(f"Regime detection: {current_regime_desc} (H={hurst_values[-1]:.3f}, M={multifractality_values[-1]:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            raise
    
    def plot_regime_history(self, save_plot: bool = True, filename: str = 'regime_history.png'):
        """
        Plot regime history.
        
        Args:
            save_plot: Whether to save plot
            filename: Filename for saved plot
        """
        if not self.regime_history:
            raise ValueError("No regime history available. Run detect_regime first.")
        
        try:
            # Get most recent history
            history = self.regime_history[-1]
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot Hurst exponent
            ax1.plot(history['timestamps'], history['hurst_values'], 'o-')
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Walk (H=0.5)')
            ax1.set_ylabel('Hurst Exponent')
            ax1.set_title('Market Regime Analysis')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Plot multifractality
            ax2.plot(history['timestamps'], history['multifractality_values'], 'o-')
            ax2.set_ylabel('Multifractality')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Plot regimes
            regime_colors = {
                0: 'gray',    # Random Walk
                1: 'green',   # Persistent
                2: 'blue',    # Anti-Persistent
                3: 'red'      # Multifractal
            }
            
            regime_labels = {
                0: 'Random Walk',
                1: 'Persistent (Trending)',
                2: 'Anti-Persistent (Mean-Reverting)',
                3: 'Multifractal (Complex)'
            }
            
            # Create colored background for regimes
            for i, regime in enumerate(history['regimes']):
                if i < len(history['timestamps']) - 1:
                    ax3.axvspan(history['timestamps'][i], history['timestamps'][i+1], 
                               alpha=0.3, color=regime_colors.get(regime, 'gray'))
                else:
                    # For the last point, extend a bit
                    if len(history['timestamps']) > 1:
                        width = history['timestamps'][-1] - history['timestamps'][-2]
                        ax3.axvspan(history['timestamps'][-1], history['timestamps'][-1] + width, 
                                   alpha=0.3, color=regime_colors.get(regime, 'gray'))
            
            # Add regime labels
            handles = [plt.Rectangle((0,0), 1, 1, color=color, alpha=0.3) for color in regime_colors.values()]
            ax3.legend(handles, regime_labels.values(), loc='upper right')
            
            ax3.set_ylabel('Regime')
            ax3.set_xlabel('Time')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Regime history plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting regime history: {e}")
            raise
    
    def get_trading_recommendations(self, current_regime: int) -> Dict[str, Any]:
        """
        Get trading recommendations based on detected regime.
        
        Args:
            current_regime: Current market regime code
            
        Returns:
            Dict containing trading recommendations
        """
        # Define regime-specific recommendations
        recommendations = {
            0: {  # Random Walk
                'description': 'Market is following a random walk (efficient market)',
                'strategy': 'Consider passive strategies or high-frequency statistical arbitrage',
                'position_sizing': 'Reduce position sizes',
                'stop_loss': 'Moderate stop losses',
                'take_profit': 'Moderate take profits',
                'timeframe': 'Short timeframes may be more effective',
                'indicators': 'Focus on volatility-based indicators'
            },
            1: {  # Persistent (Trending)
                'description': 'Market shows persistence (trending behavior)',
                'strategy': 'Trend-following strategies are optimal',
                'position_sizing': 'Can increase position sizes',
                'stop_loss': 'Wider stop losses to avoid noise',
                'take_profit': 'Let profits run with trailing stops',
                'timeframe': 'Longer timeframes capture the trend better',
                'indicators': 'Moving averages, MACD, ADX'
            },
            2: {  # Anti-Persistent (Mean-Reverting)
                'description': 'Market shows anti-persistence (mean-reverting behavior)',
                'strategy': 'Mean-reversion strategies are optimal',
                'position_sizing': 'Moderate position sizes',
                'stop_loss': 'Tighter stop losses',
                'take_profit': 'Take profits at resistance/support levels',
                'timeframe': 'Medium timeframes balance noise and opportunity',
                'indicators': 'RSI, Bollinger Bands, Stochastics'
            },
            3: {  # Multifractal (Complex)
                'description': 'Market shows complex multifractal behavior',
                'strategy': 'Adaptive strategies that combine trend and mean-reversion',
                'position_sizing': 'Reduce position sizes due to unpredictability',
                'stop_loss': 'Tighter stop losses to manage increased risk',
                'take_profit': 'Multiple take profit levels',
                'timeframe': 'Multi-timeframe analysis is crucial',
                'indicators': 'Combine volatility, trend, and momentum indicators'
            }
        }
        
        # Get recommendations for current regime
        if current_regime in recommendations:
            return recommendations[current_regime]
        else:
            return recommendations[0]  # Default to random walk recommendations
