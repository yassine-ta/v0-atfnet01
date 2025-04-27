#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced backtesting framework for ATFNet.
Implements walk-forward optimization, Monte Carlo simulation, and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import os
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time
import multiprocessing as mp

from utils.performance_optimizer import ParallelBacktester

logger = logging.getLogger('atfnet.backtest.advanced')


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_metrics(
        equity_curve: np.ndarray,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: Equity curve
            trades: List of trade dictionaries
            initial_capital: Initial capital
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Annualized return (assuming 252 trading days)
        n_days = len(equity_curve)
        if n_days > 1:
            annual_return = ((equity_curve[-1] / equity_curve[0]) ** (252 / n_days) - 1) * 100
        else:
            annual_return = 0.0
        
        # Volatility
        if len(returns) > 1:
            daily_vol = np.std(returns, ddof=1)
            annual_vol = daily_vol * np.sqrt(252) * 100
        else:
            daily_vol = 0.0
            annual_vol = 0.0
        
        # Sharpe ratio
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        if daily_vol > 0:
            sharpe = (np.mean(returns) - daily_rf) / daily_vol * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio (downside risk)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns, ddof=1)
            sortino = (np.mean(returns) - daily_rf) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0.0
        else:
            sortino = float('inf') if np.mean(returns) > daily_rf else 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Trade metrics
        n_trades = len(trades)
        if n_trades > 0:
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
            
            n_winning = len(winning_trades)
            n_losing = len(losing_trades)
            
            win_rate = n_winning / n_trades if n_trades > 0 else 0.0
            
            avg_profit = np.mean([t.get('profit', 0) for t in winning_trades]) if n_winning > 0 else 0.0
            avg_loss = np.mean([abs(t.get('profit', 0)) for t in losing_trades]) if n_losing > 0 else 0.0
            
            profit_factor = sum(t.get('profit', 0) for t in winning_trades) / abs(sum(t.get('profit', 0) for t in losing_trades)) if n_losing > 0 and sum(t.get('profit', 0) for t in losing_trades) != 0 else float('inf')
            
            # Average trade
            avg_trade = np.mean([t.get('profit', 0) for t in trades])
            
            # Average holding period
            holding_periods = []
            for trade in trades:
                if 'open_time' in trade and 'close_time' in trade:
                    open_time = trade['open_time']
                    close_time = trade['close_time']
                    if isinstance(open_time, str):
                        open_time = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                    if isinstance(close_time, str):
                        close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                    
                    holding_period = (close_time - open_time).total_seconds() / 3600  # in hours
                    holding_periods.append(holding_period)
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
            
            # Max consecutive wins/losses
            results = [1 if t.get('profit', 0) > 0 else -1 for t in trades]
            max_cons_wins = 0
            current_cons_wins = 0
            max_cons_losses = 0
            current_cons_losses = 0
            
            for res in results:
                if res > 0:
                    current_cons_wins += 1
                    current_cons_losses = 0
                    max_cons_wins = max(max_cons_wins, current_cons_wins)
                else:
                    current_cons_losses += 1
                    current_cons_wins = 0
                    max_cons_losses = max(max_cons_losses, current_cons_losses)
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_trade = 0.0
            avg_holding_period = 0.0
            max_cons_wins = 0
            max_cons_losses = 0
        
        # Recovery factor
        if max_drawdown > 0:
            recovery_factor = total_return / max_drawdown
        else:
            recovery_factor = float('inf') if total_return > 0 else 0.0
        
        # Regime performance
        regime_performance = {}
        for trade in trades:
            regime = trade.get('regime', 'unknown')
            if isinstance(regime, (int, np.integer)):
                regime = f"Regime_{regime}"
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0.0
                }
            
            regime_performance[regime]['count'] += 1
            if trade.get('profit', 0) > 0:
                regime_performance[regime]['wins'] += 1
            else:
                regime_performance[regime]['losses'] += 1
            
            regime_performance[regime]['profit'] += trade.get('profit', 0)
        
        # Calculate win rate and average profit for each regime
        for regime in regime_performance:
            count = regime_performance[regime]['count']
            wins = regime_performance[regime]['wins']
            
            if count > 0:
                regime_performance[regime]['win_rate'] = wins / count
                regime_performance[regime]['avg_profit'] = regime_performance[regime]['profit'] / count
            else:
                regime_performance[regime]['win_rate'] = 0.0
                regime_performance[regime]['avg_profit'] = 0.0
        
        # Compile all metrics
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'recovery_factor': recovery_factor,
            'num_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'avg_holding_period_hours': avg_holding_period,
            'max_consecutive_wins': max_cons_wins,
            'max_consecutive_losses': max_cons_losses,
            'regime_performance': regime_performance
        }
        
        return metrics
    
    @staticmethod
    def calculate_prediction_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate prediction performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of prediction metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            
            # Filter out zero changes
            valid_idx = true_direction != 0
            if np.sum(valid_idx) > 0:
                direction_accuracy = np.mean(true_direction[valid_idx] == pred_direction[valid_idx])
            else:
                direction_accuracy = 0.0
        else:
            direction_accuracy = 0.0
        
        # Correlation
        if len(y_true) > 1:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Compile metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'correlation': correlation
        }
        
        return metrics


class WalkForwardOptimizer:
    """
    Walk-forward optimization for trading strategies.
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        train_ratio: float = 0.7,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        n_jobs: int = -1,
        metric: str = 'sharpe_ratio'
    ):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            strategy_func: Strategy function to optimize
            param_grid: Parameter grid to search
            train_ratio: Ratio of training data
            window_size: Size of rolling window (if None, use expanding window)
            step_size: Step size for rolling window (if None, use window_size)
            n_jobs: Number of parallel jobs
            metric: Metric to optimize
        """
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.train_ratio = train_ratio
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.metric = metric
        
        # Initialize parallel backtester
        self.parallel_backtester = ParallelBacktester(n_processes=self.n_jobs)
        
        # Results
        self.results = []
        self.best_params = []
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from param_grid.
        
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get all parameter names and values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_dicts = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def optimize(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        min_train_size: int = 100,
        min_test_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward optimization.
        
        Args:
            data: Historical data
            date_column: Date column name
            min_train_size: Minimum training set size
            min_test_size: Minimum test set size
            
        Returns:
            List of optimization results
        """
        # Ensure data is sorted by date
        if date_column in data.columns:
            data = data.sort_values(date_column)
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Determine window parameters
        n_samples = len(data)
        
        if self.window_size is None:
            # Expanding window
            windows = []
            
            # Initial window
            train_size = int(min_train_size)
            test_size = min(min_test_size, n_samples - train_size)
            
            while train_size + test_size <= n_samples:
                train_end = train_size
                test_end = train_size + test_size
                
                windows.append((0, train_end, train_end, test_end))
                
                # Expand window
                train_size += self.step_size
                test_size = min(min_test_size, n_samples - train_size)
        else:
            # Rolling window
            windows = []
            
            # Initial window
            train_size = max(min_train_size, self.window_size)
            test_size = min(min_test_size, n_samples - train_size)
            
            start = 0
            while start + train_size + test_size <= n_samples:
                train_end = start + train_size
                test_end = train_end + test_size
                
                windows.append((start, train_end, train_end, test_end))
                
                # Roll window
                start += self.step_size
        
        logger.info(f"Created {len(windows)} windows for walk-forward optimization")
        
        # Perform optimization for each window
        self.results = []
        self.best_params = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Optimizing window {i+1}/{len(windows)}: "
                       f"Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Prepare shared data for parallel backtests
            shared_data = {
                'train_data': train_data,
                'test_data': test_data
            }
            
            # Run parallel backtests
            backtest_results = self.parallel_backtester.run_parallel_backtest(
                self.strategy_func,
                param_combinations,
                shared_data
            )
            
            # Find best parameters
            best_result = None
            best_params = None
            best_metric_value = float('-inf')
            
            for result, params in zip(backtest_results, param_combinations):
                if 'error' in result:
                    logger.warning(f"Error in backtest: {result['error']}")
                    continue
                
                # Get metric value
                if self.metric in result:
                    metric_value = result[self.metric]
                elif 'metrics' in result and self.metric in result['metrics']:
                    metric_value = result['metrics'][self.metric]
                else:
                    logger.warning(f"Metric {self.metric} not found in results")
                    continue
                
                # Update best parameters
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = result
                    best_params = params
            
            if best_params is not None:
                logger.info(f"Best parameters for window {i+1}: {best_params} "
                           f"({self.metric} = {best_metric_value:.4f})")
                
                # Store results
                window_result = {
                    'window': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_params': best_params,
                    'best_metric': best_metric_value,
                    'test_results': best_result
                }
                
                self.results.append(window_result)
                self.best_params.append(best_params)
            else:
                logger.warning(f"No valid results for window {i+1}")
        
        return self.results
    
    def get_robust_parameters(self) -> Dict[str, Any]:
        """
        Get robust parameters across all windows.
        
        Returns:
            Dictionary of robust parameters
        """
        if not self.best_params:
            raise ValueError("No optimization results available")
        
        # Get all parameter names
        param_names = list(self.best_params[0].keys())
        
        # Calculate median/mode for each parameter
        robust_params = {}
        
        for param in param_names:
            values = [p[param] for p in self.best_params]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical parameter - use median
                robust_params[param] = float(np.median(values))
                
                # Convert to int if all values are int
                if all(isinstance(v, int) for v in values):
                    robust_params[param] = int(robust_params[param])
            else:
                # Categorical parameter - use mode
                from scipy import stats
                mode_result = stats.mode(values)
                robust_params[param] = mode_result.mode[0]
        
        return robust_params
    
    def plot_parameter_stability(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot parameter stability across windows.
        
        Args:
            figsize: Figure size
        """
        if not self.best_params:
            raise ValueError("No optimization results available")
        
        # Get all parameter names
        param_names = list(self.best_params[0].keys())
        
        # Create figure
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
        
        if n_params == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, param in enumerate(param_names):
            values = [p[param] for p in self.best_params]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical parameter - line plot
                axes[i].plot(values, marker='o')
                
                # Add median line
                median = np.median(values)
                axes[i].axhline(median, color='r', linestyle='--', alpha=0.7)
                axes[i].text(0, median, f' Median: {median}', verticalalignment='bottom')
            else:
                # Categorical parameter - scatter plot with jitter
                unique_values = list(set(values))
                value_map = {val: i for i, val in enumerate(unique_values)}
                numeric_values = [value_map[val] for val in values]
                
                axes[i].scatter(range(len(numeric_values)), numeric_values)
                
                # Set y-ticks
                axes[i].set_yticks(range(len(unique_values)))
                axes[i].set_yticklabels(unique_values)
            
            axes[i].set_ylabel(param)
            axes[i].grid(True, alpha=0.3)
        
        # Set x-axis
        axes[-1].set_xlabel('Window')
        axes[-1].set_xticks(range(len(self.best_params)))
        
        plt.tight_layout()
        plt.show()
    
    def plot_metric_performance(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot metric performance across windows.
        
        Args:
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No optimization results available")
        
        # Extract metric values
        metric_values = [r['best_metric'] for r in self.results]
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(metric_values, marker='o')
        plt.axhline(np.mean(metric_values), color='r', linestyle='--', alpha=0.7)
        plt.text(0, np.mean(metric_values), f' Mean: {np.mean(metric_values):.4f}', verticalalignment='bottom')
        
        plt.xlabel('Window')
        plt.ylabel(self.metric)
        plt.title(f'{self.metric} Performance Across Windows')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategies.
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations
            confidence_level: Confidence level for intervals
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Results
        self.simulation_results = None
        self.confidence_intervals = None
    
    def simulate_returns(
        self,
        returns: np.ndarray,
        n_periods: int,
        initial_capital: float = 10000.0
    ) -> np.ndarray:
        """
        Simulate future returns using bootstrap resampling.
        
        Args:
            returns: Historical returns
            n_periods: Number of periods to simulate
            initial_capital: Initial capital
            
        Returns:
            Array of simulated equity curves
        """
        # Initialize simulations
        simulations = np.zeros((self.n_simulations, n_periods + 1))
        simulations[:, 0] = initial_capital
        
        # Run simulations
        for i in range(self.n_simulations):
            # Resample returns with replacement
            sim_returns = np.random.choice(returns, size=n_periods, replace=True)
            
            # Calculate equity curve
            for j in range(n_periods):
                simulations[i, j+1] = simulations[i, j] * (1 + sim_returns[j])
        
        self.simulation_results = simulations
        return simulations
    
    def simulate_trades(
        self,
        trades: List[Dict[str, Any]],
        n_trades: int,
        initial_capital: float = 10000.0
    ) -> np.ndarray:
        """
        Simulate future trades using bootstrap resampling.
        
        Args:
            trades: Historical trades
            n_trades: Number of trades to simulate
            initial_capital: Initial capital
            
        Returns:
            Array of simulated equity curves
        """
        # Extract profits from trades
        profits = np.array([t.get('profit', 0) for t in trades])
        
        # Initialize simulations
        simulations = np.zeros((self.n_simulations, n_trades + 1))
        simulations[:, 0] = initial_capital
        
        # Run simulations
        for i in range(self.n_simulations):
            # Resample profits with replacement
            sim_profits = np.random.choice(profits, size=n_trades, replace=True)
            
            # Calculate equity curve
            for j in range(n_trades):
                simulations[i, j+1] = simulations[i, j] + sim_profits[j]
        
        self.simulation_results = simulations
        return simulations
    
    def calculate_confidence_intervals(self) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for simulations.
        
        Returns:
            Dictionary of confidence intervals
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available")
        
        # Calculate percentiles
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        
        # Calculate confidence intervals
        lower_bound = np.percentile(self.simulation_results, lower_percentile, axis=0)
        median = np.percentile(self.simulation_results, 50, axis=0)
        upper_bound = np.percentile(self.simulation_results, upper_percentile, axis=0)
        
        self.confidence_intervals = {
            'lower': lower_bound,
            'median': median,
            'upper': upper_bound
        }
        
        return self.confidence_intervals
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate risk metrics from simulations.
        
        Returns:
            Dictionary of risk metrics
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available")
        
        # Calculate final values
        final_values = self.simulation_results[:, -1]
        initial_capital = self.simulation_results[0, 0]
        
        # Calculate returns
        returns = (final_values - initial_capital) / initial_capital
        
        # Calculate metrics
        metrics = {
            'mean_return': np.mean(returns) * 100,
            'median_return': np.median(returns) * 100,
            'std_return': np.std(returns) * 100,
            'var_95': np.percentile(returns, 5) * 100,  # 95% VaR
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]) * 100,  # 95% CVaR
            'probability_profit': np.mean(returns > 0) * 100,
            'probability_loss': np.mean(returns < 0) * 100
        }
        
        # Calculate maximum drawdowns
        max_drawdowns = []
        for sim in self.simulation_results:
            peak = np.maximum.accumulate(sim)
            drawdown = (peak - sim) / peak
            max_drawdowns.append(np.max(drawdown) * 100)
        
        metrics['mean_max_drawdown'] = np.mean(max_drawdowns)
        metrics['median_max_drawdown'] = np.median(max_drawdowns)
        metrics['worst_max_drawdown'] = np.max(max_drawdowns)
        
        return metrics
    
    def plot_simulations(
        self,
        figsize: Tuple[int, int] = (10, 6),
        alpha: float = 0.1,
        n_samples: Optional[int] = None
    ) -> None:
        """
        Plot Monte Carlo simulations.
        
        Args:
            figsize: Figure size
            alpha: Alpha for simulation lines
            n_samples: Number of sample simulations to plot (if None, plot all)
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available")
        
        if self.confidence_intervals is None:
            self.calculate_confidence_intervals()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot sample simulations
        if n_samples is None:
            n_samples = self.n_simulations
        
        for i in range(min(n_samples, self.n_simulations)):
            plt.plot(self.simulation_results[i], color='blue', alpha=alpha)
        
        # Plot confidence intervals
        plt.plot(self.confidence_intervals['median'], color='red', linewidth=2, label='Median')
        plt.plot(self.confidence_intervals['lower'], color='red', linestyle='--', linewidth=2, 
                 label=f'{self.confidence_level*100:.0f}% Confidence Interval')
        plt.plot(self.confidence_intervals['upper'], color='red', linestyle='--', linewidth=2)
        
        plt.xlabel('Period')
        plt.ylabel('Equity')
        plt.title('Monte Carlo Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_histogram(
        self,
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 50
    ) -> None:
        """
        Plot histogram of final values.
        
        Args:
            figsize: Figure size
            bins: Number of histogram bins
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available")
        
        # Calculate final values
        final_values = self.simulation_results[:, -1]
        initial_capital = self.simulation_results[0, 0]
        
        # Calculate returns
        returns = (final_values - initial_capital) / initial_capital * 100
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(returns, bins=bins, alpha=0.7)
        
        # Add vertical lines for statistics
        plt.axvline(np.mean(returns), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        plt.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.2f}%')
        plt.axvline(np.percentile(returns, 5), color='orange', linestyle='-.', linewidth=2, label=f'5th Percentile: {np.percentile(returns, 5):.2f}%')
        
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Simulated Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class AdvancedBacktester:
    """
    Advanced backtester with comprehensive analysis capabilities.
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
        transaction_costs: float = 0.0
    ):
        """
        Initialize the advanced backtester.
        
        Args:
            strategy_func: Strategy function
            initial_capital: Initial capital
            risk_free_rate: Risk-free rate (annualized)
            transaction_costs: Transaction costs as percentage
        """
        self.strategy_func = strategy_func
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        
        # Results
        self.equity_curve = None
        self.trades = None
        self.metrics = None
        self.regime_performance = None
    
    def run(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        regime_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            data: Historical data
            params: Strategy parameters
            regime_labels: Regime labels
            
        Returns:
            Backtest results
        """
        # Run strategy
        strategy_results = self.strategy_func(data, params)
        
        # Extract results
        self.equity_curve = strategy_results.get('equity_curve')
        self.trades = strategy_results.get('trades', [])
        
        # Apply transaction costs
        if self.transaction_costs > 0 and self.trades:
            for trade in self.trades:
                cost = trade.get('size', 1.0) * trade.get('open_price', 0) * self.transaction_costs
                trade['profit'] = trade.get('profit', 0) - cost
            
            # Recalculate equity curve
            if self.equity_curve is not None:
                total_costs = sum(trade.get('size', 1.0) * trade.get('open_price', 0) * self.transaction_costs for trade in self.trades)
                self.equity_curve = self.equity_curve - np.linspace(0, total_costs, len(self.equity_curve))
        
        # Add regime labels to trades
        if regime_labels is not None and len(regime_labels) == len(data):
            for trade in self.trades:
                open_idx = trade.get('open_idx')
                if open_idx is not None and 0 <= open_idx < len(regime_labels):
                    trade['regime'] = regime_labels[open_idx]
        
        # Calculate performance metrics
        self.metrics = PerformanceMetrics.calculate_metrics(
            self.equity_curve,
            self.trades,
            self.initial_capital,
            self.risk_free_rate
        )
        
        # Extract regime performance
        self.regime_performance = self.metrics.get('regime_performance', {})
        
        # Prepare results
        results = {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'metrics': self.metrics,
            'regime_performance': self.regime_performance
        }
        
        return results
    
    def run_monte_carlo(
        self,
        n_simulations: int = 1000,
        n_periods: Optional[int] = None,
        confidence_level: float = 0.95,
        method: str = 'returns'
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_simulations: Number of simulations
            n_periods: Number of periods to simulate (if None, use same as backtest)
            confidence_level: Confidence level for intervals
            method: Simulation method ('returns' or 'trades')
            
        Returns:
            Monte Carlo results
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        # Create simulator
        simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            confidence_level=confidence_level
        )
        
        if method == 'returns':
            # Calculate returns
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            # Set number of periods
            if n_periods is None:
                n_periods = len(returns)
            
            # Run simulation
            simulations = simulator.simulate_returns(
                returns,
                n_periods,
                self.initial_capital
            )
        else:  # trades
            if not self.trades:
                raise ValueError("No trades available for simulation")
            
            # Set number of trades
            if n_periods is None:
                n_periods = len(self.trades)
            
            # Run simulation
            simulations = simulator.simulate_trades(
                self.trades,
                n_periods,
                self.initial_capital
            )
        
        # Calculate confidence intervals
        confidence_intervals = simulator.calculate_confidence_intervals()
        
        # Calculate risk metrics
        risk_metrics = simulator.calculate_risk_metrics()
        
        # Prepare results
        results = {
            'simulations': simulations,
            'confidence_intervals': confidence_intervals,
            'risk_metrics': risk_metrics,
            'simulator': simulator
        }
        
        return results
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot equity curve.
        
        Args:
            figsize: Figure size
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot equity curve
        plt.plot(self.equity_curve, linewidth=2)
        
        # Add horizontal line for initial capital
        plt.axhline(self.initial_capital, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel('Period')
        plt.ylabel('Equity')
        plt.title('Equity Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot drawdown.
        
        Args:
            figsize: Figure size
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        # Calculate drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak * 100
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot drawdown
        plt.plot(drawdown, linewidth=2)
        
        # Add horizontal line for maximum drawdown
        max_dd = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        plt.axhline(max_dd, color='red', linestyle='--', alpha=0.7)
        plt.text(0, max_dd, f' Max Drawdown: {max_dd:.2f}%', verticalalignment='bottom')
        
        plt.xlabel('Period')
        plt.ylabel('Drawdown (%)')
        plt.title('Drawdown')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regime_performance(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot regime performance.
        
        Args:
            figsize: Figure size
        """
        if not self.regime_performance:
            raise ValueError("No regime performance data available")
        
        # Extract data
        regimes = list(self.regime_performance.keys())
        win_rates = [self.regime_performance[r].get('win_rate', 0) * 100 for r in regimes]
        avg_profits = [self.regime_performance[r].get('avg_profit', 0) for r in regimes]
        counts = [self.regime_performance[r].get('count', 0) for r in regimes]
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot win rate
        color = 'tab:blue'
        ax1.set_xlabel('Regime')
        ax1.set_ylabel('Win Rate (%)', color=color)
        ax1.bar(regimes, win_rates, alpha=0.7, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis
        ax2 = ax1.twinx()
        
        # Plot average profit
        color = 'tab:red'
        ax2.set_ylabel('Average Profit', color=color)
        ax2.plot(regimes, avg_profits, 'o-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add count as text
        for i, count in enumerate(counts):
            ax1.text(i, 5, f'n={count}', ha='center')
        
        plt.title('Performance by Market Regime')
        fig.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, dates: pd.Series, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            dates: Date series corresponding to equity curve
            figsize: Figure size
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        if len(dates) != len(self.equity_curve):
            raise ValueError("Length of dates must match equity curve")
        
        # Calculate daily returns
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Create DataFrame with dates and returns
        returns_df = pd.DataFrame({
            'date': dates[1:],
            'return': daily_returns
        })
        
        # Extract year and month
        returns_df['year'] = returns_df['date'].dt.year
        returns_df['month'] = returns_df['date'].dt.month
        
        # Calculate monthly returns
        monthly_returns = returns_df.groupby(['year', 'month'])['return'].sum().unstack()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        
        plt.title('Monthly Returns (%)')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, output_dir: str = 'backtest_reports') -> str:
        """
        Generate HTML backtest report.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to generated report
        """
        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'backtest_report_{timestamp}.html')
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-good {{ color: green; }}
                .metric-bad {{ color: red; }}
                .container {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="container">
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Return</td>
                        <td class="{'metric-good' if self.metrics['total_return'] > 0 else 'metric-bad'}">{self.metrics['total_return']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Annual Return</td>
                        <td class="{'metric-good' if self.metrics['annual_return'] > 0 else 'metric-bad'}">{self.metrics['annual_return']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td class="{'metric-good' if self.metrics['sharpe_ratio'] > 1 else 'metric-bad'}">{self.metrics['sharpe_ratio']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td class="{'metric-good' if self.metrics['max_drawdown'] < 20 else 'metric-bad'}">{self.metrics['max_drawdown']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td class="{'metric-good' if self.metrics['win_rate'] > 0.5 else 'metric-bad'}">{self.metrics['win_rate']*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td class="{'metric-good' if self.metrics['profit_factor'] > 1 else 'metric-bad'}">{self.metrics['profit_factor']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Number of Trades</td>
                        <td>{self.metrics['num_trades']}</td>
                    </tr>
                </table>
            </div>
            
            <!-- Add more sections as needed -->
            
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated backtest report: {report_file}")
        
        return report_file
