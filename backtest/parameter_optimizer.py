#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter optimizer for ATFNet backtesting.
This module implements functionality to optimize strategy parameters through backtesting.
"""

import os
import sys
import logging
import yaml
import json
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import multiprocessing
from functools import partial
import time
import copy
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from strategies import ATFNetStrategy
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logger = logging.getLogger('atfnet.parameter_optimizer')


class ParameterOptimizationError(Exception):
    """Exception raised for errors in the parameter optimization process."""
    pass


class ParameterOptimizer:
    """Parameter optimizer for ATFNet backtesting."""
    
    def __init__(self, strategy: Any, param_ranges_path: Optional[str] = None) -> None:
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy: ATFNet strategy instance
            param_ranges_path: Path to parameter ranges configuration file.
                              Defaults to None.
                              
        Raises:
            FileNotFoundError: If the parameter ranges file is specified but not found
            ValueError: If the parameter ranges file is invalid
        """
        self.strategy = strategy
        self.param_ranges: Dict[str, Any] = {}
        
        # Load parameter ranges
        if param_ranges_path is None:
            param_ranges_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'parameter_ranges.yaml'
            )
        
        if os.path.exists(param_ranges_path):
            try:
                with open(param_ranges_path, 'r') as f:
                    self.param_ranges = yaml.safe_load(f)
                logger.info(f"Parameter ranges loaded from {param_ranges_path}")
            except Exception as e:
                logger.error(f"Failed to load parameter ranges: {e}")
                raise ValueError(f"Failed to load parameter ranges: {e}")
        else:
            logger.warning(f"Parameter ranges file not found: {param_ranges_path}")
            if param_ranges_path is not None:
                raise FileNotFoundError(f"Parameter ranges file not found: {param_ranges_path}")
    
    def generate_parameter_combinations(
        self, 
        param_subset: Optional[List[str]] = None, 
        max_combinations: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for optimization.
        
        Args:
            param_subset: List of parameter paths to include in optimization.
                         Example: ['atfnet.sequence_length', 'attention.attention_threshold']
                         Defaults to None (use all parameters).
            max_combinations: Maximum number of combinations to generate.
                            Defaults to 100.
        
        Returns:
            List of parameter dictionaries for testing
            
        Raises:
            ValueError: If no valid parameter ranges are found
        """
        try:
            # Flatten parameter ranges
            flat_params = self._flatten_param_ranges()
            
            if not flat_params:
                raise ValueError("No valid parameter ranges found")
            
            # Filter parameters if subset is specified
            if param_subset is not None:
                flat_params = {k: v for k, v in flat_params.items() if k in param_subset}
                
                if not flat_params:
                    raise ValueError(f"No valid parameters found in subset: {param_subset}")
            
            # Generate parameter values
            param_values: Dict[str, List[Any]] = {}
            for param_name, param_range in flat_params.items():
                min_val = param_range['min']
                max_val = param_range['max']
                step = param_range['step']
                
                # Generate values based on parameter type
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    values = list(range(min_val, max_val + 1, step))
                else:
                    # For float parameters
                    values = []
                    val = min_val
                    while val <= max_val:
                        values.append(val)
                        val += step
                        # Avoid floating point precision issues
                        val = round(val, 6)
                
                param_values[param_name] = values
            
            # Calculate total combinations
            total_combinations = 1
            for values in param_values.values():
                total_combinations *= len(values)
            
            logger.info(f"Total possible parameter combinations: {total_combinations}")
            
            # If too many combinations, use random sampling
            if total_combinations > max_combinations:
                logger.info(f"Limiting to {max_combinations} random combinations")
                return self._generate_random_combinations(param_values, max_combinations)
            
            # Generate all combinations
            param_names = list(param_values.keys())
            param_value_lists = [param_values[name] for name in param_names]
            
            combinations = []
            for values in itertools.product(*param_value_lists):
                param_dict: Dict[str, Any] = {}
                for i, name in enumerate(param_names):
                    self._set_nested_dict_value(param_dict, name.split('.'), values[i])
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.exception(f"Error generating parameter combinations: {e}")
            raise ParameterOptimizationError(f"Failed to generate parameter combinations: {e}")
    
    def _generate_random_combinations(
        self, 
        param_values: Dict[str, List[Any]], 
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations.
        
        Args:
            param_values: Dictionary of parameter names and their possible values
            max_combinations: Maximum number of combinations to generate
        
        Returns:
            List of parameter dictionaries
        """
        try:
            param_names = list(param_values.keys())
            combinations = []
            
            # Set random seed for reproducibility
            np.random.seed(int(time.time()))
            
            # Generate unique combinations
            attempts = 0
            max_attempts = max_combinations * 10  # Avoid infinite loop
            
            while len(combinations) < max_combinations and attempts < max_attempts:
                param_dict: Dict[str, Any] = {}
                for name in param_names:
                    values = param_values[name]
                    value = values[np.random.randint(0, len(values))]
                    self._set_nested_dict_value(param_dict, name.split('.'), value)
                
                # Check if this combination is already in the list
                if param_dict not in combinations:
                    combinations.append(param_dict)
                
                attempts += 1
            
            if len(combinations) < max_combinations:
                logger.warning(f"Could only generate {len(combinations)} unique combinations")
            
            return combinations
            
        except Exception as e:
            logger.exception(f"Error generating random combinations: {e}")
            raise ParameterOptimizationError(f"Failed to generate random combinations: {e}")
    
    def _flatten_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        Flatten nested parameter ranges into a single-level dictionary.
        
        Returns:
            Flattened parameter ranges
        """
        flat_params: Dict[str, Dict[str, Any]] = {}
        
        def _flatten(params: Dict[str, Any], prefix: str = '') -> None:
            for key, value in params.items():
                new_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict) and 'min' in value and 'max' in value:
                    # This is a parameter range
                    flat_params[new_key] = value
                elif isinstance(value, dict):
                    # This is a nested dictionary
                    _flatten(value, new_key)
        
        _flatten(self.param_ranges)
        return flat_params
    
    def _set_nested_dict_value(self, d: Dict[str, Any], keys: List[str], value: Any) -> None:
        """
        Set a value in a nested dictionary.
        
        Args:
            d: Dictionary to modify
            keys: List of keys defining the path
            value: Value to set
        """
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    
    def run_optimization(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime, 
        param_combinations: Optional[List[Dict[str, Any]]] = None,
        param_subset: Optional[List[str]] = None, 
        max_combinations: int = 100, 
        parallel: bool = True, 
        n_jobs: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Run parameter optimization.
        
        Args:
            symbol: Symbol to backtest
            timeframe: Timeframe to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            param_combinations: List of parameter dictionaries to test.
                              Defaults to None (generate automatically).
            param_subset: List of parameter paths to include in optimization.
                         Defaults to None (use all parameters).
            max_combinations: Maximum number of combinations to generate.
                            Defaults to 100.
            parallel: Whether to run backtests in parallel.
                     Defaults to True.
            n_jobs: Number of parallel jobs. Defaults to None (use CPU count).
            progress_callback: Callback function to report progress.
                             Takes current progress and total as arguments.
        
        Returns:
            Results of optimization as a DataFrame
            
        Raises:
            ParameterOptimizationError: If optimization fails
        """
        try:
            # Generate parameter combinations if not provided
            if param_combinations is None:
                param_combinations = self.generate_parameter_combinations(
                    param_subset=param_subset,
                    max_combinations=max_combinations
                )
            
            if not param_combinations:
                raise ValueError("No parameter combinations to test")
            
            logger.info(f"Running optimization with {len(param_combinations)} parameter combinations")
            
            # Run backtests
            if parallel:
                # Use multiprocessing for parallel execution
                if n_jobs is None:
                    n_jobs = max(1, multiprocessing.cpu_count() - 1)
                
                logger.info(f"Running backtests in parallel with {n_jobs} processes")
                
                try:
                    # Create a pool of workers
                    pool = multiprocessing.Pool(processes=n_jobs)
                    
                    # Create a partial function with fixed arguments
                    backtest_func = partial(
                        self._run_backtest_with_params,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Run backtests in parallel with progress reporting
                    total = len(param_combinations)
                    results = []
                    
                    for i, result in enumerate(pool.imap(backtest_func, param_combinations)):
                        results.append(result)
                        
                        # Report progress
                        if progress_callback:
                            progress_callback(i + 1, total)
                    
                    # Close the pool
                    pool.close()
                    pool.join()
                    
                except Exception as e:
                    logger.exception(f"Error in parallel execution: {e}")
                    # Clean up in case of error
                    if 'pool' in locals():
                        pool.terminate()
                        pool.join()
                    raise
            else:
                # Run backtests sequentially
                logger.info("Running backtests sequentially")
                results = []
                total = len(param_combinations)
                
                for i, params in enumerate(param_combinations):
                    result = self._run_backtest_with_params(
                        params, symbol, timeframe, start_date, end_date
                    )
                    results.append(result)
                    
                    # Report progress
                    if progress_callback:
                        progress_callback(i + 1, total)
            
            # Process results
            if not results:
                raise ParameterOptimizationError("No results returned from backtests")
            
            # Convert results to DataFrame
            try:
                results_df = pd.DataFrame(results)
                
                # Handle case where all backtests failed
                if 'success' not in results_df.columns or not results_df['success'].any():
                    logger.error("All backtests failed")
                    # Return DataFrame with just the parameters and error info
                    if 'error' in results_df.columns:
                        return results_df[['params', 'error', 'success']]
                    else:
                        return results_df[['params', 'success']]
                
                # Sort by performance metric (e.g., profit factor, then total profit)
                results_df = results_df.sort_values(
                    by=['profit_factor', 'total_profit', 'win_rate'],
                    ascending=[False, False, False]
                )
                
                return results_df
                
            except Exception as e:
                logger.exception(f"Error processing results: {e}")
                raise ParameterOptimizationError(f"Failed to process optimization results: {e}")
        
        except Exception as e:
            logger.exception(f"Error in parameter optimization: {e}")
            raise ParameterOptimizationError(f"Parameter optimization failed: {e}")
    
    def _run_backtest_with_params(
        self, 
        params: Dict[str, Any], 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Run a backtest with specific parameters.
        
        Args:
            params: Parameter dictionary
            symbol: Symbol to backtest
            timeframe: Timeframe to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Backtest results with parameters
        """
        try:
            # Create a copy of the strategy with modified parameters
            strategy_copy = copy.deepcopy(self.strategy)
            
            # Update strategy config with parameters
            self._update_strategy_config(strategy_copy, params)
            
            # Run backtest
            backtest_results = strategy_copy.backtest(
                symbol, timeframe, start_date, end_date
            )
            
            if backtest_results is None:
                logger.warning(f"Backtest failed for parameters: {params}")
                return {
                    'params': params,
                    'total_profit': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'max_drawdown': 100,
                    'total_trades': 0,
                    'success': False
                }
            
            # Extract key metrics
            result = {
                'params': params,
                'total_profit': backtest_results['total_profit'],
                'win_rate': backtest_results['win_rate'],
                'profit_factor': backtest_results['profit_factor'],
                'max_drawdown': backtest_results['max_drawdown'],
                'total_trades': backtest_results['total_trades'],
                'success': True
            }
            
            logger.info(f"Backtest completed: Profit={result['total_profit']:.2f}, "
                       f"Win Rate={result['win_rate']:.2f}, "
                       f"Profit Factor={result['profit_factor']:.2f}")
            
            return result
        
        except Exception as e:
            logger.exception(f"Error in backtest with parameters {params}: {e}")
            return {
                'params': params,
                'total_profit': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 100,
                'total_trades': 0,
                'success': False,
                'error': str(e)
            }
    
    def _update_strategy_config(self, strategy: Any, params: Dict[str, Any]) -> None:
        """
        Update strategy configuration with parameters.
        
        Args:
            strategy: Strategy to update
            params: Parameter dictionary
        """
        try:
            # Update strategy config
            for section, section_params in params.items():
                if section == 'atfnet':
                    for key, value in section_params.items():
                        strategy.config[key] = value
                elif section == 'attention':
                    for key, value in section_params.items():
                        if key == 'attention_threshold':
                            strategy.config['attention_threshold'] = value
                        elif key == 'learning_rate':
                            strategy.config['learning_rate'] = value
                elif section == 'risk_management':
                    for key, value in section_params.items():
                        if key == 'base_risk_per_trade':
                            strategy.config['risk_per_trade'] = value
                elif section == 'indicators':
                    for key, value in section_params.items():
                        if key == 'atr_multiplier':
                            strategy.config['atr_multiplier'] = value
                        elif key == 'use_trend_filter':
                            strategy.config['use_trend_filter'] = value
            
            # Update params_config
            if hasattr(strategy, 'params_config'):
                self._update_nested_dict(strategy.params_config, params)
                
        except Exception as e:
            logger.exception(f"Error updating strategy config: {e}")
            raise ValueError(f"Failed to update strategy configuration: {e}")
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def save_results(self, results_df: pd.DataFrame, output 
                d[k] = v
        return d
    
    def save_results(self, results_df: pd.DataFrame, output_path: str) -> bool:
        """
        Save optimization results to file.
        
        Args:
            results_df: Results dataframe
            output_path: Output file path
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"Optimization results saved to {output_path}")
            return True
        
        except Exception as e:
            logger.exception(f"Error saving optimization results: {e}")
            return False
    
    def load_results(self, input_path: str) -> Optional[pd.DataFrame]:
        """
        Load optimization results from file.
        
        Args:
            input_path: Input file path
        
        Returns:
            Results dataframe or None if loading failed
        """
        try:
            # Check if file exists
            if not os.path.exists(input_path):
                logger.error(f"Results file not found: {input_path}")
                return None
            
            # Load from CSV
            results_df = pd.read_csv(input_path)
            
            # Convert params column from string to dict
            if 'params' in results_df.columns and isinstance(results_df['params'].iloc[0], str):
                results_df['params'] = results_df['params'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            
            logger.info(f"Optimization results loaded from {input_path}")
            return results_df
        
        except Exception as e:
            logger.exception(f"Error loading optimization results: {e}")
            return None
    
    def get_best_parameters(
        self, 
        results_df: pd.DataFrame, 
        metric: str = 'profit_factor', 
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get the best parameters from optimization results.
        
        Args:
            results_df: Results dataframe
            metric: Metric to sort by. Defaults to 'profit_factor'.
            n: Number of best parameter sets to return. Defaults to 5.
        
        Returns:
            List of best parameter dictionaries
        """
        try:
            # Check if dataframe is empty
            if results_df.empty:
                logger.warning("Results dataframe is empty")
                return []
            
            # Check if metric exists in dataframe
            if metric not in results_df.columns:
                logger.warning(f"Metric '{metric}' not found in results. Using 'profit_factor' instead.")
                metric = 'profit_factor'
            
            # Sort by metric
            sorted_df = results_df.sort_values(by=metric, ascending=False)
            
            # Get top n parameter sets
            best_params = sorted_df.head(n)['params'].tolist()
            
            return best_params
            
        except Exception as e:
            logger.exception(f"Error getting best parameters: {e}")
            return []
