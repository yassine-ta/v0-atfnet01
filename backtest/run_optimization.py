#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter optimization runner for ATFNet backtesting.
This script runs parameter optimization for the ATFNet strategy.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mt5_connector import MT5API
from data import DataManager
from models import ModelManager
from strategies import ATFNetStrategy
from backtest.parameter_optimizer import ParameterOptimizer
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ATFNet Parameter Optimization')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Trading timeframe')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='optimization_results',
                        help='Output directory for optimization results')
    parser.add_argument('--max_combinations', type=int, default=100,
                        help='Maximum number of parameter combinations to test')
    parser.add_argument('--parallel', action='store_true',
                        help='Run backtests in parallel')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs')
    parser.add_argument('--param_subset', type=str, default=None,
                        help='Comma-separated list of parameter paths to optimize')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def main():
    """Main function to run parameter optimization."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('atfnet_optimization', log_level)
    logger.info(f"Starting ATFNet Parameter Optimization for {args.symbol} {args.timeframe}")
    
    try:
        # Load configuration
        config_loader = ConfigLoader(args.config)
        config = config_loader.load_config()
        
        # Initialize MT5 connection
        mt5_api = MT5API(config['mt5'])
        if not mt5_api.initialize():
            logger.error("Failed to initialize MT5 connection")
            return
        
        logger.info(f"MT5 connection established: {mt5_api.account_info()}")
        
        # Initialize data manager
        data_manager = DataManager(mt5_api, config['data'])
        
        # Initialize model manager
        model_manager = ModelManager(config['models'])
        
        # Initialize strategy
        strategy = ATFNetStrategy(
            mt5_api=mt5_api,
            data_manager=data_manager,
            model_manager=model_manager,
            config=config['strategy']
        )
        
        # Initialize parameter optimizer
        optimizer = ParameterOptimizer(strategy)
        
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else datetime.now() - timedelta(days=180)
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
        
        # Parse parameter subset
        param_subset = args.param_subset.split(',') if args.param_subset else None
        
        # Run optimization
        logger.info(f"Running optimization for {args.symbol} {args.timeframe} from {start_date} to {end_date}")
        results_df = optimizer.run_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            param_subset=param_subset,
            max_combinations=args.max_combinations,
            parallel=args.parallel,
            n_jobs=args.n_jobs
        )
        
        # Create output directory
        output_dir = os.path.join(args.output, f"{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_path = os.path.join(output_dir, 'optimization_results.csv')
        optimizer.save_results(results_df, results_path)
        
        # Get best parameters
        best_params = optimizer.get_best_parameters(results_df, metric='profit_factor', n=5)
        
        # Save best parameters
        best_params_path = os.path.join(output_dir, 'best_parameters.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        
        # Print summary
        logger.info(f"Optimization completed for {args.symbol} {args.timeframe}")
        logger.info(f"Total parameter combinations tested: {len(results_df)}")
        logger.info(f"Best profit factor: {results_df['profit_factor'].max():.2f}")
        logger.info(f"Best total profit: {results_df['total_profit'].max():.2f}")
        logger.info(f"Best win rate: {results_df['win_rate'].max() * 100:.2f}%")
        logger.info(f"Results saved to {output_dir}")
        
        # Run backtest with best parameters
        logger.info("Running backtest with best parameters")
        best_params_dict = best_params[0]
        
        # Update strategy config with best parameters
        optimizer._update_strategy_config(strategy, best_params_dict)
        
        # Run backtest
        best_results = strategy.backtest(
            args.symbol,
            args.timeframe,
            start_date,
            end_date
        )
        
        # Save best backtest results
        best_results_path = os.path.join(output_dir, 'best_backtest_results.json')
        with open(best_results_path, 'w') as f:
            json.dump(best_results, f, indent=4, default=str)
        
        # Save trades to CSV
        trades_df = pd.DataFrame(best_results['trades'])
        trades_path = os.path.join(output_dir, 'best_backtest_trades.csv')
        trades_df.to_csv(trades_path, index=False)
        
        logger.info(f"Best backtest results saved to {best_results_path}")
    
    except Exception as e:
        logger.exception(f"Error in parameter optimization: {e}")
    
    finally:
        # Clean up
        if 'mt5_api' in locals() and mt5_api.is_initialized():
            mt5_api.shutdown()
            logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()
