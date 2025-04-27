#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest runner for the ATFNet Python implementation.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mt5_connector import MT5API
from data import DataManager
from models import ModelManager
from strategies import ATFNetStrategy
from utils import setup_logger, ConfigLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ATFNet Backtest Runner')
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
    parser.add_argument('--output', type=str, default='backtest_results',
                        help='Output directory for backtest results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def plot_equity_curve(results, output_dir):
    """
    Plot equity curve from backtest results.
    
    Args:
        results (dict): Backtest results
        output_dir (str): Output directory
    """
    # Create equity curve
    equity = [results['initial_balance']]
    dates = []
    
    for trade in results['trades']:
        equity.append(equity[-1] + trade['profit'])
        dates.append(trade['close_time'])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity[1:])
    plt.title(f"Equity Curve - {results['symbol']} {results['timeframe']}")
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"equity_curve_{results['symbol']}_{results['timeframe']}.png"))
    plt.close()


def plot_trade_distribution(results, output_dir):
    """
    Plot trade distribution from backtest results.
    
    Args:
        results (dict): Backtest results
        output_dir (str): Output directory
    """
    # Extract trade profits
    profits = [trade['profit'] for trade in results['trades']]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.hist(profits, bins=20)
    plt.title(f"Trade Distribution - {results['symbol']} {results['timeframe']}")
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"trade_distribution_{results['symbol']}_{results['timeframe']}.png"))
    plt.close()


def plot_regime_performance(results, output_dir):
    """
    Plot performance by market regime from backtest results.
    
    Args:
        results (dict): Backtest results
        output_dir (str): Output directory
    """
    # Group trades by regime
    regime_profits = {}
    
    for trade in results['trades']:
        regime = trade.get('regime', 0)
        if regime not in regime_profits:
            regime_profits[regime] = []
        
        regime_profits[regime].append(trade['profit'])
    
    # Calculate statistics
    regime_stats = {}
    
    for regime, profits in regime_profits.items():
        regime_stats[regime] = {
            'count': len(profits),
            'total_profit': sum(profits),
            'avg_profit': sum(profits) / len(profits) if len(profits) > 0 else 0,
            'win_rate': len([p for p in profits if p > 0]) / len(profits) if len(profits) > 0 else 0
        }
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot total profit by regime
    regimes = list(regime_stats.keys())
    total_profits = [regime_stats[r]['total_profit'] for r in regimes]
    
    plt.bar(regimes, total_profits)
    plt.title(f"Profit by Market Regime - {results['symbol']} {results['timeframe']}")
    plt.xlabel('Market Regime')
    plt.ylabel('Total Profit')
    plt.xticks(regimes, ['Unknown', 'Trending', 'Ranging', 'Volatile', 'Choppy'])
    plt.grid(True)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"regime_performance_{results['symbol']}_{results['timeframe']}.png"))
    plt.close()
    
    # Create win rate figure
    plt.figure(figsize=(12, 6))
    
    win_rates = [regime_stats[r]['win_rate'] * 100 for r in regimes]
    
    plt.bar(regimes, win_rates)
    plt.title(f"Win Rate by Market Regime - {results['symbol']} {results['timeframe']}")
    plt.xlabel('Market Regime')
    plt.ylabel('Win Rate (%)')
    plt.xticks(regimes, ['Unknown', 'Trending', 'Ranging', 'Volatile', 'Choppy'])
    plt.grid(True)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"regime_win_rate_{results['symbol']}_{results['timeframe']}.png"))
    plt.close()


def main():
    """Main function to run the backtest."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('atfnet_backtest', log_level)
    logger.info(f"Starting ATFNet Backtest for {args.symbol} {args.timeframe}")
    
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
        
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        # Run backtest
        logger.info(f"Running backtest for {args.symbol} {args.timeframe} from {start_date} to {end_date}")
        results = strategy.backtest(args.symbol, args.timeframe, start_date, end_date)
        
        if results is None:
            logger.error("Backtest failed")
            return
        
        # Create output directory
        output_dir = os.path.join(args.output, f"{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save trades to CSV
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        
        # Plot results
        plot_equity_curve(results, output_dir)
        plot_trade_distribution(results, output_dir)
        plot_regime_performance(results, output_dir)
        
        # Print summary
        logger.info(f"Backtest completed for {args.symbol} {args.timeframe}")
        logger.info(f"Total trades: {results['total_trades']}")
        logger.info(f"Win rate: {results['win_rate'] * 100:.2f}%")
        logger.info(f"Profit factor: {results['profit_factor']:.2f}")
        logger.info(f"Total profit: {results['total_profit']:.2f}")
        logger.info(f"Return: {results['return_pct']:.2f}%")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Results saved to {output_dir}")
    
    except Exception as e:
        logger.exception(f"Error in backtest: {e}")
    
    finally:
        # Clean up
        if 'mt5_api' in locals() and mt5_api.is_initialized():
            mt5_api.shutdown()
            logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()
