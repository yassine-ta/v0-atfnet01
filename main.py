#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the ATFNet Python implementation.
This script initializes the MT5 connection, loads models, and runs the trading system.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
import yaml

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use the new import structure
from mt5_connector import MT5API
from data import DataManager
from models import ModelManager
from strategies import ATFNetStrategy
from utils import setup_logger, ConfigLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ATFNet Python Implementation')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live'], default='backtest',
                        help='Operation mode: train, backtest, or live trading')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Trading timeframe')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def main():
    """Main function to run the ATFNet Python implementation."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('atfnet', log_level)
    logger.info(f"Starting ATFNet Python Implementation in {args.mode} mode")
    
    try:
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
        
        # Run in specified mode
        if args.mode == 'train':
            logger.info(f"Training models for {args.symbol} on {args.timeframe} timeframe")
            strategy.train(args.symbol, args.timeframe)
        
        elif args.mode == 'backtest':
            logger.info(f"Backtesting strategy for {args.symbol} on {args.timeframe} timeframe")
            results = strategy.backtest(args.symbol, args.timeframe)
            logger.info(f"Backtest results: {results}")
        
        elif args.mode == 'live':
            logger.info(f"Starting live trading for {args.symbol} on {args.timeframe} timeframe")
            strategy.run_live(args.symbol, args.timeframe)
        
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
    
    finally:
        # Clean up
        if 'mt5_api' in locals() and mt5_api.is_initialized():
            mt5_api.shutdown()
            logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()
