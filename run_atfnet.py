#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATFNet Launcher Script
This script launches the ATFNet application.
"""

import os
import sys
import argparse
import logging

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Import the main function from the app
from app.atfnet_app import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ATFNet Trading System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--models-dir", type=str, help="Models directory")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set environment variables if provided
    if args.data_dir:
        os.environ["ATFNET_DATA_DIR"] = args.data_dir
    
    if args.models_dir:
        os.environ["ATFNET_MODELS_DIR"] = args.models_dir
    
    if args.config:
        os.environ["ATFNET_CONFIG_PATH"] = args.config
    
    # Run the application
    main()
