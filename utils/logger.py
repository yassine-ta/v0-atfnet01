#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utility for the ATFNet Python implementation.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime


def setup_logger(name, level=logging.INFO, log_dir='logs'):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name (str): Logger name
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_dir (str, optional): Log directory. Defaults to 'logs'.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
