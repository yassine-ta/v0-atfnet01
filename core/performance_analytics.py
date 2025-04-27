#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python implementation of the OPT_ATFNet_PerformanceAnalytics.mqh file.
This module implements performance tracking and analysis.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger('atfnet.core.performance_analytics')


class PerformanceAnalytics:
    """
    Python implementation of the CPerformanceAnalytics class from MQL5.
    Tracks and analyzes trading performance.
    """
    
    def __init__(self):
        """Initialize the performance analytics."""
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Component performance
        self.attention_accuracy = 0.0
        self.forecast_accuracy = 0.0
        self.filter_effectiveness = 0.0
        
        # Trade tracking
        self.trades = []
    
    def record_trade_open(self, ticket, order_type, open_price, stop_loss, volume):
        """
        Record trade open.
        
        Args:
            ticket: Trade ticket
            order_type (str): Order type ('BUY' or 'SELL')
            open_price (float): Open price
            stop_loss (float): Stop loss price
            volume (float): Trade volume
        """
        trade = {
            'ticket': ticket,
            'type': order_type,
            'open_time': datetime.now(),
            'open_price': open_price,
            'stop_loss': stop_loss,
            'volume': volume,
            'profit': 0.0,
            'closed': False
        }
        
        self.trades.append(trade)
        self.total_trades += 1
    
    def record_trade_close(self, ticket, close_price, profit):
        """
        Record trade close.
        
        Args:
            ticket: Trade ticket
            close_price (float): Close price
            profit (float): Trade profit
        """
        # Find trade in array
        for trade in self.trades:
            if trade['ticket'] == ticket and not trade['closed']:
                trade['closed'] = True
                trade['close_time'] = datetime.now()
                trade['profit'] = profit
                
                # Update statistics
                if profit > 0:
                    self.winning_trades += 1
                    self.gross_profit += profit
                else:
                    self.gross_loss += abs(profit)
                
                break
    
    def record_component_performance(self, attention_correct, forecast_correct, filter_correct):
        """
        Record component performance.
        
        Args:
            attention_correct (bool): Whether attention was correct
            forecast_correct (bool): Whether filter was correct
        """
        # Update attention accuracy
        self.attention_accuracy = ((self.total_trades - 1) * self.attention_accuracy + (1 if attention_correct else 0)) / self.total_trades
        
        # Update forecast accuracy
        self.forecast_accuracy = ((self.total_trades - 1) * self.forecast_accuracy + (1 if forecast_correct else 0)) / self.total_trades
        
        # Update filter effectiveness
        self.filter_effectiveness = ((self.total_trades - 1) * self.filter_effectiveness + (1 if filter_correct else 0)) / self.total_trades
    
    def generate_performance_report(self):
        """
        Generate performance report.
        
        Returns:
            str: Performance report
        """
        report = "=== OPT_ATFNet Performance Report ===\n"
        
        # Calculate metrics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else 0
        net_profit = self.gross_profit - self.gross_loss
        
        # Add trade statistics
        report += f"Total Trades: {self.total_trades}\n"
        report += f"Winning Trades: {self.winning_trades} ({win_rate:.2f}%)\n"
        report += f"Gross Profit: {self.gross_profit:.2f}\n"
        report += f"Gross Loss: {self.gross_loss:.2f}\n"
        report += f"Net Profit: {net_profit:.2f}\n"
        report += f"Profit Factor: {profit_factor:.2f}\n\n"
        
        # Add component performance
        report += "Component Performance:\n"
        report += f"Attention Accuracy: {self.attention_accuracy * 100:.2f}%\n"
        report += f"Forecast Accuracy: {self.forecast_accuracy * 100:.2f}%\n"
        report += f"Filter Effectiveness: {self.filter_effectiveness * 100:.2f}%\n"
        
        return report
    
    def export_to_csv(self, filename):
        """
        Export trade history to CSV.
        
        Args:
            filename (str): Output filename
        """
        # Create DataFrame from trades
        trades_df = pd.DataFrame(self.trades)
        
        # Convert datetime objects to strings
        if 'open_time' in trades_df.columns:
            trades_df['open_time'] = trades_df['open_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '')
        
        if 'close_time' in trades_df.columns:
            trades_df['close_time'] = trades_df['close_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else '')
        
        # Save to CSV
        trades_df.to_csv(filename, index=False)
        logger.info(f"Trade history exported to {filename}")
    
    def get_win_rate(self):
        """
        Get win rate.
        
        Returns:
            float: Win rate (0-1)
        """
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0
    
    def get_profit_factor(self):
        """
        Get profit factor.
        
        Returns:
            float: Profit factor
        """
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else 0
    
    def get_net_profit(self):
        """
        Get net profit.
        
        Returns:
            float: Net profit
        """
        return self.gross_profit - self.gross_loss
