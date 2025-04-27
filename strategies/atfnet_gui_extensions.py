#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATFNet GUI Extensions
This module extends the ATFNetStrategy class with methods needed by the GUI.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('atfnet.gui')


class ATFNetGUIExtensions:
    """Extension methods for the ATFNetStrategy class to support the GUI."""
    
    def initialize_live_trading(self, symbol, timeframe):
        """
        Initialize live trading.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Get initial data
            sequence_length = self.config.get('sequence_length', 60)
            lookback = sequence_length + 100  # Extra data for indicators
            
            start_date = datetime.now() - timedelta(days=lookback * self.mt5_api.timeframe_map[timeframe] / (24 * 60 * 60))
            df = self.data_manager.get_data(symbol, timeframe, start_date)
            
            if df is None or len(df) < lookback:
                logger.error(f"Not enough data available for {symbol} {timeframe}")
                return False
            
            # Preprocess data
            df = self.data_manager.preprocess_data(df)
            
            # Add technical indicators
            df = self.data_manager.add_technical_indicators(df)
            
            # Add frequency features
            df = self.data_manager.add_frequency_features(df)
            
            # Add ATFNet features
            df = self.data_manager.add_atfnet_features(df)
            
            # Store data
            self.live_data = df
            self.live_symbol = symbol
            self.live_timeframe = timeframe
            
            # Initialize equity data
            account_info = self.mt5_api.account_info()
            self.live_equity_data = pd.DataFrame({
                'equity': [account_info['equity']],
                'balance': [account_info['balance']]
            }, index=[datetime.now()])
            
            # Initialize signals
            self.live_signals = []
            
            return True
        
        except Exception as e:
            logger.exception(f"Error initializing live trading: {e}")
            return False
    
    def update_live_trading(self):
        """
        Update live trading.
        
        Returns:
            dict: Status information
        """
        try:
            # Get latest data
            latest_bars = self.mt5_api.get_latest_bars(
                self.live_symbol,
                self.live_timeframe,
                1
            )
            
            if latest_bars is None or len(latest_bars) == 0:
                return {
                    'log': "No new data available"
                }
            
            # Preprocess latest data
            latest_df = self.data_manager.preprocess_data(latest_bars)
            
            # Add technical indicators
            latest_df = self.data_manager.add_technical_indicators(latest_df)
            
            # Add frequency features
            latest_df = self.data_manager.add_frequency_features(latest_df)
            
            # Add ATFNet features
            latest_df = self.data_manager.add_atfnet_features(latest_df)
            
            # Append to live data
            self.live_data = pd.concat([self.live_data, latest_df])
            
            # Get current price
            current_price = latest_df['close'].iloc[-1]
            
            # Create ATFNet features
            features = self.create_features_from_dataframe(latest_df.iloc[-1])
            
            # Update frequency features
            sequence_length = self.config.get('sequence_length', 60)
            price_data = self.live_data['close'].iloc[-sequence_length:].values
            self.attention.update_frequency_features(features, price_data)
            
            # Detect market regime
            regime = self.market_regime.detect_regime(self.live_data.iloc[-sequence_length:])
            regime_name = self.get_regime_name(regime)
            
            # Optimize for current regime
            self.market_regime.optimize_for_regime(regime)
            
            # Forecast price
            forecast_price = self.price_forecaster.forecast_price(price_data)
            
            # Calculate predicted return
            predicted_return = (forecast_price - current_price) / current_price
            
            # Calculate attention score
            attention_score = self.attention.calculate_score(features)
            
            # Set up signal aggregator
            self.signal_aggregator.set_predicted_return(predicted_return)
            self.signal_aggregator.set_features(features)
            self.signal_aggregator.set_trend_filter(
                self.config.get('use_trend_filter', True),
                current_price,
                self.live_data['ema_50'].iloc[-1] if 'ema_50' in self.live_data.columns else current_price
            )
            
            # Check for entry signals
            signal = "NONE"
            log_message = ""
            
            if self.signal_aggregator.should_enter_trade():
                # Get signal direction
                signal = self.signal_aggregator.get_signal_direction()
                
                # Get open positions
                open_positions = self.mt5_api.get_open_positions(self.live_symbol)
                
                if len(open_positions) < self.config.get('max_positions', 3):
                    # Calculate stop loss
                    atr = self.live_data['atr_14'].iloc[-1]
                    sl_distance = atr * self.config.get('atr_multiplier', 1.0)
                    
                    if signal == 'BUY':
                        sl_price = current_price - sl_distance
                    else:
                        sl_price = current_price + sl_distance
                    
                    # Get account info
                    account_info = self.mt5_api.account_info()
                    
                    # Get symbol info
                    symbol_info = self.mt5_api.get_symbol_info(self.live_symbol)
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        current_price, sl_price, symbol_info, account_info
                    )
                    
                    # Open position
                    result = self.mt5_api.place_order(
                        symbol=self.live_symbol,
                        order_type=signal,
                        volume=position_size,
                        price=None,  # Market price
                        sl=sl_price,
                        comment=f"ATFNet Python {self.live_timeframe}"
                    )
                    
                    if result:
                        # Record trade open
                        self.performance_analytics.record_trade_open(
                            result['order_id'],
                            signal,
                            current_price,
                            sl_price,
                            position_size
                        )
                        
                        # Add signal to list
                        self.live_signals.append({
                            'type': signal,
                            'time': latest_df.index[-1],
                            'price': current_price
                        })
                        
                        log_message = f"Opened {signal} position at {current_price} (ID: {result['order_id']})"
                        logger.info(log_message)
            
            # Manage open positions
            open_positions = self.mt5_api.get_open_positions(self.live_symbol)
            
            for position in open_positions:
                # Check for exit signals
                if self.signal_aggregator.should_exit_trade(position['ticket']):
                    # Close position
                    self.mt5_api.close_position(position['ticket'])
                    
                    # Record trade close
                    self.performance_analytics.record_trade_close(
                        position['ticket'],
                        current_price,
                        position['profit']
                    )
                    
                    # Add signal to list
                    self.live_signals.append({
                        'type': 'EXIT',
                        'time': latest_df.index[-1],
                        'price': current_price
                    })
                    
                    log_message = f"Closed {position['type']} position at {current_price} (ID: {position['ticket']}, Reason: signal)"
                    logger.info(log_message)
                
                # Update trailing stop if needed
                elif self.config.get('use_trailing_stop', True):
                    # Calculate trailing stop
                    atr = self.live_data['atr_14'].iloc[-1]
                    trail_distance = atr * self.config.get('trailing_stop_multiplier', 2.0)
                    
                    if position['type'] == 'BUY':
                        new_sl = current_price - trail_distance
                        if new_sl > position['sl']:
                            self.mt5_api.modify_position(position['ticket'], sl=new_sl)
                            log_message = f"Updated trailing stop for LONG position (ID: {position['ticket']}, New SL: {new_sl})"
                            logger.info(log_message)
                    
                    elif position['type'] == 'SELL':
                        new_sl = current_price + trail_distance
                        if new_sl < position['sl'] or position['sl'] == 0:
                            self.mt5_api.modify_position(position['ticket'], sl=new_sl)
                            log_message = f"Updated trailing stop for SHORT position (ID: {position['ticket']}, New SL: {new_sl})"
                            logger.info(log_message)
            
            # Update equity data
            account_info = self.mt5_api.account_info()
            
            new_equity_data = pd.DataFrame({
                'equity': [account_info['equity']],
                'balance': [account_info['balance']]
            }, index=[datetime.now()])
            
            self.live_equity_data = pd.concat([self.live_equity_data, new_equity_data])
            
            # Keep only the last 1000 data points
            if len(self.live_equity_data) > 1000:
                self.live_equity_data = self.live_equity_data.iloc[-1000:]
            
            # Keep only the last 1000 data points for price data
            if len(self.live_data) > 1000:
                self.live_data = self.live_data.iloc[-1000:]
            
            # Keep only the last 100 signals
            if len(self.live_signals) > 100:
                self.live_signals = self.live_signals[-100:]
            
            # Return status
            return {
                'account_balance': account_info['balance'],
                'account_equity': account_info['equity'],
                'current_profit': account_info['profit'],
                'attention_score': attention_score,
                'prediction': predicted_return,
                'regime': regime_name,
                'signal': signal,
                'positions': open_positions,
                'price_data': self.live_data.iloc[-100:],
                'equity_data': self.live_equity_data,
                'signals': self.live_signals,
                'log': log_message
            }
        
        except Exception as e:
            logger.exception(f"Error updating live trading: {e}")
            return {
                'log': f"Error: {e}"
            }
    
    def create_features_from_dataframe(self, row):
        """
        Create ATFNet features from DataFrame row.
        
        Args:
            row (pd.Series): DataFrame row
            
        Returns:
            ATFNetFeatures: Features object
        """
        from core import ATFNetFeatures
        
        features = ATFNetFeatures()
        features.price = row.get('price', 0)
        features.atr = row.get('atr', 0)
        features.z_score = row.get('z_score', 0)
        features.trend_strength = row.get('trend_strength', 0)
        features.range_position = row.get('range_position', 0)
        features.volume_profile = row.get('volume_profile', 0)
        features.pattern_strength = row.get('pattern_strength', 0)
        features.spectral_entropy = row.get('spectral_entropy', 0)
        features.freq_trend_strength = row.get('freq_trend_strength', 0)
        features.cyclicality_score = row.get('cyclicality_score', 0)
        
        return features
    
    def get_regime_name(self, regime):
        """
        Get regime name from regime code.
        
        Args:
            regime (int): Regime code
            
        Returns:
            str: Regime name
        """
        regime_names = {
            0: "Trending",
            1: "Ranging",
            2: "Volatile",
            3: "Choppy",
            4: "Unknown"
        }
        
        return regime_names.get(regime, "Unknown")
