#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATFNet strategy implementation for the Python version.
This module implements the ATFNet trading strategy with machine learning enhancements.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import pickle
from collections import deque
import copy

# Import core ATFNet components
from core import (
    ATFNetAttention, 
    ATFNetFeatures,
    PriceForecaster, 
    FrequencyFilter,
    MarketRegimeDetector,
    RiskManager,
    SignalAggregator,
    PerformanceAnalytics
)

# Import GUI extensions
from .atfnet_gui_extensions import ATFNetGUIExtensions

logger = logging.getLogger('atfnet.strategy')


class ATFNetStrategy(ATFNetGUIExtensions):
    """ATFNet strategy implementation with machine learning enhancements."""
    
    def __init__(
        self, 
        mt5_api: Any, 
        data_manager: Any, 
        model_manager: Any, 
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the ATFNet strategy.
        
        Args:
            mt5_api: MT5 API connector
            data_manager: Data manager
            model_manager: Model manager
            config: Strategy configuration
        """
        self.mt5_api = mt5_api
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.config = config
        
        # Load parameters config if it exists
        self.params_config = self._load_parameters_config()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize feature cache
        self.feature_cache: deque = deque(maxlen=config.get('sequence_length', 60))
        
        # Initialize positions
        self.positions: Dict[str, Dict[str, Any]] = {}
    
    def _load_parameters_config(self) -> Dict[str, Any]:
        """
        Load parameters configuration from file.
        
        Returns:
            Dictionary containing parameters configuration
        """
        try:
            # Get the path to the parameters config file
            params_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'config', 
                'atfnet_parameters.yaml'
            )
            
            # Check if the file exists
            if not os.path.exists(params_config_path):
                logger.warning(f"Parameters config file not found: {params_config_path}")
                return {}
            
            # Load the parameters config
            import yaml
            with open(params_config_path, 'r') as f:
                params_config = yaml.safe_load(f)
            
            logger.info(f"Parameters loaded from {params_config_path}")
            return params_config
        
        except Exception as e:
            logger.error(f"Failed to load parameters: {e}")
            return {}
    
    def _initialize_components(self) -> None:
        """Initialize ATFNet components with current parameters."""
        try:
            # Initialize core ATFNet components with parameters
            self.attention = ATFNetAttention(
                initial_learning_rate=self.config.get('learning_rate', 0.01),
                lookback_period=self.config.get('sequence_length', 60),
                config=self.params_config
            )
            
            self.price_forecaster = PriceForecaster(
                forecast_horizon=self.config.get('prediction_length', 20),
                lookback_period=self.config.get('sequence_length', 60)
            )
            
            self.freq_filter = FrequencyFilter(
                lookback_period=self.config.get('sequence_length', 60),
                entropy_threshold=0.6,
                cyclicality_threshold=0.5
            )
            
            self.market_regime = MarketRegimeDetector(config=self.params_config)
            self.risk_manager = RiskManager(config=self.params_config)
            self.signal_aggregator = SignalAggregator(config=self.params_config)
            self.performance_analytics = PerformanceAnalytics()
            
            # Connect components
            self.signal_aggregator.set_attention(self.attention)
            self.signal_aggregator.set_forecaster(self.price_forecaster)
            self.signal_aggregator.set_frequency_filter(self.freq_filter)
            self.signal_aggregator.set_threshold(self.config.get('attention_threshold', 0.6))
            
            # Set initial balance for risk manager
            self.risk_manager.initial_balance = self.config.get('initial_balance', 10000)
            
            logger.info("ATFNet components initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    def update_parameters(self, params_config: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            params_config: New parameter configuration
        """
        try:
            # Update params_config
            self.params_config = copy.deepcopy(params_config)
            
            # Re-initialize components with new parameters
            self._initialize_components()
            
            logger.info("Strategy parameters updated successfully")
        
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            raise ValueError(f"Parameter update failed: {e}")
    
    def train(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Train models for the ATFNet strategy.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_date: Start date for training data. Defaults to None.
            end_date: End date for training data. Defaults to None.
            
        Returns:
            Dictionary containing training results or None if training failed
        """
        logger.info(f"Training models for {symbol} {timeframe}")
        
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                # Default to 1 year of data
                start_date = end_date - timedelta(days=365)
            
            # Get historical data
            df = self.data_manager.get_data(symbol, timeframe, start_date, end_date)
            
            if df is None or len(df) == 0:
                logger.error(f"No data available for {symbol} {timeframe}")
                return None
            
            # Preprocess data
            df = self.data_manager.preprocess_data(df)
            
            # Add technical indicators
            df = self.data_manager.add_technical_indicators(df)
            
            # Add frequency features
            df = self.data_manager.add_frequency_features(df)
            
            # Add ATFNet features
            df = self.data_manager.add_atfnet_features(df)
            
            # Save features
            self.data_manager.save_features(symbol, timeframe, df)
            
            # Prepare features for training
            sequence_length = self.config.get('sequence_length', 60)
            X, y, feature_names = self.data_manager.prepare_features(
                df, 
                target_column=self.config.get('target_column', 'returns'),
                target_shift=self.config.get('target_shift', -1),
                sequence_length=sequence_length
            )
            
            # Save feature scaler
            self.data_manager.save_scaler('feature')
            
            # Train models
            results: Dict[str, Any] = {}
            
            # Train price prediction model
            model_name = f"{symbol}_{timeframe}_price_prediction"
            model, history = self.model_manager.train_model(
                model_type=self.config.get('model_type', 'lstm'),
                X_train=X,
                y_train=y,
                model_name=model_name,
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32)
            )
            
            results['price_prediction'] = {
                'model_name': model_name,
                'training_loss': history.history['loss'][-1],
                'validation_loss': history.history['val_loss'][-1]
            }
            
            # Train market regime classifier
            if self.config.get('train_regime_classifier', True):
                # Create regime labels
                regimes = self._detect_regimes(df)
                df['regime'] = regimes
                
                # Prepare features for regime classification
                X_regime, y_regime, _ = self.data_manager.prepare_features(
                    df,
                    target_column='regime',
                    target_shift=0,
                    sequence_length=sequence_length
                )
                
                # Train regime classifier
                model_name = f"{symbol}_{timeframe}_regime_classifier"
                model, history = self.model_manager.train_model(
                    model_type=self.config.get('model_type', 'lstm'),
                    X_train=X_regime,
                    y_train=y_regime,
                    model_name=model_name,
                    epochs=self.config.get('epochs', 100),
                    batch_size=self.config.get('batch_size', 32)
                )
                
                results['regime_classifier'] = {
                    'model_name': model_name,
                    'training_loss': history.history['loss'][-1],
                    'validation_loss': history.history['val_loss'][-1]
                }
            
            logger.info(f"Training completed for {symbol} {timeframe}")
            return results
        
        except Exception as e:
            logger.exception(f"Error in training: {e}")
            return None
    
    def backtest(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None, 
        params_override: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest the ATFNet strategy.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_date: Start date for backtest. Defaults to None.
            end_date: End date for backtest. Defaults to None.
            params_override: Parameter overrides for this backtest. Defaults to None.
            
        Returns:
            Dictionary containing backtest results or None if backtest failed
        """
        logger.info(f"Backtesting strategy for {symbol} {timeframe}")
        
        # Apply parameter overrides if provided
        original_params = None
        if params_override is not None:
            original_params = copy.deepcopy(self.params_config)
            self.update_parameters(params_override)
        
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                # Default to 6 months of data
                start_date = end_date - timedelta(days=180)
            
            # Get historical data
            df = self.data_manager.get_data(symbol, timeframe, start_date, end_date)
            
            if df is None or len(df) == 0:
                logger.error(f"No data available for {symbol} {timeframe}")
                return None
            
            # Preprocess data
            df = self.data_manager.preprocess_data(df)
            
            # Add technical indicators
            df = self.data_manager.add_technical_indicators(df)
            
            # Add frequency features
            df = self.data_manager.add_frequency_features(df)
            
            # Add ATFNet features
            df = self.data_manager.add_atfnet_features(df)
            
            # Initialize backtest variables
            balance = self.config.get('initial_balance', 10000)
            equity = balance
            positions: Dict[str, Dict[str, Any]] = {}
            trades: List[Dict[str, Any]] = []
            
            # Get symbol info
            symbol_info = self.mt5_api.get_symbol_info(symbol)
            
            if symbol_info is None:
                logger.error(f"Symbol info for {symbol} not available")
                return None
            
            # Set initial balance for risk manager
            self.risk_manager.initial_balance = balance
            
            # Prepare account info for position sizing
            account_info = {'balance': balance, 'equity': equity}
            
            # Run backtest
            sequence_length = self.config.get('sequence_length', 60)
            
            for i in range(sequence_length, len(df)):
                # Current timestamp
                timestamp = df.index[i]
                
                # Current price data
                current_price = df['close'].iloc[i]
                
                # Create ATFNet features
                features = ATFNetFeatures()
                features.price = df['price'].iloc[i]
                features.atr = df['atr'].iloc[i]
                features.z_score = df['z_score'].iloc[i]
                features.trend_strength = df['trend_strength'].iloc[i]
                features.range_position = df['range_position'].iloc[i]
                features.volume_profile = df['volume_profile'].iloc[i]
                features.pattern_strength = df['pattern_strength'].iloc[i]
                features.spectral_entropy = df['spectral_entropy'].iloc[i]
                features.freq_trend_strength = df['freq_trend_strength'].iloc[i]
                features.cyclicality_score = df['cyclicality_score'].iloc[i]
                
                # Update frequency features
                price_data = df['close'].iloc[i-sequence_length:i+1].values
                self.attention.update_frequency_features(features, price_data)
                
                # Detect market regime
                regime = self.market_regime.detect_regime(df.iloc[i-sequence_length:i+1])
                
                # Optimize for current regime
                self.market_regime.optimize_for_regime(regime)
                
                # Calculate predicted return
                predicted_return = (df['close'].iloc[i+1] - current_price) / current_price if i+1 < len(df) else 0
                
                # Set up signal aggregator
                self.signal_aggregator.set_predicted_return(predicted_return)
                self.signal_aggregator.set_features(features)
                self.signal_aggregator.set_trend_filter(
                    self.config.get('use_trend_filter', True),
                    current_price,
                    df['ema_50'].iloc[i] if 'ema_50' in df.columns else current_price
                )
                
                # Check for entry signals
                if len(positions) < self.config.get('max_positions', 3):
                    if self.signal_aggregator.should_enter_trade():
                        # Get signal direction
                        direction = self.signal_aggregator.get_signal_direction()
                        
                        # Calculate stop loss
                        atr = df['atr_14'].iloc[i]
                        sl_distance = atr * self.config.get('atr_multiplier', 1.0)
                        
                        if direction == 'BUY':
                            sl_price = current_price - sl_distance
                        else:
                            sl_price = current_price + sl_distance
                        
                        # Calculate position size
                        position_size = self.risk_manager.calculate_position_size(
                            current_price, sl_price, symbol_info, account_info
                        )
                        
                        # Open position
                        position_id = f"{direction.lower()}_{timestamp.strftime('%Y%m%d%H%M%S')}"
                        positions[position_id] = {
                            'type': direction,
                            'open_time': timestamp,
                            'open_price': current_price,
                            'size': position_size,
                            'sl': sl_price,
                            'tp': None,
                            'regime': regime
                        }
                        
                        logger.info(f"Backtest: Opened {direction} position at {current_price} (ID: {position_id})")
                
                # Manage open positions
                positions_to_close: List[Tuple[str, float, float, str]] = []
                
                for position_id, position in positions.items():
                    # Check stop loss
                    if position['type'] == 'BUY' and current_price <= position['sl']:
                        # Close position at stop loss
                        profit = (position['sl'] - position['open_price']) * position['size']
                        positions_to_close.append((position_id, position['sl'], profit, 'sl'))
                        
                    elif position['type'] == 'SELL' and current_price >= position['sl']:
                        # Close position at stop loss
                        profit = (position['open_price'] - position['sl']) * position['size']
                        positions_to_close.append((position_id, position['sl'], profit, 'sl'))
                    
                    # Check take profit if set
                    elif position['tp'] is not None:
                        if position['type'] == 'BUY' and current_price >= position['tp']:
                            # Close position at take profit
                            profit = (position['tp'] - position['open_price']) * position['size']
                            positions_to_close.append((position_id, position['tp'], profit, 'tp'))
                            
                        elif position['type'] == 'SELL' and current_price <= position['tp']:
                            # Close position at take profit
                            profit = (position['open_price'] - position['tp']) * position['size']
                            positions_to_close.append((position_id, position['tp'], profit, 'tp'))
                    
                    # Check for exit signals
                    elif self.signal_aggregator.should_exit_trade(position_id):
                        if position['type'] == 'BUY':
                            profit = (current_price - position['open_price']) * position['size']
                        else:
                            profit = (position['open_price'] - current_price) * position['size']
                        
                        positions_to_close.append((position_id, current_price, profit, 'signal'))
                
                # Close positions
                for position_id, close_price, profit, reason in positions_to_close:
                    position = positions.pop(position_id)
                    
                    # Update balance
                    balance += profit
                    account_info['balance'] = balance
                    
                    # Record trade
                    trade = {
                        'id': position_id,
                        'type': position['type'],
                        'open_time': position['open_time'],
                        'close_time': timestamp,
                        'open_price': position['open_price'],
                        'close_price': close_price,
                        'size': position['size'],
                        'profit': profit,
                        'reason': reason,
                        'regime': position['regime']
                    }
                    
                    trades.append(trade)
                    
                    # Update risk parameters
                    self.risk_manager.update_risk_parameters(profit)
                    
                    # Update signal weights
                    self.signal_aggregator.update_weights(profit)
                    
                    # Update attention weights
                    self.attention.update_weights(features, profit)
                    
                    logger.info(f"Backtest: Closed {position['type']} position at {close_price} with profit {profit:.2f} (ID: {position_id}, Reason: {reason})")
                
                # Update equity
                open_positions_value = 0
                for position in positions.values():
                    if position['type'] == 'BUY':
                        open_positions_value += (current_price - position['open_price']) * position['size']
                    else:
                        open_positions_value += (position['open_price'] - current_price) * position['size']
                
                equity = balance + open_positions_value
                account_info['equity'] = equity
            
            # Close any remaining positions at the last price
            last_price = df['close'].iloc[-1]
            last_timestamp = df.index[-1]
            
            for position_id, position in list(positions.items()):
                if position['type'] == 'BUY':
                    profit = (last_price - position['open_price']) * position['size']
                else:
                    profit = (position['open_price'] - last_price) * position['size']
                
                # Update balance
                balance += profit
                
                # Record trade
                trade = {
                    'id': position_id,
                    'type': position['type'],
                    'open_time': position['open_time'],
                    'close_time': last_timestamp,
                    'open_price': position['open_price'],
                    'close_price': last_price,
                    'size': position['size'],
                    'profit': profit,
                    'reason': 'end',
                    'regime': position['regime']
                }
                
                trades.append(trade)
                
                logger.info(f"Backtest: Closed {position['type']} position at {last_price} with profit {profit:.2f} (ID: {position_id}, Reason: end)")
            
            # Calculate backtest metrics
            initial_balance = self.config.get('initial_balance', 10000)
            total_profit = balance - initial_balance
            profit_factor = 0.0
            win_rate = 0.0
            
            if len(trades) > 0:
                winning_trades = [t for t in trades if t['profit'] > 0]
                losing_trades = [t for t in trades if t['profit'] <= 0]
                
                win_rate = len(winning_trades) / len(trades)
                
                gross_profit = sum(t['profit'] for t in winning_trades)
                gross_loss = abs(sum(t['profit'] for t in losing_trades))
                
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
            
            # Calculate drawdown
            equity_curve = []
            current_equity = initial_balance
            peak_equity = initial_balance
            max_drawdown = 0.0
            
            for trade in trades:
                current_equity += trade['profit']
                equity_curve.append(current_equity)
                
                if current_equity > peak_equity:
                    peak_equity = current_equity
                
                drawdown = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Prepare results
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_profit': total_profit,
                'return_pct': (total_profit / initial_balance) * 100,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'trades': trades,
                'parameters': self.params_config
            }
            
            logger.info(f"Backtest completed for {symbol} {timeframe}: Profit: {total_profit:.2f}, Win Rate: {win_rate:.2f}, Drawdown: {max_drawdown:.2f}%")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in backtest: {e}")
            return None
            
        finally:
            # Restore original parameters if they were overridden
            if original_params is not None:
                self.update_parameters(original_params)
    
    def _detect_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect market regimes in historical data.
        
        Args:
            df: Historical price data
            
        Returns:
            Array of regime labels
        """
        # Calculate regime features
        trend_strength = np.abs(df['trend_strength_20'])
        volatility = df['atr_14'] / df['close']
        cyclicality = df['cyclicality_20'] if 'cyclicality_20' in df.columns else np.zeros(len(df))
        
        # Initialize regimes
        regimes = np.zeros(len(df))
        
        # Get regime thresholds from parameters
        trend_threshold = 0.6
        volatility_threshold = 0.01
        
        if self.params_config and 'regime_detection' in self.params_config:
            trend_threshold = self.params_config['regime_detection'].get('trend_strength_threshold', trend_threshold)
            volatility_threshold = self.params_config['regime_detection'].get('volatility_threshold', volatility_threshold)
        
        # Classify regimes
        # 0: Trending
        # 1: Ranging
        # 2: Volatile
        # 3: Choppy
        
        for i in range(len(df)):
            if trend_strength.iloc[i] > trend_threshold:
                if volatility.iloc[i] > volatility_threshold:
                    regimes[i] = 2  # Volatile
                else:
                    regimes[i] = 0  # Trending
            else:
                if volatility.iloc[i] > volatility_threshold:
                    regimes[i] = 3  # Choppy
                else:
                    regimes[i] = 1  # Ranging
        
        return regimes
