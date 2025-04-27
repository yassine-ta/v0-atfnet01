#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Risk Management module for ATFNet.
Implements sophisticated risk management techniques for trading strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger('atfnet.core.advanced_risk')


class AdvancedRiskManager:
    """
    Advanced risk management system for ATFNet.
    Implements sophisticated risk management techniques including:
    - Dynamic position sizing
    - Tail risk hedging
    - Adaptive stop-loss placement
    - Portfolio-level risk management
    - Drawdown control
    - Volatility targeting
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'risk_management'):
        """
        Initialize the advanced risk manager.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% per trade
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.05)  # 5% portfolio risk
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)  # 15% max drawdown
        self.target_volatility = self.config.get('target_volatility', 0.10)  # 10% annualized vol
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% risk-free rate
        self.confidence_level = self.config.get('confidence_level', 0.95)  # 95% confidence for VaR
        
        # Initialize state variables
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.positions = {}
        self.portfolio_history = []
        self.risk_metrics_history = []
        self.trade_history = []
        
        # Initialize risk models
        self.volatility_model = ExponentialVolatilityModel(
            alpha=self.config.get('volatility_alpha', 0.94),
            lookback=self.config.get('volatility_lookback', 20)
        )
        
        self.correlation_model = DynamicCorrelationModel(
            lookback=self.config.get('correlation_lookback', 50),
            min_periods=self.config.get('correlation_min_periods', 20)
        )
        
        self.tail_risk_model = TailRiskModel(
            confidence_level=self.confidence_level,
            lookback=self.config.get('tail_risk_lookback', 100)
        )
        
        # Initialize regime-based parameters
        self.regime_risk_adjustments = {
            'trending': 1.0,      # Normal risk in trending markets
            'ranging': 0.8,       # Reduced risk in ranging markets
            'volatile': 0.6,      # Significantly reduced risk in volatile markets
            'choppy': 0.5         # Minimal risk in choppy markets
        }
        
        # Initialize hedging parameters
        self.hedging_enabled = self.config.get('hedging_enabled', False)
        self.hedge_instruments = self.config.get('hedge_instruments', [])
        self.hedge_allocation = self.config.get('hedge_allocation', 0.05)  # 5% allocation to hedges
        
        logger.info("Advanced Risk Manager initialized")
    
    def update_capital(self, new_capital: float) -> None:
        """
        Update current capital.
        
        Args:
            new_capital: New capital amount
        """
        self.current_capital = new_capital
        
        # Update peak capital and drawdown
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        self.current_drawdown = 1.0 - (self.current_capital / self.peak_capital)
        
        # Log significant drawdown events
        if self.current_drawdown > 0.1:  # 10% drawdown
            logger.warning(f"Significant drawdown: {self.current_drawdown:.2%}")
        
        # Update portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': self.current_drawdown
        })
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                               market_state: Dict[str, Any] = None) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            market_state: Current market state information
            
        Returns:
            Position size in units
        """
        # Calculate risk per trade based on current capital
        risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Adjust risk based on market regime
        if market_state and 'regime' in market_state:
            regime = market_state['regime']
            regime_adjustment = self.regime_risk_adjustments.get(regime, 1.0)
            risk_amount *= regime_adjustment
        
        # Adjust risk based on current drawdown
        drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown_limit)
        drawdown_factor = max(0.25, min(1.0, drawdown_factor))  # Limit between 0.25 and 1.0
        risk_amount *= drawdown_factor
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.warning(f"Invalid risk per unit: {risk_per_unit}. Using 1% of price.")
            risk_per_unit = entry_price * 0.01  # Default to 1% of price
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Adjust for portfolio risk concentration
        portfolio_adjustment = self._calculate_portfolio_adjustment(symbol, position_size, entry_price)
        position_size *= portfolio_adjustment
        
        # Adjust for volatility targeting
        if market_state and 'volatility' in market_state:
            volatility_adjustment = self._calculate_volatility_adjustment(market_state['volatility'])
            position_size *= volatility_adjustment
        
        logger.info(f"Position size for {symbol}: {position_size:.2f} units "
                   f"(Risk: ${risk_amount:.2f}, DD Factor: {drawdown_factor:.2f}, "
                   f"Portfolio Adj: {portfolio_adjustment:.2f})")
        
        return position_size
    
    def _calculate_portfolio_adjustment(self, symbol: str, position_size: float, 
                                       price: float) -> float:
        """
        Calculate portfolio adjustment factor based on concentration risk.
        
        Args:
            symbol: Trading symbol
            position_size: Calculated position size
            price: Current price
            
        Returns:
            Adjustment factor (0.0 to 1.0)
        """
        # Calculate position value
        position_value = position_size * price
        
        # Calculate current portfolio exposure
        portfolio_exposure = 0.0
        for sym, pos in self.positions.items():
            portfolio_exposure += pos['size'] * pos['current_price']
        
        # Add new position
        total_exposure = portfolio_exposure + position_value
        
        # Calculate concentration
        concentration = position_value / self.current_capital
        
        # Adjust based on concentration
        if concentration > 0.2:  # More than 20% in one position
            return 0.2 / concentration  # Scale down to 20%
        
        # Adjust based on total exposure
        max_exposure = self.current_capital * 1.5  # Allow up to 150% exposure (with leverage)
        if total_exposure > max_exposure:
            return max_exposure / total_exposure
        
        return 1.0
    
    def _calculate_volatility_adjustment(self, current_volatility: float) -> float:
        """
        Calculate adjustment factor based on volatility targeting.
        
        Args:
            current_volatility: Current volatility estimate
            
        Returns:
            Adjustment factor
        """
        if current_volatility <= 0:
            return 1.0
        
        # Target constant volatility
        adjustment = self.target_volatility / current_volatility
        
        # Limit adjustment factor
        adjustment = max(0.2, min(2.0, adjustment))
        
        return adjustment
    
    def update_position(self, symbol: str, current_price: float, 
                       market_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update position information and risk metrics.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            market_state: Current market state
            
        Returns:
            Updated position information
        """
        if symbol not in self.positions:
            logger.warning(f"Position for {symbol} not found")
            return {}
        
        position = self.positions[symbol]
        
        # Update price
        position['current_price'] = current_price
        
        # Calculate profit/loss
        if position['direction'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
        
        position['unrealized_pnl_pct'] = position['unrealized_pnl'] / (position['entry_price'] * position['size'])
        
        # Update stop loss if trailing
        if position.get('trailing_stop', False):
            self._update_trailing_stop(position, current_price)
        
        # Update take profit if dynamic
        if position.get('dynamic_take_profit', False):
            self._update_dynamic_take_profit(position, current_price, market_state)
        
        # Check for stop loss or take profit hit
        exit_signal = self._check_exit_conditions(position, current_price)
        
        return {
            'position': position,
            'exit_signal': exit_signal
        }
    
    def _update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> None:
        """
        Update trailing stop loss.
        
        Args:
            position: Position information
            current_price: Current price
        """
        if position['direction'] == 'long':
            # For long positions, move stop loss up
            if current_price > position.get('highest_price', position['entry_price']):
                position['highest_price'] = current_price
                
                # Calculate new stop loss
                trail_amount = position['highest_price'] * position.get('trailing_stop_pct', 0.02)
                new_stop = position['highest_price'] - trail_amount
                
                # Only move stop loss up, never down
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.info(f"Updated trailing stop for {position['symbol']} to {new_stop:.4f}")
        else:
            # For short positions, move stop loss down
            if current_price < position.get('lowest_price', position['entry_price']):
                position['lowest_price'] = current_price
                
                # Calculate new stop loss
                trail_amount = position['lowest_price'] * position.get('trailing_stop_pct', 0.02)
                new_stop = position['lowest_price'] + trail_amount
                
                # Only move stop loss down, never up
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    logger.info(f"Updated trailing stop for {position['symbol']} to {new_stop:.4f}")
    
    def _update_dynamic_take_profit(self, position: Dict[str, Any], 
                                   current_price: float,
                                   market_state: Dict[str, Any] = None) -> None:
        """
        Update dynamic take profit levels based on market conditions.
        
        Args:
            position: Position information
            current_price: Current price
            market_state: Current market state
        """
        if not market_state:
            return
        
        # Get volatility from market state
        volatility = market_state.get('volatility', 0.01)
        
        # Get regime from market state
        regime = market_state.get('regime', 'unknown')
        
        # Adjust take profit based on regime and volatility
        if regime == 'trending':
            # In trending markets, set wider take profits
            tp_multiple = 3.0
        elif regime == 'ranging':
            # In ranging markets, set tighter take profits
            tp_multiple = 1.5
        elif regime == 'volatile':
            # In volatile markets, set moderate take profits
            tp_multiple = 2.0
        else:
            # Default
            tp_multiple = 2.0
        
        # Calculate take profit distance based on volatility
        tp_distance = volatility * current_price * tp_multiple
        
        if position['direction'] == 'long':
            new_take_profit = position['entry_price'] + tp_distance
            
            # Only update if new take profit is higher than current (for unrealized profits)
            if position.get('take_profit') is None or new_take_profit > position['take_profit']:
                position['take_profit'] = new_take_profit
                logger.info(f"Updated take profit for {position['symbol']} to {new_take_profit:.4f}")
        else:
            new_take_profit = position['entry_price'] - tp_distance
            
            # Only update if new take profit is lower than current (for unrealized profits)
            if position.get('take_profit') is None or new_take_profit < position['take_profit']:
                position['take_profit'] = new_take_profit
                logger.info(f"Updated take profit for {position['symbol']} to {new_take_profit:.4f}")
    
    def _check_exit_conditions(self, position: Dict[str, Any], current_price: float) -> bool:
        """
        Check if exit conditions are met.
        
        Args:
            position: Position information
            current_price: Current price
            
        Returns:
            True if exit conditions are met, False otherwise
        """
        # Check stop loss
        if position['direction'] == 'long' and current_price <= position['stop_loss']:
            logger.info(f"Stop loss triggered for {position['symbol']} at {current_price:.4f}")
            return True
        
        if position['direction'] == 'short' and current_price >= position['stop_loss']:
            logger.info(f"Stop loss triggered for {position['symbol']} at {current_price:.4f}")
            return True
        
        # Check take profit
        if position.get('take_profit') is not None:
            if position['direction'] == 'long' and current_price >= position['take_profit']:
                logger.info(f"Take profit triggered for {position['symbol']} at {current_price:.4f}")
                return True
            
            if position['direction'] == 'short' and current_price <= position['take_profit']:
                logger.info(f"Take profit triggered for {position['symbol']} at {current_price:.4f}")
                return True
        
        # Check time-based exit
        if position.get('max_holding_time') is not None:
            holding_time = datetime.now() - position['entry_time']
            if holding_time.total_seconds() / 3600 >= position['max_holding_time']:
                logger.info(f"Time-based exit triggered for {position['symbol']} after {holding_time}")
                return True
        
        return False
    
    def add_position(self, symbol: str, direction: str, entry_price: float, 
                    size: float, stop_loss: float, take_profit: float = None,
                    trailing_stop: bool = False, trailing_stop_pct: float = 0.02,
                    dynamic_take_profit: bool = False, max_holding_time: float = None) -> None:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            trailing_stop: Whether to use trailing stop
            trailing_stop_pct: Trailing stop percentage
            dynamic_take_profit: Whether to use dynamic take profit
            max_holding_time: Maximum holding time in hours
        """
        # Validate direction
        if direction not in ['long', 'short']:
            raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'.")
        
        # Create position
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'trailing_stop': trailing_stop,
            'trailing_stop_pct': trailing_stop_pct,
            'dynamic_take_profit': dynamic_take_profit,
            'max_holding_time': max_holding_time,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0
        }
        
        # Add position tracking for trailing stop
        if trailing_stop:
            if direction == 'long':
                position['highest_price'] = entry_price
            else:
                position['lowest_price'] = entry_price
        
        # Add to positions
        self.positions[symbol] = position
        
        # Log position
        logger.info(f"Added {direction} position for {symbol}: {size} units at {entry_price:.4f}, "
                   f"SL: {stop_loss:.4f}, TP: {take_profit if take_profit else 'None'}")
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime = None) -> Dict[str, Any]:
        """
        Close a position and record the trade.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit time (defaults to now)
            
        Returns:
            Closed position information
        """
        if symbol not in self.positions:
            logger.warning(f"Position for {symbol} not found")
            return {}
        
        position = self.positions.pop(symbol)
        
        # Set exit time
        if exit_time is None:
            exit_time = datetime.now()
        
        # Calculate realized P&L
        if position['direction'] == 'long':
            realized_pnl = (exit_price - position['entry_price']) * position['size']
        else:
            realized_pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Calculate realized P&L percentage
        realized_pnl_pct = realized_pnl / (position['entry_price'] * position['size'])
        
        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'holding_time': (exit_time - position['entry_time']).total_seconds() / 3600,  # in hours
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'stop_loss': position['stop_loss'],
            'take_profit': position.get('take_profit'),
            'exit_reason': 'unknown'  # This should be set by the calling code
        }
        
        # Add to trade history
        self.trade_history.append(trade)
        
        # Update capital
        self.update_capital(self.current_capital + realized_pnl)
        
        # Log trade
        logger.info(f"Closed {position['direction']} position for {symbol}: {position['size']} units, "
                   f"Entry: {position['entry_price']:.4f}, Exit: {exit_price:.4f}, "
                   f"P&L: {realized_pnl:.2f} ({realized_pnl_pct:.2%})")
        
        return trade
    
    def calculate_portfolio_risk(self, price_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            price_data: Dictionary of price data for each symbol
            
        Returns:
            Dictionary of risk metrics
        """
        if not self.positions:
            return {
                'var': 0.0,
                'cvar': 0.0,
                'portfolio_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'portfolio_beta': 0.0,
                'position_concentration': {},
                'sector_concentration': {},
                'drawdown': self.current_drawdown
            }
        
        # Calculate position values and weights
        position_values = {}
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            position_value = position['size'] * position['current_price']
            position_values[symbol] = position_value
            total_value += position_value
        
        # Calculate position weights
        position_weights = {symbol: value / total_value for symbol, value in position_values.items()}
        
        # Calculate concentration metrics
        position_concentration = {symbol: weight for symbol, weight in position_weights.items()}
        
        # Calculate portfolio volatility and VaR if price data is provided
        portfolio_volatility = 0.0
        var = 0.0
        cvar = 0.0
        sharpe_ratio = 0.0
        portfolio_beta = 0.0
        
        if price_data:
            # Calculate returns
            returns = {}
            for symbol in self.positions:
                if symbol in price_data:
                    returns[symbol] = price_data[symbol]['close'].pct_change().dropna()
            
            if returns:
                # Calculate covariance matrix
                returns_df = pd.DataFrame(returns)
                cov_matrix = returns_df.cov()
                
                # Calculate portfolio volatility
                portfolio_variance = 0.0
                for i, symbol_i in enumerate(position_weights):
                    for j, symbol_j in enumerate(position_weights):
                        if symbol_i in cov_matrix.index and symbol_j in cov_matrix.columns:
                            portfolio_variance += (position_weights[symbol_i] * position_weights[symbol_j] * 
                                                 cov_matrix.loc[symbol_i, symbol_j])
                
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate VaR and CVaR
                portfolio_returns = sum(returns_df[symbol] * position_weights.get(symbol, 0) 
                                      for symbol in returns_df.columns)
                
                var = self.tail_risk_model.calculate_var(portfolio_returns)
                cvar = self.tail_risk_model.calculate_cvar(portfolio_returns)
                
                # Calculate Sharpe ratio
                avg_return = portfolio_returns.mean()
                sharpe_ratio = (avg_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Calculate portfolio beta (assuming first symbol is market)
                if len(returns_df.columns) > 0:
                    market_symbol = returns_df.columns[0]
                    market_returns = returns_df[market_symbol]
                    
                    portfolio_beta = sum(
                        position_weights.get(symbol, 0) * 
                        (returns_df[symbol].cov(market_returns) / market_returns.var())
                        for symbol in returns_df.columns if symbol != market_symbol
                    )
        
        # Prepare risk metrics
        risk_metrics = {
            'var': var,
            'cvar': cvar,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_beta': portfolio_beta,
            'position_concentration': position_concentration,
            'sector_concentration': {},  # Would need sector data
            'drawdown': self.current_drawdown,
            'total_exposure': total_value / self.current_capital
        }
        
        # Add to risk metrics history
        self.risk_metrics_history.append({
            'timestamp': datetime.now(),
            **risk_metrics
        })
        
        return risk_metrics
    
    def check_portfolio_limits(self, risk_metrics: Dict[str, Any] = None) -> Dict[str, bool]:
        """
        Check if portfolio risk limits are exceeded.
        
        Args:
            risk_metrics: Risk metrics (if None, will be calculated)
            
        Returns:
            Dictionary of limit checks
        """
        if risk_metrics is None:
            risk_metrics = self.calculate_portfolio_risk()
        
        # Check drawdown limit
        drawdown_exceeded = risk_metrics['drawdown'] > self.max_drawdown_limit
        
        # Check VaR limit
        var_limit = self.max_portfolio_risk
        var_exceeded = risk_metrics['var'] > var_limit
        
        # Check exposure limit
        exposure_limit = 1.5  # 150% of capital
        exposure_exceeded = risk_metrics['total_exposure'] > exposure_limit
        
        # Check concentration limit
        concentration_limit = 0.3  # 30% in any one position
        concentration_exceeded = any(weight > concentration_limit 
                                   for weight in risk_metrics['position_concentration'].values())
        
        # Prepare limit checks
        limit_checks = {
            'drawdown_exceeded': drawdown_exceeded,
            'var_exceeded': var_exceeded,
            'exposure_exceeded': exposure_exceeded,
            'concentration_exceeded': concentration_exceeded
        }
        
        # Log warnings for exceeded limits
        for limit, exceeded in limit_checks.items():
            if exceeded:
                logger.warning(f"Risk limit exceeded: {limit}")
        
        return limit_checks
    
    def get_risk_reduction_actions(self, limit_checks: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Get recommended actions to reduce risk when limits are exceeded.
        
        Args:
            limit_checks: Results from check_portfolio_limits
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        if any(limit_checks.values()):
            # Calculate position to reduce for each position
            for symbol, position in self.positions.items():
                position_value = position['size'] * position['current_price']
                position_risk = position_value / self.current_capital
                
                # Determine reduction percentage based on which limits are exceeded
                reduction_pct = 0.0
                
                if limit_checks.get('drawdown_exceeded', False):
                    reduction_pct = max(reduction_pct, 0.5)  # Reduce by 50% for drawdown
                
                if limit_checks.get('var_exceeded', False):
                    reduction_pct = max(reduction_pct, 0.3)  # Reduce by 30% for VaR
                
                if limit_checks.get('exposure_exceeded', False):
                    reduction_pct = max(reduction_pct, 0.2)  # Reduce by 20% for exposure
                
                if limit_checks.get('concentration_exceeded', False) and position_risk > 0.3:
                    # Reduce concentrated positions more aggressively
                    reduction_pct = max(reduction_pct, 0.4)
                
                if reduction_pct > 0:
                    actions.append({
                        'action': 'reduce_position',
                        'symbol': symbol,
                        'current_size': position['size'],
                        'reduction_pct': reduction_pct,
                        'target_size': position['size'] * (1 - reduction_pct),
                        'reason': 'risk_limit_exceeded'
                    })
        
        return actions
    
    def calculate_optimal_portfolio(self, expected_returns: Dict[str, float], 
                                   covariance_matrix: pd.DataFrame,
                                   risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Dictionary of expected returns for each symbol
            covariance_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary of optimal weights for each symbol
        """
        # Convert inputs to numpy arrays
        symbols = list(expected_returns.keys())
        returns_vector = np.array([expected_returns[s] for s in symbols])
        
        # Ensure covariance matrix has the same symbols
        cov_matrix = covariance_matrix.loc[symbols, symbols].values
        
        # Define objective function to minimize (negative utility)
        def objective(weights):
            portfolio_return = np.dot(weights, returns_vector)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
            return -utility
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum of weights = 1
        ]
        
        # Define bounds (allow short selling up to 20%)
        bounds = [(-0.2, 1.2) for _ in range(len(symbols))]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(len(symbols)) / len(symbols)
        
        # Solve optimization problem
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            # Convert result to dictionary
            optimal_weights = {symbol: weight for symbol, weight in zip(symbols, result.x)}
            
            # Log result
            logger.info(f"Optimal portfolio calculated: {optimal_weights}")
            
            return optimal_weights
        else:
            logger.error(f"Portfolio optimization failed: {result.message}")
            return {symbol: 1.0 / len(symbols) for symbol in symbols}  # Equal weights as fallback
    
    def plot_risk_metrics(self, save_plot: bool = True, 
                         filename: str = 'risk_metrics.png') -> None:
        """
        Plot risk metrics history.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.risk_metrics_history:
            logger.warning("No risk metrics history to plot")
            return
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.risk_metrics_history)
        history_df.set_index('timestamp', inplace=True)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot portfolio volatility
        ax1.plot(history_df.index, history_df['portfolio_volatility'], color='blue')
        ax1.set_ylabel('Portfolio Volatility')
        ax1.set_title('Risk Metrics History')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot VaR and CVaR
        ax2.plot(history_df.index, history_df['var'], color='red', label='VaR')
        ax2.plot(history_df.index, history_df['cvar'], color='darkred', label='CVaR')
        ax2.set_ylabel('Value at Risk')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot drawdown
        ax3.plot(history_df.index, history_df['drawdown'], color='orange')
        ax3.set_ylabel('Drawdown')
        ax3.set_xlabel('Time')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Risk metrics plot saved to {self.output_dir}/{filename}")
        
        plt.close()
    
    def plot_portfolio_composition(self, save_plot: bool = True,
                                  filename: str = 'portfolio_composition.png') -> None:
        """
        Plot current portfolio composition.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.positions:
            logger.warning("No positions to plot")
            return
        
        # Calculate position values
        position_values = {}
        for symbol, position in self.positions.items():
            position_values[symbol] = position['size'] * position['current_price']
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create pie chart
        plt.pie(position_values.values(), labels=position_values.keys(), autopct='%1.1f%%',
               shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Portfolio Composition')
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Portfolio composition plot saved to {self.output_dir}/{filename}")
        
        plt.close()


class ExponentialVolatilityModel:
    """
    Exponential volatility model for estimating asset volatility.
    Uses exponentially weighted moving average (EWMA) for volatility estimation.
    """
    
    def __init__(self, alpha: float = 0.94, lookback: int = 20):
        """
        Initialize the exponential volatility model.
        
        Args:
            alpha: Decay factor (higher = more weight on historical data)
            lookback: Lookback period for initialization
        """
        self.alpha = alpha
        self.lookback = lookback
        self.volatility = None
        self.returns_history = []
    
    def update(self, returns: Union[float, List[float]]) -> float:
        """
        Update volatility estimate with new returns.
        
        Args:
            returns: New return(s) to incorporate
            
        Returns:
            Updated volatility estimate
        """
        # Handle single return or list of returns
        if isinstance(returns, (int, float)):
            self.returns_history.append(returns)
        else:
            self.returns_history.extend(returns)
        
        # Keep only lookback period
        if len(self.returns_history) > self.lookback:
            self.returns_history = self.returns_history[-self.lookback:]
        
        # Initialize volatility if needed
        if self.volatility is None and len(self.returns_history) > 1:
            self.volatility = np.std(self.returns_history)
        
        # Update volatility using EWMA
        if self.volatility is not None and isinstance(returns, (int, float)):
            self.volatility = np.sqrt(self.alpha * self.volatility**2 + (1 - self.alpha) * returns**2)
        
        return self.volatility


class DynamicCorrelationModel:
    """
    Dynamic correlation model for estimating asset correlations.
    Uses rolling window for correlation estimation.
    """
    
    def __init__(self, lookback: int = 50, min_periods: int = 20):
        """
        Initialize the dynamic correlation model.
        
        Args:
            lookback: Lookback period for correlation calculation
            min_periods: Minimum periods required for calculation
        """
        self.lookback = lookback
        self.min_periods = min_periods
        self.returns_history = {}
        self.correlation_matrix = None
    
    def update(self, returns_dict: Dict[str, float]) -> pd.DataFrame:
        """
        Update correlation estimates with new returns.
        
        Args:
            returns_dict: Dictionary of returns for each asset
            
        Returns:
            Updated correlation matrix
        """
        # Update returns history
        for symbol, ret in returns_dict.items():
            if symbol not in self.returns_history:
                self.returns_history[symbol] = []
            
            self.returns_history[symbol].append(ret)
            
            # Keep only lookback period
            if len(self.returns_history[symbol]) > self.lookback:
                self.returns_history[symbol] = self.returns_history[symbol][-self.lookback:]
        
        # Calculate correlation matrix if enough data
        min_length = min(len(hist) for hist in self.returns_history.values()) if self.returns_history else 0
        
        if min_length >= self.min_periods:
            # Convert to DataFrame
            returns_df = pd.DataFrame(self.returns_history)
            
            # Calculate correlation matrix
            self.correlation_matrix = returns_df.corr()
        
        return self.correlation_matrix


class TailRiskModel:
    """
    Tail risk model for estimating Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    
    def __init__(self, confidence_level: float = 0.95, lookback: int = 100):
        """
        Initialize the tail risk model.
        
        Args:
            confidence_level: Confidence level for VaR calculation
            lookback: Lookback period for calculation
        """
        self.confidence_level = confidence_level
        self.lookback = lookback
        self.returns_history = []
    
    def update(self, returns: Union[float, List[float]]) -> None:
        """
        Update returns history.
        
        Args:
            returns: New return(s) to incorporate
        """
        # Handle single return or list of returns
        if isinstance(returns, (int, float)):
            self.returns_history.append(returns)
        else:
            self.returns_history.extend(returns)
        
        # Keep only lookback period
        if len(self.returns_history) > self.lookback:
            self.returns_history = self.returns_history[-self.lookback:]
    
    def calculate_var(self, returns: pd.Series = None) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Returns series (uses history if None)
            
        Returns:
            VaR estimate
        """
        if returns is None:
            if not self.returns_history:
                return 0.0
            returns = pd.Series(self.returns_history)
        
        # Calculate VaR using historical method
        var = -returns.quantile(1 - self.confidence_level)
        
        return var
    
    def calculate_cvar(self, returns: pd.Series = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Returns series (uses history if None)
            
        Returns:
            CVaR estimate
        """
        if returns is None:
            if not self.returns_history:
                return 0.0
            returns = pd.Series(self.returns_history)
        
        # Calculate VaR
        var = self.calculate_var(returns)
        
        # Calculate CVaR as average of returns beyond VaR
        cvar = -returns[returns < -var].mean()
        
        # Handle case where no returns exceed VaR
        if np.isnan(cvar):
            cvar = var
        
        return cvar
