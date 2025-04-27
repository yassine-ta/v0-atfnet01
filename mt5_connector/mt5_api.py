#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MetaTrader 5 API connector for the ATFNet Python implementation.
This module handles all interactions with the MT5 platform.
"""

import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import MetaTrader5 as mt5
import os

logger = logging.getLogger('atfnet.mt5_api')


class MT5API:
    """MetaTrader 5 API connector class."""
    
    def __init__(self, config):
        """
        Initialize the MT5 API connector.
        
        Args:
            config (dict): MT5 connection configuration
        """
        self.config = config
        self.initialized = False
        self.timezone = pytz.timezone(config.get('timezone', 'UTC'))
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
    
    def initialize(self):
        """
        Initialize connection to MetaTrader 5 terminal.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if self.initialized:
            return True
        
        # Initialize MT5 connection
        if not mt5.initialize(
            path=self.config.get('terminal_path', ''),
            login=self.config.get('login', 0),
            password=self.config.get('password', ''),
            server=self.config.get('server', ''),
            timeout=self.config.get('timeout', 60000)
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        self.initialized = True
        logger.info("MT5 connection initialized successfully")
        return True
    
    def shutdown(self):
        """Shutdown connection to MetaTrader 5 terminal."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 connection shutdown")
    
    def is_initialized(self):
        """Check if MT5 connection is initialized."""
        return self.initialized
    
    def account_info(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
        
        # Convert to dict
        return {
            'login': account_info.login,
            'server': account_info.server,
            'currency': account_info.currency,
            'leverage': account_info.leverage,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level
        }
    
    def get_symbols(self):
        """
        Get available symbols.
        
        Returns:
            list: List of available symbols
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(f"Failed to get symbols: {mt5.last_error()}")
            return []
        
        return [symbol.name for symbol in symbols]
    
    def get_symbol_info(self, symbol):
        """
        Get symbol information.
        
        Args:
            symbol (str): Symbol name
            
        Returns:
            dict: Symbol information
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None
        
        # Convert to dict
        return {
            'name': symbol_info.name,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'tick_value': symbol_info.trade_tick_value,
            'tick_size': symbol_info.trade_tick_size,
            'contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step
        }
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date=None, count=None):
        """
        Get historical price data.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'H1')
            start_date (datetime): Start date
            end_date (datetime, optional): End date. Defaults to None.
            count (int, optional): Number of bars to fetch. Defaults to None.
            
        Returns:
            pd.DataFrame: Historical price data
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return None
        
        # Convert timeframe string to MT5 timeframe
        mt5_timeframe = self.timeframe_map.get(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        # Convert dates to UTC timestamp
        start_date = pd.Timestamp(start_date).tz_localize(self.timezone).tz_convert('UTC')
        
        if end_date is not None:
            end_date = pd.Timestamp(end_date).tz_localize(self.timezone).tz_convert('UTC')
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        elif count is not None:
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, count)
        else:
            end_date = datetime.now(self.timezone).astimezone(pytz.UTC)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get historical data for {symbol} {timeframe}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_latest_bars(self, symbol, timeframe, count=100):
        """
        Get the latest price bars.
        
        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'H1')
            count (int, optional): Number of bars to fetch. Defaults to 100.
            
        Returns:
            pd.DataFrame: Latest price bars
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return None
        
        # Convert timeframe string to MT5 timeframe
        mt5_timeframe = self.timeframe_map.get(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get latest bars for {symbol} {timeframe}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def place_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=None):
        """
        Place a trading order.
        
        Args:
            symbol (str): Symbol name
            order_type (str): Order type ('BUY', 'SELL')
            volume (float): Order volume
            price (float, optional): Order price. Defaults to None (market price).
            sl (float, optional): Stop loss price. Defaults to None.
            tp (float, optional): Take profit price. Defaults to None.
            comment (str, optional): Order comment. Defaults to None.
            
        Returns:
            dict: Order result
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return None
        
        # Map order type string to MT5 order type
        order_type_map = {
            'BUY': mt5.ORDER_TYPE_BUY,
            'SELL': mt5.ORDER_TYPE_SELL,
            'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
            'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
            'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
            'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
        }
        
        mt5_order_type = order_type_map.get(order_type)
        if mt5_order_type is None:
            logger.error(f"Invalid order type: {order_type}")
            return None
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None
        
        # Enable symbol for trading if needed
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}")
                return None
        
        # Get current price if not provided
        if price is None:
            if mt5_order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_LIMIT]:
                price = mt5.symbol_info_tick(symbol).ask
            else:
                price = mt5.symbol_info_tick(symbol).bid
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": price,
            "deviation": self.config.get('deviation', 10),
            "magic": self.config.get('magic', 123456),
            "comment": comment or "ATFNet Python",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}, {result.comment}")
            return None
        
        logger.info(f"Order placed successfully: {result.order}")
        
        # Return order info
        return {
            'order_id': result.order,
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': price,
            'sl': sl,
            'tp': tp
        }
    
    def close_position(self, position_id):
        """
        Close an open position.
        
        Args:
            position_id (int): Position ID
            
        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            logger.error(f"Position {position_id} not found: {mt5.last_error()}")
            return False
        
        position = position[0]
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position_id,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": self.config.get('deviation', 10),
            "magic": self.config.get('magic', 123456),
            "comment": "ATFNet Python Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close position failed: {result.retcode}, {result.comment}")
            return False
        
        logger.info(f"Position {position_id} closed successfully")
        return True
    
    def get_open_positions(self, symbol=None):
        """
        Get open positions.
        
        Args:
            symbol (str, optional): Symbol to filter positions. Defaults to None.
            
        Returns:
            list: List of open positions
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return []
        
        # Convert to list of dicts
        result = []
        for position in positions:
            result.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': position.volume,
                'open_price': position.price_open,
                'current_price': position.price_current,
                'sl': position.sl,
                'tp': position.tp,
                'profit': position.profit,
                'swap': position.swap,
                'time': datetime.fromtimestamp(position.time, tz=self.timezone)
            })
        
        return result
    
    def modify_position(self, position_id, sl=None, tp=None):
        """
        Modify an open position.
        
        Args:
            position_id (int): Position ID
            sl (float, optional): New stop loss price. Defaults to None.
            tp (float, optional): New take profit price. Defaults to None.
            
        Returns:
            bool: True if position was modified successfully, False otherwise
        """
        if not self.initialized:
            logger.error("MT5 not initialized")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=position_id)
        if position is None or len(position) == 0:
            logger.error(f"Position {position_id} not found: {mt5.last_error()}")
            return False
        
        position = position[0]
        
        # Prepare modify request
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "symbol": position.symbol,
            "position": position_id,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
            "magic": self.config.get('magic', 123456)
        }
        
        # Send modify request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify position failed: {result.retcode}, {result.comment}")
            return False
        
        logger.info(f"Position {position_id} modified successfully")
        return True

    def connect(self):
        """Initialize & log into MT5 using terminal_path (or path)."""
        import MetaTrader5 as mt5

        term = self.config.get("terminal_path") or self.config.get("path")
        # if user passed a directory, look for terminal executables
        if term and os.path.isdir(term):
            for exe in ("terminal64.exe", "terminal.exe"):
                cand = os.path.join(term, exe)
                if os.path.exists(cand):
                    term = cand
                    break

        if not term or not os.path.exists(term):
            logger.error(f"MT5 executable not found: {term}")
            return False

        login = self.config.get("login")
        pwd   = self.config.get("password")
        srv   = self.config.get("server")
        if not mt5.initialize(term, login=login, password=pwd, server=srv):
            err = mt5.last_error()
            logger.error(f"MT5 connection failed: {err}")
            return False

        self.initialized = True
        logger.info("MT5 connected")
        return True

    def disconnect(self):
        """Shutdown MT5 if initialized."""
        import MetaTrader5 as mt5
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 disconnected")
            return True
        return False

    # alias old names
    get_account_info = account_info
    get_positions     = get_open_positions

    def get_trades(self, limit=10, symbol=None):
        """
        Return up to `limit` recent deals.
        """
        if not self.initialized:
            return []
        # fetch last 1 day by default
        end = datetime.now(self.timezone)
        start = end - timedelta(days=1)
        if symbol:
            deals = mt5.history_deals_get(start, end, symbol=symbol)
        else:
            deals = mt5.history_deals_get(start, end)
        if not deals:
            return []
        # take last `limit`
        deals = deals[-limit:]
        result = []
        for d in deals:
            result.append({
                'ticket': d.ticket,
                'symbol': d.symbol,
                'type': 'BUY' if d.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP) else 'SELL',
                'volume': d.volume,
                'price_open': getattr(d, 'price_open', d.price),
                'price_close': getattr(d, 'price_close', d.price),
                'profit': d.profit,
                'time': datetime.fromtimestamp(d.time, tz=self.timezone)
            })
        return result
