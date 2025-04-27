#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization tools for ATFNet.
Implements advanced charts and plots for analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

logger = logging.getLogger('atfnet.utils.visualization')


class ChartPlotter:
    """
    Advanced chart plotter for financial data.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: str = 'seaborn-v0_8-darkgrid',
        save_dir: Optional[str] = None
    ):
        """
        Initialize the chart plotter.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.style = style
        self.save_dir = save_dir
        
        # Set style
        plt.style.use(style)
        
        # Create save directory if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_price_chart(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        title: str = 'Price Chart',
        show_volume: bool = True,
        ma_periods: Optional[List[int]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot price chart with optional volume and moving averages.
        
        Args:
            df: DataFrame with price data
            date_col: Date column name
            open_col: Open price column name
            high_col: High price column name
            low_col: Low price column name
            close_col: Close price column name
            volume_col: Volume column name
            title: Chart title
            show_volume: Whether to show volume
            ma_periods: List of moving average periods
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        if show_volume and volume_col in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None
        
        # Plot price
        ax1.plot(df[date_col], df[close_col], label='Close')
        
        # Add moving averages
        if ma_periods is not None:
            for period in ma_periods:
                ma_col = f'MA_{period}'
                if ma_col not in df.columns:
                    df[ma_col] = df[close_col].rolling(window=period).mean()
                ax1.plot(df[date_col], df[ma_col], label=f'MA {period}')
        
        # Add volume
        if show_volume and volume_col in df.columns and ax2 is not None:
            ax2.bar(df[date_col], df[volume_col], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
        
        # Set labels and title
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_range = df[date_col].max() - df[date_col].min()
            
            if date_range.days > 365:
                # More than a year, show months
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif date_range.days > 30:
                # More than a month, show weeks
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                # Less than a month, show days
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_candlestick_chart(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        title: str = 'Candlestick Chart',
        show_volume: bool = True,
        ma_periods: Optional[List[int]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot candlestick chart with optional volume and moving averages.
        
        Args:
            df: DataFrame with price data
            date_col: Date column name
            open_col: Open price column name
            high_col: High price column name
            low_col: Low price column name
            close_col: Close price column name
            volume_col: Volume column name
            title: Chart title
            show_volume: Whether to show volume
            ma_periods: List of moving average periods
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        if show_volume and volume_col in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None
        
        # Plot candlesticks
        width = 0.6
        width2 = 0.05
        
        up = df[close_col] > df[open_col]
        down = df[close_col] <= df[open_col]
        
        # Plot candle bodies
        ax1.bar(df.index[up], df[close_col][up] - df[open_col][up], width, bottom=df[open_col][up], color='green', alpha=0.7)
        ax1.bar(df.index[down], df[open_col][down] - df[close_col][down], width, bottom=df[close_col][down], color='red', alpha=0.7)
        
        # Plot candle wicks
        ax1.bar(df.index[up], df[high_col][up] - df[close_col][up], width2, bottom=df[close_col][up], color='green', alpha=0.7)
        ax1.bar(df.index[up], df[open_col][up] - df[low_col][up], width2, bottom=df[low_col][up], color='green', alpha=0.7)
        ax1.bar(df.index[down], df[high_col][down] - df[open_col][down], width2, bottom=df[open_col][down], color='red', alpha=0.7)
        ax1.bar(df.index[down], df[close_col][down] - df[low_col][down], width2, bottom=df[low_col][down], color='red', alpha=0.7)
        
        # Add moving averages
        if ma_periods is not None:
            for period in ma_periods:
                ma_col = f'MA_{period}'
                if ma_col not in df.columns:
                    df[ma_col] = df[close_col].rolling(window=period).mean()
                ax1.plot(df.index, df[ma_col], label=f'MA {period}')
        
        # Add volume
        if show_volume and volume_col in df.columns and ax2 is not None:
            ax2.bar(df.index[up], df[volume_col][up], width, color='green', alpha=0.5)
            ax2.bar(df.index[down], df[volume_col][down], width, color='red', alpha=0.5)
            ax2.set_ylabel('Volume')
        
        # Set labels and title
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if len(df) > 30:
            step = len(df) // 10
            ax1.set_xticks(df.index[::step])
            ax1.set_xticklabels(df[date_col].iloc[::step])
        else:
            ax1.set_xticks(df.index)
            ax1.set_xticklabels(df[date_col])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_indicator_chart(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, List[str]],
        date_col: str = 'date',
        price_col: str = 'close',
        title: str = 'Technical Indicators',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot technical indicators.
        
        Args:
            df: DataFrame with price and indicator data
            date_col: Date column name
            price_col: Price column name
            indicators: Dictionary mapping subplot names to indicator columns
            title: Chart title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        n_subplots = len(indicators) + 1  # +1 for price
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
        
        if n_subplots == 1:
            axes = [axes]
        
        # Plot price
        axes[0].plot(df[date_col], df[price_col])
        axes[0].set_title(title)
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        # Plot indicators
        for i, (indicator_name, indicator_cols) in enumerate(indicators.items(), 1):
            for col in indicator_cols:
                if col in df.columns:
                    axes[i].plot(df[date_col], df[col], label=col)
            
            axes[i].set_ylabel(indicator_name)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Format x-axis
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_range = df[date_col].max() - df[date_col].min()
            
            if date_range.days > 365:
                # More than a year, show months
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif date_range.days > 30:
                # More than a month, show weeks
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                # Less than a month, show days
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_regime_chart(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        price_col: str = 'close',
        regime_col: str = 'regime',
        regime_names: Optional[Dict[int, str]] = None,
        title: str = 'Market Regimes',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot price with market regime background.
        
        Args:
            df: DataFrame with price and regime data
            date_col: Date column name
            price_col: Price column name
            regime_col: Regime column name
            regime_names: Dictionary mapping regime values to names
            title: Chart title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        ax.plot(df[date_col], df[price_col], color='black', linewidth=1.5, zorder=5)
        
        # Default regime names if not provided
        if regime_names is None:
            unique_regimes = df[regime_col].unique()
            regime_names = {
                regime: f"Regime {regime}" for regime in unique_regimes
            }
        
        # Default colors for regimes
        colors = ['lightblue', 'lightgreen', 'lightsalmon', 'lightgray', 'lightpink', 'lightyellow']
        
        # Plot regime backgrounds
        regime_changes = df[regime_col].ne(df[regime_col].shift()).cumsum()
        
        for i, (_, group) in enumerate(df.groupby(regime_changes)):
            regime = group[regime_col].iloc[0]
            color = colors[int(regime) % len(colors)]
            
            ax.axvspan(
                group[date_col].iloc[0],
                group[date_col].iloc[-1],
                alpha=0.3,
                color=color,
                label=regime_names.get(regime, f"Regime {regime}") if i == 0 or regime != df.iloc[regime_changes == i-1][regime_col].iloc[0] else ""
            )
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_range = df[date_col].max() - df[date_col].min()
            
            if date_range.days > 365:
                # More than a year, show months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif date_range.days > 30:
                # More than a month, show weeks
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                # Less than a month, show days
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trades(
        self,
        df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        date_col: str = 'date',
        price_col: str = 'close',
        title: str = 'Trading Performance',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot price chart with trade entries and exits.
        
        Args:
            df: DataFrame with price data
            trades: List of trade dictionaries
            date_col: Date column name
            price_col: Price column name
            title: Chart title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        ax.plot(df[date_col], df[price_col], color='black', linewidth=1.5)
        
        # Plot trades
        for trade in trades:
            # Get trade details
            open_time = trade.get('open_time')
            close_time = trade.get('close_time')
            open_price = trade.get('open_price')
            close_price = trade.get('close_price')
            trade_type = trade.get('type', 'BUY')
            profit = trade.get('profit', 0)
            
            # Skip if missing required data
            if open_time is None or close_time is None or open_price is None or close_price is None:
                continue
            
            # Convert string dates to datetime if needed
            if isinstance(open_time, str):
                open_time = pd.to_datetime(open_time)
            if isinstance(close_time, str):
                close_time = pd.to_datetime(close_time)
            
            # Determine color based on profit
            color = 'green' if profit > 0 else 'red'
            
            # Plot entry point
            ax.scatter(open_time, open_price, color=color, marker='^' if trade_type == 'BUY' else 'v', s=100, zorder=5)
            
            # Plot exit point
            ax.scatter(close_time, close_price, color=color, marker='o', s=100, zorder=5)
            
            # Connect entry and exit with line
            ax.plot([open_time, close_time], [open_price, close_price], color=color, linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            date_range = df[date_col].max() - df[date_col].min()
            
            if date_range.days > 365:
                # More than a year, show months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif date_range.days > 30:
                # More than a month, show weeks
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                # Less than a month, show days
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_equity_curve(
        self,
        equity_curve: np.ndarray,
        dates: Optional[np.ndarray] = None,
        benchmark: Optional[np.ndarray] = None,
        title: str = 'Equity Curve',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve with optional benchmark.
        
        Args:
            equity_curve: Equity curve array
            dates: Date array
            benchmark: Benchmark equity curve
            title: Chart title
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Set figure size
        if figsize is None:
            figsize = self.figsize
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create x-axis
        x = dates if dates is not None else np.arange(len(equity_curve))
        
        # Plot equity curve
        ax.plot(x, equity_curve, label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark is not None:
            if len(benchmark) != len(equity_curve):
                logger.warning(f"Benchmark length ({len(benchmark)}) doesn't match equity curve length ({len(equity_curve)})")
                benchmark = benchmark[:len(equity_curve)]
            
            ax.plot(x, benchmark, label='Benchmark', linewidth=2, alpha=0.7)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        max_dd = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        # Add max drawdown annotation
        if max_dd > 0:
            ax.annotate(
                f'Max DD: {max_dd:.2f}%',
                xy=(x[max_dd_idx], equity_curve[max_dd_idx]),
                xytext=(x[max_dd_idx], equity_curve[max_dd_idx] * 1.1),
                arrowprops=dict(facecolor='red', shrink=0.05),
                color='red'
            )
        
        # Set labels and title
        ax.set_title(title)
        if dates is not None:
            ax.set_xlabel('Date')
        else:
            ax.set_xlabel('Period')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if dates is not None and pd.api.types.is_datetime64_any_dtype(dates):
            date_range = dates[-1] - dates[0]
            
            if isinstance(date_range, timedelta) and date_range.days > 365:
                # More than a year, show months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif isinstance(date_range, timedelta) and date_range.days > 30:
                # More than a month, show weeks
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                # Less than a month, show days
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class InteractivePlotter:
    """
    Interactive chart plotter using Plotly.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the interactive plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        
        # Create save directory if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_candlestick_chart(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: Optional[str] = 'volume',
        title: str = 'Candlestick Chart',
        show_volume: bool = True,
        ma_periods: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot interactive candlestick chart.
        
        Args:
            df: DataFrame with price data
            date_col: Date column name
            open_col: Open price column name
            high_col: High price column name
            low_col: Low price column name
            close_col: Close price column name
            volume_col: Volume column name
            title: Chart title
            show_volume: Whether to show volume
            ma_periods: List of moving average periods
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Create subplots
        if show_volume and volume_col in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.8, 0.2]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df[date_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if ma_periods is not None:
            for period in ma_periods:
                ma_col = f'MA_{period}'
                if ma_col not in df.columns:
                    df[ma_col] = df[close_col].rolling(window=period).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=df[ma_col],
                        name=f'MA {period}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add volume
        if show_volume and volume_col in df.columns:
            # Color volume bars based on price direction
            colors = ['green' if row[close_col] >= row[open_col] else 'red' for _, row in df.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=df[date_col],
                    y=df[volume_col],
                    name='Volume',
                    marker=dict(color=colors, opacity=0.5)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_indicator_chart(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, List[str]],
        date_col: str = 'date',
        price_col: str = 'close',
        title: str = 'Technical Indicators',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot interactive technical indicators.
        
        Args:
            df: DataFrame with price and indicator data
            date_col: Date column name
            price_col: Price column name
            indicators: Dictionary mapping subplot names to indicator columns
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Create subplots
        n_subplots = len(indicators) + 1  # +1 for price
        fig = make_subplots(
            rows=n_subplots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[price_col] + list(indicators.keys())
        )
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[price_col],
                name=price_col,
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Add indicator traces
        for i, (indicator_name, indicator_cols) in enumerate(indicators.items(), 1):
            for col in indicator_cols:
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df[date_col],
                            y=df[col],
                            name=col,
                            line=dict(width=1)
                        ),
                        row=i+1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            template='plotly_white',
            height=300 * n_subplots
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_trades(
        self,
        df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        date_col: str = 'date',
        price_col: str = 'close',
        title: str = 'Trading Performance',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot interactive price chart with trade entries and exits.
        
        Args:
            df: DataFrame with price data
            trades: List of trade dictionaries
            date_col: Date column name
            price_col: Price column name
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[price_col],
                name='Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Prepare trade data
        buy_entries_x = []
        buy_entries_y = []
        buy_exits_x = []
        buy_exits_y = []
        
        sell_entries_x = []
        sell_entries_y = []
        sell_exits_x = []
        sell_exits_y = []
        
        win_trades_x = []
        win_trades_y = []
        loss_trades_x = []
        loss_trades_y = []
        
        # Process trades
        for trade in trades:
            # Get trade details
            open_time = trade.get('open_time')
            close_time = trade.get('close_time')
            open_price = trade.get('open_price')
            close_price = trade.get('close_price')
            trade_type = trade.get('type', 'BUY')
            profit = trade.get('profit', 0)
            
            # Skip if missing required data
            if open_time is None or close_time is None or open_price is None or close_price is None:
                continue
            
            # Convert string dates to datetime if needed
            if isinstance(open_time, str):
                open_time = pd.to_datetime(open_time)
            if isinstance(close_time, str):
                close_time = pd.to_datetime(close_time)
            
            # Process trade based on type and profit
            if trade_type == 'BUY':
                buy_entries_x.append(open_time)
                buy_entries_y.append(open_price)
                buy_exits_x.append(close_time)
                buy_exits_y.append(close_price)
            else:  # SELL
                sell_entries_x.append(open_time)
                sell_entries_y.append(open_price)
                sell_exits_x.append(close_time)
                sell_exits_y.append(close_price)
            
            # Add to win/loss lists for line connections
            if profit > 0:
                win_trades_x.extend([open_time, close_time, None])
                win_trades_y.extend([open_price, close_price, None])
            else:
                loss_trades_x.extend([open_time, close_time, None])
                loss_trades_y.extend([open_price, close_price, None])
        
        # Add buy entry markers
        if buy_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_entries_x,
                    y=buy_entries_y,
                    mode='markers',
                    name='Buy Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    )
                )
            )
        
        # Add buy exit markers
        if buy_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_exits_x,
                    y=buy_exits_y,
                    mode='markers',
                    name='Buy Exit',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    )
                )
            )
        
        # Add sell entry markers
        if sell_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_entries_x,
                    y=sell_entries_y,
                    mode='markers',
                    name='Sell Entry',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=1, color='darkred')
                    )
                )
            )
        
        # Add sell exit markers
        if sell_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_exits_x,
                    y=sell_exits_y,
                    mode='markers',
                    name='Sell Exit',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='red',
                        line=dict(width=1, color='darkred')
                    )
                )
            )
        
        # Add winning trade lines
        if win_trades_x:
            fig.add_trace(
                go.Scatter(
                    x=win_trades_x,
                    y=win_trades_y,
                    mode='lines',
                    name='Winning Trades',
                    line=dict(color='green', width=1, dash='dash')
                )
            )
        
        # Add losing trade lines
        if loss_trades_x:
            fig.add_trace(
                go.Scatter(
                    x=loss_trades_x,
                    y=loss_trades_y,
                    mode='lines',
                    name='Losing Trades',
                    line=dict(color='red', width=1, dash='dash')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_equity_curve(
        self,
        equity_curve: np.ndarray,
        dates: Optional[np.ndarray] = None,
        benchmark: Optional[np.ndarray] = None,
        drawdowns: bool = True,
        title: str = 'Equity Curve',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot interactive equity curve with optional benchmark and drawdowns.
        
        Args:
            equity_curve: Equity curve array
            dates: Date array
            benchmark: Benchmark equity curve
            drawdowns: Whether to show drawdowns
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Create figure
        if drawdowns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=['Equity', 'Drawdown %']
            )
        else:
            fig = go.Figure()
        
        # Create x-axis
        x = dates if dates is not None else np.arange(len(equity_curve))
        
        # Add equity curve trace
        if drawdowns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=equity_curve,
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=equity_curve,
                    name='Strategy',
                    line=dict(color='blue', width=2)
                )
            )
        
        # Add benchmark if provided
        if benchmark is not None:
            if len(benchmark) != len(equity_curve):
                logger.warning(f"Benchmark length ({len(benchmark)}) doesn't match equity curve length ({len(equity_curve)})")
                benchmark = benchmark[:len(equity_curve)]
            
            if drawdowns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=benchmark,
                        name='Benchmark',
                        line=dict(color='gray', width=2, dash='dot')
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=benchmark,
                        name='Benchmark',
                        line=dict(color='gray', width=2, dash='dot')
                    )
                )
        
        # Calculate and add drawdown
        if drawdowns:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Add max drawdown line
            max_dd = np.max(drawdown)
            fig.add_trace(
                go.Scatter(
                    x=[x[0], x[-1]],
                    y=[max_dd, max_dd],
                    name=f'Max DD: {max_dd:.2f}%',
                    line=dict(color='darkred', width=1, dash='dash')
                ),
                row=2, col=1
            )
        
        # Update layout
        if drawdowns:
            fig.update_layout(
                title=title,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=700
            )
            
            fig.update_yaxes(title_text='Equity', row=1, col=1)
            fig.update_yaxes(title_text='Drawdown %', row=2, col=1)
            
            if dates is not None:
                fig.update_xaxes(title_text='Date', row=2, col=1)
            else:
                fig.update_xaxes(title_text='Period', row=2, col=1)
        else:
            fig.update_layout(
                title=title,
                xaxis_title='Date' if dates is not None else 'Period',
                yaxis_title='Equity',
                template='plotly_white'
            )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_performance_metrics(
        self,
        metrics: Dict[str, float],
        benchmark_metrics: Optional[Dict[str, float]] = None,
        title: str = 'Performance Metrics',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot performance metrics comparison.
        
        Args:
            metrics: Dictionary of performance metrics
            benchmark_metrics: Dictionary of benchmark metrics for comparison
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Prepare data
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create figure
        if benchmark_metrics is not None:
            benchmark_values = [benchmark_metrics.get(name, 0) for name in metric_names]
            
            fig = go.Figure(data=[
                go.Bar(name='Strategy', x=metric_names, y=metric_values),
                go.Bar(name='Benchmark', x=metric_names, y=benchmark_values)
            ])
        else:
            fig = go.Figure(data=[
                go.Bar(name='Strategy', x=metric_names, y=metric_values)
            ])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white',
            barmode='group'
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = 'Correlation Matrix',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame with data to correlate
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            width=700
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_regime_distribution(
        self,
        regimes: np.ndarray,
        regime_names: Optional[Dict[int, str]] = None,
        title: str = 'Market Regime Distribution',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot market regime distribution pie chart.
        
        Args:
            regimes: Array of regime labels
            regime_names: Dictionary mapping regime values to names
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Count regimes
        unique_regimes, counts = np.unique(regimes, return_counts=True)
        
        # Default regime names if not provided
        if regime_names is None:
            regime_names = {
                regime: f"Regime {regime}" for regime in unique_regimes
            }
        
        # Prepare labels
        labels = [regime_names.get(regime, f"Regime {regime}") for regime in unique_regimes]
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=counts,
            hole=.3
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_feature_importance(
        self,
        features: List[str],
        importances: np.ndarray,
        title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            features: List of feature names
            importances: Array of importance values
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        # Create figure
        fig = go.Figure(data=[go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h'
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white'
        )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig
    
    def plot_optimization_results(
        self,
        param_values: Dict[str, List[float]],
        performance: np.ndarray,
        metric_name: str = 'Sharpe Ratio',
        title: str = 'Parameter Optimization Results',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot parameter optimization results.
        
        Args:
            param_values: Dictionary mapping parameter names to lists of values
            performance: Array of performance metric values
            metric_name: Name of the performance metric
            title: Chart title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Check if we have 1 or 2 parameters
        if len(param_values) == 1:
            # 1D optimization
            param_name = list(param_values.keys())[0]
            values = param_values[param_name]
            
            # Create figure
            fig = go.Figure(data=go.Scatter(
                x=values,
                y=performance,
                mode='lines+markers'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=param_name,
                yaxis_title=metric_name,
                template='plotly_white'
            )
        
        elif len(param_values) == 2:
            # 2D optimization
            param_names = list(param_values.keys())
            x_values = param_values[param_names[0]]
            y_values = param_values[param_names[1]]
            
            # Create meshgrid
            X, Y = np.meshgrid(x_values, y_values)
            Z = performance.reshape(len(y_values), len(x_values))
            
            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=Z,
                x=x_values,
                y=y_values,
                colorscale='Viridis',
                colorbar=dict(title=metric_name)
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=param_names[0],
                yaxis_title=param_names[1],
                template='plotly_white'
            )
        
        else:
            # More than 2 parameters - show best results
            fig = go.Figure()
            
            # Find top 10 parameter combinations
            top_indices = np.argsort(performance)[-10:]
            
            # Create table data
            table_data = []
            for idx in reversed(top_indices):
                row = []
                for param_name, values in param_values.items():
                    param_idx = idx % len(values)
                    row.append(values[param_idx])
                    idx = idx // len(values)
                row.append(performance[top_indices[idx]])
                table_data.append(row)
            
            # Create table
            fig.add_trace(go.Table(
                header=dict(
                    values=list(param_values.keys()) + [metric_name],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lavender',
                    align='left'
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                template='plotly_white'
            )
        
        # Save plot if requested
        if save_path is not None:
            fig.write_html(save_path)
        elif self.save_dir is not None:
            # Save to default directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            fig.write_html(save_path)
        
        return fig


class DashboardVisualizer:
    """
    Dashboard visualizer for ATFNet.
    Creates interactive dashboards with multiple plots.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the dashboard visualizer.
        
        Args:
            save_dir: Directory to save dashboards
        """
        self.save_dir = save_dir
        self.plotter = InteractivePlotter(save_dir)
        
        # Create save directory if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
    def create_trading_dashboard(
        self,
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        equity_curve: np.ndarray,
        performance_metrics: Dict[str, float],
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_equity: Optional[np.ndarray] = None,
        benchmark_metrics: Optional[Dict[str, float]] = None,
        date_col: str = 'date',
        title: str = 'Trading Dashboard',
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a trading dashboard with multiple plots.
        
        Args:
            price_data: DataFrame with price data
            trades: List of trade dictionaries
            equity_curve: Equity curve array
            performance_metrics: Dictionary of performance metrics
            benchmark_data: DataFrame with benchmark price data
            benchmark_equity: Benchmark equity curve
            benchmark_metrics: Dictionary of benchmark metrics
            date_col: Date column name
            title: Dashboard title
            save_path: Path to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create HTML file
        if save_path is None:
            if self.save_dir is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"{title.replace(' ', '_')}_{timestamp}.html"
        
        # Create plots
        price_fig = self.plotter.plot_candlestick_chart(
            df=price_data,
            date_col=date_col,
            title='Price Chart',
            save_path=None
        )
        
        trades_fig = self.plotter.plot_trades(
            df=price_data,
            trades=trades,
            date_col=date_col,
            title='Trading Performance',
            save_path=None
        )
        
        equity_fig = self.plotter.plot_equity_curve(
            equity_curve=equity_curve,
            dates=price_data[date_col] if date_col in price_data.columns else None,
            benchmark=benchmark_equity,
            title='Equity Curve',
            save_path=None
        )
        
        metrics_fig = self.plotter.plot_performance_metrics(
            metrics=performance_metrics,
            benchmark_metrics=benchmark_metrics,
            title='Performance Metrics',
            save_path=None
        )
        
        # Convert plots to HTML
        price_html = price_fig.to_html(full_html=False, include_plotlyjs='cdn')
        trades_html = trades_fig.to_html(full_html=False, include_plotlyjs=False)
        equity_html = equity_fig.to_html(full_html=False, include_plotlyjs=False)
        metrics_html = metrics_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Create dashboard HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: -10px;
                }}
                .col {{
                    flex: 1;
                    padding: 10px;
                    min-width: 300px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .card-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #2c3e50;
                }}
                .full-width {{
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="container">
                <div class="row">
                    <div class="col full-width">
                        <div class="card">
                            <div class="card-title">Price Chart with Trades</div>
                            {trades_html}
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <div class="card">
                            <div class="card-title">Equity Curve</div>
                            {equity_html}
                        </div>
                    </div>
                    <div class="col">
                        <div class="card">
                            <div class="card-title">Performance Metrics</div>
                            {metrics_html}
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(save_path, 'w') as f:
            f.write(html)
        
        return save_path
    
    def create_analysis_dashboard(
        self,
        price_data: pd.DataFrame,
        indicators: Dict[str, List[str]],
        regimes: Optional[np.ndarray] = None,
        regime_names: Optional[Dict[int, str]] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
        date_col: str = 'date',
        title: str = 'Market Analysis Dashboard',
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a market analysis dashboard with multiple plots.
        
        Args:
            price_data: DataFrame with price data
            indicators: Dictionary mapping subplot names to indicator columns
            regimes: Array of regime labels
            regime_names: Dictionary mapping regime values to names
            feature_names: List of feature names
            feature_importances: Array of feature importance values
            date_col: Date column name
            title: Dashboard title
            save_path: Path to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create HTML file
        if save_path is None:
            if self.save_dir is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"{title.replace(' ', '_')}_{timestamp}.html"
        
        # Create plots
        price_fig = self.plotter.plot_candlestick_chart(
            df=price_data,
            date_col=date_col,
            title='Price Chart',
            save_path=None
        )
        
        indicator_fig = self.plotter.plot_indicator_chart(
            df=price_data,
            date_col=date_col,
            indicators=indicators,
            title='Technical Indicators',
            save_path=None
        )
        
        # Create regime plot if data is provided
        regime_html = ""
        if regimes is not None:
            regime_fig = self.plotter.plot_regime_distribution(
                regimes=regimes,
                regime_names=regime_names,
                title='Market Regime Distribution',
                save_path=None
            )
            regime_html = regime_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Create feature importance plot if data is provided
        feature_html = ""
        if feature_names is not None and feature_importances is not None:
            feature_fig = self.plotter.plot_feature_importance(
                features=feature_names,
                importances=feature_importances,
                title='Feature Importance',
                save_path=None
            )
            feature_html = feature_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Convert plots to HTML
        price_html = price_fig.to_html(full_html=False, include_plotlyjs='cdn')
        indicator_html = indicator_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Create dashboard HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: -10px;
                }}
                .col {{
                    flex: 1;
                    padding: 10px;
                    min-width: 300px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .card-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #2c3e50;
                }}
                .full-width {{
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="container">
                <div class="row">
                    <div class="col full-width">
                        <div class="card">
                            <div class="card-title">Price Chart</div>
                            {price_html}
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col full-width">
                        <div class="card">
                            <div class="card-title">Technical Indicators</div>
                            {indicator_html}
                        </div>
                    </div>
                </div>
                <div class="row">
                    {f'''
                    <div class="col">
                        <div class="card">
                            <div class="card-title">Market Regime Distribution</div>
                            {regime_html}
                        </div>
                    </div>
                    ''' if regime_html else ''}
                    {f'''
                    <div class="col">
                        <div class="card">
                            <div class="card-title">Feature Importance</div>
                            {feature_html}
                        </div>
                    </div>
                    ''' if feature_html else ''}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(save_path, 'w') as f:
            f.write(html)
        
        return save_path
    
    def create_optimization_dashboard(
        self,
        param_values: Dict[str, List[float]],
        performance: np.ndarray,
        metric_name: str = 'Sharpe Ratio',
        best_params: Optional[Dict[str, float]] = None,
        best_performance: Optional[float] = None,
        convergence_history: Optional[np.ndarray] = None,
        title: str = 'Optimization Dashboard',
        save_path: Optional[str] = None
    ) -> str:
        """
        Create an optimization dashboard with multiple plots.
        
        Args:
            param_values: Dictionary mapping parameter names to lists of values
            performance: Array of performance metric values
            metric_name: Name of the performance metric
            best_params: Dictionary of best parameter values
            best_performance: Best performance value
            convergence_history: Array of performance values over iterations
            title: Dashboard title
            save_path: Path to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create HTML file
        if save_path is None:
            if self.save_dir is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}_{timestamp}.html")
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"{title.replace(' ', '_')}_{timestamp}.html"
        
        # Create optimization results plot
        opt_fig = self.plotter.plot_optimization_results(
            param_values=param_values,
            performance=performance,
            metric_name=metric_name,
            title='Parameter Optimization Results',
            save_path=None
        )
        
        # Create convergence plot if data is provided
        convergence_html = ""
        if convergence_history is not None:
            # Create figure
            conv_fig = go.Figure(data=go.Scatter(
                x=list(range(1, len(convergence_history) + 1)),
                y=convergence_history,
                mode='lines+markers'
            ))
            
            # Update layout
            conv_fig.update_layout(
                title='Optimization Convergence',
                xaxis_title='Iteration',
                yaxis_title=metric_name,
                template='plotly_white'
            )
            
            convergence_html = conv_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Convert plots to HTML
        opt_html = opt_fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Create best parameters HTML if provided
        best_params_html = ""
        if best_params is not None and best_performance is not None:
            best_params_html = f"""
            <div class="card">
                <div class="card-title">Best Parameters</div>
                <table class="params-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    {''.join([f'<tr><td>{param}</td><td>{value}</td></tr>' for param, value in best_params.items()])}
                    <tr class="highlight">
                        <td>{metric_name}</td>
                        <td>{best_performance:.4f}</td>
                    </tr>
                </table>
            </div>
            """
        
        # Create dashboard HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: -10px;
                }}
                .col {{
                    flex: 1;
                    padding: 10px;
                    min-width: 300px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .card-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #2c3e50;
                }}
                .full-width {{
                    width: 100%;
                }}
                .params-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .params-table th, .params-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .params-table th {{
                    background-color: #f2f2f2;
                }}
                .highlight {{
                    font-weight: bold;
                    background-color: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="container">
                <div class="row">
                    <div class="col {'' if best_params_html else 'full-width'}">
                        <div class="card">
                            <div class="card-title">Optimization Results</div>
                            {opt_html}
                        </div>
                    </div>
                    {f'''
                    <div class="col">
                        {best_params_html}
                    </div>
                    ''' if best_params_html else ''}
                </div>
                {f'''
                <div class="row">
                    <div class="col full-width">
                        <div class="card">
                            <div class="card-title">Optimization Convergence</div>
                            {convergence_html}
                        </div>
                    </div>
                </div>
                ''' if convergence_html else ''}
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(save_path, 'w') as f:
            f.write(html)
        
        return save_path
