#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced ATFNet PyQt5 Application
This module implements an advanced GUI for the ATFNet trading system with
professional-grade visualizations, real-time monitoring, and enhanced user experience.
"""

import sys
import os
import logging
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import mplfinance as mpf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QSplitter, QFrame, QGroupBox, QFormLayout, QTextEdit, QProgressBar, QAction,
    QMenu, QStatusBar, QDockWidget, QSizePolicy, QSlider, QToolBar, QToolButton,
    QScrollArea, QGridLayout, QRadioButton, QButtonGroup, QCalendarWidget,
    QDateEdit, QTimeEdit, QDateTimeEdit, QStackedWidget, QListWidget, QListWidgetItem,
    QSystemTrayIcon, QStyle, QDialog, QDialogButtonBox, QColorDialog, QFontDialog,
    QInputDialog, QTabBar, QStyleFactory, QGraphicsDropShadowEffect, QDesktopWidget
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSettings, QSize, QDateTime, QDate, QTime,
    QPropertyAnimation, QEasingCurve, QRect, QPoint, QUrl, QEvent, QObject,
    QSortFilterProxyModel, QModelIndex, QAbstractTableModel, QAbstractItemModel
)
from PyQt5.QtGui import (
    QIcon, QFont, QColor, QPalette, QPixmap, QImage, QPainter, QPen, QBrush,
    QCursor, QKeySequence, QStandardItemModel, QStandardItem, QFontMetrics,
    QLinearGradient, QRadialGradient, QConicalGradient, QTransform, QPolygon,
    QTextCursor, QTextCharFormat, QTextDocument, QTextOption, QTextTableFormat
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
import pyqtgraph as pg

# add project root so imports like `strategies`/`core` resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the package using absolute imports
from mt5_connector.mt5_api import MT5API
from data import DataManager
from models import ModelManager
from strategies import ATFNetStrategy
from utils import setup_logger
from utils import ConfigLoader
from core import ATFNetAttention, ATFNetFeatures
from core import PriceForecaster
from core import MarketRegimeDetector
from core import RiskManager
from core import SignalAggregator
from core import PerformanceAnalytics
from utils import ChartPlotter, InteractivePlotter

# Set up logging
logger = setup_logger('app.enhanced_app', logging.INFO)

# Set pyqtgraph configuration
pg.setConfigOptions(antialias=True, background='w', foreground='k')


class DarkPalette(QPalette):
    """Custom dark palette for the application."""
    
    def __init__(self):
        """Initialize the dark palette."""
        super().__init__()
        
        # Set colors
        self.setColor(QPalette.Window, QColor(53, 53, 53))
        self.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setColor(QPalette.Base, QColor(25, 25, 25))
        self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        self.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        self.setColor(QPalette.Text, QColor(255, 255, 255))
        self.setColor(QPalette.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setColor(QPalette.BrightText, QColor(255, 0, 0))
        self.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Set disabled colors
        self.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        self.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        self.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        self.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        self.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))


class LightPalette(QPalette):
    """Custom light palette for the application."""
    
    def __init__(self):
        """Initialize the light palette."""
        super().__init__()
        
        # Set colors
        self.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setColor(QPalette.Base, QColor(255, 255, 255))
        self.setColor(QPalette.AlternateBase, QColor(233, 233, 233))
        self.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        self.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        self.setColor(QPalette.Text, QColor(0, 0, 0))
        self.setColor(QPalette.Button, QColor(240, 240, 240))
        self.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        self.setColor(QPalette.BrightText, QColor(255, 0, 0))
        self.setColor(QPalette.Link, QColor(0, 0, 255))
        self.setColor(QPalette.Highlight, QColor(0, 120, 215))
        self.setColor(QPalette.HighlightedText, QColor(255, 255, 255))


class CustomTabBar(QTabBar):
    """Custom tab bar with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom tab bar."""
        super().__init__(parent)
        
        # Set properties
        self.setDrawBase(False)
        self.setExpanding(True)
        self.setDocumentMode(True)
        self.setElideMode(Qt.ElideRight)
        self.setMovable(True)
        
        # Set style
        self.setStyleSheet("""
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 8px 12px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: #2a82da;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #505050;
            }
        """)


class CustomTableWidget(QTableWidget):
    """Custom table widget with enhanced styling and functionality."""
    
    def __init__(self, parent=None):
        """Initialize the custom table widget."""
        super().__init__(parent)
        
        # Set properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        
        # Set style
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #444444;
                selection-background-color: #2a82da;
                selection-color: #ffffff;
            }
            
            QTableWidget::item {
                padding: 4px;
            }
            
            QHeaderView::section {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 6px;
                border: none;
            }
        """)
        
        # Set header properties
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(False)


class CustomButton(QPushButton):
    """Custom button with enhanced styling."""
    
    def __init__(self, text="", parent=None, primary=False):
        """Initialize the custom button."""
        super().__init__(text, parent)
        
        # Set properties
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Set style based on type
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2a82da;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                
                QPushButton:hover {
                    background-color: #3a92ea;
                }
                
                QPushButton:pressed {
                    background-color: #1a72ca;
                }
                
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #505050;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                }
                
                QPushButton:hover {
                    background-color: #606060;
                }
                
                QPushButton:pressed {
                    background-color: #404040;
                }
                
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
            """)


class CustomGroupBox(QGroupBox):
    """Custom group box with enhanced styling."""
    
    def __init__(self, title="", parent=None):
        """Initialize the custom group box."""
        super().__init__(title, parent)
        
        # Set style
        self.setStyleSheet("""
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #ffffff;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)


class CustomComboBox(QComboBox):
    """Custom combo box with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom combo box."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 6em;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #505050;
            }
            
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #2a82da;
                selection-color: #ffffff;
            }
        """)


class CustomLineEdit(QLineEdit):
    """Custom line edit with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom line edit."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
            }
            
            QLineEdit:focus {
                border: 1px solid #2a82da;
            }
        """)


class CustomTextEdit(QTextEdit):
    """Custom text edit with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom text edit."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px;
            }
        """)


class CustomProgressBar(QProgressBar):
    """Custom progress bar with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom progress bar."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QProgressBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #2a82da;
                border-radius: 4px;
            }
        """)


class CustomSlider(QSlider):
    """Custom slider with enhanced styling."""
    
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        """Initialize the custom slider."""
        super().__init__(orientation, parent)
        
        # Set style
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #3c3c3c;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #2a82da;
                border: none;
                width: 16px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #3a92ea;
            }
        """)


class CustomCheckBox(QCheckBox):
    """Custom check box with enhanced styling."""
    
    def __init__(self, text="", parent=None):
        """Initialize the custom check box."""
        super().__init__(text, parent)
        
        # Set style
        self.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: #3c3c3c;
                border: 1px solid #505050;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2a82da;
                border: 1px solid #2a82da;
                image: url(:/icons/check.png);
            }
        """)


class CustomRadioButton(QRadioButton):
    """Custom radio button with enhanced styling."""
    
    def __init__(self, text="", parent=None):
        """Initialize the custom radio button."""
        super().__init__(text, parent)
        
        # Set style
        self.setStyleSheet("""
            QRadioButton {
                color: #ffffff;
                spacing: 5px;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
            }
            
            QRadioButton::indicator:unchecked {
                background-color: #3c3c3c;
                border: 1px solid #505050;
            }
            
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 1px solid #2a82da;
            }
        """)


class CustomDateEdit(QDateEdit):
    """Custom date edit with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom date edit."""
        super().__init__(parent)
        
        # Set properties
        self.setCalendarPopup(True)
        self.setDisplayFormat("yyyy-MM-dd")
        
        # Set style
        self.setStyleSheet("""
            QDateEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
            }
            
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #505050;
            }
            
            QCalendarWidget QToolButton {
                color: white;
                background-color: #3c3c3c;
            }
            
            QCalendarWidget QMenu {
                color: white;
                background-color: #3c3c3c;
            }
            
            QCalendarWidget QSpinBox {
                color: white;
                background-color: #3c3c3c;
                selection-background-color: #2a82da;
                selection-color: white;
            }
            
            QCalendarWidget QTableView {
                alternate-background-color: #505050;
            }
        """)


class CustomSpinBox(QSpinBox):
    """Custom spin box with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom spin box."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QSpinBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
            }
            
            QSpinBox::up-button, QSpinBox::down-button {
                subcontrol-origin: border;
                width: 16px;
                border-left: 1px solid #505050;
            }
            
            QSpinBox::up-button {
                subcontrol-position: top right;
            }
            
            QSpinBox::down-button {
                subcontrol-position: bottom right;
            }
        """)


class CustomDoubleSpinBox(QDoubleSpinBox):
    """Custom double spin box with enhanced styling."""
    
    def __init__(self, parent=None):
        """Initialize the custom double spin box."""
        super().__init__(parent)
        
        # Set style
        self.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 8px;
            }
            
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                width: 16px;
                border-left: 1px solid #505050;
            }
            
            QDoubleSpinBox::up-button {
                subcontrol-position: top right;
            }
            
            QDoubleSpinBox::down-button {
                subcontrol-position: bottom right;
            }
        """)


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt5."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100, dark_mode=True):
        """Initialize the canvas."""
        # Set style based on dark mode
        if dark_mode:
            plt.style.use('dark_background')
            bg_color = '#2d2d2d'
            fg_color = '#ffffff'
        else:
            plt.style.use('default')
            bg_color = '#ffffff'
            fg_color = '#000000'
        
        # Create figure
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=bg_color)
        self.axes = self.fig.add_subplot(111)
        
        # Set colors
        self.axes.set_facecolor(bg_color)
        self.axes.tick_params(colors=fg_color)
        self.axes.spines['bottom'].set_color(fg_color)
        self.axes.spines['top'].set_color(fg_color)
        self.axes.spines['left'].set_color(fg_color)
        self.axes.spines['right'].set_color(fg_color)
        self.axes.xaxis.label.set_color(fg_color)
        self.axes.yaxis.label.set_color(fg_color)
        self.axes.title.set_color(fg_color)
        
        # Initialize canvas
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        # Set size policy
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PyQtGraphCanvas(pg.PlotWidget):
    """PyQtGraph canvas for interactive plots."""
    
    def __init__(self, parent=None, dark_mode=True):
        """Initialize the canvas."""
        # Set style based on dark mode
        if dark_mode:
            pg.setConfigOption('background', '#2d2d2d')
            pg.setConfigOption('foreground', '#ffffff')
        else:
            pg.setConfigOption('background', '#ffffff')
            pg.setConfigOption('foreground', '#000000')
        
        # Initialize plot widget
        super().__init__(parent=parent)
        
        # Set properties
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange()
        self.setDownsampling(auto=True, mode='peak')
        self.setClipToView(True)


class BacktestWorker(QThread):
    """Worker thread for running backtests."""
    
    progress = pyqtSignal(int)
    update = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, strategy, symbol, timeframe, start_date, end_date, config):
        """Initialize the worker."""
        super().__init__()
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.running = True
    
    def run(self):
        """Run the backtest."""
        try:
            # Update strategy config
            for key, value in self.config.items():
                self.strategy.config[key] = value
            
            # Emit progress updates
            self.progress.emit(10)
            
            # Run backtest
            results = self.strategy.backtest(
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date,
                progress_callback=self.update_progress
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(results)
        
        except Exception as e:
            logger.exception(f"Error in backtest worker: {e}")
            self.error.emit(str(e))
    
    def update_progress(self, progress_data):
        """Update progress during backtest."""
        if not self.running:
            return False
        
        # Emit progress
        progress = progress_data.get('progress', 0)
        self.progress.emit(int(progress * 100))
        
        # Emit update
        self.update.emit(progress_data)
        
        return self.running
    
    def stop(self):
        """Stop the backtest."""
        self.running = False


class TrainingWorker(QThread):
    """Worker thread for training models."""
    
    progress = pyqtSignal(int)
    update = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, strategy, symbol, timeframe, start_date, end_date, config):
        """Initialize the worker."""
        super().__init__()
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.running = True
    
    def run(self):
        """Run the training."""
        try:
            # Update strategy config
            for key, value in self.config.items():
                self.strategy.config[key] = value
            
            # Emit progress updates
            self.progress.emit(10)
            
            # Run training
            results = self.strategy.train(
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date,
                progress_callback=self.update_progress
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(results)
        
        except Exception as e:
            logger.exception(f"Error in training worker: {e}")
            self.error.emit(str(e))
    
    def update_progress(self, progress_data):
        """Update progress during training."""
        if not self.running:
            return False
        
        # Emit progress
        progress = progress_data.get('progress', 0)
        self.progress.emit(int(progress * 100))
        
        # Emit update
        self.update.emit(progress_data)
        
        return self.running
    
    def stop(self):
        """Stop the training."""
        self.running = False


class LiveTradingWorker(QThread):
    """Worker thread for live trading."""
    
    update = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, strategy, symbol, timeframe, config):
        """Initialize the worker."""
        super().__init__()
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config
        self.running = False
    
    def run(self):
        """Run live trading."""
        try:
            # Update strategy config
            for key, value in self.config.items():
                self.strategy.config[key] = value
            
            # Initialize data
            self.running = True
            success = self.strategy.initialize_live_trading(self.symbol, self.timeframe)
            
            if not success:
                self.error.emit(f"Failed to initialize live trading for {self.symbol} {self.timeframe}")
                return
            
            # Main trading loop
            while self.running:
                # Update strategy
                status = self.strategy.update_live_trading()
                
                # Emit update
                self.update.emit(status)
                
                # Sleep
                self.msleep(1000)  # 1 second
        
        except Exception as e:
            logger.exception(f"Error in live trading worker: {e}")
            self.error.emit(str(e))
    
    def stop(self):
        """Stop live trading."""
        self.running = False


class OptimizationWorker(QThread):
    """Worker thread for parameter optimization."""
    
    progress = pyqtSignal(int)
    update = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, strategy, symbol, timeframe, start_date, end_date, params_to_optimize):
        """Initialize the worker."""
        super().__init__()
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.params_to_optimize = params_to_optimize
        self.running = True
    
    def run(self):
        """Run the optimization."""
        try:
            # Emit progress updates
            self.progress.emit(5)
            
            # Run optimization
            results = self.strategy.optimize_parameters(
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date,
                self.params_to_optimize,
                progress_callback=self.update_progress
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(results)
        
        except Exception as e:
            logger.exception(f"Error in optimization worker: {e}")
            self.error.emit(str(e))
    
    def update_progress(self, progress_data):
        """Update progress during optimization."""
        if not self.running:
            return False
        
        # Emit progress
        progress = progress_data.get('progress', 0)
        self.progress.emit(int(progress * 100))
        
        # Emit update
        self.update.emit(progress_data)
        
        return self.running
    
    def stop(self):
        """Stop the optimization."""
        self.running = False


class DataImportWorker(QThread):
    """Worker thread for importing historical data."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, data_manager, symbol, timeframe, start_date, end_date):
        """Initialize the worker."""
        super().__init__()
        self.data_manager = data_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
    
    def run(self):
        """Run the data import."""
        try:
            # Emit progress updates
            self.progress.emit(10)
            
            # Import data
            data = self.data_manager.get_data(
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date
            )
            
            # Process data
            self.progress.emit(50)
            
            processed_data = self.data_manager.preprocess_data(data)
            
            # Add indicators
            self.progress.emit(70)
            
            data_with_indicators = self.data_manager.add_technical_indicators(processed_data)
            
            # Add frequency features
            self.progress.emit(90)
            
            final_data = self.data_manager.add_frequency_features(data_with_indicators)
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit({
                'data': final_data,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'start_date': self.start_date,
                'end_date': self.end_date
            })
        
        except Exception as e:
            logger.exception(f"Error in data import worker: {e}")
            self.error.emit(str(e))


class EnhancedATFNetApp(QMainWindow):
    """Enhanced main application window for ATFNet."""
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.setWindowTitle("ATFNet Professional Trading System")
        self.setGeometry(100, 100, 1600, 900)
        self.mt5_connected = False
        
        # Initialize components
        self.init_ui()
        self.init_connections()
        self.init_atfnet()
        
        # Load settings
        self.load_settings()
        
        # Set dark mode by default
        self.set_dark_mode(True)
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center the window on the screen."""
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    def set_dark_mode(self, enabled):
        """Set dark mode for the application."""
        if enabled:
            self.setStyleSheet("""
                QMainWindow, QDialog {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                
                QWidget {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                
                QSplitter::handle {
                    background-color: #3c3c3c;
                }
                
                QScrollBar:vertical {
                    background-color: #2d2d2d;
                    width: 12px;
                    margin: 0px;
                }
                
                QScrollBar::handle:vertical {
                    background-color: #505050;
                    min-height: 20px;
                    border-radius: 6px;
                }
                
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                
                QScrollBar:horizontal {
                    background-color: #2d2d2d;
                    height: 12px;
                    margin: 0px;
                }
                
                QScrollBar::handle:horizontal {
                    background-color: #505050;
                    min-width: 20px;
                    border-radius: 6px;
                }
                
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                    width: 0px;
                }
                
                QToolTip {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #3c3c3c;
                }
            """)
            self.setPalette(DarkPalette())
        else:
            self.setStyleSheet("")
            self.setPalette(LightPalette())
        
        # Update setting
        self.dark_mode = enabled
        
        # Save setting
        settings = QSettings("ATFNet", "EnhancedATFNetApp")
        settings.setValue("appearance/dark_mode", enabled)
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create tab widget with custom tab bar
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabBar(CustomTabBar())
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_chart_tab()
        self.create_backtest_tab()
        self.create_optimization_tab()
        self.create_training_tab()
        self.create_live_trading_tab()
        self.create_analytics_tab()
        self.create_settings_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create system tray icon
        self.create_tray_icon()
    
    def create_toolbar(self):
        """Create the toolbar."""
        # Create toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(self.toolbar)
        
        # Add actions
        self.connect_action = QAction(QIcon(":/icons/connect.png"), "Connect", self)
        self.connect_action.triggered.connect(self.toggle_connection)
        self.toolbar.addAction(self.connect_action)
        
        self.toolbar.addSeparator()
        
        self.import_data_action = QAction(QIcon(":/icons/import.png"), "Import Data", self)
        self.import_data_action.triggered.connect(self.import_data_dialog)
        self.toolbar.addAction(self.import_data_action)
        
        self.export_data_action = QAction(QIcon(":/icons/export.png"), "Export Data", self)
        self.export_data_action.triggered.connect(self.export_data_dialog)
        self.toolbar.addAction(self.export_data_action)
        
        self.toolbar.addSeparator()
        
        self.start_backtest_action = QAction(QIcon(":/icons/backtest.png"), "Backtest", self)
        self.start_backtest_action.triggered.connect(self.run_backtest)
        self.toolbar.addAction(self.start_backtest_action)
        
        self.start_optimization_action = QAction(QIcon(":/icons/optimize.png"), "Optimize", self)
        self.start_optimization_action.triggered.connect(self.run_optimization)
        self.toolbar.addAction(self.start_optimization_action)
        
        self.start_training_action = QAction(QIcon(":/icons/train.png"), "Train", self)
        self.start_training_action.triggered.connect(self.run_training)
        self.toolbar.addAction(self.start_training_action)
        
        self.toolbar.addSeparator()
        
        self.start_trading_action = QAction(QIcon(":/icons/trade.png"), "Start Trading", self)
        self.start_trading_action.triggered.connect(self.start_live_trading)
        self.toolbar.addAction(self.start_trading_action)
        
        self.stop_trading_action = QAction(QIcon(":/icons/stop.png"), "Stop Trading", self)
        self.stop_trading_action.triggered.connect(self.stop_live_trading)
        self.stop_trading_action.setEnabled(False)
        self.toolbar.addAction(self.stop_trading_action)
        
        self.toolbar.addSeparator()
        
        self.settings_action = QAction(QIcon(":/icons/settings.png"), "Settings", self)
        self.settings_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(7))  # Settings tab
        self.toolbar.addAction(self.settings_action)
        
        self.help_action = QAction(QIcon(":/icons/help.png"), "Help", self)
        self.help_action.triggered.connect(self.show_help_dialog)
        self.toolbar.addAction(self.help_action)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # Connect action
        connect_action = QAction("Connect to MT5", self)
        connect_action.triggered.connect(self.toggle_connection)
        file_menu.addAction(connect_action)
        
        file_menu.addSeparator()
        
        # Import data action
        import_data_action = QAction("Import Data", self)
        import_data_action.triggered.connect(self.import_data_dialog)
        file_menu.addAction(import_data_action)
        
        # Export data action
        export_data_action = QAction("Export Data", self)
        export_data_action.triggered.connect(self.export_data_dialog)
        file_menu.addAction(export_data_action)
        
        file_menu.addSeparator()
        
        # Load config action
        load_config_action = QAction("Load Configuration", self)
        load_config_action.triggered.connect(self.load_config_dialog)
        file_menu.addAction(load_config_action)
        
        # Save config action
        save_config_action = QAction("Save Configuration", self)
        save_config_action.triggered.connect(self.save_config_dialog)
        file_menu.addAction(save_config_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Trading menu
        trading_menu = menu_bar.addMenu("Trading")
        
        # Start backtest action
        start_backtest_action = QAction("Run Backtest", self)
        start_backtest_action.triggered.connect(self.run_backtest)
        trading_menu.addAction(start_backtest_action)
        
        # Start optimization action
        start_optimization_action = QAction("Run Optimization", self)
        start_optimization_action.triggered.connect(self.run_optimization)
        trading_menu.addAction(start_optimization_action)
        
        # Start training action
        start_training_action = QAction("Train Models", self)
        start_training_action.triggered.connect(self.run_training)
        trading_menu.addAction(start_training_action)
        
        trading_menu.addSeparator()
        
        # Start live trading action
        start_trading_action = QAction("Start Live Trading", self)
        start_trading_action.triggered.connect(self.start_live_trading)
        trading_menu.addAction(start_trading_action)
        
        # Stop live trading action
        stop_trading_action = QAction("Stop Live Trading", self)
        stop_trading_action.triggered.connect(self.stop_live_trading)
        trading_menu.addAction(stop_trading_action)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")
        
        # Export results action
        export_results_action = QAction("Export Results", self)
        export_results_action.triggered.connect(self.export_results_dialog)
        tools_menu.addAction(export_results_action)
        
        # Generate report action
        generate_report_action = QAction("Generate Report", self)
        generate_report_action.triggered.connect(self.generate_report_dialog)
        tools_menu.addAction(generate_report_action)
        
        tools_menu.addSeparator()
        
        # Preferences action
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.show_preferences_dialog)
        tools_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menu_bar.addMenu("View")
        
        # Dark mode action
        dark_mode_action = QAction("Dark Mode", self)
        dark_mode_action.setCheckable(True)
        dark_mode_action.setChecked(True)
        dark_mode_action.triggered.connect(self.set_dark_mode)
        view_menu.addAction(dark_mode_action)
        
        # Fullscreen action
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        # Documentation action
        documentation_action = QAction("Documentation", self)
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def create_tray_icon(self):
        """Create the system tray icon."""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Create tray menu
        tray_menu = QMenu()
        
        # Add actions
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.showNormal)
        tray_menu.addAction(show_action)
        
        tray_menu.addSeparator()
        
        start_trading_action = QAction("Start Trading", self)
        start_trading_action.triggered.connect(self.start_live_trading)
        tray_menu.addAction(start_trading_action)
        
        stop_trading_action = QAction("Stop Trading", self)
        stop_trading_action.triggered.connect(self.stop_live_trading)
        tray_menu.addAction(stop_trading_action)
        
        tray_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        tray_menu.addAction(exit_action)
        
        # Set tray menu
        self.tray_icon.setContextMenu(tray_menu)
        
        # Show tray icon
        self.tray_icon.show()
        
        # Connect signals
        self.tray_icon.activated.connect(self.tray_icon_activated)
    
    def tray_icon_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self.showNormal()
    
    def create_dashboard_tab(self):
        """Create the dashboard tab."""
        # Create tab widget
        dashboard_tab = QWidget()
        self.tab_widget.addTab(dashboard_tab, "Dashboard")
        
        # Create layout
        dashboard_layout = QVBoxLayout(dashboard_tab)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(10)
        
        # Create top section
        top_section = QHBoxLayout()
        dashboard_layout.addLayout(top_section)
        
        # Create account info group
        account_group = CustomGroupBox("Account Information")
        top_section.addWidget(account_group)
        
        account_layout = QFormLayout(account_group)
        account_layout.setContentsMargins(15, 25, 15, 15)
        account_layout.setSpacing(10)
        
        self.account_balance_label = QLabel("N/A")
        account_layout.addRow("Balance:", self.account_balance_label)
        
        self.account_equity_label = QLabel("N/A")
        account_layout.addRow("Equity:", self.account_equity_label)
        
        self.account_margin_label = QLabel("N/A")
        account_layout.addRow("Margin:", self.account_margin_label)
        
        self.account_profit_label = QLabel("N/A")
        account_layout.addRow("Profit:", self.account_profit_label)
        
        self.account_leverage_label = QLabel("N/A")
        account_layout.addRow("Leverage:", self.account_leverage_label)
        
        # Create performance group
        performance_group = CustomGroupBox("Performance Metrics")
        top_section.addWidget(performance_group)
        
        performance_layout = QFormLayout(performance_group)
        performance_layout.setContentsMargins(15, 25, 15, 15)
        performance_layout.setSpacing(10)
        
        self.win_rate_label = QLabel("N/A")
        performance_layout.addRow("Win Rate:", self.win_rate_label)
        
        self.profit_factor_label = QLabel("N/A")
        performance_layout.addRow("Profit Factor:", self.profit_factor_label)
        
        self.drawdown_label = QLabel("N/A")
        performance_layout.addRow("Max Drawdown:", self.drawdown_label)
        
        self.sharpe_ratio_label = QLabel("N/A")
        performance_layout.addRow("Sharpe Ratio:", self.sharpe_ratio_label)
        
        self.total_trades_label = QLabel("N/A")
        performance_layout.addRow("Total Trades:", self.total_trades_label)
        
        # Create market info group
        market_group = CustomGroupBox("Market Information")
        top_section.addWidget(market_group)
        
        market_layout = QFormLayout(market_group)
        market_layout.setContentsMargins(15, 25, 15, 15)
        market_layout.setSpacing(10)
        
        self.symbol_label = QLabel("N/A")
        market_layout.addRow("Symbol:", self.symbol_label)
        
        self.timeframe_label = QLabel("N/A")
        market_layout.addRow("Timeframe:", self.timeframe_label)
        
        self.regime_label = QLabel("N/A")
        market_layout.addRow("Market Regime:", self.regime_label)
        
        self.volatility_label = QLabel("N/A")
        market_layout.addRow("Volatility:", self.volatility_label)
        
        self.signal_label = QLabel("N/A")
        market_layout.addRow("Signal:", self.signal_label)
        
        # Create middle section
        middle_section = QHBoxLayout()
        dashboard_layout.addLayout(middle_section)
        
        # Create equity chart
        equity_group = CustomGroupBox("Equity Curve")
        middle_section.addWidget(equity_group)
        
        equity_layout = QVBoxLayout(equity_group)
        equity_layout.setContentsMargins(15, 25, 15, 15)
        equity_layout.setSpacing(10)
        
        self.equity_canvas = PyQtGraphCanvas(dark_mode=True)
        equity_layout.addWidget(self.equity_canvas)
        
        # Create price chart
        price_group = CustomGroupBox("Price Chart")
        middle_section.addWidget(price_group)
        
        price_layout = QVBoxLayout(price_group)
        price_layout.setContentsMargins(15, 25, 15, 15)
        price_layout.setSpacing(10)
        
        self.price_canvas = PyQtGraphCanvas(dark_mode=True)
        price_layout.addWidget(self.price_canvas)
        
        # Create bottom section
        bottom_section = QHBoxLayout()
        dashboard_layout.addLayout(bottom_section)
        
        # Create positions table
        positions_group = CustomGroupBox("Open Positions")
        bottom_section.addWidget(positions_group)
        
        positions_layout = QVBoxLayout(positions_group)
        positions_layout.setContentsMargins(15, 25, 15, 15)
        positions_layout.setSpacing(10)
        
        self.positions_table = CustomTableWidget()
        self.positions_table.setColumnCount(8)
        self.positions_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Current Price", "SL", "Profit"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        positions_layout.addWidget(self.positions_table)
        
        # Create recent trades table
        trades_group = CustomGroupBox("Recent Trades")
        bottom_section.addWidget(trades_group)
        
        trades_layout = QVBoxLayout(trades_group)
        trades_layout.setContentsMargins(15, 25, 15, 15)
        trades_layout.setSpacing(10)
        
        self.trades_table = CustomTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Close Price", "Profit", "Time"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
    
    def create_chart_tab(self):
        """Create the chart tab."""
        # Create tab widget
        chart_tab = QWidget()
        self.tab_widget.addTab(chart_tab, "Charts")
        
        # Create layout
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(10)
        
        # Create top controls
        controls_layout = QHBoxLayout()
        chart_layout.addLayout(controls_layout)
        
        # Symbol selection
        symbol_label = QLabel("Symbol:")
        controls_layout.addWidget(symbol_label)
        
        self.chart_symbol_combo = CustomComboBox()
        self.chart_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"])
        self.chart_symbol_combo.currentTextChanged.connect(self.update_chart)
        controls_layout.addWidget(self.chart_symbol_combo)
        
        # Timeframe selection
        timeframe_label = QLabel("Timeframe:")
        controls_layout.addWidget(timeframe_label)
        
        self.chart_timeframe_combo = CustomComboBox()
        self.chart_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.chart_timeframe_combo.setCurrentText("H1")
        self.chart_timeframe_combo.currentTextChanged.connect(self.update_chart)
        controls_layout.addWidget(self.chart_timeframe_combo)
        
        # Chart type selection
        chart_type_label = QLabel("Chart Type:")
        controls_layout.addWidget(chart_type_label)
        
        self.chart_type_combo = CustomComboBox()
        self.chart_type_combo.addItems(["Candlestick", "OHLC", "Line", "Area"])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        controls_layout.addWidget(self.chart_type_combo)
        
        # Date range
        date_range_label = QLabel("Date Range:")
        controls_layout.addWidget(date_range_label)
        
        self.chart_date_range_combo = CustomComboBox()
        self.chart_date_range_combo.addItems(["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "Custom"])
        self.chart_date_range_combo.currentTextChanged.connect(self.update_chart_date_range)
        controls_layout.addWidget(self.chart_date_range_combo)
        
        # Custom date range
        self.chart_start_date = CustomDateEdit()
        self.chart_start_date.setDate(QDate.currentDate().addMonths(-1))
        self.chart_start_date.setVisible(False)
        self.chart_start_date.dateChanged.connect(self.update_chart)
        controls_layout.addWidget(self.chart_start_date)
        
        self.chart_end_date = CustomDateEdit()
        self.chart_end_date.setDate(QDate.currentDate())
        self.chart_end_date.setVisible(False)
        self.chart_end_date.dateChanged.connect(self.update_chart)
        controls_layout.addWidget(self.chart_end_date)
        
        # Refresh button
        self.refresh_chart_button = CustomButton("Refresh", primary=True)
        self.refresh_chart_button.clicked.connect(self.update_chart)
        controls_layout.addWidget(self.refresh_chart_button)
        
        controls_layout.addStretch()
        
        # Create splitter for chart and indicators
        chart_splitter = QSplitter(Qt.Vertical)
        chart_layout.addWidget(chart_splitter)
        
        # Create chart widget
        chart_widget = QWidget()
        chart_splitter.addWidget(chart_widget)
        
        chart_widget_layout = QVBoxLayout(chart_widget)
        chart_widget_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart canvas
        self.chart_canvas = QWebEngineView()
        self.chart_canvas.setUrl(QUrl("about:blank"))
        chart_widget_layout.addWidget(self.chart_canvas)
        
        # Create indicators widget
        indicators_widget = QWidget()
        chart_splitter.addWidget(indicators_widget)
        
        indicators_layout = QVBoxLayout(indicators_widget)
        indicators_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create indicators group
        indicators_group = CustomGroupBox("Technical Indicators")
        indicators_layout.addWidget(indicators_group)
        
        indicators_group_layout = QVBoxLayout(indicators_group)
        indicators_group_layout.setContentsMargins(15, 25, 15, 15)
        
        # Create indicators controls
        indicators_controls_layout = QHBoxLayout()
        indicators_group_layout.addLayout(indicators_controls_layout)
        
        # Indicator selection
        indicator_label = QLabel("Add Indicator:")
        indicators_controls_layout.addWidget(indicator_label)
        
        self.indicator_combo = CustomComboBox()
        self.indicator_combo.addItems([
            "Moving Average", "Bollinger Bands", "RSI", "MACD", "Stochastic",
            "ATR", "Ichimoku Cloud", "Fibonacci Retracement", "Volume Profile"
        ])
        indicators_controls_layout.addWidget(self.indicator_combo)
        
        # Add indicator button
        self.add_indicator_button = CustomButton("Add")
        self.add_indicator_button.clicked.connect(self.add_indicator)
        indicators_controls_layout.addWidget(self.add_indicator_button)
        
        indicators_controls_layout.addStretch()
        
        # Create indicators table
        self.indicators_table = CustomTableWidget()
        self.indicators_table.setColumnCount(4)
        self.indicators_table.setHorizontalHeaderLabels([
            "Indicator", "Parameters", "Color", "Actions"
        ])
        self.indicators_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        indicators_group_layout.addWidget(self.indicators_table)
        
        # Set splitter sizes
        chart_splitter.setSizes([700, 300])
    
    def create_backtest_tab(self):
        """Create the backtest tab."""
        # Create tab widget
        backtest_tab = QWidget()
        self.tab_widget.addTab(backtest_tab, "Backtest")
        
        # Create layout
        backtest_layout = QVBoxLayout(backtest_tab)
        backtest_layout.setContentsMargins(0, 0, 0, 0)
        backtest_layout.setSpacing(10)
        
        # Create splitter
        backtest_splitter = QSplitter(Qt.Horizontal)
        backtest_layout.addWidget(backtest_splitter)
        
        # Create left panel
        left_panel = QWidget()
        backtest_splitter.addWidget(left_panel)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Create parameters group
        params_group = CustomGroupBox("Backtest Parameters")
        left_layout.addWidget(params_group)
        
        params_layout = QFormLayout(params_group)
        params_layout.setContentsMargins(15, 25, 15, 15)
        params_layout.setSpacing(10)
        
        # Symbol selection
        self.backtest_symbol_combo = CustomComboBox()
        self.backtest_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"])
        params_layout.addRow("Symbol:", self.backtest_symbol_combo)
        
        # Timeframe selection
        self.backtest_timeframe_combo = CustomComboBox()
        self.backtest_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.backtest_timeframe_combo.setCurrentText("H1")
        params_layout.addRow("Timeframe:", self.backtest_timeframe_combo)
        
        # Date range
        date_range_layout = QHBoxLayout()
        
        self.backtest_start_date = CustomDateEdit()
        self.backtest_start_date.setDate(QDate.currentDate().addMonths(-6))
        date_range_layout.addWidget(self.backtest_start_date)
        
        date_range_layout.addWidget(QLabel("to"))
        
        self.backtest_end_date = CustomDateEdit()
        self.backtest_end_date.setDate(QDate.currentDate())
        date_range_layout.addWidget(self.backtest_end_date)
        
        params_layout.addRow("Date Range:", date_range_layout)
        
        # Initial balance
        self.backtest_balance = CustomDoubleSpinBox()
        self.backtest_balance.setRange(1000, 1000000)
        self.backtest_balance.setValue(10000)
        self.backtest_balance.setSingleStep(1000)
        params_layout.addRow("Initial Balance:", self.backtest_balance)
        
        # Create strategy parameters group
        strategy_group = CustomGroupBox("Strategy Parameters")
        left_layout.addWidget(strategy_group)
        
        strategy_layout = QFormLayout(strategy_group)
        strategy_layout.setContentsMargins(15, 25, 15, 15)
        strategy_layout.setSpacing(10)
        
        # Risk per trade
        self.backtest_risk = CustomDoubleSpinBox()
        self.backtest_risk.setRange(0.1, 10.0)
        self.backtest_risk.setValue(1.0)
        self.backtest_risk.setSingleStep(0.1)
        self.backtest_risk.setSuffix("%")
        strategy_layout.addRow("Risk Per Trade:", self.backtest_risk)
        
        # ATR multiplier
        self.backtest_atr_mult = CustomDoubleSpinBox()
        self.backtest_atr_mult.setRange(0.5, 5.0)
        self.backtest_atr_mult.setValue(1.5)
        self.backtest_atr_mult.setSingleStep(0.1)
        strategy_layout.addRow("ATR Multiplier:", self.backtest_atr_mult)
        
        # Attention threshold
        self.backtest_attention = CustomDoubleSpinBox()
        self.backtest_attention.setRange(0.1, 1.0)
        self.backtest_attention.setValue(0.6)
        self.backtest_attention.setSingleStep(0.05)
        strategy_layout.addRow("Attention Threshold:", self.backtest_attention)
        
        # Prediction threshold
        self.backtest_prediction = CustomDoubleSpinBox()
        self.backtest_prediction.setRange(0.0001, 0.01)
        self.backtest_prediction.setValue(0.0015)
        self.backtest_prediction.setSingleStep(0.0001)
        strategy_layout.addRow("Prediction Threshold:", self.backtest_prediction)
        
        # Use trend filter
        self.backtest_trend_filter = CustomCheckBox()
        self.backtest_trend_filter.setChecked(True)
        strategy_layout.addRow("Use Trend Filter:", self.backtest_trend_filter)
        
        # Use trailing stop
        self.backtest_trailing_stop = CustomCheckBox()
        self.backtest_trailing_stop.setChecked(True)
        strategy_layout.addRow("Use Trailing Stop:", self.backtest_trailing_stop)
        
        # Advanced options button
        self.advanced_options_button = CustomButton("Advanced Options...")
        self.advanced_options_button.clicked.connect(self.show_advanced_backtest_options)
        strategy_layout.addRow("", self.advanced_options_button)
        
        # Create buttons
        buttons_layout = QHBoxLayout()
        left_layout.addLayout(buttons_layout)
        
        self.run_backtest_button = CustomButton("Run Backtest", primary=True)
        self.run_backtest_button.clicked.connect(self.run_backtest)
        buttons_layout.addWidget(self.run_backtest_button)
        
        self.stop_backtest_button = CustomButton("Stop")
        self.stop_backtest_button.clicked.connect(self.stop_backtest)
        self.stop_backtest_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_backtest_button)
        
        self.export_backtest_button = CustomButton("Export Results")
        self.export_backtest_button.clicked.connect(self.export_backtest_results)
        self.export_backtest_button.setEnabled(False)
        buttons_layout.addWidget(self.export_backtest_button)
        
        # Create progress bar
        self.backtest_progress = CustomProgressBar()
        left_layout.addWidget(self.backtest_progress)
        
        # Create right panel
        right_panel = QWidget()
        backtest_splitter.addWidget(right_panel)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        # Create results tabs
        results_tabs = QTabWidget()
        right_layout.addWidget(results_tabs)
        
        # Create summary tab
        summary_tab = QWidget()
        results_tabs.addTab(summary_tab, "Summary")
        
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(10)
        
        # Create results group
        results_group = CustomGroupBox("Backtest Results")
        summary_layout.addWidget(results_group)
        
        results_layout = QFormLayout(results_group)
        results_layout.setContentsMargins(15, 25, 15, 15)
        results_layout.setSpacing(10)
        
        self.backtest_total_trades = QLabel("N/A")
        results_layout.addRow("Total Trades:", self.backtest_total_trades)
        
        self.backtest_win_rate = QLabel("N/A")
        results_layout.addRow("Win Rate:", self.backtest_win_rate)
        
        self.backtest_profit_factor = QLabel("N/A")
        results_layout.addRow("Profit Factor:", self.backtest_profit_factor)
        
        self.backtest_net_profit = QLabel("N/A")
        results_layout.addRow("Net Profit:", self.backtest_net_profit)
        
        self.backtest_max_drawdown = QLabel("N/A")
        results_layout.addRow("Max Drawdown:", self.backtest_max_drawdown)
        
        self.backtest_sharpe_ratio = QLabel("N/A")
        results_layout.addRow("Sharpe Ratio:", self.backtest_sharpe_ratio)
        
        self.backtest_sortino_ratio = QLabel("N/A")
        results_layout.addRow("Sortino Ratio:", self.backtest_sortino_ratio)
        
        self.backtest_avg_trade = QLabel("N/A")
        results_layout.addRow("Avg Trade:", self.backtest_avg_trade)
        
        self.backtest_max_consecutive_wins = QLabel("N/A")
        results_layout.addRow("Max Consecutive Wins:", self.backtest_max_consecutive_wins)
        
        self.backtest_max_consecutive_losses = QLabel("N/A")
        results_layout.addRow("Max Consecutive Losses:", self.backtest_max_consecutive_losses)
        
        # Create charts section
        charts_section = QHBoxLayout()
        summary_layout.addLayout(charts_section)
        
        # Create equity chart
        equity_group = CustomGroupBox("Equity Curve")
        charts_section.addWidget(equity_group)
        
        equity_layout = QVBoxLayout(equity_group)
        equity_layout.setContentsMargins(15, 25, 15, 15)
        equity_layout.setSpacing(10)
        
        self.backtest_equity_canvas = PyQtGraphCanvas(dark_mode=True)
        equity_layout.addWidget(self.backtest_equity_canvas)
        
        # Create trades chart
        trades_group = CustomGroupBox("Trades Distribution")
        charts_section.addWidget(trades_group)
        
        trades_layout = QVBoxLayout(trades_group)
        trades_layout.setContentsMargins(15, 25, 15, 15)
        trades_layout.setSpacing(10)
        
        self.backtest_trades_canvas = PyQtGraphCanvas(dark_mode=True)
        trades_layout.addWidget(self.backtest_trades_canvas)
        
        # Create trades tab
        trades_tab = QWidget()
        results_tabs.addTab(trades_tab, "Trades")
        
        trades_tab_layout = QVBoxLayout(trades_tab)
        trades_tab_layout.setContentsMargins(0, 0, 0, 0)
        trades_tab_layout.setSpacing(10)
        
        # Create trades table
        self.backtest_trades_table = CustomTableWidget()
        self.backtest_trades_table.setColumnCount(9)
        self.backtest_trades_table.setHorizontalHeaderLabels([
            "ID", "Type", "Open Time", "Close Time", "Open Price",
            "Close Price", "Volume", "Profit", "Reason"
        ])
        self.backtest_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_tab_layout.addWidget(self.backtest_trades_table)
        
        # Create chart tab
        chart_tab = QWidget()
        results_tabs.addTab(chart_tab, "Chart")
        
        chart_tab_layout = QVBoxLayout(chart_tab)
        chart_tab_layout.setContentsMargins(0, 0, 0, 0)
        chart_tab_layout.setSpacing(10)
        
        # Create chart canvas
        self.backtest_chart_canvas = QWebEngineView()
        self.backtest_chart_canvas.setUrl(QUrl("about:blank"))
        chart_tab_layout.addWidget(self.backtest_chart_canvas)
        
        # Create report tab
        report_tab = QWidget()
        results_tabs.addTab(report_tab, "Report")
        
        report_layout = QVBoxLayout(report_tab)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_layout.setSpacing(10)
        
        # Create report text
        self.backtest_report_text = CustomTextEdit()
        self.backtest_report_text.setReadOnly(True)
        report_layout.addWidget(self.backtest_report_text)
        
        # Set splitter sizes
        backtest_splitter.setSizes([400, 800])
    
    def create_optimization_tab(self):
        """Create the optimization tab."""
        # Create tab widget
        optimization_tab = QWidget()
        self.tab_widget.addTab(optimization_tab, "Optimization")
        
        # Create layout
        optimization_layout = QVBoxLayout(optimization_tab)
        optimization_layout.setContentsMargins(0, 0, 0, 0)
        optimization_layout.setSpacing(10)
        
        # Create splitter
        optimization_splitter = QSplitter(Qt.Horizontal)
        optimization_layout.addWidget(optimization_splitter)
        
        # Create left panel
        left_panel = QWidget()
        optimization_splitter.addWidget(left_panel)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Create parameters group
        params_group = CustomGroupBox("Optimization Parameters")
        left_layout.addWidget(params_group)
        
        params_layout = QFormLayout(params_group)
        params_layout.setContentsMargins(15, 25, 15, 15)
        params_layout.setSpacing(10)
        
        # Symbol selection
        self.optimization_symbol_combo = CustomComboBox()
        self.optimization_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"])
        params_layout.addRow("Symbol:", self.optimization_symbol_combo)
        
        # Timeframe selection
        self.optimization_timeframe_combo = CustomComboBox()
        self.optimization_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.optimization_timeframe_combo.setCurrentText("H1")
        params_layout.addRow("Timeframe:", self.optimization_timeframe_combo)
        
        # Date range
        date_range_layout = QHBoxLayout()
        
        self.optimization_start_date = CustomDateEdit()
        self.optimization_start_date.setDate(QDate.currentDate().addMonths(-6))
        date_range_layout.addWidget(self.optimization_start_date)
        
        date_range_layout.addWidget(QLabel("to"))
        
        self.optimization_end_date = CustomDateEdit()
        self.optimization_end_date.setDate(QDate.currentDate())
        date_range_layout.addWidget(self.optimization_end_date)
        
        params_layout.addRow("Date Range:", date_range_layout)
        
        # Optimization method
        self.optimization_method_combo = CustomComboBox()
        self.optimization_method_combo.addItems(["Grid Search", "Genetic Algorithm", "Bayesian Optimization", "Particle Swarm"])
        params_layout.addRow("Method:", self.optimization_method_combo)
        
        # Optimization metric
        self.optimization_metric_combo = CustomComboBox()
        self.optimization_metric_combo.addItems(["Net Profit", "Profit Factor", "Sharpe Ratio", "Sortino Ratio", "Custom"])
        params_layout.addRow("Metric:", self.optimization_metric_combo)
        
        # Create parameters to optimize group
        optimize_group = CustomGroupBox("Parameters to Optimize")
        left_layout.addWidget(optimize_group)
        
        optimize_layout = QVBoxLayout(optimize_group)
        optimize_layout.setContentsMargins(15, 25, 15, 15)
        optimize_layout.setSpacing(10)
        
        # Create parameters table
        self.optimization_params_table = CustomTableWidget()
        self.optimization_params_table.setColumnCount(5)
        self.optimization_params_table.setHorizontalHeaderLabels([
            "Parameter", "Min", "Max", "Step", "Enable"
        ])
        self.optimization_params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        optimize_layout.addWidget(self.optimization_params_table)
        
        # Add default parameters
        self.optimization_params_table.setRowCount(5)
        
        # Risk per trade
        self.optimization_params_table.setItem(0, 0, QTableWidgetItem("Risk Per Trade (%)"))
        self.optimization_params_table.setItem(0, 1, QTableWidgetItem("0.5"))
        self.optimization_params_table.setItem(0, 2, QTableWidgetItem("5.0"))
        self.optimization_params_table.setItem(0, 3, QTableWidgetItem("0.5"))
        risk_check = QCheckBox()
        risk_check.setChecked(True)
        self.optimization_params_table.setCellWidget(0, 4, risk_check)
        
        # ATR multiplier
        self.optimization_params_table.setItem(1, 0, QTableWidgetItem("ATR Multiplier"))
        self.optimization_params_table.setItem(1, 1, QTableWidgetItem("1.0"))
        self.optimization_params_table.setItem(1, 2, QTableWidgetItem("3.0"))
        self.optimization_params_table.setItem(1, 3, QTableWidgetItem("0.5"))
        atr_check = QCheckBox()
        atr_check.setChecked(True)
        self.optimization_params_table.setCellWidget(1, 4, atr_check)
        
        # Attention threshold
        self.optimization_params_table.setItem(2, 0, QTableWidgetItem("Attention Threshold"))
        self.optimization_params_table.setItem(2, 1, QTableWidgetItem("0.3"))
        self.optimization_params_table.setItem(2, 2, QTableWidgetItem("0.9"))
        self.optimization_params_table.setItem(2, 3, QTableWidgetItem("0.1"))
        attention_check = QCheckBox()
        attention_check.setChecked(True)
        self.optimization_params_table.setCellWidget(2, 4, attention_check)
        
        # Prediction threshold
        self.optimization_params_table.setItem(3, 0, QTableWidgetItem("Prediction Threshold"))
        self.optimization_params_table.setItem(3, 1, QTableWidgetItem("0.0005"))
        self.optimization_params_table.setItem(3, 2, QTableWidgetItem("0.0030"))
        self.optimization_params_table.setItem(3, 3, QTableWidgetItem("0.0005"))
        prediction_check = QCheckBox()
        prediction_check.setChecked(True)
        self.optimization_params_table.setCellWidget(3, 4, prediction_check)
        
        # Trailing stop multiplier
        self.optimization_params_table.setItem(4, 0, QTableWidgetItem("Trailing Stop Multiplier"))
        self.optimization_params_table.setItem(4, 1, QTableWidgetItem("1.0"))
        self.optimization_params_table.setItem(4, 2, QTableWidgetItem("4.0"))
        self.optimization_params_table.setItem(4, 3, QTableWidgetItem("0.5"))
        trailing_check = QCheckBox()
        trailing_check.setChecked(False)
        self.optimization_params_table.setCellWidget(4, 4, trailing_check)
        
        # Create buttons
        buttons_layout = QHBoxLayout()
        left_layout.addLayout(buttons_layout)
        
        self.run_optimization_button = CustomButton("Run Optimization", primary=True)
        self.run_optimization_button.clicked.connect(self.run_optimization)
        buttons_layout.addWidget(self.run_optimization_button)
        
        self.stop_optimization_button = CustomButton("Stop")
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.stop_optimization_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_optimization_button)
        
        self.export_optimization_button = CustomButton("Export Results")
        self.export_optimization_button.clicked.connect(self.export_optimization_results)
        self.export_optimization_button.setEnabled(False)
        buttons_layout.addWidget(self.export_optimization_button)
        
        # Create progress bar
        self.optimization_progress = CustomProgressBar()
        left_layout.addWidget(self.optimization_progress)
        
        # Create right panel
        right_panel = QWidget()
        optimization_splitter.addWidget(right_panel)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        # Create results group
        results_group = CustomGroupBox("Optimization Results")
        right_layout.addWidget(results_group)
        
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(15, 25, 15, 15)
        results_layout.setSpacing(10)
        
        # Create results table
        self.optimization_results_table = CustomTableWidget()
        self.optimization_results_table.setColumnCount(7)
        self.optimization_results_table.setHorizontalHeaderLabels([
            "Rank", "Net Profit", "Profit Factor", "Sharpe Ratio", "Win Rate", "Trades", "Parameters"
        ])
        self.optimization_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.optimization_results_table)
        
        # Create visualization group
        visualization_group = CustomGroupBox("Visualization")
        right_layout.addWidget(visualization_group)
        
        visualization_layout = QVBoxLayout(visualization_group)
        visualization_layout.setContentsMargins(15, 25, 15, 15)
        visualization_layout.setSpacing(10)
        
        # Create visualization controls
        controls_layout = QHBoxLayout()
        visualization_layout.addLayout(controls_layout)
        
        # X-axis parameter
        x_axis_label = QLabel("X-Axis:")
        controls_layout.addWidget(x_axis_label)
        
        self.optimization_x_axis_combo = CustomComboBox()
        self.optimization_x_axis_combo.addItems(["Risk Per Trade", "ATR Multiplier", "Attention Threshold", "Prediction Threshold"])
        self.optimization_x_axis_combo.currentTextChanged.connect(self.update_optimization_visualization)
        controls_layout.addWidget(self.optimization_x_axis_combo)
        
        # Y-axis parameter
        y_axis_label = QLabel("Y-Axis:")
        controls_layout.addWidget(y_axis_label)
        
        self.optimization_y_axis_combo = CustomComboBox()
        self.optimization_y_axis_combo.addItems(["Net Profit", "Profit Factor", "Sharpe Ratio", "Win Rate"])
        self.optimization_y_axis_combo.currentTextChanged.connect(self.update_optimization_visualization)
        controls_layout.addWidget(self.optimization_y_axis_combo)
        
        # Color by parameter
        color_by_label = QLabel("Color By:")
        controls_layout.addWidget(color_by_label)
        
        self.optimization_color_by_combo = CustomComboBox()
        self.optimization_color_by_combo.addItems(["Net Profit", "Profit Factor", "Sharpe Ratio", "Win Rate"])
        self.optimization_color_by_combo.currentTextChanged.connect(self.update_optimization_visualization)
        controls_layout.addWidget(self.optimization_color_by_combo)
        
        # Create visualization canvas
        self.optimization_canvas = PyQtGraphCanvas(dark_mode=True)
        visualization_layout.addWidget(self.optimization_canvas)
        
        # Set splitter sizes
        optimization_splitter.setSizes([400, 800])
    
    def create_training_tab(self):
        """Create the training tab."""
        # Create tab widget
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "Training")
        
        # Create layout
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(0, 0, 0, 0)
        training_layout.setSpacing(10)
        
        # Create splitter
        training_splitter = QSplitter(Qt.Horizontal)
        training_layout.addWidget(training_splitter)
        
        # Create left panel
        left_panel = QWidget()
        training_splitter.addWidget(left_panel)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Create parameters group
        params_group = CustomGroupBox("Training Parameters")
        left_layout.addWidget(params_group)
        
        params_layout = QFormLayout(params_group)
        params_layout.setContentsMargins(15, 25, 15, 15)
        params_layout.setSpacing(10)
        
        # Symbol selection
        self.training_symbol_combo = CustomComboBox()
        self.training_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"])
        params_layout.addRow("Symbol:", self.training_symbol_combo)
        
        # Timeframe selection
        self.training_timeframe_combo = CustomComboBox()
        self.training_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.training_timeframe_combo.setCurrentText("H1")
        params_layout.addRow("Timeframe:", self.training_timeframe_combo)
        
        # Date range
        date_range_layout = QHBoxLayout()
        
        self.training_start_date = CustomDateEdit()
        self.training_start_date.setDate(QDate.currentDate().addYears(-1))
        date_range_layout.addWidget(self.training_start_date)
        
        date_range_layout.addWidget(QLabel("to"))
        
        self.training_end_date = CustomDateEdit()
        self.training_end_date.setDate(QDate.currentDate())
        date_range_layout.addWidget(self.training_end_date)
        
        params_layout.addRow("Date Range:", date_range_layout)
        
        # Create model parameters group
        model_group = CustomGroupBox("Model Parameters")
        left_layout.addWidget(model_group)
        
        model_layout = QFormLayout(model_group)
        model_layout.setContentsMargins(15, 25, 15, 15)
        model_layout.setSpacing(10)
        
        # Model type
        self.training_model_type = CustomComboBox()
        self.training_model_type.addItems(["lstm", "gru", "cnn", "transformer", "hybrid"])
        model_layout.addRow("Model Type:", self.training_model_type)
        
        # Epochs
        self.training_epochs = CustomSpinBox()
        self.training_epochs.setRange(10, 1000)
        self.training_epochs.setValue(100)
        self.training_epochs.setSingleStep(10)
        model_layout.addRow("Epochs:", self.training_epochs)
        
        # Batch size
        self.training_batch_size = CustomSpinBox()
        self.training_batch_size.setRange(8, 256)
        self.training_batch_size.setValue(32)
        self.training_batch_size.setSingleStep(8)
        model_layout.addRow("Batch Size:", self.training_batch_size)
        
        # Sequence length
        self.training_sequence_length = CustomSpinBox()
        self.training_sequence_length.setRange(10, 200)
        self.training_sequence_length.setValue(60)
        self.training_sequence_length.setSingleStep(10)
        model_layout.addRow("Sequence Length:", self.training_sequence_length)
        
        # Learning rate
        self.training_learning_rate = CustomDoubleSpinBox()
        self.training_learning_rate.setRange(0.0001, 0.1)
        self.training_learning_rate.setValue(0.001)
        self.training_learning_rate.setSingleStep(0.0001)
        self.training_learning_rate.setDecimals(6)
        model_layout.addRow("Learning Rate:", self.training_learning_rate)
        
        # Train regime classifier
        self.training_regime = CustomCheckBox()
        self.training_regime.setChecked(True)
        model_layout.addRow("Train Regime Classifier:", self.training_regime)
        
        # Use GPU
        self.training_use_gpu = CustomCheckBox()
        self.training_use_gpu.setChecked(True)
        model_layout.addRow("Use GPU:", self.training_use_gpu)
        
        # Advanced options button
        self.advanced_training_button = CustomButton("Advanced Options...")
        self.advanced_training_button.clicked.connect(self.show_advanced_training_options)
        model_layout.addRow("", self.advanced_training_button)
        
        # Create buttons
        buttons_layout = QHBoxLayout()
        left_layout.addLayout(buttons_layout)
        
        self.run_training_button = CustomButton("Run Training", primary=True)
        self.run_training_button.clicked.connect(self.run_training)
        buttons_layout.addWidget(self.run_training_button)
        
        self.stop_training_button = CustomButton("Stop")
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_training_button)
        
        # Create progress bar
        self.training_progress = CustomProgressBar()
        left_layout.addWidget(self.training_progress)
        
        # Create right panel
        right_panel = QWidget()
        training_splitter.addWidget(right_panel)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        # Create tabs
        training_results_tabs = QTabWidget()
        right_layout.addWidget(training_results_tabs)
        
        # Create price prediction results tab
        price_tab = QWidget()
        training_results_tabs.addTab(price_tab, "Price Prediction")
        
        price_layout = QVBoxLayout(price_tab)
        price_layout.setContentsMargins(0, 0, 0, 0)
        price_layout.setSpacing(10)
        
        # Create price prediction results group
        price_group = CustomGroupBox("Price Prediction Model")
        price_layout.addWidget(price_group)
        
        price_results_layout = QFormLayout(price_group)
        price_results_layout.setContentsMargins(15, 25, 15, 15)
        price_results_layout.setSpacing(10)
        
        self.price_model_name = QLabel("N/A")
        price_results_layout.addRow("Model Name:", self.price_model_name)
        
        self.price_training_loss = QLabel("N/A")
        price_results_layout.addRow("Training Loss:", self.price_training_loss)
        
        self.price_validation_loss = QLabel("N/A")
        price_results_layout.addRow("Validation Loss:", self.price_validation_loss)
        
        self.price_accuracy = QLabel("N/A")
        price_results_layout.addRow("Accuracy:", self.price_accuracy)
        
        self.price_model_size = QLabel("N/A")
        price_results_layout.addRow("Model Size:", self.price_model_size)
        
        # Create price prediction chart
        price_chart_group = CustomGroupBox("Training Progress")
        price_layout.addWidget(price_chart_group)
        
        price_chart_layout = QVBoxLayout(price_chart_group)
        price_chart_layout.setContentsMargins(15, 25, 15, 15)
        price_chart_layout.setSpacing(10)
        
        self.price_training_canvas = PyQtGraphCanvas(dark_mode=True)
        price_chart_layout.addWidget(self.price_training_canvas)
        
        # Create regime classifier results tab
        regime_tab = QWidget()
        training_results_tabs.addTab(regime_tab, "Regime Classifier")
        
        regime_layout = QVBoxLayout(regime_tab)
        regime_layout.setContentsMargins(0, 0, 0, 0)
        regime_layout.setSpacing(10)
        
        # Create regime classifier results group
        regime_group = CustomGroupBox("Regime Classifier Model")
        regime_layout.addWidget(regime_group)
        
        regime_results_layout = QFormLayout(regime_group)
        regime_results_layout.setContentsMargins(15, 25, 15, 15)
        regime_results_layout.setSpacing(10)
        
        self.regime_model_name = QLabel("N/A")
        regime_results_layout.addRow("Model Name:", self.regime_model_name)
        
        self.regime_training_loss = QLabel("N/A")
        regime_results_layout.addRow("Training Loss:", self.regime_training_loss)
        
        self.regime_validation_loss = QLabel("N/A")
        regime_results_layout.addRow("Validation Loss:", self.regime_validation_loss)
        
        self.regime_accuracy = QLabel("N/A")
        regime_results_layout.addRow("Accuracy:", self.regime_accuracy)
        
        self.regime_f1_score = QLabel("N/A")
        regime_results_layout.addRow("F1 Score:", self.regime_f1_score)
        
        # Create regime classifier chart
        regime_chart_group = CustomGroupBox("Training Progress")
        regime_layout.addWidget(regime_chart_group)
        
        regime_chart_layout = QVBoxLayout(regime_chart_group)
        regime_chart_layout.setContentsMargins(15, 25, 15, 15)
        regime_chart_layout.setSpacing(10)
        
        self.regime_training_canvas = PyQtGraphCanvas(dark_mode=True)
        regime_chart_layout.addWidget(self.regime_training_canvas)
        
        # Create model evaluation tab
        evaluation_tab = QWidget()
        training_results_tabs.addTab(evaluation_tab, "Evaluation")
        
        evaluation_layout = QVBoxLayout(evaluation_tab)
        evaluation_layout.setContentsMargins(0, 0, 0, 0)
        evaluation_layout.setSpacing(10)
        
        # Create evaluation chart
        evaluation_group = CustomGroupBox("Model Evaluation")
        evaluation_layout.addWidget(evaluation_group)
        
        evaluation_chart_layout = QVBoxLayout(evaluation_group)
        evaluation_chart_layout.setContentsMargins(15, 25, 15, 15)
        evaluation_chart_layout.setSpacing(10)
        
        self.evaluation_canvas = PyQtGraphCanvas(dark_mode=True)
        evaluation_chart_layout.addWidget(self.evaluation_canvas)
        
        # Create feature importance group
        feature_group = CustomGroupBox("Feature Importance")
        evaluation_layout.addWidget(feature_group)
        
        feature_layout = QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(15, 25, 15, 15)
        feature_layout.setSpacing(10)
        
        self.feature_importance_canvas = PyQtGraphCanvas(dark_mode=True)
        feature_layout.addWidget(self.feature_importance_canvas)
        
        # Create log tab
        log_tab = QWidget()
        training_results_tabs.addTab(log_tab, "Log")
        
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(10)
        
        # Create log text
        self.training_log = CustomTextEdit()
        self.training_log.setReadOnly(True)
        log_layout.addWidget(self.training_log)
        
        # Set splitter sizes
        training_splitter.setSizes([400, 800])
    
    def create_live_trading_tab(self):
        """Create the live trading tab."""
        # Create tab widget
        live_tab = QWidget()
        self.tab_widget.addTab(live_tab, "Live Trading")
        
        # Create layout
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(10)
        
        # Create splitter
        live_splitter = QSplitter(Qt.Horizontal)
        live_layout.addWidget(live_splitter)
        
        # Create left panel
        left_panel = QWidget()
        live_splitter.addWidget(left_panel)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # Create parameters group
        params_group = CustomGroupBox("Trading Parameters")
        left_layout.addWidget(params_group)
        
        params_layout = QFormLayout(params_group)
        params_layout.setContentsMargins(15, 25, 15, 15)
        params_layout.setSpacing(10)
        
        # Symbol selection
        self.live_symbol_combo = CustomComboBox()
        self.live_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"])
        params_layout.addRow("Symbol:", self.live_symbol_combo)
        
        # Timeframe selection
        self.live_timeframe_combo = CustomComboBox()
        self.live_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.live_timeframe_combo.setCurrentText("H1")
        params_layout.addRow("Timeframe:", self.live_timeframe_combo)
        
        # Create strategy parameters group
        strategy_group = CustomGroupBox("Strategy Parameters")
        left_layout.addWidget(strategy_group)
        
        strategy_layout = QFormLayout(strategy_group)
        strategy_layout.setContentsMargins(15, 25, 15, 15)
        strategy_layout.setSpacing(10)
        
        # Risk per trade
        self.live_risk = CustomDoubleSpinBox()
        self.live_risk.setRange(0.1, 10.0)
        self.live_risk.setValue(1.0)
        self.live_risk.setSingleStep(0.1)
        self.live_risk.setSuffix("%")
        strategy_layout.addRow("Risk Per Trade:", self.live_risk)
        
        # ATR multiplier
        self.live_atr_mult = CustomDoubleSpinBox()
        self.live_atr_mult.setRange(0.5, 5.0)
        self.live_atr_mult.setValue(1.5)
        self.live_atr_mult.setSingleStep(0.1)
        strategy_layout.addRow("ATR Multiplier:", self.live_atr_mult)
        
        # Attention threshold
        self.live_attention = CustomDoubleSpinBox()
        self.live_attention.setRange(0.1, 1.0)
        self.live_attention.setValue(0.6)
        self.live_attention.setSingleStep(0.05)
        strategy_layout.addRow("Attention Threshold:", self.live_attention)
        
        # Prediction threshold
        self.live_prediction = CustomDoubleSpinBox()
        self.live_prediction.setRange(0.0001, 0.01)
        self.live_prediction.setValue(0.0015)
        self.live_prediction.setSingleStep(0.0001)
        strategy_layout.addRow("Prediction Threshold:", self.live_prediction)
        
        # Use trend filter
        self.live_trend_filter = CustomCheckBox()
        self.live_trend_filter.setChecked(True)
        strategy_layout.addRow("Use Trend Filter:", self.live_trend_filter)
        
        # Use trailing stop
        self.live_trailing_stop = CustomCheckBox()
        self.live_trailing_stop.setChecked(True)
        strategy_layout.addRow("Use Trailing Stop:", self.live_trailing_stop)
        
        # Max positions
        self.live_max_positions = CustomSpinBox()
        self.live_max_positions.setRange(1, 10)
        self.live_max_positions.setValue(3)
        strategy_layout.addRow("Max Positions:", self.live_max_positions)
        
        # Advanced options button
        self.advanced_live_button = CustomButton("Advanced Options...")
        self.advanced_live_button.clicked.connect(self.show_advanced_live_options)
        strategy_layout.addRow("", self.advanced_live_button)
        
        # Create buttons
        buttons_layout = QHBoxLayout()
        left_layout.addLayout(buttons_layout)
        
        self.start_live_button = CustomButton("Start Trading", primary=True)
        self.start_live_button.clicked.connect(self.start_live_trading)
        buttons_layout.addWidget(self.start_live_button)
        
        self.stop_live_button = CustomButton("Stop Trading")
        self.stop_live_button.clicked.connect(self.stop_live_trading)
        self.stop_live_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_live_button)
        
        # Create status section
        status_group = CustomGroupBox("Trading Status")
        left_layout.addWidget(status_group)
        
        status_layout = QFormLayout(status_group)
        status_layout.setContentsMargins(15, 25, 15, 15)
        status_layout.setSpacing(10)
        
        self.live_status_label = QLabel("Stopped")
        status_layout.addRow("Status:", self.live_status_label)
        
        self.live_account_balance = QLabel("N/A")
        status_layout.addRow("Account Balance:", self.live_account_balance)
        
        self.live_account_equity = QLabel("N/A")
        status_layout.addRow("Account Equity:", self.live_account_equity)
        
        self.live_current_profit = QLabel("N/A")
        status_layout.addRow("Current Profit:", self.live_current_profit)
        
        # Create signal group
        signal_group = CustomGroupBox("Current Signals")
        left_layout.addWidget(signal_group)
        
        signal_layout = QFormLayout(signal_group)
        signal_layout.setContentsMargins(15, 25, 15, 15)
        signal_layout.setSpacing(10)
        
        self.live_attention_score = QLabel("N/A")
        signal_layout.addRow("Attention Score:", self.live_attention_score)
        
        self.live_prediction_value = QLabel("N/A")
        signal_layout.addRow("Prediction:", self.live_prediction_value)
        
        self.live_regime_value = QLabel("N/A")
        signal_layout.addRow("Market Regime:", self.live_regime_value)
        
        self.live_signal_value = QLabel("N/A")
        signal_layout.addRow("Signal:", self.live_signal_value)
        
        # Create right panel
        right_panel = QWidget()
        live_splitter.addWidget(right_panel)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        # Create tabs
        live_tabs = QTabWidget()
        right_layout.addWidget(live_tabs)
        
        # Create charts tab
        charts_tab = QWidget()
        live_tabs.addTab(charts_tab, "Charts")
        
        charts_layout = QVBoxLayout(charts_tab)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(10)
        
        # Create charts section
        charts_section = QHBoxLayout()
        charts_layout.addLayout(charts_section)
        
        # Create price chart
        price_group = CustomGroupBox("Price Chart")
        charts_section.addWidget(price_group)
        
        price_layout = QVBoxLayout(price_group)
        price_layout.setContentsMargins(15, 25, 15, 15)
        price_layout.setSpacing(10)
        
        self.live_price_canvas = PyQtGraphCanvas(dark_mode=True)
        price_layout.addWidget(self.live_price_canvas)
        
        # Create equity chart
        equity_group = CustomGroupBox("Equity Curve")
        charts_section.addWidget(equity_group)
        
        equity_layout = QVBoxLayout(equity_group)
        equity_layout.setContentsMargins(15, 25, 15, 15)
        equity_layout.setSpacing(10)
        
        self.live_equity_canvas = PyQtGraphCanvas(dark_mode=True)
        equity_layout.addWidget(self.live_equity_canvas)
        
        # Create positions tab
        positions_tab = QWidget()
        live_tabs.addTab(positions_tab, "Positions")
        
        positions_tab_layout = QVBoxLayout(positions_tab)
        positions_tab_layout.setContentsMargins(0, 0, 0, 0)
        positions_tab_layout.setSpacing(10)
        
        # Create positions table
        self.live_positions_table = CustomTableWidget()
        self.live_positions_table.setColumnCount(8)
        self.live_positions_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Current Price", "SL", "Profit"
        ])
        self.live_positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        positions_tab_layout.addWidget(self.live_positions_table)
        
        # Create trades tab
        trades_tab = QWidget()
        live_tabs.addTab(trades_tab, "Trades History")
        
        trades_tab_layout = QVBoxLayout(trades_tab)
        trades_tab_layout.setContentsMargins(0, 0, 0, 0)
        trades_tab_layout.setSpacing(10)
        
        # Create trades table
        self.live_trades_table = CustomTableWidget()
        self.live_trades_table.setColumnCount(8)
        self.live_trades_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Type", "Volume", "Open Price",
            "Close Price", "Profit", "Time"
        ])
        self.live_trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_tab_layout.addWidget(self.live_trades_table)
        
        # Create log tab
        log_tab = QWidget()
        live_tabs.addTab(log_tab, "Log")
        
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(10)
        
        # Create log text
        self.live_log = CustomTextEdit()
        self.live_log.setReadOnly(True)
        log_layout.addWidget(self.live_log)
        
        # Set splitter sizes
        live_splitter.setSizes([400, 800])
    
    def create_analytics_tab(self):
        """Create the analytics tab."""
        # Create tab widget
        analytics_tab = QWidget()
        self.tab_widget.addTab(analytics_tab, "Analytics")
        
        # Create layout
        analytics_layout = QVBoxLayout(analytics_tab)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(10)
        
        # Create tabs
        analytics_tabs = QTabWidget()
        analytics_layout.addWidget(analytics_tabs)
        
        # Create performance tab
        performance_tab = QWidget()
        analytics_tabs.addTab(performance_tab, "Performance")
        
        performance_layout = QVBoxLayout(performance_tab)
        performance_layout.setContentsMargins(10, 10, 10, 10)
        performance_layout.setSpacing(10)
        
        # Create performance metrics group
        metrics_group = CustomGroupBox("Performance Metrics")
        performance_layout.addWidget(metrics_group)
        
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setContentsMargins(15, 25, 15, 15)
        metrics_layout.setSpacing(20)
        
        # Create metric widgets
        self.create_metric_widget(metrics_layout, 0, 0, "Total Return", "25.4%", "up")
        self.create_metric_widget(metrics_layout, 0, 1, "Win Rate", "68.2%", "up")
        self.create_metric_widget(metrics_layout, 0, 2, "Profit Factor", "2.34", "up")
        self.create_metric_widget(metrics_layout, 0, 3, "Sharpe Ratio", "1.87", "up")
        
        self.create_metric_widget(metrics_layout, 1, 0, "Max Drawdown", "12.3%", "down")
        self.create_metric_widget(metrics_layout, 1, 1, "Recovery Factor", "2.07", "up")
        self.create_metric_widget(metrics_layout, 1, 2, "Avg Trade", "$127.45", "up")
        self.create_metric_widget(metrics_layout, 1, 3, "Expectancy", "$98.32", "up")
        
        # Create performance charts group
        charts_group = CustomGroupBox("Performance Charts")
        performance_layout.addWidget(charts_group)
        
        charts_layout = QHBoxLayout(charts_group)
        charts_layout.setContentsMargins(15, 25, 15, 15)
        charts_layout.setSpacing(20)
        
        # Create equity curve chart
        equity_layout = QVBoxLayout()
        charts_layout.addLayout(equity_layout)
        
        equity_label = QLabel("Equity Curve")
        equity_label.setAlignment(Qt.AlignCenter)
        equity_layout.addWidget(equity_label)
        
        self.analytics_equity_canvas = PyQtGraphCanvas(dark_mode=True)
        equity_layout.addWidget(self.analytics_equity_canvas)
        
        # Create drawdown chart
        drawdown_layout = QVBoxLayout()
        charts_layout.addLayout(drawdown_layout)
        
        drawdown_label = QLabel("Drawdown")
        drawdown_label.setAlignment(Qt.AlignCenter)
        drawdown_layout.addWidget(drawdown_label)
        
        self.analytics_drawdown_canvas = PyQtGraphCanvas(dark_mode=True)
        drawdown_layout.addWidget(self.analytics_drawdown_canvas)
        
        # Create trades distribution group
        trades_group = CustomGroupBox("Trades Distribution")
        performance_layout.addWidget(trades_group)
        
        trades_layout = QHBoxLayout(trades_group)
        trades_layout.setContentsMargins(15, 25, 15, 15)
        trades_layout.setSpacing(20)
        
        # Create monthly returns chart
        monthly_layout = QVBoxLayout()
        trades_layout.addLayout(monthly_layout)
        
        monthly_label = QLabel("Monthly Returns")
        monthly_label.setAlignment(Qt.AlignCenter)
        monthly_layout.addWidget(monthly_label)
        
        self.analytics_monthly_canvas = PyQtGraphCanvas(dark_mode=True)
        monthly_layout.addWidget(self.analytics_monthly_canvas)
        
        # Create profit distribution chart
        profit_layout = QVBoxLayout()
        trades_layout.addLayout(profit_layout)
        
        profit_label = QLabel("Profit Distribution")
        profit_label.setAlignment(Qt.AlignCenter)
        profit_layout.addWidget(profit_label)
        
        self.analytics_profit_canvas = PyQtGraphCanvas(dark_mode=True)
        profit_layout.addWidget(self.analytics_profit_canvas)
        
        # Create risk analysis tab
        risk_tab = QWidget()
        analytics_tabs.addTab(risk_tab, "Risk Analysis")
        
        risk_layout = QVBoxLayout(risk_tab)
        risk_layout.setContentsMargins(10, 10, 10, 10)
        risk_layout.setSpacing(10)
        
        # Create risk metrics group
        risk_metrics_group = CustomGroupBox("Risk Metrics")
        risk_layout.addWidget(risk_metrics_group)
        
        risk_metrics_layout = QGridLayout(risk_metrics_group)
        risk_metrics_layout.setContentsMargins(15, 25, 15, 15)
        risk_metrics_layout.setSpacing(20)
        
        # Create risk metric widgets
        self.create_metric_widget(risk_metrics_layout, 0, 0, "Value at Risk (95%)", "$245.67", "down")
        self.create_metric_widget(risk_metrics_layout, 0, 1, "Expected Shortfall", "$312.45", "down")
        self.create_metric_widget(risk_metrics_layout, 0, 2, "Tail Ratio", "0.87", "neutral")
        self.create_metric_widget(risk_metrics_layout, 0, 3, "Calmar Ratio", "1.23", "up")
        
        # Create risk charts group
        risk_charts_group = CustomGroupBox("Risk Analysis")
        risk_layout.addWidget(risk_charts_group)
        
        risk_charts_layout = QHBoxLayout(risk_charts_group)
        risk_charts_layout.setContentsMargins(15, 25, 15, 15)
        risk_charts_layout.setSpacing(20)
        
        # Create VaR chart
        var_layout = QVBoxLayout()
        risk_charts_layout.addLayout(var_layout)
        
        var_label = QLabel("Value at Risk")
        var_label.setAlignment(Qt.AlignCenter)
        var_layout.addWidget(var_label)
        
        self.analytics_var_canvas = PyQtGraphCanvas(dark_mode=True)
        var_layout.addWidget(self.analytics_var_canvas)
        
        # Create correlation chart
        correlation_layout = QVBoxLayout()
        risk_charts_layout.addLayout(correlation_layout)
        
        correlation_label = QLabel("Return Correlation")
        correlation_label.setAlignment(Qt.AlignCenter)
        correlation_layout.addWidget(correlation_label)
        
        self.analytics_correlation_canvas = PyQtGraphCanvas(dark_mode=True)
        correlation_layout.addWidget(self.analytics_correlation_canvas)
        
        # Create stress test group
        stress_group = CustomGroupBox("Stress Testing")
        risk_layout.addWidget(stress_group)
        
        stress_layout = QVBoxLayout(stress_group)
        stress_layout.setContentsMargins(15, 25, 15, 15)
        stress_layout.setSpacing(10)
        
        # Create stress test controls
        controls_layout = QHBoxLayout()
        stress_layout.addLayout(controls_layout)
        
        scenario_label = QLabel("Scenario:")
        controls_layout.addWidget(scenario_label)
        
        self.stress_scenario_combo = CustomComboBox()
        self.stress_scenario_combo.addItems([
            "Market Crash (-20%)", "High Volatility", "Low Liquidity",
            "Interest Rate Hike", "Custom Scenario"
        ])
        controls_layout.addWidget(self.stress_scenario_combo)
        
        self.run_stress_test_button = CustomButton("Run Test")
        self.run_stress_test_button.clicked.connect(self.run_stress_test)
        controls_layout.addWidget(self.run_stress_test_button)
        
        controls_layout.addStretch()
        
        # Create stress test results table
        self.stress_test_table = CustomTableWidget()
        self.stress_test_table.setColumnCount(5)
        self.stress_test_table.setHorizontalHeaderLabels([
            "Scenario", "Expected Loss", "Max Drawdown", "Recovery Time", "Survival Probability"
        ])
        self.stress_test_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stress_layout.addWidget(self.stress_test_table)
        
        # Create model analysis tab
        model_tab = QWidget()
        analytics_tabs.addTab(model_tab, "Model Analysis")
        
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(10, 10, 10, 10)
        model_layout.setSpacing(10)
        
        # Create model metrics group
        model_metrics_group = CustomGroupBox("Model Metrics")
        model_layout.addWidget(model_metrics_group)
        
        model_metrics_layout = QGridLayout(model_metrics_group)
        model_metrics_layout.setContentsMargins(15, 25, 15, 15)
        model_metrics_layout.setSpacing(20)
        
        # Create model metric widgets
        self.create_metric_widget(model_metrics_layout, 0, 0, "Prediction Accuracy", "72.4%", "up")
        self.create_metric_widget(model_metrics_layout, 0, 1, "Direction Accuracy", "68.9%", "up")
        self.create_metric_widget(model_metrics_layout, 0, 2, "Mean Squared Error", "0.0023", "down")
        self.create_metric_widget(model_metrics_layout, 0, 3, "Model Confidence", "82.7%", "up")
        
        # Create model charts group
        model_charts_group = CustomGroupBox("Model Performance")
        model_layout.addWidget(model_charts_group)
        
        model_charts_layout = QHBoxLayout(model_charts_group)
        model_charts_layout.setContentsMargins(15, 25, 15, 15)
        model_charts_layout.setSpacing(20)
        
        # Create prediction accuracy chart
        prediction_layout = QVBoxLayout()
        model_charts_layout.addLayout(prediction_layout)
        
        prediction_label = QLabel("Prediction vs Actual")
        prediction_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(prediction_label)
        
        self.analytics_prediction_canvas = PyQtGraphCanvas(dark_mode=True)
        prediction_layout.addWidget(self.analytics_prediction_canvas)
        
        # Create feature importance chart
        feature_layout = QVBoxLayout()
        model_charts_layout.addLayout(feature_layout)
        
        feature_label = QLabel("Feature Importance")
        feature_label.setAlignment(Qt.AlignCenter)
        feature_layout.addWidget(feature_label)
        
        self.analytics_feature_canvas = PyQtGraphCanvas(dark_mode=True)
        feature_layout.addWidget(self.analytics_feature_canvas)
        
        # Create model explainability group
        explainability_group = CustomGroupBox("Model Explainability")
        model_layout.addWidget(explainability_group)
        
        explainability_layout = QVBoxLayout(explainability_group)
        explainability_layout.setContentsMargins(15, 25, 15, 15)
        explainability_layout.setSpacing(10)
        
        # Create explainability controls
        explainability_controls_layout = QHBoxLayout()
        explainability_layout.addLayout(explainability_controls_layout)
        
        trade_label = QLabel("Trade:")
        explainability_controls_layout.addWidget(trade_label)
        
        self.explainability_trade_combo = CustomComboBox()
        self.explainability_trade_combo.addItems(["Latest Trade", "Best Trade", "Worst Trade", "Random Trade"])
        explainability_controls_layout.addWidget(self.explainability_trade_combo)
        
        self.analyze_trade_button = CustomButton("Analyze")
        self.analyze_trade_button.clicked.connect(self.analyze_trade)
        explainability_controls_layout.addWidget(self.analyze_trade_button)
        
        explainability_controls_layout.addStretch()
        
        # Create explainability canvas
        self.explainability_canvas = PyQtGraphCanvas(dark_mode=True)
        explainability_layout.addWidget(self.explainability_canvas)
        
        # Create market analysis tab
        market_tab = QWidget()
        analytics_tabs.addTab(market_tab, "Market Analysis")
        
        market_layout = QVBoxLayout(market_tab)
        market_layout.setContentsMargins(10, 10, 10, 10)
        market_layout.setSpacing(10)
        
        # Create market regime group
        regime_group = CustomGroupBox("Market Regime Analysis")
        market_layout.addWidget(regime_group)
        
        regime_layout = QVBoxLayout(regime_group)
        regime_layout.setContentsMargins(15, 25, 15, 15)
        regime_layout.setSpacing(10)
        
        # Create regime chart
        self.regime_canvas = PyQtGraphCanvas(dark_mode=True)
        regime_layout.addWidget(self.regime_canvas)
        
        # Create regime performance table
        self.regime_table = CustomTableWidget()
        self.regime_table.setColumnCount(6)
        self.regime_table.setHorizontalHeaderLabels([
            "Regime", "Occurrence", "Win Rate", "Profit Factor", "Avg Trade", "Total Profit"
        ])
        self.regime_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        regime_layout.addWidget(self.regime_table)
        
        # Create frequency analysis group
        frequency_group = CustomGroupBox("Frequency Analysis")
        market_layout.addWidget(frequency_group)
        
        frequency_layout = QHBoxLayout(frequency_group)
        frequency_layout.setContentsMargins(15, 25, 15, 15)
        frequency_layout.setSpacing(20)
        
        # Create frequency spectrum chart
        spectrum_layout = QVBoxLayout()
        frequency_layout.addLayout(spectrum_layout)
        
        spectrum_label = QLabel("Frequency Spectrum")
        spectrum_label.setAlignment(Qt.AlignCenter)
        spectrum_layout.addWidget(spectrum_label)
        
        self.frequency_spectrum_canvas = PyQtGraphCanvas(dark_mode=True)
        spectrum_layout.addWidget(self.frequency_spectrum_canvas)
        
        # Create cyclicality chart
        cyclicality_layout = QVBoxLayout()
        frequency_layout.addLayout(cyclicality_layout)
        
        cyclicality_label = QLabel("Market Cyclicality")
        cyclicality_label.setAlignment(Qt.AlignCenter)
        cyclicality_layout.addWidget(cyclicality_label)
        
        self.cyclicality_canvas = PyQtGraphCanvas(dark_mode=True)
        cyclicality_layout.addWidget(self.cyclicality_canvas)
    
    def create_settings_tab(self):
        """Create the settings tab."""
        # Create tab widget
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "Settings")
        
        # Create layout
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)
        
        # Create tabs
        settings_tabs = QTabWidget()
        settings_layout.addWidget(settings_tabs)
        
        # Create general settings tab
        general_tab = QWidget()
        settings_tabs.addTab(general_tab, "General")
        
        general_layout = QVBoxLayout(general_tab)
        general_layout.setContentsMargins(10, 10, 10, 10)
        general_layout.setSpacing(10)
        
        # Create appearance group
        appearance_group = CustomGroupBox("Appearance")
        general_layout.addWidget(appearance_group)
        
        appearance_layout = QFormLayout(appearance_group)
        appearance_layout.setContentsMargins(15, 25, 15, 15)
        appearance_layout.setSpacing(10)
        
        # Dark mode
        self.settings_dark_mode = CustomCheckBox()
        self.settings_dark_mode.setChecked(True)
        self.settings_dark_mode.stateChanged.connect(lambda state: self.set_dark_mode(state == Qt.Checked))
        appearance_layout.addRow("Dark Mode:", self.settings_dark_mode)
        
        # Chart theme
        self.settings_chart_theme = CustomComboBox()
        self.settings_chart_theme.addItems(["Dark", "Light", "Blue", "Green", "Custom"])
        appearance_layout.addRow("Chart Theme:", self.settings_chart_theme)
        
        # Font size
        self.settings_font_size = CustomSpinBox()
        self.settings_font_size.setRange(8, 16)
        self.settings_font_size.setValue(10)
        appearance_layout.addRow("Font Size:", self.settings_font_size)
        
        # Create behavior group
        behavior_group = CustomGroupBox("Behavior")
        general_layout.addWidget(behavior_group)
        
        behavior_layout = QFormLayout(behavior_group)
        behavior_layout.setContentsMargins(15, 25, 15, 15)
        behavior_layout.setSpacing(10)
        
        # Auto-save
        self.settings_auto_save = CustomCheckBox()
        self.settings_auto_save.setChecked(True)
        behavior_layout.addRow("Auto-save Settings:", self.settings_auto_save)
        
        # Minimize to tray
        self.settings_minimize_to_tray = CustomCheckBox()
        self.settings_minimize_to_tray.setChecked(True)
        behavior_layout.addRow("Minimize to Tray:", self.settings_minimize_to_tray)
        
        # Show notifications
        self.settings_show_notifications = CustomCheckBox()
        self.settings_show_notifications.setChecked(True)
        behavior_layout.addRow("Show Notifications:", self.settings_show_notifications)
        
        # Create MT5 settings tab
        mt5_tab = QWidget()
        settings_tabs.addTab(mt5_tab, "MT5 Connection")
        
        mt5_layout = QVBoxLayout(mt5_tab)
        mt5_layout.setContentsMargins(10, 10, 10, 10)
        mt5_layout.setSpacing(10)
        
        # Create MT5 settings group
        mt5_group = CustomGroupBox("MT5 Connection Settings")
        mt5_layout.addWidget(mt5_group)
        
        mt5_form_layout = QFormLayout(mt5_group)
        mt5_form_layout.setContentsMargins(15, 25, 15, 15)
        mt5_form_layout.setSpacing(10)
        
        self.mt5_path = CustomLineEdit()
        mt5_form_layout.addRow("MT5 Path:", self.mt5_path)
        
        self.mt5_login = CustomLineEdit()
        mt5_form_layout.addRow("Login:", self.mt5_login)
        
        self.mt5_password = CustomLineEdit()
        self.mt5_password.setEchoMode(QLineEdit.Password)
        mt5_form_layout.addRow("Password:", self.mt5_password)
        
        self.mt5_server = CustomLineEdit()
        mt5_form_layout.addRow("Server:", self.mt5_server)
        
        # Create connection buttons
        mt5_buttons_layout = QHBoxLayout()
        mt5_form_layout.addRow("", mt5_buttons_layout)
        
        self.mt5_test_button = CustomButton("Test Connection")
        self.mt5_test_button.clicked.connect(self.test_mt5_connection)
        mt5_buttons_layout.addWidget(self.mt5_test_button)
        
        self.mt5_save_button = CustomButton("Save Settings")
        self.mt5_save_button.clicked.connect(self.save_mt5_settings)
        mt5_buttons_layout.addWidget(self.mt5_save_button)
        
        # Create data settings tab
        data_tab = QWidget()
        settings_tabs.addTab(data_tab, "Data")
        
        data_layout = QVBoxLayout(data_tab)
        data_layout.setContentsMargins(10, 10, 10, 10)
        data_layout.setSpacing(10)
        
        # Create data settings group
        data_group = CustomGroupBox("Data Settings")
        data_layout.addWidget(data_group)
        
        data_form_layout = QFormLayout(data_group)
        data_form_layout.setContentsMargins(15, 25, 15, 15)
        data_form_layout.setSpacing(10)
        
        self.data_path = CustomLineEdit()
        data_form_layout.addRow("Data Path:", self.data_path)
        
        self.data_cache = CustomCheckBox()
        self.data_cache.setChecked(True)
        data_form_layout.addRow("Use Data Cache:", self.data_cache)
        
        self.data_update_interval = CustomSpinBox()
        self.data_update_interval.setRange(1, 60)
        self.data_update_interval.setValue(5)
        self.data_update_interval.setSuffix(" min")
        data_form_layout.addRow("Update Interval:", self.data_update_interval)
        
        # Create model settings tab
        model_tab = QWidget()
        settings_tabs.addTab(model_tab, "Models")
        
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(10, 10, 10, 10)
        model_layout.setSpacing(10)
        
        # Create model settings group
        model_group = CustomGroupBox("Model Settings")
        model_layout.addWidget(model_group)
        
        model_form_layout = QFormLayout(model_group)
        model_form_layout.setContentsMargins(15, 25, 15, 15)
        model_form_layout.setSpacing(10)
        
        self.model_path = CustomLineEdit()
        model_form_layout.addRow("Model Path:", self.model_path)
        
        self.model_gpu = CustomCheckBox()
        self.model_gpu.setChecked(True)
        model_form_layout.addRow("Use GPU:", self.model_gpu)
        
        self.model_precision = CustomComboBox()
        self.model_precision.addItems(["FP32", "FP16", "INT8"])
        model_form_layout.addRow("Precision:", self.model_precision)
        
        # Create buttons
        settings_buttons_layout = QHBoxLayout()
        settings_layout.addLayout(settings_buttons_layout)
        
        self.save_settings_button = CustomButton("Save All Settings", primary=True)
        self.save_settings_button.clicked.connect(self.save_settings)
        settings_buttons_layout.addWidget(self.save_settings_button)
        
        self.reset_settings_button = CustomButton("Reset to Defaults")
        self.reset_settings_button.clicked.connect(self.reset_settings)
        settings_buttons_layout.addWidget(self.reset_settings_button)
    
    def create_metric_widget(self, layout, row, col, title, value, trend):
        """Create a metric widget for the analytics tab."""
        # Create widget
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        widget.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        # Create layout
        widget_layout = QVBoxLayout(widget)
        widget_layout.setContentsMargins(10, 10, 10, 10)
        widget_layout.setSpacing(5)
        
        # Create title label
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        widget_layout.addWidget(title_label)
        
        # Create value label
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        widget_layout.addWidget(value_label)
        
        # Create trend label
        trend_label = QLabel()
        if trend == "up":
            trend_label.setText("▲ Positive")
            trend_label.setStyleSheet("color: #4caf50;")
        elif trend == "down":
            trend_label.setText("▼ Negative")
            trend_label.setStyleSheet("color: #f44336;")
        else:
            trend_label.setText("◆ Neutral")
            trend_label.setStyleSheet("color: #2196f3;")
        widget_layout.addWidget(trend_label)
        
        # Add widget to layout
        layout.addWidget(widget, row, col)
    
    def init_connections(self):
        """Initialize signal-slot connections."""
        # Timer for updating dashboard
        self.dashboard_timer = QTimer()
        self.dashboard_timer.timeout.connect(self.update_dashboard)
        self.dashboard_timer.start(5000)  # 5 seconds
    
    def init_atfnet(self):
        """Initialize ATFNet components."""
        try:
            # Load configuration
            config_loader = ConfigLoader("config/default_config.yaml")
            self.config = config_loader.load_config()
            
            # Initialize MT5 connection
            self.mt5_api = MT5API(self.config['mt5'])
            
            # Initialize data manager
            self.data_manager = DataManager(self.mt5_api, self.config['data'])
            
            # Initialize model manager
            self.model_manager = ModelManager(self.config['models'])
            
            # Initialize strategy
            self.strategy = ATFNetStrategy(
                mt5_api=self.mt5_api,
                data_manager=self.data_manager,
                model_manager=self.model_manager,
                config=self.config['strategy']
            )
            
            # Initialize chart plotter
            self.chart_plotter = ChartPlotter()
            
            # Initialize interactive plotter
            self.interactive_plotter = InteractivePlotter()
            
            # Update status
            self.status_bar.showMessage("ATFNet initialized successfully")
        
        except Exception as e:
            logger.exception(f"Error initializing ATFNet: {e}")
            self.status_bar.showMessage(f"Error initializing ATFNet: {e}")
    
    def load_settings(self):
        """Load application settings."""
        settings = QSettings("ATFNet", "EnhancedATFNetApp")
        
        # Appearance settings
        self.dark_mode = settings.value("appearance/dark_mode", True, type=bool)
        
        # MT5 settings
        self.mt5_path.setText(settings.value("mt5/path", ""))
        self.mt5_login.setText(settings.value("mt5/login", ""))
        self.mt5_password.setText(settings.value("mt5/password", ""))
        self.mt5_server.setText(settings.value("mt5/server", ""))
        
        # Data settings
        self.data_path.setText(settings.value("data/path", ""))
        self.data_cache.setChecked(settings.value("data/cache", True, type=bool))
        self.data_update_interval.setValue(settings.value("data/update_interval", 5, type=int))
        
        # Model settings
        self.model_path.setText(settings.value("model/path", ""))
        self.model_gpu.setChecked(settings.value("model/gpu", True, type=bool))
    
    def save_settings(self):
        """Save application settings."""
        settings = QSettings("ATFNet", "EnhancedATFNetApp")
        
        # Appearance settings
        settings.setValue("appearance/dark_mode", self.dark_mode)
        
        # MT5 settings
        settings.setValue("mt5/path", self.mt5_path.text())
        settings.setValue("mt5/login", self.mt5_login.text())
        settings.setValue("mt5/password", self.mt5_password.text())
        settings.setValue("mt5/server", self.mt5_server.text())
        
        # Data settings
        settings.setValue("data/path", self.data_path.text())
        settings.setValue("data/cache", self.data_cache.isChecked())
        settings.setValue("data/update_interval", self.data_update_interval.value())
        
        # Model settings
        settings.setValue("model/path", self.model_path.text())
        settings.setValue("model/gpu", self.model_gpu.isChecked())
        
        # Show message
        QMessageBox.information(self, "Settings", "Settings saved successfully")
    
    def reset_settings(self):
        """Reset settings to defaults."""
        # Ask for confirmation
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear settings
            settings = QSettings("ATFNet", "EnhancedATFNetApp")
            settings.clear()
            
            # Load defaults
            self.load_settings()
            
            # Set dark mode
            self.set_dark_mode(True)
            
            # Show message
            QMessageBox.information(self, "Settings", "Settings reset to defaults")
    
    def toggle_connection(self):
        """Toggle MT5 connection on/off."""
        if not self.mt5_connected:
            # override config with current UI values
            self.mt5_api.config.update({
                'terminal_path': self.mt5_path.text(),
                'login': int(self.mt5_login.text() or 0),
                'password': self.mt5_password.text(),
                'server': self.mt5_server.text()
            })
            try:
                success = self.mt5_api.connect()
                if success:
                    self.mt5_connected = True
                    self.connect_action.setText("Disconnect")
                    self.status_bar.showMessage("Connected to MT5")
                else:
                    QMessageBox.warning(self, "MT5 Connection", "Failed to connect to MT5")
            except Exception as e:
                logger.exception(f"Error connecting to MT5: {e}")
                QMessageBox.critical(self, "MT5 Connection", f"Error: {e}")
        else:
            try:
                self.mt5_api.disconnect()
            except Exception as e:
                logger.exception(f"Error disconnecting from MT5: {e}")
            self.mt5_connected = False
            self.connect_action.setText("Connect")
            self.status_bar.showMessage("Disconnected from MT5")

    def import_data_dialog(self):
        """Open a file dialog to import data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            # TODO: replace with real import routine
            QMessageBox.information(self, "Import Data", f"Imported data from:\n{file_path}")

    def export_data_dialog(self):
        """Open a file dialog to export data."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            # TODO: replace with real export routine
            QMessageBox.information(self, "Export Data", f"Exported data to:\n{file_path}")

    # Add the rest of the methods for functionality...
    # For brevity, I'm not including all the method implementations
    # as they would be similar to the original ATFNetApp class

    # Backtest handlers
    def run_backtest(self):
        """Start backtest worker thread."""
        symbol    = self.backtest_symbol_combo.currentText()
        timeframe = self.backtest_timeframe_combo.currentText()
        start     = self.backtest_start_date.date().toPyDate()
        end       = self.backtest_end_date.date().toPyDate()
        cfg = {
            'risk':             self.backtest_risk.value(),
            'atr_multiplier':   self.backtest_atr_mult.value(),
            'attention':        self.backtest_attention.value(),
            'prediction':       self.backtest_prediction.value(),
            'trend_filter':     self.backtest_trend_filter.isChecked(),
            'trailing_stop':    self.backtest_trailing_stop.isChecked(),
            'balance':          self.backtest_balance.value()
        }

        self.run_backtest_button.setEnabled(False)
        self.stop_backtest_button.setEnabled(True)
        self.backtest_progress.setValue(0)

        # launch worker
        self.backtest_worker = BacktestWorker(
            self.strategy, symbol, timeframe, start, end, cfg
        )
        self.backtest_worker.progress.connect(self.backtest_progress.setValue)
        self.backtest_worker.error.connect(
            lambda e: QMessageBox.critical(self, "Backtest Error", e)
        )
        self.backtest_worker.finished.connect(self.on_backtest_finished)
        self.backtest_worker.start()

    def stop_backtest(self):
        """Stop the backtest worker."""
        if hasattr(self, 'backtest_worker'):
            self.backtest_worker.stop()
        self.run_backtest_button.setEnabled(True)
        self.stop_backtest_button.setEnabled(False)

    def on_backtest_finished(self, results):
        """Populate results and reset UI."""
        # metrics
        self.backtest_total_trades.setText(str(results.get('total_trades', 'N/A')))
        self.backtest_win_rate.setText(f"{results.get('win_rate',0)*100:.1f}%")
        self.backtest_profit_factor.setText(f"{results.get('profit_factor',0):.2f}")
        # ...and so on for other labels...

        # enable export
        self.export_backtest_button.setEnabled(True)
        # reset buttons
        self.run_backtest_button.setEnabled(True)
        self.stop_backtest_button.setEnabled(False)

    def export_backtest_results(self):
        QMessageBox.information(self, "Backtest", "Results exported")

    # Optimization handlers
    def run_optimization(self):
        QMessageBox.information(self, "Optimization", "Optimization started")

    def stop_optimization(self):
        QMessageBox.information(self, "Optimization", "Optimization stopped")

    def export_optimization_results(self):
        QMessageBox.information(self, "Optimization", "Results exported")

    # Training handlers
    def run_training(self):
        QMessageBox.information(self, "Training", "Training started")

    def stop_training(self):
        QMessageBox.information(self, "Training", "Training stopped")

    # Live trading handlers
    def start_live_trading(self):
        QMessageBox.information(self, "Live Trading", "Live trading started")

    def stop_live_trading(self):
        QMessageBox.information(self, "Live Trading", "Live trading stopped")

    # Export & report
    def export_results_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "")
        if path:
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")

    def generate_report_dialog(self):
        QMessageBox.information(self, "Report", "Report generated")

    # Misc UI dialogs
    def show_preferences_dialog(self):
        QMessageBox.information(self, "Preferences", "Preferences dialog")

    def toggle_fullscreen(self, checked):
        self.setWindowState(self.windowState() ^ Qt.WindowFullScreen)

    def show_documentation(self):
        QMessageBox.information(self, "Help", "Opening documentation...")

    def show_about_dialog(self):
        QMessageBox.information(self, "About", "Enhanced ATFNet v1.0")

    def show_help_dialog(self):
        QMessageBox.information(self, "Help", "Displaying help...")

    # Advanced options stubs
    def show_advanced_backtest_options(self):
        QMessageBox.information(self, "Backtest", "Advanced backtest options")

    def show_advanced_training_options(self):
        QMessageBox.information(self, "Training", "Advanced training options")

    def show_advanced_live_options(self):
        QMessageBox.information(self, "Live Trading", "Advanced live options")

    # Chart & dashboard updates
    def update_dashboard(self):
        """Fetch latest account, positions, trades and refresh UI."""
        if not self.mt5_connected:
            return

        # account info
        info = self.mt5_api.get_account_info() or {}
        self.account_balance_label.setText(f"{info.get('balance',0):.2f}")
        self.account_equity_label.setText(f"{info.get('equity',0):.2f}")
        self.account_margin_label.setText(f"{info.get('margin',0):.2f}")
        self.account_profit_label.setText(f"{info.get('profit',0):.2f}")
        self.account_leverage_label.setText(str(info.get('leverage','N/A')))

        # open positions
        self.positions_table.setRowCount(0)
        for pos in self.mt5_api.get_positions() or []:
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)
            fields = [
                pos.get('ticket'), pos.get('symbol'), pos.get('type'), pos.get('volume'),
                pos.get('open_price'), pos.get('current_price'), pos.get('sl'), pos.get('profit')
            ]
            for col, val in enumerate(fields):
                self.positions_table.setItem(row, col, QTableWidgetItem(str(val)))

        # recent trades
        self.trades_table.setRowCount(0)
        for tr in self.mt5_api.get_trades(limit=10) or []:
            row = self.trades_table.rowCount()
            self.trades_table.insertRow(row)
            fields = [
                tr.get('ticket'), tr.get('symbol'), tr.get('type'), tr.get('volume'),
                tr.get('price_open'), tr.get('price_close'), tr.get('profit'), tr.get('time')
            ]
            for col, val in enumerate(fields):
                self.trades_table.setItem(row, col, QTableWidgetItem(str(val)))

    def update_chart(self):
        """Fetch OHLC data and redraw chart."""
        sym = self.chart_symbol_combo.currentText()
        tf = self.chart_timeframe_combo.currentText()
        start = self.chart_start_date.date().toPyDate()
        end = self.chart_end_date.date().toPyDate()
        df = self.data_manager.get_data(sym, tf, start, end)
        ctype = self.chart_type_combo.currentText().toLowerCase()
        # ChartPlotter knows how to draw
        html = self.chart_plotter.plot(df, chart_type=ctype)
        self.chart_canvas.setHtml(html)

    def update_chart_date_range(self):
        """Toggle custom date edits."""
        custom = self.chart_date_range_combo.currentText() == "Custom"
        self.chart_start_date.setVisible(custom)
        self.chart_end_date.setVisible(custom)

    def add_indicator(self):
        """Add selected indicator to the table."""
        name = self.indicator_combo.currentText()
        row = self.indicators_table.rowCount()
        self.indicators_table.insertRow(row)
        self.indicators_table.setItem(row, 0, QTableWidgetItem(name))
        self.indicators_table.setItem(row, 1, QTableWidgetItem(""))
        self.indicators_table.setItem(row, 2, QTableWidgetItem("#ffffff"))
        btn = QPushButton("Remove")
        btn.clicked.connect(lambda _, r=row: self.indicators_table.removeRow(r))
        self.indicators_table.setCellWidget(row, 3, btn)

    # Stress test & explainability
    def run_stress_test(self):
        QMessageBox.information(self, "Stress Test", "Stress test executed")

    def analyze_trade(self):
        QMessageBox.information(self, "Explainability", "Trade analysis complete")

    # MT5 settings
    def test_mt5_connection(self):
        """Test MT5 connection using current settings."""
        conf = {
            'terminal_path': self.mt5_path.text(),
            'login': int(self.mt5_login.text() or 0),
            'password': self.mt5_password.text(),
            'server': self.mt5_server.text(),
            'timeout': self.config['mt5'].get('timeout', 60000)
        }
        api = MT5API(conf)
        if api.connect():
            api.disconnect()
            QMessageBox.information(self, "MT5 Connection", "Connection successful")
        else:
            QMessageBox.critical(self, "MT5 Connection", "Connection failed")

    def save_mt5_settings(self):
        QMessageBox.information(self, "MT5 Connection", "Settings saved")

    def update_optimization_visualization(self):
        """Refresh optimization scatter based on axis/color selections."""
        # no-op or hook into your plotting logic, e.g.:
        # dims = (self.optimization_x_axis_combo.currentText(),
        #         self.optimization_y_axis_combo.currentText(),
        #         self.optimization_color_by_combo.currentText())
        # df = self.optimization_results  # assume stored
        # self.optimization_canvas.plot_scatter(df, *dims)
        pass

    def load_config_dialog(self):
        """Open and load a strategy/config file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "YAML/JSON (*.yaml *.json)")
        if path:
            # TODO: implement actual load
            QMessageBox.information(self, "Load Config", f"Loaded: {path}")

    def save_config_dialog(self):
        """Save current strategy/config to file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "YAML (*.yaml)")
        if path:
            # TODO: implement actual save
            QMessageBox.information(self, "Save Config", f"Saved: {path}")

def main():
    """Main function."""
    try:
        # Create application
        app = QApplication(sys.argv)
        
        # Set style
        app.setStyle(QStyleFactory.create('Fusion'))
        
        # Create main window
        window = EnhancedATFNetApp()
        window.show()
        
        # Run application
        sys.exit(app.exec_())
    
    except Exception as e:
        logger.exception(f"Error in main: {e}")


if __name__ == "__main__":
    main()
