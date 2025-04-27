# ATFNet Python Implementation Architecture

## Overview

The ATFNet Python implementation is a comprehensive trading system that combines adaptive time-frequency analysis with machine learning for enhanced trading performance. The system is built with a modular architecture that separates concerns and allows for easy extension and customization.

This document provides a detailed explanation of the architecture and behavior of the ATFNet Python implementation, including the PyQt5 GUI application.

## System Architecture

The ATFNet system is organized into several key components:

\`\`\`
ATFNet Python Implementation
├── Core Components
│   ├── ATFNetAttention
│   ├── PriceForecaster
│   ├── FrequencyFilter
│   ├── MarketRegimeDetector
│   ├── RiskManager
│   ├── SignalAggregator
│   └── PerformanceAnalytics
├── Infrastructure
│   ├── MT5 Connector
│   ├── Data Manager
│   └── Model Manager
├── Strategy
│   └── ATFNetStrategy
└── GUI Application
    ├── Dashboard
    ├── Backtest Module
    ├── Training Module
    ├── Live Trading Module
    └── Settings Module
\`\`\`

### Core Components

The core components implement the fundamental trading logic of the ATFNet system:

1. **ATFNetAttention**: Implements the enhanced attention mechanism with dual-domain analysis. It combines time-domain and frequency-domain features to generate attention scores for potential trading opportunities.

2. **PriceForecaster**: Predicts future price movements using multiple methods, including time-domain analysis, frequency-domain analysis, and recurrent neural networks.

3. **FrequencyFilter**: Filters trades based on frequency domain analysis, identifying favorable market conditions for trading.

4. **MarketRegimeDetector**: Identifies the current market regime (trending, ranging, volatile, choppy) and optimizes strategy parameters accordingly.

5. **RiskManager**: Handles position sizing, risk calculation, and drawdown protection to ensure proper risk management.

6. **SignalAggregator**: Combines signals from different sources to make trading decisions, applying weights and thresholds to generate entry and exit signals.

7. **PerformanceAnalytics**: Tracks and analyzes trading performance, calculating metrics such as win rate, profit factor, and drawdown.

### Infrastructure

The infrastructure components provide the foundation for the ATFNet system:

1. **MT5 Connector**: Handles all interactions with the MetaTrader 5 platform, including data retrieval, order placement, and account information.

2. **Data Manager**: Manages data fetching, preprocessing, and feature engineering. It handles data caching, normalization, and preparation for machine learning models.

3. **Model Manager**: Manages machine learning models, including training, evaluation, and prediction. It supports multiple model architectures and provides utilities for model persistence.

### Strategy

The ATFNetStrategy class integrates all components into a cohesive trading strategy:

1. **ATFNetStrategy**: Coordinates the core components and implements the main trading logic. It handles backtesting, live trading, and model training.

### GUI Application

The PyQt5 GUI application provides a user-friendly interface for interacting with the ATFNet system:

1. **Dashboard**: Displays real-time account information, performance metrics, and market data.

2. **Backtest Module**: Allows users to configure and run backtests, visualize results, and export data for further analysis.

3. **Training Module**: Provides tools for training and evaluating machine learning models with customizable parameters.

4. **Live Trading Module**: Enables users to monitor and control live trading operations with real-time updates.

5. **Settings Module**: Allows users to configure system parameters, including MT5 connection settings, data management options, and model settings.

## Data Flow

The data flow in the ATFNet system follows a clear path:

1. **Data Acquisition**: The MT5 Connector retrieves historical and real-time data from the MetaTrader 5 platform.

2. **Data Preprocessing**: The Data Manager preprocesses the raw data, adding technical indicators and calculating derived features.

3. **Feature Engineering**: The Data Manager performs feature engineering, including frequency domain analysis and ATFNet-specific features.

4. **Model Training**: The Model Manager trains machine learning models using the prepared features and historical data.

5. **Signal Generation**: The core components analyze the data and generate trading signals based on their respective algorithms.

6. **Signal Aggregation**: The SignalAggregator combines the signals from different sources to make trading decisions.

7. **Risk Management**: The RiskManager calculates position sizes and manages risk based on account information and market conditions.

8. **Order Execution**: The MT5 Connector executes trades based on the trading decisions.

9. **Performance Tracking**: The PerformanceAnalytics tracks and analyzes trading performance.

10. **Visualization**: The GUI Application visualizes data, signals, and performance metrics for user interaction.

## GUI Structure and Functionality

The PyQt5 GUI application is organized into tabs, each providing specific functionality:

### Dashboard Tab

The Dashboard tab provides an overview of the current state of the trading system:

- **Account Information**: Displays balance, equity, margin, and profit.
- **Performance Metrics**: Shows win rate, profit factor, drawdown, and total trades.
- **Market Information**: Displays symbol, timeframe, market regime, and current signal.
- **Equity Curve**: Visualizes the equity curve over time.
- **Price Chart**: Displays the price chart with indicators and signals.
- **Open Positions**: Lists current open positions with details.
- **Recent Trades**: Shows recent trade history.

### Backtest Tab

The Backtest tab allows users to configure and run backtests:

- **Backtest Parameters**: Includes symbol, timeframe, date range, and initial balance.
- **Strategy Parameters**: Allows customization of risk, ATR multiplier, attention threshold, prediction threshold, and trend filter.
- **Backtest Controls**: Provides buttons to run, stop, and export backtest results.
- **Backtest Results**: Displays total trades, win rate, profit factor, net profit, and max drawdown.
- **Equity Curve**: Visualizes the equity curve from the backtest.
- **Trades Distribution**: Shows the distribution of trade profits.
- **Trades Table**: Lists all trades from the backtest with details.

### Training Tab

The Training tab provides tools for training machine learning models:

- **Training Parameters**: Includes symbol, timeframe, and date range.
- **Model Parameters**: Allows selection of model type, epochs, batch size, sequence length, and regime classifier option.
- **Training Controls**: Provides buttons to run and stop training.
- **Training Results**: Displays model names, training loss, and validation loss.
- **Training Charts**: Visualizes training progress for price prediction and regime classifier models.
- **Training Log**: Shows detailed training log messages.

### Live Trading Tab

The Live Trading tab enables users to monitor and control live trading:

- **Trading Parameters**: Includes symbol, timeframe, and strategy parameters.
- **Trading Controls**: Provides buttons to start and stop trading.
- **Trading Status**: Displays current status, account balance, equity, and profit.
- **Signal Information**: Shows attention score, prediction, market regime, and current signal.
- **Price Chart**: Displays real-time price chart with signals.
- **Equity Curve**: Visualizes real-time equity curve.
- **Open Positions**: Lists current open positions with details.
- **Trading Log**: Shows detailed trading log messages.

### Settings Tab

The Settings tab allows users to configure system parameters:

- **MT5 Connection Settings**: Includes MT5 path, login, password, and server.
- **Data Settings**: Allows configuration of data path, cache usage, and update interval.
- **Model Settings**: Provides options for model path and GPU usage.
- **Settings Controls**: Includes buttons to save settings and reset to defaults.

## Trading Logic

The trading logic of the ATFNet system is based on the integration of multiple analysis methods:

### Signal Generation

1. **Time Domain Analysis**: Analyzes price patterns, trends, and technical indicators in the time domain.

2. **Frequency Domain Analysis**: Applies Fast Fourier Transform (FFT) to identify dominant frequencies and cyclical patterns in the price data.

3. **Machine Learning Prediction**: Uses trained models to predict future price movements based on historical patterns.

4. **Attention Mechanism**: Applies the ATFNetAttention mechanism to focus on the most relevant features for trading decisions.

### Entry Criteria

A trade is entered when the following conditions are met:

1. The combined signal strength exceeds the entry threshold.
2. The frequency filter allows trading in the current market conditions.
3. The trend filter (if enabled) confirms the trade direction.
4. The number of open positions is below the maximum allowed.

### Exit Criteria

A trade is exited when one of the following conditions is met:

1. The stop loss is hit.
2. The take profit is hit.
3. The exit signal is generated by the SignalAggregator.

### Risk Management

The RiskManager handles position sizing and risk control:

1. **Position Sizing**: Calculates position size based on account balance, risk per trade, and stop loss distance.
2. **Drawdown Protection**: Reduces risk or stops trading when drawdown exceeds specified limits.
3. **Dynamic Risk Adjustment**: Adjusts risk based on trading performance and market conditions.

## Integration with MT5

The ATFNet system integrates with MetaTrader 5 through the MT5 Connector:

### Data Retrieval

1. **Historical Data**: Retrieves historical price data for backtesting and model training.
2. **Real-time Data**: Retrieves real-time price data for live trading.
3. **Account Information**: Retrieves account balance, equity, margin, and other account details.
4. **Open Positions**: Retrieves information about current open positions.
5. **Trade History**: Retrieves historical trade data for performance analysis.

### Order Execution

1. **Order Placement**: Places market and pending orders with specified parameters.
2. **Order Modification**: Modifies existing orders, including stop loss and take profit levels.
3. **Order Cancellation**: Cancels pending orders.
4. **Position Closure**: Closes open positions.

## Machine Learning Components

The ATFNet system incorporates machine learning for enhanced trading performance:

### Model Architectures

The system supports multiple model architectures:

1. **LSTM**: Long Short-Term Memory networks for sequence prediction.
2. **GRU**: Gated Recurrent Units for efficient sequence modeling.
3. **CNN**: Convolutional Neural Networks for pattern recognition.
4. **Transformer**: Transformer-based models for advanced sequence modeling.

### Model Training

The Model Manager handles model training:

1. **Data Preparation**: Prepares features and targets for training.
2. **Model Configuration**: Configures model architecture and hyperparameters.
3. **Training Process**: Trains models with early stopping and learning rate scheduling.
4. **Model Evaluation**: Evaluates models on validation data.
5. **Model Persistence**: Saves trained models for later use.

### Model Applications

The trained models are used for:

1. **Price Prediction**: Predicting future price movements.
2. **Regime Classification**: Identifying market regimes for adaptive strategy parameters.
3. **Signal Enhancement**: Enhancing trading signals with machine learning insights.

## Customization Options

The ATFNet system provides numerous customization options:

### Strategy Parameters

1. **Risk Per Trade**: Percentage of account balance risked per trade.
2. **ATR Multiplier**: Multiplier for Average True Range used in stop loss calculation.
3. **Attention Threshold**: Minimum attention score required for trade entry.
4. **Prediction Threshold**: Minimum predicted return required for trade entry.
5. **Trend Filter**: Option to use trend filter for trade direction confirmation.

### Model Parameters

1. **Model Type**: Selection of model architecture (LSTM, GRU, CNN, Transformer).
2. **Epochs**: Number of training epochs.
3. **Batch Size**: Batch size for training.
4. **Sequence Length**: Length of input sequences for time series models.
5. **Regime Classifier**: Option to train and use market regime classifier.

### Data Parameters

1. **Data Cache**: Option to use cached data for improved performance.
2. **Update Interval**: Interval for data updates in live trading.
3. **Feature Selection**: Selection of features used for analysis and model training.

## Deployment Considerations

When deploying the ATFNet system, consider the following:

### Hardware Requirements

1. **CPU**: Multi-core processor for parallel data processing.
2. **RAM**: Sufficient memory for data handling and model training (8GB minimum, 16GB+ recommended).
3. **GPU**: Optional but recommended for faster model training.
4. **Storage**: SSD storage for improved data access speed.

### Software Requirements

1. **Python**: Python 3.8 or higher.
2. **Dependencies**: All packages listed in requirements.txt.
3. **MetaTrader 5**: Installed and configured MetaTrader 5 terminal.
4. **Operating System**: Windows for direct MT5 integration, or any OS with Wine for Windows emulation.

### Network Requirements

1. **Internet Connection**: Stable internet connection for MT5 communication.
2. **Latency**: Low latency connection for real-time trading.

### Operational Considerations

1. **Monitoring**: Regular monitoring of system performance and trading results.
2. **Maintenance**: Periodic retraining of models with new data.
3. **Backup**: Regular backup of configuration, models, and data.
4. **Testing**: Thorough testing of any changes before deployment.

## Conclusion

The ATFNet Python implementation provides a comprehensive trading system with a user-friendly PyQt5 GUI. Its modular architecture allows for easy extension and customization, while the integration of multiple analysis methods and machine learning enhances trading performance.

By understanding the architecture and behavior of the system, users can effectively leverage its capabilities for algorithmic trading in various market conditions.
