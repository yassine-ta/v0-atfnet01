# OPT_ATFNet: Advanced Trading System Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Trading Strategy](#trading-strategy)
4. [Signal Generation](#signal-generation)
5. [Risk Management](#risk-management)
6. [Market Regime Detection](#market-regime-detection)
7. [Performance Analytics](#performance-analytics)
8. [Implementation Details](#implementation-details)
9. [Usage Guide](#usage-guide)
10. [Parameter Optimization](#parameter-optimization)

## Introduction

OPT_ATFNet is an advanced algorithmic trading system that combines time-domain and frequency-domain analysis with machine learning techniques to identify high-probability trading opportunities across various market conditions. The system represents a significant evolution of the original ATFNet framework, with enhanced architecture, improved risk management, and adaptive capabilities.

Key features of OPT_ATFNet include:

- **Dual-domain analysis**: Combines traditional time-domain technical analysis with frequency-domain spectral analysis
- **Attention mechanism**: Dynamically weights different market factors based on their predictive power
- **Adaptive forecasting**: Combines multiple forecasting methods with machine learning for price prediction
- **Market regime detection**: Identifies current market conditions and optimizes strategy parameters accordingly
- **Dynamic risk management**: Adjusts position sizing and risk parameters based on account performance and market conditions
- **Comprehensive performance analytics**: Tracks and analyzes trading performance for continuous improvement

## System Architecture

OPT_ATFNet employs a modular, component-based architecture that promotes maintainability, extensibility, and separation of concerns. The system is organized into the following key components:

![OPT_ATFNet Architecture](https://via.placeholder.com/800x500?text=OPT_ATFNet+Architecture)

### Core Components

1. **Controller (CATFNetController)**: The central component that coordinates all other modules and implements the main trading logic. It processes market ticks, manages the trading cycle, and orchestrates interactions between components.

2. **Attention Mechanism (CATFNetAttention)**: Dynamically weights different market factors based on their historical predictive power. This component learns from past trades to focus on the most relevant signals for current market conditions.

3. **Price Forecaster (CPriceForecaster)**: Combines multiple forecasting methods (time-domain, frequency-domain, and RNN-based) to predict future price movements. It uses an adaptive combiner to weight different forecasting methods based on their recent accuracy.

4. **Signal Aggregator (CSignalAggregator)**: Combines signals from different sources (attention mechanism, price forecaster, frequency filter) to make trading decisions. It applies thresholds and filters to generate high-quality entry and exit signals.

5. **Risk Manager (CRiskManager)**: Handles position sizing, risk calculation, and drawdown protection. It dynamically adjusts risk parameters based on account performance and market conditions.

6. **Market Regime Detector (CMarketRegimeDetector)**: Identifies the current market condition (trending, ranging, volatile, choppy) and optimizes strategy parameters accordingly.

7. **Performance Analytics (CPerformanceAnalytics)**: Tracks and analyzes trading performance, component effectiveness, and strategy metrics for continuous improvement.

### Supporting Components

1. **Frequency Features (CFrequencyFeatures)**: Extracts and analyzes frequency-domain features from price data using Fast Fourier Transform (FFT).

2. **Frequency Filter (CFrequencyFilter)**: Uses frequency-domain analysis to filter out low-probability trades in certain market conditions.

3. **Position Manager (CPositionManager)**: Tracks and manages open positions, including trailing stops and partial close operations.

4. **Pattern Detector (CPatternDetector)**: Identifies candlestick patterns and chart formations that may indicate potential trading opportunities.

## Trading Strategy

OPT_ATFNet employs a multi-factor, adaptive trading strategy that combines technical analysis, statistical methods, and machine learning techniques. The strategy operates across different market regimes and adapts its parameters accordingly.

### Strategy Overview

1. **Market Analysis Phase**:
   - Analyze price data in both time and frequency domains
   - Detect current market regime (trending, ranging, volatile, choppy)
   - Calculate key technical indicators and statistical measures
   - Extract frequency-domain features using FFT

2. **Signal Generation Phase**:
   - Generate price forecasts using multiple methods
   - Calculate attention scores for different market factors
   - Apply frequency filters to eliminate low-probability setups
   - Aggregate signals with dynamic weighting

3. **Trade Execution Phase**:
   - Calculate optimal position size based on risk parameters
   - Determine precise entry points and stop loss levels
   - Execute trades with appropriate risk management
   - Implement multi-stage take profit strategy

4. **Position Management Phase**:
   - Monitor open positions for trailing stop adjustments
   - Implement partial close at predefined profit targets
   - Adjust stop loss levels based on market conditions
   - Close positions when exit signals are triggered

5. **Learning and Adaptation Phase**:
   - Update attention weights based on trade outcomes
   - Adjust forecaster weights based on prediction accuracy
   - Optimize parameters for detected market regime
   - Update risk parameters based on account performance

### Key Strategy Elements

#### Dual-Domain Analysis

OPT_ATFNet analyzes price data in both time and frequency domains:

- **Time Domain**: Traditional technical analysis focusing on price levels, trends, patterns, and momentum
- **Frequency Domain**: Spectral analysis using FFT to identify dominant cycles, periodicities, and hidden patterns

The system dynamically adjusts the weighting between these domains based on their predictive performance in current market conditions.

#### Attention Mechanism

The attention mechanism is a key innovation that allows the system to focus on the most relevant market factors. It works by:

1. Assigning weights to different market factors (price, ATR, Z-Score, trend strength, etc.)
2. Calculating a weighted score based on current market conditions
3. Learning from trade outcomes to adjust weights over time
4. Normalizing weights to maintain balanced decision-making

This approach allows the system to adapt to changing market conditions and focus on the factors that are most predictive in the current environment.

#### Adaptive Forecasting

OPT_ATFNet uses multiple forecasting methods and combines them adaptively:

1. **Time-Domain Forecasting**: Uses regression and statistical methods to predict future prices
2. **Frequency-Domain Forecasting**: Uses FFT to identify cycles and project them forward
3. **RNN-Based Forecasting**: Uses a simple recurrent neural network for sequence prediction

These forecasts are combined using an adaptive weighting mechanism that adjusts based on the recent accuracy of each method.

## Signal Generation

The signal generation process in OPT_ATFNet involves several steps:

### 1. Feature Extraction

The system extracts a comprehensive set of features from market data:

- **Price-based features**: Current price, recent highs/lows, range position
- **Volatility features**: ATR, Z-Score, price velocity
- **Trend features**: EMA trend strength, directional movement
- **Pattern features**: Candlestick patterns, chart formations
- **Frequency features**: Spectral entropy, dominant frequencies, cyclicality

### 2. Attention Scoring

The attention mechanism calculates a weighted score based on current feature values:

\`\`\`
attentionScore = Σ(featureValue_i * featureWeight_i)
\`\`\`

Where:
- `featureValue_i` is the normalized value of feature i
- `featureWeight_i` is the learned weight for feature i

### 3. Price Forecasting

The system generates price forecasts using multiple methods:

- **Time-domain forecast**: Uses linear regression and statistical methods
- **Frequency-domain forecast**: Projects dominant cycles forward
- **RNN forecast**: Uses a simple recurrent neural network

These forecasts are combined using adaptive weights:

\`\`\`
combinedForecast = (timeWeight * timeForecast) + (freqWeight * freqForecast) + (rnnWeight * rnnForecast)
\`\`\`

### 4. Signal Aggregation

The signal aggregator combines attention scores and forecasts to generate trading signals:

\`\`\`
entrySignal = (attentionWeight * attentionScore) + (forecastWeight * forecastStrength) + (filterWeight * filterScore)
\`\`\`

A trade is triggered when:
1. The entry signal exceeds the threshold
2. The trend filter conditions are met (if enabled)
3. The frequency filter allows trading in current conditions
4. Risk management conditions are satisfied

## Risk Management

OPT_ATFNet implements a sophisticated risk management system that operates at multiple levels:

### Position Sizing

Position size is calculated based on:

1. **Risk per trade**: The percentage of account balance risked on each trade
2. **Stop loss distance**: The distance from entry to stop loss in price units
3. **Account conditions**: Current drawdown, recent performance, etc.

The formula used is:

\`\`\`
positionSize = (accountBalance * riskPercentage) / (entryPrice - stopLossPrice)
\`\`\`

This is then adjusted based on:
- Current drawdown (reduced size when in drawdown)
- Recent performance (increased size after consistent profits)
- Market volatility (reduced size in highly volatile conditions)

### Dynamic Risk Adjustment

The system dynamically adjusts risk parameters based on:

1. **Account growth**: Gradually increases risk as the account grows
2. **Drawdown protection**: Reduces risk during drawdown periods
3. **Consecutive losses**: Reduces risk after multiple consecutive losses
4. **Market regime**: Adjusts risk based on current market conditions

### Multi-Level Protection

OPT_ATFNet implements multiple layers of protection:

1. **Position-level protection**: Stop loss for each individual trade
2. **Strategy-level protection**: Maximum positions, correlation filters
3. **Account-level protection**: Maximum daily loss, maximum drawdown
4. **Emergency protection**: Automatic trading pause after severe drawdown

### Trailing Stop Strategy

The system uses a sophisticated trailing stop strategy:

1. **Initial stop loss**: Based on ATR or key market levels
2. **First target**: Partial position close at first R:R target
3. **Breakeven move**: Stop loss moved to breakeven after first target
4. **Trailing activation**: Trailing stop activated after swing target
5. **ATR-based trailing**: Trail distance based on ATR multiple

## Market Regime Detection

OPT_ATFNet can detect different market regimes and adapt its strategy accordingly:

### Regime Classification

The system classifies market conditions into four main regimes:

1. **Trending**: Strong directional movement with good R-squared fit
2. **Ranging**: Oscillating price within defined boundaries
3. **Volatile**: Large price movements with high standard deviation
4. **Choppy**: Erratic price movement with frequent direction changes

### Detection Metrics

Market regimes are detected using several metrics:

1. **Trend strength**: R-squared of linear regression on recent prices
2. **Volatility**: Standard deviation of recent price returns
3. **Cyclicality**: Frequency of direction changes in recent price movement

### Strategy Adaptation

For each market regime, the system adapts its parameters:

1. **Trending markets**:
   - Increase trend-following weight
   - Reduce mean-reversion weight
   - Extend position holding time

2. **Ranging markets**:
   - Increase mean-reversion weight
   - Reduce trend-following weight
   - Shorten position holding time

3. **Volatile markets**:
   - Widen stop losses
   - Reduce position sizes
   - Increase profit targets

4. **Choppy markets**:
   - Increase signal thresholds
   - Reduce overall exposure
   - Consider pausing trading

## Performance Analytics

OPT_ATFNet includes comprehensive performance tracking and analysis:

### Trade Metrics

The system tracks key performance metrics:

1. **Win rate**: Percentage of profitable trades
2. **Profit factor**: Ratio of gross profit to gross loss
3. **Average win/loss**: Average profit of winning trades vs. losing trades
4. **Maximum drawdown**: Largest peak-to-trough decline
5. **Sharpe ratio**: Risk-adjusted return measure

### Component Effectiveness

The system also tracks the effectiveness of individual components:

1. **Attention accuracy**: How often the attention mechanism correctly identifies profitable setups
2. **Forecast accuracy**: How accurately the price forecaster predicts future price movements
3. **Filter effectiveness**: How effectively the frequency filter eliminates poor trades

### Performance Reporting

The performance analytics module can generate detailed reports:

1. **Trade history**: Complete record of all trades with entry/exit details
2. **Performance summary**: Key metrics and statistics
3. **Component analysis**: Effectiveness of individual components
4. **Equity curve**: Visual representation of account balance over time

## Implementation Details

### Code Organization

OPT_ATFNet is organized into the following files:

1. **OPT_ATFNet_EA.mq5**: Main EA file and entry point
2. **OPT_ATFNet_Includes.mqh**: Includes all necessary files
3. **OPT_ATFNet_Parameters.mqh**: Defines all input parameters
4. **OPT_ATFNet_Controller.mqh**: Main controller class
5. **OPT_ATFNet_RiskManager.mqh**: Risk management module
6. **OPT_ATFNet_SignalAggregator.mqh**: Signal aggregation module
7. **OPT_ATFNet_MarketRegime.mqh**: Market regime detection
8. **OPT_ATFNet_PerformanceAnalytics.mqh**: Performance tracking

Plus the original ATFNet components:
- ATFNetAttention.mqh
- FrequencyFeatures.mqh
- FrequencyFilter.mqh
- PriceForecaster.mqh
- AdaptiveForecastCombiner.mqh
- FFTUtility.mqh

### Key Classes

1. **CATFNetController**: Central controller that coordinates all components
2. **CRiskManager**: Handles position sizing and risk management
3. **CSignalAggregator**: Combines signals from different sources
4. **CMarketRegimeDetector**: Identifies current market regime
5. **CPerformanceAnalytics**: Tracks and analyzes performance
6. **CATFNetAttention**: Implements the attention mechanism
7. **CPriceForecaster**: Generates price forecasts
8. **CFrequencyFeatures**: Extracts frequency-domain features
9. **CPositionManager**: Manages open positions

### Execution Flow

1. **Initialization**:
   - Load parameters
   - Initialize all components
   - Set up initial state

2. **Tick Processing**:
   - Receive new market tick
   - Update market data
   - Process through controller

3. **Trading Logic**:
   - Update position tracking
   - Manage existing positions
   - Check for new entry opportunities
   - Execute trades if conditions met

4. **Learning and Adaptation**:
   - Update attention weights
   - Adjust forecaster weights
   - Optimize for current market regime

## Usage Guide

### Installation

1. Copy all files to the MQL5/Experts/OPT_ATFNet/ directory
2. Compile OPT_ATFNet_EA.mq5 in MetaEditor
3. Attach the EA to a chart in MetaTrader 5

### Configuration

OPT_ATFNet has numerous parameters organized into logical groups:

1. **General Settings**:
   - EXPERT_MAGIC: Magic number for trade identification
   - ENABLE_LOGGING: Enable detailed logging
   - ENABLE_ALERTS: Enable alerts for trade events

2. **ATFNet Parameters**:
   - SEQUENCE_LENGTH: Input sequence length for analysis
   - PREDICTION_LENGTH: Prediction horizon
   - TIME_DOMAIN_WEIGHT: Weight for time-domain analysis
   - FREQ_DOMAIN_WEIGHT: Weight for frequency-domain analysis

3. **Position Management**:
   - MAX_POSITIONS: Maximum allowed open positions
   - MIN_LOTS/MAX_LOTS: Position size limits
   - MAX_DAILY_LOSS: Maximum allowed daily loss
   - MAX_POSITION_DD: Maximum drawdown per position

4. **Risk Management**:
   - BASE_RISK_PER_TRADE: Base risk percentage per trade
   - MAX_RISK_PER_TRADE: Maximum risk percentage
   - ENABLE_DYNAMIC_RISK: Enable dynamic risk adjustment

5. **Strategy Parameters**:
   - ATTENTION_THRESHOLD: Minimum attention score for trade
   - LEARNING_RATE: Learning rate for weight updates
   - RANGE_LOOKBACK: Range calculation period
   - ENTRY_CHECK_SECONDS: Entry check interval

6. **Technical Indicators**:
   - ATR_PERIOD: ATR calculation period
   - ATR_MULTIPLIER: ATR multiplier for stop loss
   - USE_TREND_FILTER: Enable EMA trend filter

7. **Session Trading**:
   - ENABLE_SESSION_TRADING: Enable session-based trading
   - TRADE_ASIAN_SESSION: Trade during Asian session
   - TRADE_EUROPEAN_SESSION: Trade during European session
   - TRADE_AMERICAN_SESSION: Trade during American session

8. **Market Regime Detection**:
   - ENABLE_REGIME_DETECTION: Enable market regime detection
   - REGIME_LOOKBACK_BARS: Bars to analyze for regime detection
   - TREND_STRENGTH_THRESHOLD: Threshold for trend detection

### Recommended Workflow

1. **Initial Setup**:
   - Start with default parameters
   - Use small position sizes for initial testing
   - Enable detailed logging

2. **Monitoring**:
   - Monitor performance metrics
   - Review trade history and reasons
   - Check component effectiveness

3. **Optimization**:
   - Adjust parameters based on performance
   - Focus on one parameter group at a time
   - Use the built-in optimization tools

4. **Scaling Up**:
   - Gradually increase position sizes
   - Monitor drawdown carefully
   - Continue to refine parameters

## Parameter Optimization

OPT_ATFNet parameters can be optimized using the MetaTrader Strategy Tester:

### Optimization Approach

1. **Sequential Optimization**:
   - Optimize one parameter group at a time
   - Start with strategy parameters, then risk parameters
   - Keep other parameters fixed during optimization

2. **Walk-Forward Testing**:
   - Use walk-forward optimization to prevent curve-fitting
   - Test optimized parameters on out-of-sample data
   - Verify robustness across different market conditions

3. **Multi-Criteria Optimization**:
   - Consider multiple performance metrics (not just profit)
   - Balance return, drawdown, and stability
   - Optimize for risk-adjusted returns (Sharpe ratio)

### Key Parameters to Optimize

1. **Strategy Parameters**:
   - ATTENTION_THRESHOLD: 0.5-0.8 (step 0.05)
   - PREDICTION_THRESHOLD: 0.0005-0.0025 (step 0.0005)
   - TIME_DOMAIN_WEIGHT/FREQ_DOMAIN_WEIGHT: 0.3-0.7 (step 0.1)

2. **Risk Parameters**:
   - BASE_RISK_PER_TRADE: 0.5-2.0 (step 0.25)
   - MAX_POSITION_DD: 1.0-3.0 (step 0.5)
   - RR_RATIO/SWING_RR_RATIO: 1.0-3.0 (step 0.5)

3. **Technical Parameters**:
   - ATR_PERIOD: 10-20 (step 2)
   - ATR_MULTIPLIER: 0.8-1.5 (step 0.1)
   - Z_SCORE_PERIOD: 15-30 (step 5)

### Optimization Tips

1. Use a representative testing period that includes different market conditions
2. Start with coarse optimization, then refine with smaller steps
3. Verify results with forward testing before live trading
4. Re-optimize periodically as market conditions change
5. Keep a record of optimization results for different market conditions

---

## Conclusion

OPT_ATFNet represents a sophisticated trading system that combines advanced technical analysis, frequency-domain analysis, and machine learning techniques. Its modular architecture, adaptive capabilities, and comprehensive risk management make it suitable for a wide range of market conditions.

By understanding the system's components and how they interact, traders can effectively configure and optimize OPT_ATFNet for their specific trading goals and risk tolerance. The system's ability to learn from past trades and adapt to changing market conditions provides a significant advantage in today's dynamic markets.

Remember that no trading system is perfect, and proper risk management is essential. Always start with small position sizes, monitor performance carefully, and adjust parameters based on actual results.

---

*Disclaimer: Trading involves risk. Past performance is not indicative of future results. This documentation is for educational purposes only and does not constitute financial advice.*
