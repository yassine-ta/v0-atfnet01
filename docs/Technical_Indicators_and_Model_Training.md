# Technical Indicators and Model Training in ATFNet

This document explains the technical indicators used in the ATFNet Python implementation and how the model trains features and targets.

## Technical Indicators

ATFNet uses a comprehensive set of technical indicators to analyze market conditions and make trading decisions. These indicators are calculated from price and volume data and serve as inputs to the machine learning models.

### Moving Averages

Moving averages smooth price data to identify trends:

- **Simple Moving Average (SMA)**: Calculated for periods 5, 10, 20, 50, 100, and 200
- **Exponential Moving Average (EMA)**: Calculated for periods 5, 10, 20, 50, 100, and 200

Moving averages help identify trend direction and potential support/resistance levels. Shorter periods (5, 10) are more responsive to recent price changes, while longer periods (50, 100, 200) show long-term trends.

### Bollinger Bands

Bollinger Bands measure volatility and potential overbought/oversold conditions:

- **Middle Band**: 20-period SMA
- **Upper Band**: Middle Band + (2 × Standard Deviation)
- **Lower Band**: Middle Band - (2 × Standard Deviation)
- **Bandwidth**: (Upper Band - Lower Band) / Middle Band
- **%B**: Position of price within the bands

Bollinger Bands help identify volatility expansions and contractions. When bands narrow, it often precedes a significant price move. The %B indicator shows where the price is relative to the bands, helping identify potential reversals.

### Relative Strength Index (RSI)

RSI measures the speed and change of price movements, identifying overbought/oversold conditions:

- **RSI-14**: 14-period RSI

RSI values above 70 typically indicate overbought conditions, while values below 30 indicate oversold conditions. The indicator also helps identify divergences between price and momentum.

### Moving Average Convergence Divergence (MACD)

MACD identifies changes in the strength, direction, momentum, and duration of a trend:

- **MACD Line**: 12-period EMA - 26-period EMA
- **Signal Line**: 9-period EMA of MACD Line
- **Histogram**: MACD Line - Signal Line

MACD crossovers (MACD Line crossing the Signal Line) can indicate potential trend changes. The histogram shows the acceleration of price movement, with increasing values indicating strengthening momentum.

### Average True Range (ATR)

ATR measures market volatility:

- **ATR-14**: 14-period ATR
- **ATR Percentage**: ATR as a percentage of price

ATR helps set appropriate stop-loss levels and position sizes based on market volatility. Higher ATR values indicate more volatile market conditions.

### Stochastic Oscillator

Stochastic Oscillator compares a closing price to its price range over a specific period:

- **%K**: 14-period stochastic
- **%D**: 3-period SMA of %K

Like RSI, the Stochastic Oscillator helps identify overbought (above 80) and oversold (below 20) conditions. Crossovers between %K and %D can signal potential entry or exit points.

### Z-Score

Z-Score measures how many standard deviations a price is from its mean:

- **Z-Score-20**: 20-period Z-Score

Z-Score helps identify when prices are statistically unusual. Values above 2 or below -2 indicate prices that are significantly higher or lower than the recent average.

### Trend Strength

Trend Strength measures the strength of a trend relative to volatility:

- **Trend Strength-20**: (Close - SMA20) / ATR14

Higher absolute values indicate stronger trends. Positive values indicate uptrends, while negative values indicate downtrends.

### Range Position

Range Position measures where the current price is within its recent range:

- **Range Position-10**: Position within 10-period high-low range

Values close to 1 indicate prices near the high of the range, while values close to 0 indicate prices near the low of the range. This helps identify potential reversal points.

### Volume Indicators

Volume indicators measure the strength of price movements:

- **Volume SMA-20**: 20-period SMA of volume
- **Volume Ratio**: Volume / Volume SMA-20
- **Money Flow Index (MFI)**: 14-period MFI

Volume confirmation is important for validating price movements. The Money Flow Index combines price and volume to identify potential reversals, similar to RSI but with volume weighting.

## Frequency Domain Features

In addition to traditional technical indicators, ATFNet also uses frequency domain features derived from Fast Fourier Transform (FFT) analysis. These features help identify cyclical patterns in price data.

### FFT Features

- **FFT Magnitudes**: Magnitudes of dominant frequencies
- **FFT Phases**: Phases of dominant frequencies
- **Spectral Entropy**: Measure of randomness in frequency distribution
- **Dominant Frequency Strength**: Ratio of dominant frequency to total
- **Cyclicality Score**: Measure of cyclical behavior

FFT analysis transforms time-domain price data into the frequency domain, revealing hidden cycles and periodicities. Spectral entropy measures the randomness of the frequency distribution, with lower values indicating stronger cyclical patterns. The cyclicality score specifically measures the strength of low-frequency components, which correspond to longer-term cycles.

## ATFNet-Specific Features

ATFNet also uses a set of specialized features that combine information from multiple indicators:

- **Price**: Current closing price
- **ATR**: Average True Range (volatility)
- **Z-Score**: Standardized price deviation
- **Trend Strength**: Strength of current trend
- **Range Position**: Position within recent price range
- **Volume Profile**: Volume relative to recent average
- **Pattern Strength**: Strength of chart patterns
- **Spectral Entropy**: Randomness in frequency domain
- **Frequency Trend Strength**: Strength of dominant frequency
- **Cyclicality Score**: Strength of cyclical behavior

These combined features provide a comprehensive view of market conditions, incorporating information about price, volatility, trend, cycles, and volume. They serve as the primary inputs to the ATFNet attention mechanism and machine learning models.

## Feature Preparation for Model Training

The ATFNet implementation prepares features for model training using a systematic process that creates sequences of features and corresponding targets for time series prediction.

### Feature Selection

The following features are selected for model training:

- Core ATFNet features (price, ATR, Z-Score, etc.)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Frequency domain features (spectral entropy, cyclicality, etc.)

This comprehensive feature set captures different aspects of market behavior, from trend and momentum to volatility and cyclical patterns.

### Feature Scaling

Features are scaled using StandardScaler to normalize their values. This ensures that all features have similar scales, which is important for neural network training. The scaler is fitted on the training data and then applied to both training and testing data.

### Sequence Creation

Features are organized into sequences for time series prediction. Each sequence consists of a window of consecutive time periods (e.g., 60 periods) and serves as input to the recurrent neural network. The corresponding target is the value to be predicted (e.g., the return in the next period).

## Target Definition

The target for model training is defined based on the target_column and target_shift parameters:

- **Target Column**: Typically "returns" (percentage price change)
- **Target Shift**: Typically -1 (next period's return)

By default, the target is the future return (returns column with a shift of -1), which means the model is trained to predict the next period's return. This allows the model to forecast price movements and make trading decisions accordingly.

## Model Training Process

The ATFNet implementation uses a systematic process to train machine learning models. The training process includes:

1. **Data Splitting**: The data is split into training and validation sets (typically 80% training, 20% validation).
2. **Model Building**: The appropriate model architecture is constructed based on the specified type.
3. **Callback Setup**: Callbacks are set up for early stopping, model checkpointing, and learning rate reduction.
4. **Model Training**: The model is trained on the training data with validation on the validation set.
5. **Model Saving**: The trained model and its training history are saved for later use.

### Model Types

ATFNet supports several types of models:

1. **LSTM (Long Short-Term Memory)**: LSTM networks are a type of recurrent neural network (RNN) that can learn long-term dependencies in time series data. They use memory cells and gates to control the flow of information, allowing them to remember patterns over long sequences.

2. **GRU (Gated Recurrent Unit)**: GRU is a simplified version of LSTM that uses fewer parameters while maintaining similar performance. GRU networks use update and reset gates to control information flow, making them computationally more efficient than LSTMs.

3. **Hybrid**: The hybrid model combines LSTM and GRU layers in parallel branches, followed by dense layers. This architecture leverages the strengths of both LSTM and GRU, potentially capturing different aspects of the time series patterns.

### Model Architecture

All model architectures follow a similar pattern:

1. **Recurrent Layers**: One or more LSTM or GRU layers to process the time series data
2. **Batch Normalization**: After each recurrent layer to normalize activations
3. **Dropout**: After each batch normalization to prevent overfitting
4. **Dense Output Layer**: Final layer to produce the prediction

The hybrid model has parallel LSTM and GRU branches that are concatenated before the dense layers.

### Training Configuration

The training process uses the following configuration:

- **Optimizer**: Adam with configurable learning rate
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Early Stopping**: Monitors validation loss with patience of 10 epochs
- **Model Checkpoint**: Saves the best model based on validation loss
- **Learning Rate Reduction**: Reduces learning rate when validation loss plateaus

These settings help ensure efficient training and prevent overfitting. Early stopping halts training when the validation loss stops improving, while learning rate reduction helps fine-tune the model when progress slows.

## Model Evaluation

After training, models are evaluated using several metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predictions and actual values. Lower values indicate better performance.

- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predictions and actual values. This metric is less sensitive to outliers than MSE.

- **Root Mean Squared Error (RMSE)**: Square root of MSE, provides error in the same units as the target. This makes it more interpretable than MSE.

- **R-squared (R²)**: Measures the proportion of variance in the target that is predictable from the features. Values range from 0 to 1, with higher values indicating better fit.

- **Directional Accuracy**: Measures how often the model correctly predicts the direction of price movement (up or down). This is particularly important for trading strategies, as the direction of movement often matters more than the exact magnitude.

These metrics provide a comprehensive evaluation of model performance, considering both the accuracy of the predictions and their usefulness for trading decisions.

## Model Application in Trading

The trained models are used in the ATFNet strategy to make trading decisions. The process involves:

1. **Feature Calculation**: Technical indicators and frequency features are calculated from recent price data.
2. **Feature Scaling**: Features are scaled using the saved scaler.
3. **Sequence Creation**: A sequence of recent features is created.
4. **Prediction**: The model predicts the target (e.g., next period's return).
5. **Signal Generation**: The prediction is combined with other factors (market regime, risk parameters, etc.) to generate trading signals.
6. **Position Management**: Positions are opened, managed, and closed based on the signals and risk management rules.

The ATFNet attention mechanism plays a crucial role in this process, dynamically weighting different features based on their predictive power in the current market conditions. This adaptive approach helps the strategy perform well across different market regimes.

## Market Regime Detection

ATFNet includes a market regime detection component that classifies market conditions into different regimes:

- **Trending**: Strong directional movement with moderate volatility
- **Ranging**: Sideways movement with low volatility
- **Volatile**: Strong directional movement with high volatility
- **Choppy**: Sideways movement with high volatility

The regime is detected based on trend strength and volatility metrics. Different trading parameters and strategies are applied depending on the detected regime, allowing the system to adapt to changing market conditions.

## Adaptive Learning

One of the key features of ATFNet is its adaptive learning capability. The system continuously updates its parameters based on trading results:

- **Attention Weights**: Updated based on the profitability of trades, giving more weight to features that lead to profitable trades.
- **Signal Weights**: Adjusted based on the performance of different signal sources.
- **Risk Parameters**: Modified based on recent trading performance to optimize risk-reward ratio.

This adaptive approach allows ATFNet to evolve with the market, continuously improving its performance and adapting to changing market conditions.

## Conclusion

The ATFNet Python implementation uses a comprehensive set of technical indicators and frequency domain features for market analysis and prediction. The model training process involves preparing sequences of scaled features and corresponding targets, training recurrent neural network models (LSTM, GRU, or hybrid), and evaluating their performance using various metrics.

The combination of traditional technical indicators, frequency domain analysis, and deep learning models allows ATFNet to capture complex patterns in financial time series data and make accurate predictions for trading decisions. The adaptive learning and market regime detection components further enhance the system's ability to perform well across different market conditions.

By understanding the technical indicators used and the model training process, users can customize and optimize the ATFNet system for their specific trading needs and market conditions.
