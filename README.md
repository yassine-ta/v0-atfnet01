# ATFNet Python Implementation

This folder contains the Python implementation of the ATFNet trading system with machine learning and deep learning enhancements. It connects to MetaTrader 5 via the Python API to fetch market data, train models, and execute trades.

## Project Structure

- `config/` - Configuration files for different models and strategies
- `data/` - Data storage and preprocessing modules
- `models/` - Machine learning and deep learning models
- `strategies/` - Trading strategy implementations
- `utils/` - Utility functions and helper classes
- `mt5_connector/` - MetaTrader 5 connection and API interaction
- `notebooks/` - Jupyter notebooks for analysis and visualization
- `backtest/` - Backtesting framework
- `main.py` - Main entry point for the application
- `requirements.txt` - Python dependencies

## Setup

1. Install MetaTrader 5 and set up API access
2. Install Python dependencies: `pip install -r requirements.txt`
3. Configure MT5 connection in `config/mt5_config.json`
4. Run the application: `python main.py`

## Features

- Real-time data fetching from MT5
- Feature engineering based on ATFNet principles
- Machine learning models for market regime detection
- Deep learning models for price prediction
- Reinforcement learning for optimizing trade execution
- Automated trading via MT5 API
- Performance analytics and visualization
\`\`\`

Now, let's create the requirements file:

```plaintext file="python_implementation/requirements.txt"
# Core dependencies
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
pytorch>=1.10.0
statsmodels>=0.13.0
ta>=0.10.0  # Technical analysis library

# MT5 connection
MetaTrader5>=5.0.34
pytz>=2021.3

# Data processing
pandas-ta>=0.3.14b0
ta-lib>=0.4.24
numba>=0.55.0

# Machine learning
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=2.10.0
shap>=0.40.0

# Deep learning
torch>=1.10.0
transformers>=4.16.0

# Reinforcement learning
stable-baselines3>=1.5.0
gym>=0.21.0

# Utilities
tqdm>=4.62.0
joblib>=1.1.0
pyyaml>=6.0
python-dotenv>=0.19.0
