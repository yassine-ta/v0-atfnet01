//+------------------------------------------------------------------+
//|                                   OPT_ATFNet_Parameters.mqh      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

// This file contains extern variables that will be defined in the main EA file
// These are just declarations, not definitions

// General Settings
extern int EXPERT_MAGIC;
extern bool ENABLE_LOGGING;
extern bool ENABLE_ALERTS;

// ATFNet Parameters
extern int SEQUENCE_LENGTH;
extern int PREDICTION_LENGTH;
extern double PREDICTION_THRESHOLD;
extern bool USE_FREQUENCY_DOMAIN;
extern double TIME_DOMAIN_WEIGHT;
extern double FREQ_DOMAIN_WEIGHT;

// Position Management
extern int MAX_POSITIONS;
extern double MIN_LOTS;
extern double MAX_LOTS;
extern double MAX_DAILY_LOSS;
extern double MAX_POSITION_DD;
extern double MAX_ALLOWED_DD;
extern double MAX_SL_PIPS;
extern double MIN_PROFIT_PERCENT;

// Dynamic Risk Management
extern double BASE_RISK_PER_TRADE;
extern double MAX_RISK_PER_TRADE;
extern double PROFIT_THRESHOLD;
extern double RISK_INCREASE_STEP;
extern bool ENABLE_DYNAMIC_RISK;

// Drawdown Protection
extern double MAX_FLOATING_DD;
extern double EMERGENCY_CLOSE_DD;
extern bool SCALE_POSITION_SIZE;
extern double DD_SCALE_FACTOR;

// Attention Mechanism Parameters
extern double ATTENTION_THRESHOLD;
extern double LEARNING_RATE;
extern int WEIGHT_UPDATE_HOURS;

// Strategy Parameters
extern int RANGE_LOOKBACK;
extern int ENTRY_CHECK_SECONDS;
extern ENUM_TIMEFRAMES PERIOD_TIME;
extern double TARGET_PROFIT;

// Technical Indicators
extern int ATR_PERIOD;
extern double ATR_MULTIPLIER;
extern bool USE_TREND_FILTER;
extern int TREND_EMA_PERIOD;
extern ENUM_APPLIED_PRICE PRICE_SOURCE;

// Pattern Recognition
extern int CANDLE_RATES;
extern double BODY_RATIO;
extern double WICK_RATIO;
extern double DOJI_RATIO;
extern double PINBAR_RATIO;
extern double INSIDE_BAR_MAX;
extern double OUTSIDE_BAR_MIN;

// Z-Score Settings
extern bool USE_WEIGHTED_Z_SCORE;
extern int Z_SCORE_PERIOD;
extern double Z_SCORE_THRESHOLD;

// Session Trading
extern bool ENABLE_SESSION_TRADING;
extern bool TRADE_ASIAN_SESSION;
extern bool TRADE_EUROPEAN_SESSION;
extern bool TRADE_AMERICAN_SESSION;
extern int ASIAN_START;
extern int ASIAN_END;
extern int EUROPEAN_START;
extern int EUROPEAN_END;
extern int AMERICAN_START;
extern int AMERICAN_END;

// Market Hours
extern bool CLOSE_ALL_FRIDAY;
extern int FRIDAY_CLOSE_HOUR;
extern bool PREVENT_WEEKEND_TRADES;

// Trailing Stop Settings
extern double RR_RATIO;
extern double SWING_RR_RATIO;
extern double FIRST_TARGET_CLOSE_PCT;
extern double TRAILING_STOP_ATR_MULT;
extern double MIN_TRAILING_STOP_PIPS;

// Market Regime Detection
extern bool ENABLE_REGIME_DETECTION;
extern int REGIME_LOOKBACK_BARS;
extern double TREND_STRENGTH_THRESHOLD;
extern double VOLATILITY_THRESHOLD;
