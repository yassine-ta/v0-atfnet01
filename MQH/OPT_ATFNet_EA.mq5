//+------------------------------------------------------------------+
//|                                                OPT_ATFNet_EA.mq5 |
//|                                                                   |
//|           Optimized ATFNet with Enhanced Architecture            |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property link      ""
#property version   "2.00"
#property strict

// Include all necessary files
#include <Trade\Trade.mqh>
#include "OPT_ATFNet_Includes.mqh"

// Define input parameters with proper grouping
// General Settings
input string GeneralGroupLabel = "===== General Settings ====="; // General Settings
input int EXPERT_MAGIC_NUM = 123456; // Expert Magic Number
input bool ENABLE_LOGGING_FLAG = true; // Enable Detailed Logging
input bool ENABLE_ALERTS_FLAG = false; // Enable Alerts

// ATFNet Parameters
input string ATFNetGroupLabel = "===== ATFNet Parameters ====="; // ATFNet Parameters
input int SEQUENCE_LENGTH_PARAM = 60; // Input sequence length
input int PREDICTION_LENGTH_PARAM = 20; // Prediction length
input double PREDICTION_THRESHOLD_PARAM = 0.0015; // Prediction threshold (%)
input bool USE_FREQUENCY_DOMAIN_FLAG = true; // Use frequency domain analysis
input double TIME_DOMAIN_WEIGHT_PARAM = 0.5; // Time domain weight (0-1)
input double FREQ_DOMAIN_WEIGHT_PARAM = 0.5; // Frequency domain weight (0-1)

// Position Management
input string PositionGroupLabel = "===== Position Management ====="; // Position Management
input int MAX_POSITIONS_PARAM = 3; // Maximum allowed positions
input double MIN_LOTS_PARAM = 0.01; // Min Lots
input double MAX_LOTS_PARAM = 10.0; // Max Lots
input double MAX_DAILY_LOSS_PARAM = 5.0; // Maximum daily loss (%)
input double MAX_POSITION_DD_PARAM = 2.0; // Max position drawdown (%)
input double MAX_ALLOWED_DD_PARAM = 6.0; // Max allowed drawdown (%)
input double MAX_SL_PIPS_PARAM = 200.0; // Maximum Stop Loss in Pips
input double MIN_PROFIT_PERCENT_PARAM = 0.015; // Min Profit % to open trades

// Dynamic Risk Management
input string RiskGroupLabel = "===== Dynamic Risk Management ====="; // Dynamic Risk Management
input double BASE_RISK_PER_TRADE_PARAM = 1.0; // Base risk per trade (%)
input double MAX_RISK_PER_TRADE_PARAM = 3.0; // Maximum risk per trade (%)
input double PROFIT_THRESHOLD_PARAM = 5.0; // Profit threshold for risk increase (%)
input double RISK_INCREASE_STEP_PARAM = 0.5; // Risk increase step (%)
input bool ENABLE_DYNAMIC_RISK_FLAG = true; // Enable dynamic risk management

// Drawdown Protection
input string DrawdownGroupLabel = "===== Drawdown Protection ====="; // Drawdown Protection
input double MAX_FLOATING_DD_PARAM = 12.0; // Maximum floating drawdown (%)
input double EMERGENCY_CLOSE_DD_PARAM = 15.0; // Emergency close all positions drawdown (%)
input bool SCALE_POSITION_SIZE_FLAG = true; // Scale position size based on drawdown
input double DD_SCALE_FACTOR_PARAM = 0.5; // Position size scale factor when in drawdown

// Attention Mechanism Parameters
input string AttentionGroupLabel = "===== Attention Parameters ====="; // Attention Parameters
input double ATTENTION_THRESHOLD_PARAM = 0.6; // Minimum attention score for trade
input double LEARNING_RATE_PARAM = 0.01; // Learning rate for weight updates
input int WEIGHT_UPDATE_HOURS_PARAM = 1; // Hours between weight updates

// Strategy Parameters
input string StrategyGroupLabel = "===== Strategy Parameters ====="; // Strategy Parameters
input int RANGE_LOOKBACK_PARAM = 10; // Range Lookback Period
input int ENTRY_CHECK_SECONDS_PARAM = 15; // Entry Check Interval (seconds)
input ENUM_TIMEFRAMES PERIOD_TIME_PARAM = PERIOD_CURRENT; // Timeframe for Calculations
input double TARGET_PROFIT_PARAM = 50.0; // Target Profit (currency)

// Technical Indicators
input string IndicatorsGroupLabel = "===== Technical Indicators ====="; // Technical Indicators
input int ATR_PERIOD_PARAM = 14; // ATR Period
input double ATR_MULTIPLIER_PARAM = 1.0; // ATR Multiplier
input bool USE_TREND_FILTER_FLAG = true; // Use EMA Trend Filter
input int TREND_EMA_PERIOD_PARAM = 50; // EMA Period
input ENUM_APPLIED_PRICE PRICE_SOURCE_PARAM = PRICE_CLOSE; // Price Source for EMA

// Pattern Recognition
input string PatternGroupLabel = "===== Pattern Recognition ====="; // Pattern Recognition
input int CANDLE_RATES_PARAM = 3; // Min candlesticks required
input double BODY_RATIO_PARAM = 0.3; // Body/Total Ratio
input double WICK_RATIO_PARAM = 0.3; // Wick/Total Ratio
input double DOJI_RATIO_PARAM = 0.1; // Doji Body/Total Ratio
input double PINBAR_RATIO_PARAM = 0.3; // Pinbar nose ratio
input double INSIDE_BAR_MAX_PARAM = 0.8; // Inside bar maximum size ratio
input double OUTSIDE_BAR_MIN_PARAM = 1.2; // Outside bar minimum size ratio

// Z-Score Settings
input string ZScoreGroupLabel = "===== Z-Score Settings ====="; // Z-Score Settings
input bool USE_WEIGHTED_Z_SCORE_FLAG = true; // Use Weighted Z-Score
input int Z_SCORE_PERIOD_PARAM = 20; // Z-Score Period
input double Z_SCORE_THRESHOLD_PARAM = 1.5; // Z-Score Threshold

// Session Trading
input string SessionGroupLabel = "===== Session Trading ====="; // Session Trading
input bool ENABLE_SESSION_TRADING_FLAG = true; // Enable Session Trading
input bool TRADE_ASIAN_SESSION_FLAG = true; // Trade Asian Session
input bool TRADE_EUROPEAN_SESSION_FLAG = true; // Trade European Session
input bool TRADE_AMERICAN_SESSION_FLAG = true; // Trade American Session
input int ASIAN_START_PARAM = 0; // Asian Session Start (Server Hour)
input int ASIAN_END_PARAM = 9; // Asian Session End (Server Hour)
input int EUROPEAN_START_PARAM = 7; // European Session Start (Server Hour)
input int EUROPEAN_END_PARAM = 16; // European Session End (Server Hour)
input int AMERICAN_START_PARAM = 13; // American Session Start (Server Hour)
input int AMERICAN_END_PARAM = 22; // American Session End (Server Hour)

// Market Hours
input string MarketHoursGroupLabel = "===== Market Hours ====="; // Market Hours
input bool CLOSE_ALL_FRIDAY_FLAG = true; // Close all positions before weekend
input int FRIDAY_CLOSE_HOUR_PARAM = 21; // Hour to close positions on Friday
input bool PREVENT_WEEKEND_TRADES_FLAG = true; // Prevent new trades near weekend

// Trailing Stop Settings
input string TrailingGroupLabel = "===== Position Management Settings ====="; // Position Management Settings
input double RR_RATIO_PARAM = 1.5; // Risk:Reward Ratio for first target
input double SWING_RR_RATIO_PARAM = 3.0; // Risk:Reward Ratio for swing target
input double FIRST_TARGET_CLOSE_PCT_PARAM = 70.0; // Percentage to close at first RR target
input double TRAILING_STOP_ATR_MULT_PARAM = 2.0; // Trailing stop ATR multiplier
input double MIN_TRAILING_STOP_PIPS_PARAM = 10.0; // Minimum trailing stop in pips

// Market Regime Detection
input string RegimeGroupLabel = "===== Market Regime Detection ====="; // Market Regime Detection
input bool ENABLE_REGIME_DETECTION_FLAG = true; // Enable market regime detection
input int REGIME_LOOKBACK_BARS_PARAM = 100; // Bars to analyze for regime detection
input double TREND_STRENGTH_THRESHOLD_PARAM = 0.6; // Trend strength threshold
input double VOLATILITY_THRESHOLD_PARAM = 1.5; // Volatility threshold (ATR multiple)

// Global objects
CTrade g_trade;
CATFNetController* g_controller = NULL;

// Map input parameters to extern variables
void MapParameters() {
    // Set extern variables in OPT_ATFNet_Controller.mqh
    EXPERT_MAGIC = EXPERT_MAGIC_NUM;
    ENABLE_LOGGING = ENABLE_LOGGING_FLAG;
    ENABLE_ALERTS = ENABLE_ALERTS_FLAG;
    SEQUENCE_LENGTH = SEQUENCE_LENGTH_PARAM;
    PREDICTION_LENGTH = PREDICTION_LENGTH_PARAM;
    PREDICTION_THRESHOLD = PREDICTION_THRESHOLD_PARAM;
    USE_FREQUENCY_DOMAIN = USE_FREQUENCY_DOMAIN_FLAG;
    TIME_DOMAIN_WEIGHT = TIME_DOMAIN_WEIGHT_PARAM;
    FREQ_DOMAIN_WEIGHT = FREQ_DOMAIN_WEIGHT_PARAM;
    MAX_POSITIONS = MAX_POSITIONS_PARAM;
    MIN_LOTS = MIN_LOTS_PARAM;
    MAX_LOTS = MAX_LOTS_PARAM;
    MAX_DAILY_LOSS = MAX_DAILY_LOSS_PARAM;
    MAX_POSITION_DD = MAX_POSITION_DD_PARAM;
    MAX_ALLOWED_DD = MAX_ALLOWED_DD_PARAM;
    MAX_SL_PIPS = MAX_SL_PIPS_PARAM;
    MIN_PROFIT_PERCENT = MIN_PROFIT_PERCENT_PARAM;
    BASE_RISK_PER_TRADE = BASE_RISK_PER_TRADE_PARAM;
    MAX_RISK_PER_TRADE = MAX_RISK_PER_TRADE_PARAM;
    PROFIT_THRESHOLD = PROFIT_THRESHOLD_PARAM;
    RISK_INCREASE_STEP = RISK_INCREASE_STEP_PARAM;
    ENABLE_DYNAMIC_RISK = ENABLE_DYNAMIC_RISK_FLAG;
    MAX_FLOATING_DD = MAX_FLOATING_DD_PARAM;
    EMERGENCY_CLOSE_DD = EMERGENCY_CLOSE_DD_PARAM;
    SCALE_POSITION_SIZE = SCALE_POSITION_SIZE_FLAG;
    DD_SCALE_FACTOR = DD_SCALE_FACTOR_PARAM;
    ATTENTION_THRESHOLD = ATTENTION_THRESHOLD_PARAM;
    LEARNING_RATE = LEARNING_RATE_PARAM;
    WEIGHT_UPDATE_HOURS = WEIGHT_UPDATE_HOURS_PARAM;
    RANGE_LOOKBACK = RANGE_LOOKBACK_PARAM;
    ENTRY_CHECK_SECONDS = ENTRY_CHECK_SECONDS_PARAM;
    PERIOD_TIME = PERIOD_TIME_PARAM;
    TARGET_PROFIT = TARGET_PROFIT_PARAM;
    ATR_PERIOD = ATR_PERIOD_PARAM;
    ATR_MULTIPLIER = ATR_MULTIPLIER_PARAM;
    USE_TREND_FILTER = USE_TREND_FILTER_FLAG;
    TREND_EMA_PERIOD = TREND_EMA_PERIOD_PARAM;
    PRICE_SOURCE = PRICE_SOURCE_PARAM;
    CANDLE_RATES = CANDLE_RATES_PARAM;
    BODY_RATIO = BODY_RATIO_PARAM;
    WICK_RATIO = WICK_RATIO_PARAM;
    DOJI_RATIO = DOJI_RATIO_PARAM;
    PINBAR_RATIO = PINBAR_RATIO_PARAM;
    INSIDE_BAR_MAX = INSIDE_BAR_MAX_PARAM;
    OUTSIDE_BAR_MIN = OUTSIDE_BAR_MIN_PARAM;
    USE_WEIGHTED_Z_SCORE = USE_WEIGHTED_Z_SCORE_FLAG;
    Z_SCORE_PERIOD = Z_SCORE_PERIOD_PARAM;
    Z_SCORE_THRESHOLD = Z_SCORE_THRESHOLD_PARAM;
    ENABLE_SESSION_TRADING = ENABLE_SESSION_TRADING_FLAG;
    TRADE_ASIAN_SESSION = TRADE_ASIAN_SESSION_FLAG;
    TRADE_EUROPEAN_SESSION = TRADE_EUROPEAN_SESSION_FLAG;
    TRADE_AMERICAN_SESSION = TRADE_AMERICAN_SESSION_FLAG;
    ASIAN_START = ASIAN_START_PARAM;
    ASIAN_END = ASIAN_END_PARAM;
    EUROPEAN_START = EUROPEAN_START_PARAM;
    EUROPEAN_END = EUROPEAN_END_PARAM;
    AMERICAN_START = AMERICAN_START_PARAM;
    AMERICAN_END = AMERICAN_END_PARAM;
    CLOSE_ALL_FRIDAY = CLOSE_ALL_FRIDAY_FLAG;
    FRIDAY_CLOSE_HOUR = FRIDAY_CLOSE_HOUR_PARAM;
    PREVENT_WEEKEND_TRADES = PREVENT_WEEKEND_TRADES_FLAG;
    RR_RATIO = RR_RATIO_PARAM;
    SWING_RR_RATIO = SWING_RR_RATIO_PARAM;
    FIRST_TARGET_CLOSE_PCT = FIRST_TARGET_CLOSE_PCT_PARAM;
    TRAILING_STOP_ATR_MULT = TRAILING_STOP_ATR_MULT_PARAM;
    MIN_TRAILING_STOP_PIPS = MIN_TRAILING_STOP_PIPS_PARAM;
    ENABLE_REGIME_DETECTION = ENABLE_REGIME_DETECTION_FLAG;
    REGIME_LOOKBACK_BARS = REGIME_LOOKBACK_BARS_PARAM;
    TREND_STRENGTH_THRESHOLD = TREND_STRENGTH_THRESHOLD_PARAM;
    VOLATILITY_THRESHOLD = VOLATILITY_THRESHOLD_PARAM;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit() {
    // Parameter validation
    if(!ValidateParameters()) {
        Print("Invalid parameters. Please check input values.");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    // Map input parameters to extern variables
    MapParameters();
    
    // Set up trade object
    g_trade.SetDeviationInPoints(10); // 1 pip deviation for market orders
    g_trade.SetTypeFilling(ORDER_FILLING_FOK);
    g_trade.SetExpertMagicNumber(EXPERT_MAGIC_NUM); // Set magic number from parameters
    
    // Initialize controller with all components
    g_controller = new CATFNetController(&g_trade);
    if(!g_controller.Initialize()) {
        Print("Failed to initialize OPT_ATFNet controller");
        return INIT_FAILED;
    }
    
    Print("=== OPT_ATFNet EA Initialized ===");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Clean up controller
    if(g_controller != NULL) {
        delete g_controller;
        g_controller = NULL;
    }
    
    Print("=== OPT_ATFNet EA Deinitialized ===");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick() {
    // Process tick through controller
    if(g_controller != NULL) {
        g_controller.ProcessTick();
    }
}

//+------------------------------------------------------------------+
//| Validate input parameters                                         |
//+------------------------------------------------------------------+
bool ValidateParameters() {
    // Validate sequence length
    if(SEQUENCE_LENGTH_PARAM < 10) {
        Print("Invalid SEQUENCE_LENGTH value. Must be at least 10.");
        return false;
    }
    
    // Validate prediction length
    if(PREDICTION_LENGTH_PARAM < 1) {
        Print("Invalid PREDICTION_LENGTH value. Must be at least 1.");
        return false;
    }
    
    // Validate Z-Score period
    if(Z_SCORE_PERIOD_PARAM < 2) {
        Print("Invalid Z_SCORE_PERIOD value. Must be at least 2.");
        return false;
    }
    
    // Validate domain weights
    if(TIME_DOMAIN_WEIGHT_PARAM < 0 || TIME_DOMAIN_WEIGHT_PARAM > 1 || 
       FREQ_DOMAIN_WEIGHT_PARAM < 0 || FREQ_DOMAIN_WEIGHT_PARAM > 1) {
        Print("Invalid domain weights. Must be between 0 and 1.");
        return false;
    }
    
    // Validate risk parameters
    if(BASE_RISK_PER_TRADE_PARAM <= 0 || MAX_RISK_PER_TRADE_PARAM <= 0) {
        Print("Invalid risk parameters. Must be greater than 0.");
        return false;
    }
    
    return true;
}
