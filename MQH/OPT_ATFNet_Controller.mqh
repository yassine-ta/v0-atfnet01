//+------------------------------------------------------------------+
//|                                 OPT_ATFNet_Controller.mqh        |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

// Include necessary files
#include <Trade\Trade.mqh>
#include "OPT_ATFNet_Types.mqh"
#include "OPT_ATFNet_Stubs.mqh"

// Forward declarations
class CFrequencyFeatures;
class CFrequencyFilter;
class CPriceForecaster;
class CATFNetAttention;
class CRiskManager;
class CSignalAggregator;
class CMarketRegimeDetector;

// External variables from parameters file
extern int EXPERT_MAGIC;
extern bool ENABLE_LOGGING;
extern bool ENABLE_ALERTS;
extern int SEQUENCE_LENGTH;
extern int PREDICTION_LENGTH;
extern double PREDICTION_THRESHOLD;
extern bool USE_FREQUENCY_DOMAIN;
extern double TIME_DOMAIN_WEIGHT;
extern double FREQ_DOMAIN_WEIGHT;
extern int MAX_POSITIONS;
extern double MIN_LOTS;
extern double MAX_LOTS;
extern double MAX_DAILY_LOSS;
extern double MAX_POSITION_DD;
extern double MAX_ALLOWED_DD;
extern double MAX_SL_PIPS;
extern double MIN_PROFIT_PERCENT;
extern double BASE_RISK_PER_TRADE;
extern double MAX_RISK_PER_TRADE;
extern double PROFIT_THRESHOLD;
extern double RISK_INCREASE_STEP;
extern bool ENABLE_DYNAMIC_RISK;
extern double MAX_FLOATING_DD;
extern double EMERGENCY_CLOSE_DD;
extern bool SCALE_POSITION_SIZE;
extern double DD_SCALE_FACTOR;
extern double ATTENTION_THRESHOLD;
extern double LEARNING_RATE;
extern int WEIGHT_UPDATE_HOURS;
extern int RANGE_LOOKBACK;
extern int ENTRY_CHECK_SECONDS;
extern ENUM_TIMEFRAMES PERIOD_TIME;
extern double TARGET_PROFIT;
extern int ATR_PERIOD;
extern double ATR_MULTIPLIER;
extern bool USE_TREND_FILTER;
extern int TREND_EMA_PERIOD;
extern ENUM_APPLIED_PRICE PRICE_SOURCE;
extern int CANDLE_RATES;
extern double BODY_RATIO;
extern double WICK_RATIO;
extern double DOJI_RATIO;
extern double PINBAR_RATIO;
extern double INSIDE_BAR_MAX;
extern double OUTSIDE_BAR_MIN;
extern bool USE_WEIGHTED_Z_SCORE;
extern int Z_SCORE_PERIOD;
extern double Z_SCORE_THRESHOLD;
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
extern bool CLOSE_ALL_FRIDAY;
extern int FRIDAY_CLOSE_HOUR;
extern bool PREVENT_WEEKEND_TRADES;
extern double RR_RATIO;
extern double SWING_RR_RATIO;
extern double FIRST_TARGET_CLOSE_PCT;
extern double TRAILING_STOP_ATR_MULT;
extern double MIN_TRAILING_STOP_PIPS;
extern bool ENABLE_REGIME_DETECTION;
extern int REGIME_LOOKBACK_BARS;
extern double TREND_STRENGTH_THRESHOLD;
extern double VOLATILITY_THRESHOLD;

//+------------------------------------------------------------------+
//| Main controller class for OPT_ATFNet                             |
//+------------------------------------------------------------------+
class CATFNetController {
private:
    // Trading components
    CTrade* m_trade;
    
    // Analysis components
    CFrequencyFeatures* m_freqFeatures;
    CFrequencyFilter* m_freqFilter;
    CPriceForecaster* m_priceForecaster;
    CATFNetAttention* m_attention;
    
    // Management components
    CPositionManager* m_posManager;
    CRiskManager* m_riskManager;
    CSignalAggregator* m_signalAggregator;
    CMarketRegimeDetector* m_regimeDetector;
    CPerformanceAnalytics* m_performance;
    
    // Price data arrays
    double m_closeArray[];
    double m_openArray[];
    double m_highArray[];
    double m_lowArray[];
    long m_volumeArray[];
    
    // State variables
    bool m_isEnabled;
    datetime m_lastUpdateTime;
    datetime m_lastEntryCheckTime;
    datetime m_lastWeightUpdateTime;
    double m_rangeHigh;
    double m_rangeLow;
    
    // Internal methods
    bool UpdateMarketData();
    bool GenerateSignals();
    void ManagePositions();
    void UpdateWeights();
    double CalculateStopLoss(ENUM_ORDER_TYPE orderType);
    bool IsValidTradingSession();
    bool IsFridayCloseTime();
    bool IsNearWeekend();
    void CheckDailyLossLimit();
    
public:
    // Constructor/Destructor
    CATFNetController(CTrade* trade);
    ~CATFNetController();
    
    // Main methods
    bool Initialize();
    void ProcessTick();
    void CloseAllPositions(string reason = "");
    
    // Getters/Setters
    void SetEnabled(bool enabled) { m_isEnabled = enabled; }
    bool IsEnabled() const { return m_isEnabled; }
    
    // Position management
    bool OpenPosition(ENUM_ORDER_TYPE orderType, double positionSize, double stopLoss);
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CATFNetController::CATFNetController(CTrade* trade) {
    m_trade = trade;
    m_freqFeatures = NULL;
    m_freqFilter = NULL;
    m_priceForecaster = NULL;
    m_attention = NULL;
    m_posManager = NULL;
    m_riskManager = NULL;
    m_signalAggregator = NULL;
    m_regimeDetector = NULL;
    m_performance = NULL;
    
    m_isEnabled = true;
    m_lastUpdateTime = 0;
    m_lastEntryCheckTime = 0;
    m_lastWeightUpdateTime = 0;
    m_rangeHigh = 0;
    m_rangeLow = 0;
    
    // Initialize arrays
    ArrayResize(m_closeArray, SEQUENCE_LENGTH);
    ArrayResize(m_openArray, SEQUENCE_LENGTH);
    ArrayResize(m_highArray, SEQUENCE_LENGTH);
    ArrayResize(m_lowArray, SEQUENCE_LENGTH);
    ArrayResize(m_volumeArray, SEQUENCE_LENGTH);
    ArrayInitialize(m_closeArray, 0);
    ArrayInitialize(m_openArray, 0);
    ArrayInitialize(m_highArray, 0);
    ArrayInitialize(m_lowArray, 0);
    ArrayInitialize(m_volumeArray, 0);
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CATFNetController::~CATFNetController() {
    // Clean up objects
    if(m_freqFeatures != NULL) delete m_freqFeatures;
    if(m_freqFilter != NULL) delete m_freqFilter;
    if(m_priceForecaster != NULL) delete m_priceForecaster;
    if(m_attention != NULL) delete m_attention;
    if(m_posManager != NULL) delete m_posManager;
    if(m_riskManager != NULL) delete m_riskManager;
    if(m_signalAggregator != NULL) delete m_signalAggregator;
    if(m_regimeDetector != NULL) delete m_regimeDetector;
    if(m_performance != NULL) delete m_performance;
}

//+------------------------------------------------------------------+
//| Initialize all components                                         |
//+------------------------------------------------------------------+
bool CATFNetController::Initialize() {
    // Initialize frequency features
    m_freqFeatures = new CFrequencyFeatures(SEQUENCE_LENGTH);
    if(m_freqFeatures == NULL) {
        Print("Failed to initialize frequency features");
        return false;
    }
    
    // Initialize frequency filter
    m_freqFilter = new CFrequencyFilter(SEQUENCE_LENGTH);
    if(m_freqFilter == NULL) {
        Print("Failed to initialize frequency filter");
        return false;
    }
    
    // Initialize price forecaster
    m_priceForecaster = new CPriceForecaster(PREDICTION_LENGTH, SEQUENCE_LENGTH);
    if(m_priceForecaster == NULL) {
        Print("Failed to initialize price forecaster");
        return false;
    }
    
    // Initialize attention mechanism
    m_attention = new CATFNetAttention(LEARNING_RATE, SEQUENCE_LENGTH);
    if(m_attention == NULL) {
        Print("Failed to initialize attention mechanism");
        return false;
    }
    
    // Initialize position manager
    m_posManager = new CPositionManager();
    if(m_posManager == NULL) {
        Print("Failed to initialize position manager");
        return false;
    }
    
    // Initialize risk manager
    m_riskManager = new CRiskManager();
    if(m_riskManager == NULL) {
        Print("Failed to initialize risk manager");
        return false;
    }
    
    // Initialize signal aggregator
    m_signalAggregator = new CSignalAggregator();
    if(m_signalAggregator == NULL) {
        Print("Failed to initialize signal aggregator");
        return false;
    }
    
    // Initialize market regime detector
    m_regimeDetector = new CMarketRegimeDetector();
    if(m_regimeDetector == NULL) {
        Print("Failed to initialize market regime detector");
        return false;
    }
    
    // Initialize performance analytics
    m_performance = new CPerformanceAnalytics();
    if(m_performance == NULL) {
        Print("Failed to initialize performance analytics");
        return false;
    }
    
    // Set up risk manager parameters
    m_riskManager.SetBaseRisk(BASE_RISK_PER_TRADE);
    m_riskManager.SetMaxRisk(MAX_RISK_PER_TRADE);
    m_riskManager.SetMaxDailyLoss(MAX_DAILY_LOSS);
    m_riskManager.SetMaxDrawdown(MAX_ALLOWED_DD);
    m_riskManager.SetEmergencyDrawdown(EMERGENCY_CLOSE_DD);
    m_riskManager.SetDynamicRiskEnabled(ENABLE_DYNAMIC_RISK);
    
    // Set up signal aggregator
    m_signalAggregator.SetAttention(m_attention);
    m_signalAggregator.SetForecaster(m_priceForecaster);
    m_signalAggregator.SetFrequencyFilter(m_freqFilter);
    m_signalAggregator.SetThreshold(ATTENTION_THRESHOLD);
    
    return true;
}

//+------------------------------------------------------------------+
//| Process new tick                                                  |
//+------------------------------------------------------------------+
void CATFNetController::ProcessTick() {
    // Skip if EA is disabled
    if(!m_isEnabled) return;
    
    // Check for daily loss limit
    CheckDailyLossLimit();
    
    // Check for weekend closing
    if(CLOSE_ALL_FRIDAY && IsFridayCloseTime()) {
        CloseAllPositions("Friday closing time reached");
        return;
    }
    
    // Prevent weekend trades
    if(PREVENT_WEEKEND_TRADES && IsNearWeekend()) {
        return;
    }
    
    // Update position tracking
    m_posManager.UpdatePositions();
    
    // Manage existing positions
    ManagePositions();
    
    // Check if it's time to check for new entries
    datetime currentTime = TimeCurrent();
    if(currentTime > m_lastEntryCheckTime + ENTRY_CHECK_SECONDS) {
        // Update market data
        if(UpdateMarketData()) {
            // Detect market regime if enabled
            if(ENABLE_REGIME_DETECTION) {
                ENUM_MARKET_REGIME regime = m_regimeDetector.DetectRegime();
                m_regimeDetector.OptimizeForRegime(regime);
            }
            
            // Check for new entries
            if(m_posManager.GetPositionCount() < MAX_POSITIONS) {
                // Check if we're in a valid trading session
                if(!ENABLE_SESSION_TRADING || IsValidTradingSession()) {
                    // Generate signals
                    if(GenerateSignals()) {
                        // Get signal direction
                        ENUM_ORDER_TYPE direction = m_signalAggregator.GetSignalDirection();
                        
                        // Check if we should enter a trade
                        if(m_signalAggregator.ShouldEnterTrade()) {
                            // Calculate stop loss
                            double stopLoss = CalculateStopLoss(direction);
                            
                            // Calculate position size
                            double entryPrice = (direction == ORDER_TYPE_BUY) ? 
                                               SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                                               SymbolInfoDouble(_Symbol, SYMBOL_BID);
                            
                            double positionSize = m_riskManager.CalculatePositionSize(entryPrice, stopLoss, _Symbol);
                            
                            // Open position
                            OpenPosition(direction, positionSize, stopLoss);
                        }
                    }
                }
            }
        }
        
        m_lastEntryCheckTime = currentTime;
    }
    
    // Check if it's time to update weights
    if(currentTime > m_lastWeightUpdateTime + WEIGHT_UPDATE_HOURS * 3600) {
        UpdateWeights();
        m_lastWeightUpdateTime = currentTime;
    }
}

//+------------------------------------------------------------------+
//| Update market data                                                |
//+------------------------------------------------------------------+
bool CATFNetController::UpdateMarketData() {
    // Copy price data
    if(CopyClose(_Symbol, PERIOD_TIME, 0, SEQUENCE_LENGTH, m_closeArray) != SEQUENCE_LENGTH) {
        Print("Failed to copy close prices");
        return false;
    }
    
    if(CopyOpen(_Symbol, PERIOD_TIME, 0, SEQUENCE_LENGTH, m_openArray) != SEQUENCE_LENGTH) {
        Print("Failed to copy open prices");
        return false;
    }
    
    if(CopyHigh(_Symbol, PERIOD_TIME, 0, SEQUENCE_LENGTH, m_highArray) != SEQUENCE_LENGTH) {
        Print("Failed to copy high prices");
        return false;
    }
    
    if(CopyLow(_Symbol, PERIOD_TIME, 0, SEQUENCE_LENGTH, m_lowArray) != SEQUENCE_LENGTH) {
        Print("Failed to copy low prices");
        return false;
    }
    
    if(CopyTickVolume(_Symbol, PERIOD_TIME, 0, SEQUENCE_LENGTH, m_volumeArray) != SEQUENCE_LENGTH) {
        Print("Failed to copy tick volumes");
        return false;
    }
    
    // Update frequency features
    m_freqFeatures.UpdateFeatures();
    
    return true;
}

//+------------------------------------------------------------------+
//| Generate trading signals                                          |
//+------------------------------------------------------------------+
bool CATFNetController::GenerateSignals() {
    // Get current price
    double currentPrice = m_closeArray[SEQUENCE_LENGTH-1];
    
    // Get price forecasts
    double forecastPrice = m_priceForecaster.ForecastPrice();
    
    // Calculate predicted return
    double predictedReturn = (forecastPrice - currentPrice) / currentPrice;
    
    // Update the forecaster with actual price outcomes
    m_priceForecaster.UpdateCombiner(currentPrice);
    
    // Get technical indicators from ATFNet components
    double atrValue = 0;
    double emaValue = 0;
    double zScore = 0;
    
    // These would normally come from the ATFNet indicators
    // For now, we'll calculate them directly
    
    // Calculate ATR
    double atrSum = 0;
    for(int i = 0; i < ATR_PERIOD; i++) {
        double trueRange = MathMax(m_highArray[i] - m_lowArray[i],
                                  MathMax(MathAbs(m_highArray[i] - m_closeArray[i+1]),
                                         MathAbs(m_lowArray[i] - m_closeArray[i+1])));
        atrSum += trueRange;
    }
    atrValue = atrSum / ATR_PERIOD;
    
    // Calculate EMA
    emaValue = m_closeArray[0];
    double alpha = 2.0 / (TREND_EMA_PERIOD + 1.0);
    for(int i = 1; i < SEQUENCE_LENGTH; i++) {
        emaValue = alpha * m_closeArray[i] + (1.0 - alpha) * emaValue;
    }
    
    // Calculate Z-Score
    double sum = 0;
    for(int i = 0; i < Z_SCORE_PERIOD; i++) {
        sum += m_closeArray[i];
    }
    double mean = sum / Z_SCORE_PERIOD;
    
    double variance = 0;
    for(int i = 0; i < Z_SCORE_PERIOD; i++) {
        variance += MathPow(m_closeArray[i] - mean, 2);
    }
    variance /= Z_SCORE_PERIOD;
    double stdDev = MathSqrt(variance);
    
    if(stdDev > 0) {
        zScore = (currentPrice - mean) / stdDev;
    } else {
        zScore = 0;
    }
    
    // Calculate range position
    double rangeHigh = m_highArray[ArrayMaximum(m_highArray, 0, RANGE_LOOKBACK)];
    double rangeLow = m_lowArray[ArrayMinimum(m_lowArray, 0, RANGE_LOOKBACK)];
    double rangeSize = rangeHigh - rangeLow;
    double rangePosition = (rangeSize > 0) ? (currentPrice - rangeLow) / rangeSize : 0.5;
    
    // Create ATFNet features
    ATFNetFeatures features;
    features.price = currentPrice;
    features.atr = atrValue;
    features.zScore = zScore;
    features.trendStrength = (currentPrice - emaValue) / emaValue;
    features.rangePosition = rangePosition;
    features.volumeProfile = 0.5; // Placeholder
    features.patternStrength = 0.5; // Placeholder
    
    // Update frequency features
    m_attention.UpdateFrequencyFeatures(features);
    
    // Set signal parameters
    m_signalAggregator.SetPredictedReturn(predictedReturn);
    m_signalAggregator.SetFeatures(features);
    m_signalAggregator.SetTrendFilter(USE_TREND_FILTER, currentPrice, emaValue);
    
    return true;
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void CATFNetController::ManagePositions() {
    // Get current account state
    double currentDrawdown = m_riskManager.GetCurrentDrawdown();
    
    // Emergency close all positions if drawdown is too high
    if(currentDrawdown >= EMERGENCY_CLOSE_DD) {
        CloseAllPositions("Emergency drawdown protection triggered");
        m_isEnabled = false;
        Print("Trading stopped due to emergency drawdown protection: ", currentDrawdown, "%");
        return;
    }
    
    // Regular position management
    double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        
        // Get position details
        double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double stopLoss = PositionGetDouble(POSITION_SL);
        double positionSize = PositionGetDouble(POSITION_VOLUME);
        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        
        // Calculate position drawdown and profit
        double profitPercent = 0;
        double positionDrawdown = 0;
        
        if(posType == POSITION_TYPE_BUY) {
            profitPercent = (currentBid - entryPrice) / entryPrice * 100;
            positionDrawdown = (entryPrice - currentBid) / entryPrice * 100;
        } else {
            profitPercent = (entryPrice - currentAsk) / entryPrice * 100;
            positionDrawdown = (currentAsk - entryPrice) / entryPrice * 100;
        }
        
        // Close position if drawdown exceeds limit
        if(positionDrawdown > MAX_POSITION_DD) {
            m_trade.PositionClose(ticket);
            Print("Position closed due to exceeding max position drawdown: ", positionDrawdown, "%");
            continue;
        }
        
        // Get initial stop loss and calculate distances
        double initialStopLoss = m_posManager.GetInitialStopLoss(ticket);
        double initialStopDistance = MathAbs(entryPrice - initialStopLoss);
        
        // Calculate RR target percentages
        double rrTargetPercent = (initialStopDistance * RR_RATIO / entryPrice) * 100;
        double swingRRTargetPercent = (initialStopDistance * SWING_RR_RATIO / entryPrice) * 100;
        
        // Check if position reached first RR target
        if(!m_posManager.IsFirstTargetHit(ticket) && profitPercent >= rrTargetPercent) {
            m_posManager.SetFirstTargetHit(ticket, true);
            // Calculate partial close size
            double closeSize = positionSize * (FIRST_TARGET_CLOSE_PCT / 100.0);
            double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            closeSize = MathFloor(closeSize / lotStep) * lotStep;
            
            // Execute partial close if size is valid
            if(closeSize >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
                bool result = false;
                if(posType == POSITION_TYPE_BUY) {
                    result = m_trade.Sell(closeSize, _Symbol, 0, 0, 0, "First RR target hit");
                } else {
                    result = m_trade.Buy(closeSize, _Symbol, 0, 0, 0, "First RR target hit");
                }
                if(result) {
                    Print("First RR target hit - Partial close executed for ticket ", ticket);
                }
            }
        }
        
        // Check if position reached swing RR target
        if(m_posManager.IsFirstTargetHit(ticket) && !m_posManager.IsSwingTargetHit(ticket) && 
           profitPercent >= swingRRTargetPercent) {
            m_posManager.SetSwingTargetHit(ticket, true);
            m_posManager.SetTrailReferencePrice(ticket, posType == POSITION_TYPE_BUY ? currentBid : currentAsk);
            Print("Swing RR target hit - Starting trailing stop for ticket ", ticket);
        }
        
        // Handle trailing stop after swing target hit
        if(m_posManager.IsSwingTargetHit(ticket)) {
            double trailReferencePrice = m_posManager.GetTrailReferencePrice(ticket);
            
            // Update trail reference price
            if(posType == POSITION_TYPE_BUY) {
                if(currentBid > trailReferencePrice) {
                    m_posManager.SetTrailReferencePrice(ticket, currentBid);
                    trailReferencePrice = currentBid;
                }
            } else {
                if(currentAsk < trailReferencePrice) {
                    m_posManager.SetTrailReferencePrice(ticket, currentAsk);
                    trailReferencePrice = currentAsk;
                }
            }
            
            // Calculate trailing stop distance
            double atrValue = 0; // This would come from indicators
            double trailingStopDistance = MathMax(
                atrValue * TRAILING_STOP_ATR_MULT,
                MIN_TRAILING_STOP_PIPS * _Point
            );
            
            // Calculate new stop loss level
            double newStopLoss;
            if(posType == POSITION_TYPE_BUY) {
                newStopLoss = trailReferencePrice - trailingStopDistance;
                if(newStopLoss > stopLoss) {
                    m_trade.PositionModify(ticket, newStopLoss, 0);
                }
            } else {
                newStopLoss = trailReferencePrice + trailingStopDistance;
                if(newStopLoss < stopLoss || stopLoss == 0) {
                    m_trade.PositionModify(ticket, newStopLoss, 0);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Close all positions                                               |
//+------------------------------------------------------------------+
void CATFNetController::CloseAllPositions(string reason = "") {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        
        // Skip positions for other symbols
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) {
            continue;
        }
        
        // Close position
        m_trade.PositionClose(ticket);
        Print("Position closed. Ticket: ", ticket, " Reason: ", reason);
    }
}

//+------------------------------------------------------------------+
//| Calculate stop loss level based on ATR                            |
//+------------------------------------------------------------------+
double CATFNetController::CalculateStopLoss(ENUM_ORDER_TYPE orderType) {
    // Get current price
    double currentPrice = (orderType == ORDER_TYPE_BUY) ? 
                         SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                         SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Use ATR for stop loss calculation
    double atrValue = 0; // This would come from indicators
    double stopDistance = atrValue * ATR_MULTIPLIER;
    
    // Cap stop distance to max SL pips
    double maxSlDistance = MAX_SL_PIPS * _Point;
    stopDistance = MathMin(stopDistance, maxSlDistance);
    
    // Calculate stop loss level
    if(orderType == ORDER_TYPE_BUY) {
        return currentPrice - stopDistance;
    } else {
        return currentPrice + stopDistance;
    }
}

//+------------------------------------------------------------------+
//| Check if current time is within valid trading session             |
//+------------------------------------------------------------------+
bool CATFNetController::IsValidTradingSession() {
    // Get current hour
    MqlDateTime dt;
    TimeCurrent(dt);
    int currentHour = dt.hour;
    
    // Check Asian session
    bool inAsian = (TRADE_ASIAN_SESSION && 
                   currentHour >= ASIAN_START && 
                   currentHour < ASIAN_END);
    
    // Check European session
    bool inEuropean = (TRADE_EUROPEAN_SESSION && 
                      currentHour >= EUROPEAN_START && 
                      currentHour < EUROPEAN_END);
    
    // Check American session
    bool inAmerican = (TRADE_AMERICAN_SESSION && 
                      currentHour >= AMERICAN_START && 
                      currentHour < AMERICAN_END);
    
    return inAsian || inEuropean || inAmerican;
}

//+------------------------------------------------------------------+
//| Check if it's time to close positions on Friday                   |
//+------------------------------------------------------------------+
bool CATFNetController::IsFridayCloseTime() {
    MqlDateTime dt;
    TimeCurrent(dt);
    
    // Check if it's Friday and after the specified hour
    return (dt.day_of_week == 5 && dt.hour >= FRIDAY_CLOSE_HOUR);
}

//+------------------------------------------------------------------+
//| Check if we're near weekend (Friday evening)                      |
//+------------------------------------------------------------------+
bool CATFNetController::IsNearWeekend() {
    MqlDateTime dt;
    TimeCurrent(dt);
    
    // Check if it's Friday and after the specified hour - 1
    return (dt.day_of_week == 5 && dt.hour >= (FRIDAY_CLOSE_HOUR - 1));
}

//+------------------------------------------------------------------+
//| Check daily loss limit                                            |
//+------------------------------------------------------------------+
void CATFNetController::CheckDailyLossLimit() {
    if(m_riskManager.CheckDailyLossLimit()) {
        CloseAllPositions("Daily loss limit reached");
        m_isEnabled = false;
        Print("Daily loss limit reached. Trading stopped.");
    }
}

//+------------------------------------------------------------------+
//| Update attention weights                                          |
//+------------------------------------------------------------------+
void CATFNetController::UpdateWeights() {
    // This would normally be called after a trade is closed
    // For now, we'll just update the weights periodically
    
    // Create sample features (would normally be from closed trade)
    ATFNetFeatures features;
    features.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    features.atr = 0; // This would come from indicators
    features.zScore = 0; // This would come from indicators
    features.trendStrength = 0; // This would come from indicators
    features.rangePosition = 0.5; // This would come from indicators
    features.volumeProfile = 0.5; // Placeholder
    features.patternStrength = 0.5; // Placeholder
    
    // Update frequency features
    m_attention.UpdateFrequencyFeatures(features);
    
    // Calculate a sample trade result (would normally be actual P/L)
    double tradeResult = 0.1 * (MathRand() / 32767.0 * 2 - 1);
    
    // Update weights
    m_attention.UpdateWeights(features, tradeResult);
}

//+------------------------------------------------------------------+
//| Open a new position                                               |
//+------------------------------------------------------------------+
bool CATFNetController::OpenPosition(ENUM_ORDER_TYPE orderType, double positionSize, double stopLoss) {
    // Get entry price
    double entryPrice = (orderType == ORDER_TYPE_BUY) ? 
                       SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                       SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Open the position
    bool result = false;
    if(orderType == ORDER_TYPE_BUY) {
        result = m_trade.Buy(positionSize, _Symbol, 0, stopLoss);
    } else {
        result = m_trade.Sell(positionSize, _Symbol, 0, stopLoss);
    }
    
    // Track the position if opened successfully
    if(result) {
        ulong ticket = m_trade.ResultOrder();
        m_posManager.AddTracker(ticket);
        
        // Get forecaster weights for debug output
        double timeWeight = 0, freqWeight = 0;
        m_priceForecaster.GetCombinerWeights(timeWeight, freqWeight);
        
        string orderTypeStr = (orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL";
        Print(orderTypeStr, " position opened. Ticket: ", ticket, 
              " Size: ", positionSize, 
              " Entry: ", entryPrice, 
              " SL: ", stopLoss,
              " Forecaster Weights - Time: ", timeWeight,
              " Freq: ", freqWeight);
              
        // Record trade for performance analytics
        m_performance.RecordTradeOpen(ticket, orderType, entryPrice, stopLoss, positionSize);
    }
    
    return result;
}
