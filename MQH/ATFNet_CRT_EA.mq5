//+------------------------------------------------------------------+
//|                                                 ATFNet_CRT_EA.mq5 |
//|                                                                   |
//|                 Enhanced ATFNet with OptimizedCRT Strategy        |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property link      ""
#property version   "1.00"
#property strict

// Include necessary files
#include <Trade\Trade.mqh>
#include "ATFNet_CRT_Includes.mqh"
#include "PriceForecaster.mqh"
// Create an instance of CTrade for trade operations
CTrade trade;
//+------------------------------------------------------------------+
//| Expert Parameters                                                 |
//+------------------------------------------------------------------+|
//+------------------------------------------------------------------+
// ATFNet Parameters
input group "ATFNet Parameters"
input int               SEQUENCE_LENGTH = 60;                                   // Input sequence length
input int               PREDICTION_LENGTH = 20;                                 // Prediction length
input double            PREDICTION_THRESHOLD = 0.0015;                          // Prediction threshold (%)
input bool              USE_FREQUENCY_DOMAIN = true;                            // Use frequency domain analysis
input double            TIME_DOMAIN_WEIGHT = 0.5;                               // Time domain weight (0-1)
input double            FREQ_DOMAIN_WEIGHT = 0.5;                               // Frequency domain weight (0-1)

// Position Management
input group "Position Management"
input int               MAX_POSITIONS    = 3;                                   // Maximum allowed positions
input double            minLots          = 1;                                   // Min Lots
input double            maxLots          = 80;                                  // Max Lots
input double            MaxDailyLoss     = 5.0;                                 // Maximum daily loss (%)
input double            MaxPositionDrawdownPercent = 2.0;                       // Max position drawdown (%)
input double            MaxAllowedDrawdownPercent = 6.0;                        // Max allowed drawdown (%)
input double            MAX_SL_PIPS      = 200.0;                               // Maximum Stop Loss in Pips
input double            minProfitPercent = 0.015;                               // Min Profit % to open trades

// Dynamic Risk Management
input group "Dynamic Risk Management"
input double BaseRiskPerTrade = 1.0;      // Base risk per trade (%)
input double MaxRiskPerTrade = 3.0;       // Maximum risk per trade (%)
input double ProfitThreshold = 5.0;       // Profit threshold for risk increase (%)
input double RiskIncreaseStep = 0.5;      // Risk increase step (%)
input bool EnableDynamicRisk = true;      // Enable dynamic risk management

// Add these input parameters under "Dynamic Risk Management" group
input group "Drawdown Protection"
input double            MaxFloatingDrawdown = 12.0;        // Maximum floating drawdown (%)
input double            EmergencyCloseDrawdown = 15.0;     // Emergency close all positions drawdown (%)
input double            MaxPositionDrawdown = 4.0;         // Maximum drawdown per position (%)
input bool              ScalePositionSize = true;          // Scale position size based on drawdown
input double            DrawdownScaleFactor = 0.5;         // Position size scale factor when in drawdown

// Attention Mechanism Parameters
input group "Attention Parameters"
input double            ATTENTION_THRESHOLD = 0.6;                              // Minimum attention score for trade
input double            LEARNING_RATE = 0.01;                                   // Learning rate for weight updates
input int               WEIGHT_UPDATE_HOURS = 1;                                // Hours between weight updates

// Strategy Parameters
input group "Strategy Parameters"
input int               NBars            = 1;                                   // Copy Last N Bars
input int               rangeLookback    = 10;                                  // Range Lookback Period
input int               entrySeconds     = 15;                                  // Entry Check Interval (5-60s)
input ENUM_TIMEFRAMES   PeriodTime       = PERIOD_CURRENT;                      // Timeframe for Calculations
input double            TargetProfit     = 50.0;                                // Target Profit (currency)
input double            SweepPips        = 1;                                   // Sweep Pips for High/Low

// Technical Indicators
input group "Technical Indicators"
input int               atrPeriod        = 14;                                  // ATR Period
input double            atrMultiplier    = 1.0;                                 // ATR Multiplier
input bool              UseTrendFilter   = true;                                // Use EMA Trend Filter
input int               TrendEMAPeriod   = 50;                                  // EMA Period
input ENUM_APPLIED_PRICE PriceSource    = PRICE_CLOSE;                          // Price Source for EMA

// Pattern Recognition
input group "Pattern Recognition"
input int               Candlerates      = 3;                                   // Min candlesticks required
input double            BodyCan          = 0.3;                                 // Body/Total Ratio
input double            WickCan          = 0.3;                                 // Wick/Total Ratio
input double            DojiCan          = 0.1;                                 // Doji Body/Total Ratio
input double            feature_attention = 0.5;                                // Feature Attention
input int               number_patterns = 7;                                    // Number of patterns to check

// Additional Pattern Parameters
input group "Additional Pattern Parameters"
input double PinbarRatio      = 0.3;    // Pinbar nose ratio
input double StarGapPercent   = 0.1;    // Star pattern gap percent
input double ThreeBarPercent  = 0.5;    // Three-bar pattern percent
input double InsideBarMaxSize = 0.8;    // Inside bar maximum size ratio
input double OutsideBarMinSize = 1.2;   // Outside bar minimum size ratio

// Z-Score Settings
input group "Z-Score Settings"
input bool              UseWeightedZScore = true;                               // Use Weighted Z-Score
input int               zScorePeriod     = 20;                                  // Z-Score Period
input double            zScoreThreshold  = 1.5;                                 // Z-Score Threshold
input double            AdjustbaseVolume = 0.1;                                 // Volume Adjustment
input double            NormalizeDoubleNu = 1;                                  // Normalization Digits

// Session Trading
input group "Session Trading"
input bool              EnableSessionTrading = true;                            // Enable Session Trading
input bool              TradeAsianSession   = true;                             // Trade Asian Session
input bool              TradeEuropeanSession = true;                            // Trade European Session
input bool              TradeAmericanSession = true;                            // Trade American Session
input int               AsianSessionStart   = 0;                                // Asian Session Start (Server Hour)
input int               AsianSessionEnd     = 9;                                // Asian Session End (Server Hour)
input int               EuropeanSessionStart = 7;                               // European Session Start (Server Hour)
input int               EuropeanSessionEnd   = 16;                              // European Session End (Server Hour)
input int               AmericanSessionStart = 13;                              // American Session Start (Server Hour)
input int               AmericanSessionEnd   = 22;                              // American Session End (Server Hour)

// Market Hours
input group "Market Hours"
input int MarketCloseBuffer = 10;     // Minutes before market close to start closing positions
input bool CloseAllOnFriday = true;   // Close all positions before weekend
input int FridayCloseHour = 21;       // Hour to close positions on Friday (server time)
input bool PreventWeekendTrades = true; // Prevent new trades near weekend

// Trailing Stop Settings
input group "Position Management Settings"
input double            RR_Ratio = 1.5;                    // Risk:Reward Ratio for first target
input double            SwingRR_Ratio = 3.0;              // Risk:Reward Ratio for swing target
input double            FirstTargetClosePercent = 70.0;   // Percentage to close at first RR target
input double            TrailingStopATRMultiplier = 2.0;  // Trailing stop ATR multiplier
input double            MinTrailingStopPips = 10.0;       // Minimum trailing stop in pips

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
double initial_balance = 0;
double rangeHigh, rangeLow;
double atrValue, emaValue;
double zScore;
datetime lastRangeCalcTime;
datetime lastEntryCheckTime;
datetime lastWeightUpdate;
bool isEAStopped = false;
datetime tradingStopTime = 0;
const int HOURS_TO_WAIT = 22;  // Trading stop duration after emergency stop (22 hours)
double riskAmount;

// Price data arrays
double closeArray[];
double openArray[];
double highArray[];
double lowArray[];
long volumeArray[];  // Changed from double to long

// Global objects
CAttentionMechanism* attention = NULL;
COptimizedIndicators* indicators = NULL;
CRangeCalculator* rangeCalc = NULL;
CPositionManager* posManager = NULL;
CPatternDetector* patternDetector = NULL;
CATFNetPredictor* atfNetPredictor = NULL;
CPriceForecaster* priceForecaster = NULL;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit() {
    // Parameter validation
    if(zScorePeriod < 2) {
        Print("Invalid zScorePeriod value. Must be at least 2.");
        return INIT_PARAMETERS_INCORRECT;
    }
    if(zScoreThreshold <= 0.0) {
        Print("Invalid zScoreThreshold value. Must be greater than 0.");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    // Initialize global objects
    if(attention != NULL) delete attention;
    attention = new CAttentionMechanism(LEARNING_RATE);
    
    if(indicators != NULL) delete indicators;
    indicators = new COptimizedIndicators(atrPeriod, TrendEMAPeriod, zScorePeriod);
    
    if(rangeCalc != NULL) delete rangeCalc;
    rangeCalc = new CRangeCalculator(rangeLookback);
    
    if(posManager != NULL) delete posManager;
    posManager = new CPositionManager();
    
    if(patternDetector != NULL) delete patternDetector;
    patternDetector = new CPatternDetector(BodyCan, WickCan, DojiCan, PinbarRatio, 
                                         InsideBarMaxSize, OutsideBarMinSize);
    
    if(atfNetPredictor != NULL) delete atfNetPredictor;
    atfNetPredictor = new CATFNetPredictor(SEQUENCE_LENGTH, PREDICTION_LENGTH, 
                                         TIME_DOMAIN_WEIGHT, FREQ_DOMAIN_WEIGHT);
    
    if(priceForecaster != NULL) delete priceForecaster;
    priceForecaster = new CPriceForecaster(PREDICTION_LENGTH, SEQUENCE_LENGTH);
    
    // Initialize arrays
    ArrayResize(closeArray, SEQUENCE_LENGTH);
    ArrayResize(openArray, SEQUENCE_LENGTH);
    ArrayResize(highArray, SEQUENCE_LENGTH);
    ArrayResize(lowArray, SEQUENCE_LENGTH);
    ArrayResize(volumeArray, SEQUENCE_LENGTH);
    ArrayInitialize(closeArray, 0);
    ArrayInitialize(openArray, 0);
    ArrayInitialize(highArray, 0);
    ArrayInitialize(lowArray, 0);
    ArrayInitialize(volumeArray, 0);
    
    // Initialize other variables
    isEAStopped = false;
    lastRangeCalcTime = 0;
    lastEntryCheckTime = 0;
    lastWeightUpdate = 0;
    initial_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    if(tradingStopTime > 0) {
        datetime currentTime = TimeCurrent();
        if(currentTime < tradingStopTime + HOURS_TO_WAIT * 3600) {
            isEAStopped = true;
            Print("EA initialized while in cool-off period. Trading remains paused.");
        } else {
            // Reset if the waiting period has already passed
            tradingStopTime = 0;
            isEAStopped = false;
        }
    }
    
    // Set up trade object
    trade.SetDeviationInPoints(10); // 1 pip deviation for market orders
    trade.SetTypeFilling(ORDER_FILLING_FOK);
    trade.SetExpertMagicNumber(123456); // Set your magic number
    
    Print("=== ATFNet with CRT Strategy EA Initialized ===");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Clean up objects
    if(attention != NULL) {
        delete attention;
        attention = NULL;
    }
    
    if(indicators != NULL) {
        delete indicators;
        indicators = NULL;
    }
    
    if(rangeCalc != NULL) {
        delete rangeCalc;
        rangeCalc = NULL;
    }
    
    if(posManager != NULL) {
        delete posManager;
        posManager = NULL;
    }
    
    if(patternDetector != NULL) {
        delete patternDetector;
        patternDetector = NULL;
    }
    
    if(atfNetPredictor != NULL) {
        delete atfNetPredictor;
        atfNetPredictor = NULL;
    }
    
    if(priceForecaster != NULL) {
        delete priceForecaster;
        priceForecaster = NULL;
    }
    
    Print("=== ATFNet with CRT Strategy EA Deinitialized ===");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick() {
    // Skip if EA is stopped
    if(isEAStopped) {
        return;
    }
    
    // Check for daily loss limit
    CheckDailyLossLimit();
    
    // Check for weekend closing
    if(CloseAllOnFriday && IsFridayCloseTime()) {
        CloseAllPositions("Friday closing time reached");
        return;
    }
    
    // Prevent weekend trades
    if(PreventWeekendTrades && IsNearWeekend()) {
        return;
    }
    
    // Update position tracking
    posManager.UpdatePositions();
    
    // Manage existing positions
    ManagePositions();
    
    // Check if it's time to update price range
    datetime currentTime = TimeCurrent();
    if(currentTime > lastRangeCalcTime + PeriodSeconds(PeriodTime)) {
        UpdatePriceRange();
    }
    
    // Check if it's time to check for new entries
    if(currentTime > lastEntryCheckTime + entrySeconds) {
        CheckForEntries();
    }
    
    // Check if it's time to update attention weights
    if(currentTime > lastWeightUpdate + WEIGHT_UPDATE_HOURS * 3600) {
        UpdateAttentionWeights();
    }
}

//+------------------------------------------------------------------+
//| Update price range                                                |
//+------------------------------------------------------------------+
void UpdatePriceRange() {
    // Update range calculation
    if(rangeCalc.UpdateRange(true)) {
        rangeHigh = rangeCalc.GetRangeHigh();
        rangeLow = rangeCalc.GetRangeLow();
        lastRangeCalcTime = TimeCurrent();
    }
}

//+------------------------------------------------------------------+
//| Check for new entry opportunities                                 |
//+------------------------------------------------------------------+
void CheckForEntries() {
    // Add drawdown check before allowing new positions
    double currentDrawdown = CalculateDrawdownPercent();
    
    // Skip new entries if drawdown is too high
    if(currentDrawdown >= MaxFloatingDrawdown) {
        return;
    }
    
    // Skip if maximum positions reached
    if(posManager.GetPositionCount() >= MAX_POSITIONS) {
        return;
    }
    
    // Skip if not in trading session
    if(EnableSessionTrading && !IsValidTradingSession()) {
        return;
    }
    
    // Update price data arrays
    UpdatePriceArrays();
    
    // Skip if we don't have enough data
    if(!HaveEnoughData()) {
        return;
    }
    
    // Get both predictions with enhanced RNN forecasting
    double atfNetPrediction = atfNetPredictor.PredictNextPrice(closeArray);
    double forecastPrice = priceForecaster.ForecastPrice();  // Now includes RNN prediction
    
    double currentPrice = closeArray[SEQUENCE_LENGTH-1];
    
    // Calculate combined prediction
    double atfNetReturn = (atfNetPrediction - currentPrice) / currentPrice;
    double forecastReturn = (forecastPrice - currentPrice) / currentPrice;
    
    // Average the predictions with more weight to RNN-enhanced forecast
    double predictedReturn = (atfNetReturn + 2 * forecastReturn) / 3.0;
    
    // Update the forecaster with actual price outcomes
    priceForecaster.UpdateCombiner(currentPrice);
    
    // Get technical indicators
    atrValue = indicators.GetATR();
    emaValue = indicators.GetEMA();
    zScore = indicators.GetZScore(UseWeightedZScore);
    
    // Calculate range position
    double rangePosition = rangeCalc.GetPositionInRange(currentPrice);
    
    // Detect patterns
    MqlRates rates[];
    CopyRates(_Symbol, PeriodTime, 0, 3, rates);
    int patterns = patternDetector.DetectPatterns(rates, 3);
    double patternStrength = patternDetector.CalculatePatternStrength(patterns);
    
    // Create attention features
    AttentionFeatures features;
    features.price = currentPrice;
    features.atr = atrValue;
    features.zScore = zScore;
    features.trendStrength = (currentPrice - emaValue) / emaValue;
    features.rangePosition = rangePosition;
    features.volumeProfile = 0.5; // Placeholder, could be calculated from volume data
    features.patternStrength = patternStrength;
    
    // Calculate attention score
    double attentionScore = attention.CalculateScore(features);
    
    // Check trend filter if enabled
    bool trendFilterPassed = true;
    if(UseTrendFilter) {
        if(predictedReturn > 0 && currentPrice < emaValue) {
            trendFilterPassed = false;
        }
        else if(predictedReturn < 0 && currentPrice > emaValue) {
            trendFilterPassed = false;
        }
    }
    
    // Trading logic
    if(attentionScore >= ATTENTION_THRESHOLD && trendFilterPassed) {
        // Long signal
        if(predictedReturn > PREDICTION_THRESHOLD) {
            OpenPosition(ORDER_TYPE_BUY);
        }
        // Short signal
        else if(predictedReturn < -PREDICTION_THRESHOLD) {
            OpenPosition(ORDER_TYPE_SELL);
        }
    }
    
    lastEntryCheckTime = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Open a new position                                               |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE orderType) {
    // Calculate position size based on risk
    double riskAmount = CalculateRiskAmount();
    double stopLossPrice = CalculateStopLoss(orderType);
    double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    if(orderType == ORDER_TYPE_SELL) {
        entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    }
    
    // Calculate risk per unit
    double riskPerUnit = MathAbs(entryPrice - stopLossPrice);
    
    // Skip if stop loss is invalid
    if(riskPerUnit <= 0) {
        Print("Invalid stop loss calculation. Entry: ", entryPrice, " SL: ", stopLossPrice);
        return;
    }
    
    // Calculate position size
    double positionSize = riskAmount / riskPerUnit;
    
    // Apply min/max lot constraints
    positionSize = MathMax(positionSize, minLots);
    positionSize = MathMin(positionSize, maxLots);
    
    // Normalize position size to lot step
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    positionSize = MathFloor(positionSize / lotStep) * lotStep;
    
    // Skip if position size is too small
    if(positionSize < SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
        Print("Position size too small: ", positionSize);
        return;
    }
    
    // Open the position
    bool result = false;
    if(orderType == ORDER_TYPE_BUY) {
        result = trade.Buy(positionSize, _Symbol, 0, stopLossPrice);
    } else {
        result = trade.Sell(positionSize, _Symbol, 0, stopLossPrice);
    }
    
    // Track the position if opened successfully
    if(result) {
        ulong ticket = trade.ResultOrder();
        posManager.AddTracker(ticket);
        
        // Get forecaster weights for debug output
        double timeWeight, freqWeight;
        priceForecaster.GetCombinerWeights(timeWeight, freqWeight);
        
        string orderTypeStr = (orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL";
        Print(orderTypeStr, " position opened. Ticket: ", ticket, 
              " Size: ", positionSize, 
              " Entry: ", entryPrice, 
              " SL: ", stopLossPrice,
              " Forecaster Weights - Time: ", timeWeight,
              " Freq: ", freqWeight);
    }
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void ManagePositions() {
    // Get current account state
    double currentDrawdown = CalculateDrawdownPercent();
    
    // Emergency close all positions if drawdown is too high
    if(currentDrawdown >= EmergencyCloseDrawdown) {
        CloseAllPositions("Emergency drawdown protection triggered");
        isEAStopped = true;
        tradingStopTime = TimeCurrent();
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
        if(positionDrawdown > MaxPositionDrawdown) {
            trade.PositionClose(ticket);
            Print("Position closed due to exceeding max position drawdown: ", positionDrawdown, "%");
            continue;
        }
        
        // Get initial stop loss and calculate distances
        double initialStopLoss = posManager.GetInitialStopLoss(ticket);
        double initialStopDistance = MathAbs(entryPrice - initialStopLoss);
        
        // Calculate RR target percentages
        double rrTargetPercent = (initialStopDistance * RR_Ratio / entryPrice) * 100;
        double swingRRTargetPercent = (initialStopDistance * SwingRR_Ratio / entryPrice) * 100;
        
        // Check if position reached first RR target
        if(!posManager.IsFirstTargetHit(ticket) && profitPercent >= rrTargetPercent) {
            posManager.SetFirstTargetHit(ticket, true);
            // Calculate partial close size
            double closeSize = positionSize * (FirstTargetClosePercent / 100.0);
            double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            closeSize = MathFloor(closeSize / lotStep) * lotStep;
            
            // Execute partial close if size is valid
            if(closeSize >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
                bool result = false;
                if(posType == POSITION_TYPE_BUY) {
                    result = trade.Sell(closeSize, _Symbol, 0, 0, 0, "First RR target hit");
                } else {
                    result = trade.Buy(closeSize, _Symbol, 0, 0, 0, "First RR target hit");
                }
                if(result) {
                    Print("First RR target hit - Partial close executed for ticket ", ticket);
                }
            }
        }
        
        // Check if position reached swing RR target
        if(posManager.IsFirstTargetHit(ticket) && !posManager.IsSwingTargetHit(ticket) && 
           profitPercent >= swingRRTargetPercent) {
            posManager.SetSwingTargetHit(ticket, true);
            posManager.SetTrailReferencePrice(ticket, posType == POSITION_TYPE_BUY ? currentBid : currentAsk);
            Print("Swing RR target hit - Starting trailing stop for ticket ", ticket);
        }
        
        // Handle trailing stop after swing target hit
        if(posManager.IsSwingTargetHit(ticket)) {
            double trailReferencePrice = posManager.GetTrailReferencePrice(ticket);
            
            // Update trail reference price
            if(posType == POSITION_TYPE_BUY) {
                if(currentBid > trailReferencePrice) {
                    posManager.SetTrailReferencePrice(ticket, currentBid);
                    trailReferencePrice = currentBid;
                }
            } else {
                if(currentAsk < trailReferencePrice) {
                    posManager.SetTrailReferencePrice(ticket, currentAsk);
                    trailReferencePrice = currentAsk;
                }
            }
            
            // Calculate trailing stop distance
            double atrValue = indicators.GetATR();
            double trailingStopDistance = MathMax(
                atrValue * TrailingStopATRMultiplier,
                MinTrailingStopPips * _Point
            );
            
            // Calculate new stop loss level
            double newStopLoss;
            if(posType == POSITION_TYPE_BUY) {
                newStopLoss = trailReferencePrice - trailingStopDistance;
                if(newStopLoss > stopLoss) {
                    trade.PositionModify(ticket, newStopLoss, 0);
                }
            } else {
                newStopLoss = trailReferencePrice + trailingStopDistance;
                if(newStopLoss < stopLoss || stopLoss == 0) {
                    trade.PositionModify(ticket, newStopLoss, 0);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Close all positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason = "") {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        
        // Skip positions for other symbols
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) {
            continue;
        }
        
        // Close position
        trade.PositionClose(ticket);
        Print("Position closed. Ticket: ", ticket, " Reason: ", reason);
    }
}

//+------------------------------------------------------------------+
//| Calculate risk amount based on account balance                    |
//+------------------------------------------------------------------+
double CalculateRiskAmount() {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double currentRisk = BaseRiskPerTrade;
    
    // Calculate current drawdown
    double currentDrawdown = CalculateDrawdownPercent();
    
    // Scale risk based on drawdown if enabled
    if(ScalePositionSize && currentDrawdown > 0) {
        currentRisk *= MathMax(0.1, 1 - (currentDrawdown * DrawdownScaleFactor / MaxFloatingDrawdown));
    }
    
    // Adjust risk based on account performance if dynamic risk is enabled
    if(EnableDynamicRisk) {
        double growthPercent = (balance - initial_balance) / initial_balance * 100;
        if(growthPercent > ProfitThreshold) {
            int thresholdCrosses = (int)(growthPercent / ProfitThreshold);
            currentRisk = MathMin(
                BaseRiskPerTrade + thresholdCrosses * RiskIncreaseStep,
                MaxRiskPerTrade
            );
        }
    }
    
    return balance * (currentRisk / 100);
}

//+------------------------------------------------------------------+
//| Calculate stop loss level based on ATR                            |
//+------------------------------------------------------------------+
double CalculateStopLoss(ENUM_ORDER_TYPE orderType) {
    // Get current price
    double currentPrice = (orderType == ORDER_TYPE_BUY) ? 
                         SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                         SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Use ATR for stop loss calculation
    double atrValue = indicators.GetATR();
    double stopDistance = atrValue * atrMultiplier;
    
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
bool IsValidTradingSession() {
    // Get current hour
    MqlDateTime dt;
    TimeCurrent(dt);
    int currentHour = dt.hour;
    
    // Check Asian session
    bool inAsian = (TradeAsianSession && 
                   currentHour >= AsianSessionStart && 
                   currentHour < AsianSessionEnd);
    
    // Check European session
    bool inEuropean = (TradeEuropeanSession && 
                      currentHour >= EuropeanSessionStart && 
                      currentHour < EuropeanSessionEnd);
    
    // Check American session
    bool inAmerican = (TradeAmericanSession && 
                      currentHour >= AmericanSessionStart && 
                      currentHour < AmericanSessionEnd);
    
    return inAsian || inEuropean || inAmerican;
}

//+------------------------------------------------------------------+
//| Check if it's time to close positions on Friday                   |
//+------------------------------------------------------------------+
bool IsFridayCloseTime() {
    MqlDateTime dt;
    TimeCurrent(dt);
    
    // Check if it's Friday and after the specified hour
    return (dt.day_of_week == 5 && dt.hour >= FridayCloseHour);
}

//+------------------------------------------------------------------+
//| Check if we're near weekend (Friday evening)                      |
//+------------------------------------------------------------------+
bool IsNearWeekend() {
    MqlDateTime dt;
    TimeCurrent(dt);
    
    // Check if it's Friday and after the specified hour - 1
    return (dt.day_of_week == 5 && dt.hour >= (FridayCloseHour - 1));
}

//+------------------------------------------------------------------+
//| Check daily loss limit                                            |
//+------------------------------------------------------------------+
void CheckDailyLossLimit() {
    static datetime lastDayChecked = 0;
    static double dayHighBalance = 0;
    
    // Get current time
    MqlDateTime dt;
    datetime currentTime = TimeCurrent(dt);
    
    // Check if it's a new day
    if(lastDayChecked == 0) {
        dayHighBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        lastDayChecked = currentTime;
    }
    else {
        MqlDateTime lastDt;
        TimeToStruct(lastDayChecked, lastDt);
        
        // Compare day, month, and year
        if(dt.day != lastDt.day || 
           dt.mon != lastDt.mon || 
           dt.year != lastDt.year) {
            // Reset daily tracking
            dayHighBalance = AccountInfoDouble(ACCOUNT_BALANCE);
            lastDayChecked = currentTime;
        }
    }
    
    // Update day high balance
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(currentBalance > dayHighBalance) {
        dayHighBalance = currentBalance;
    }
    
    // Check if daily loss limit is reached using the same drawdown calculation
    double dailyDrawdownPercent = CalculateDrawdownPercent();
    if(dailyDrawdownPercent > MaxDailyLoss) {
        CloseAllPositions("Daily loss limit reached");
        isEAStopped = true;
        tradingStopTime = currentTime;
        Print("Daily loss limit reached. Trading stopped for ", HOURS_TO_WAIT, " hours.");
    }
}

//+------------------------------------------------------------------+
//| Update price data arrays                                          |
//+------------------------------------------------------------------+
void UpdatePriceArrays() {
    // Copy price data
    CopyClose(_Symbol, PeriodTime, 0, SEQUENCE_LENGTH, closeArray);
    CopyOpen(_Symbol, PeriodTime, 0, SEQUENCE_LENGTH, openArray);
    CopyHigh(_Symbol, PeriodTime, 0, SEQUENCE_LENGTH, highArray);
    CopyLow(_Symbol, PeriodTime, 0, SEQUENCE_LENGTH, lowArray);
    CopyTickVolume(_Symbol, PeriodTime, 0, SEQUENCE_LENGTH, volumeArray);
    
    // Reverse arrays to have most recent data at the end
    ArrayReverse(closeArray);
    ArrayReverse(openArray);
    ArrayReverse(highArray);
    ArrayReverse(lowArray);
    ArrayReverse(volumeArray);
}

//+------------------------------------------------------------------+
//| Check if we have enough data for analysis                         |
//+------------------------------------------------------------------+
bool HaveEnoughData() {
    return (ArraySize(closeArray) >= SEQUENCE_LENGTH && closeArray[0] > 0);
}

//+------------------------------------------------------------------+
//| Update attention weights based on trade results                   |
//+------------------------------------------------------------------+
void UpdateAttentionWeights() {
    // This would normally be called after a trade is closed
    // For now, we'll just update the weights periodically
    
    // Create sample features (would normally be from closed trade)
    AttentionFeatures features;
    features.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    features.atr = indicators.GetATR();
    features.zScore = indicators.GetZScore(UseWeightedZScore);
    features.trendStrength = (features.price - indicators.GetEMA()) / indicators.GetEMA();
    features.rangePosition = rangeCalc.GetPositionInRange(features.price);
    features.volumeProfile = 0.5; // Placeholder
    
    // Detect patterns
    MqlRates rates[];
    CopyRates(_Symbol, PeriodTime, 0, 3, rates);
    int patterns = patternDetector.DetectPatterns(rates, 3);
    features.patternStrength = patternDetector.CalculatePatternStrength(patterns);
    
    // Calculate a sample trade result (would normally be actual P/L)
    double tradeResult = 0.1 * (MathRand() / 32767.0 * 2 - 1);
    
    // Update weights
    attention.UpdateWeights(features, tradeResult);
    lastWeightUpdate = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Calculate drawdown percentage                                     |
//+------------------------------------------------------------------+
double CalculateDrawdownPercent() {
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // If initial balance is 0, use current balance (should only happen on first run)
    double baseBalance = (initial_balance > 0) ? initial_balance : AccountInfoDouble(ACCOUNT_BALANCE);
    
    return ((baseBalance - equity) / baseBalance) * 100;
}
