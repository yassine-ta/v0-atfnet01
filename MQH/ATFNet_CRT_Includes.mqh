//+------------------------------------------------------------------+
//|                                          ATFNet_CRT_Includes.mqh |
//|                                                                   |
//|                 Enhanced ATFNet with OptimizedCRT Strategy        |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property link      ""
#property version   "1.00"
#property strict

// Include necessary files
#include <Trade\Trade.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>

//+------------------------------------------------------------------+
//| Attention Features Structure                                      |
//+------------------------------------------------------------------+
struct AttentionFeatures {
    double price;           // Current price
    double atr;             // ATR value
    double zScore;          // Z-Score
    double trendStrength;   // Trend strength
    double rangePosition;   // Position in range
    double volumeProfile;   // Volume profile
    double patternStrength; // Pattern strength
};

//+------------------------------------------------------------------+
//| Attention Weights Structure                                       |
//+------------------------------------------------------------------+
struct AttentionWeights {
    double priceWeight;     // Weight for price
    double atrWeight;       // Weight for ATR
    double zScoreWeight;    // Weight for Z-Score
    double trendWeight;     // Weight for trend
    double rangeWeight;     // Weight for range
    double volumeWeight;    // Weight for volume
    double patternWeight;   // Weight for patterns
};

//+------------------------------------------------------------------+
//| Position Tracker Structure                                        |
//+------------------------------------------------------------------+
struct PositionTracker {
    ulong ticket;           // Position ticket
    double entryPrice;      // Entry price
    double stopLoss;        // Stop loss price
    double takeProfit;      // Take profit price
    double size;            // Position size
    datetime openTime;      // Open time
    double initialStopLoss;   // Initial stop loss price
    bool firstTargetHit;      // Flag for first RR target hit
    bool swingTargetHit;      // Flag for swing target hit
    double trailReferencePrice; // Reference price for trailing stop
};

//+------------------------------------------------------------------+
//| Attention Mechanism Class                                         |
//+------------------------------------------------------------------+
class CAttentionMechanism {
private:
    double m_learningRate;          // Learning rate for weight updates
    AttentionWeights m_weights;     // Current weights
    
    // Normalize weights using softmax
    void NormalizeWeights() {
        double total = m_weights.priceWeight + m_weights.atrWeight + 
                      m_weights.zScoreWeight + m_weights.trendWeight + 
                      m_weights.rangeWeight + m_weights.volumeWeight + 
                      m_weights.patternWeight;
        
        if(total > 0) {
            m_weights.priceWeight /= total;
            m_weights.atrWeight /= total;
            m_weights.zScoreWeight /= total;
            m_weights.trendWeight /= total;
            m_weights.rangeWeight /= total;
            m_weights.volumeWeight /= total;
            m_weights.patternWeight /= total;
        } else {
            // Initialize with equal weights if total is zero
            m_weights.priceWeight = 1.0/7.0;
            m_weights.atrWeight = 1.0/7.0;
            m_weights.zScoreWeight = 1.0/7.0;
            m_weights.trendWeight = 1.0/7.0;
            m_weights.rangeWeight = 1.0/7.0;
            m_weights.volumeWeight = 1.0/7.0;
            m_weights.patternWeight = 1.0/7.0;
        }
    }
    
    // Sigmoid function for normalization
    double Sigmoid(double x) {
        return 1.0 / (1.0 + MathExp(-x));
    }

public:
    // Constructor
    CAttentionMechanism(double learningRate = 0.01) {
        m_learningRate = learningRate;
        
        // Initialize weights
        m_weights.priceWeight = 1.0;
        m_weights.atrWeight = 1.0;
        m_weights.zScoreWeight = 1.0;
        m_weights.trendWeight = 1.0;
        m_weights.rangeWeight = 1.0;
        m_weights.volumeWeight = 1.0;
        m_weights.patternWeight = 1.0;
        
        NormalizeWeights();
    }
    
    // Calculate attention score
    double CalculateScore(AttentionFeatures &features) {
        // Normalize features using sigmoid
        double normPrice = Sigmoid(features.price / 1000.0); // Adjust scaling as needed
        double normATR = Sigmoid(features.atr / 0.01);
        double normZScore = Sigmoid(features.zScore);
        double normTrend = Sigmoid(features.trendStrength * 10.0);
        double normRange = Sigmoid(features.rangePosition * 2.0 - 1.0);
        double normVolume = Sigmoid(features.volumeProfile * 2.0 - 1.0);
        double normPattern = Sigmoid(features.patternStrength * 2.0);
        
        // Calculate weighted score
        double score = m_weights.priceWeight * normPrice +
                      m_weights.atrWeight * normATR +
                      m_weights.zScoreWeight * normZScore +
                      m_weights.trendWeight * normTrend +
                      m_weights.rangeWeight * normRange +
                      m_weights.volumeWeight * normVolume +
                      m_weights.patternWeight * normPattern;
        
        return score;
    }
    
    // Update weights based on trade result
    void UpdateWeights(AttentionFeatures &features, double tradeResult) {
        // Skip if trade result is zero
        if(tradeResult == 0) return;
        
        // Calculate feature importance based on trade result
        double priceImportance = features.price * tradeResult;
        double atrImportance = features.atr * tradeResult;
        double zScoreImportance = features.zScore * tradeResult;
        double trendImportance = features.trendStrength * tradeResult;
        double rangeImportance = features.rangePosition * tradeResult;
        double volumeImportance = features.volumeProfile * tradeResult;
        double patternImportance = features.patternStrength * tradeResult;
        
        // Update weights
        m_weights.priceWeight += m_learningRate * priceImportance;
        m_weights.atrWeight += m_learningRate * atrImportance;
        m_weights.zScoreWeight += m_learningRate * zScoreImportance;
        m_weights.trendWeight += m_learningRate * trendImportance;
        m_weights.rangeWeight += m_learningRate * rangeImportance;
        m_weights.volumeWeight += m_learningRate * volumeImportance;
        m_weights.patternWeight += m_learningRate * patternImportance;
        
        // Ensure weights are positive
        m_weights.priceWeight = MathMax(0.01, m_weights.priceWeight);
        m_weights.atrWeight = MathMax(0.01, m_weights.atrWeight);
        m_weights.zScoreWeight = MathMax(0.01, m_weights.zScoreWeight);
        m_weights.trendWeight = MathMax(0.01, m_weights.trendWeight);
        m_weights.rangeWeight = MathMax(0.01, m_weights.rangeWeight);
        m_weights.volumeWeight = MathMax(0.01, m_weights.volumeWeight);
        m_weights.patternWeight = MathMax(0.01, m_weights.patternWeight);
        
        // Normalize weights
        NormalizeWeights();
    }
    
    // Get current weights
    AttentionWeights GetWeights() {
        return m_weights;
    }
    
    // Set weights manually
    void SetWeights(AttentionWeights &weights) {
        m_weights = weights;
        NormalizeWeights();
    }
};

//+------------------------------------------------------------------+
//| Optimized Indicators Class                                        |
//+------------------------------------------------------------------+
class COptimizedIndicators {
private:
    int m_atrPeriod;        // ATR period
    int m_emaPeriod;        // EMA period
    int m_zScorePeriod;     // Z-Score period
    
    double m_atrBuffer[];   // ATR buffer
    double m_emaBuffer[];   // EMA buffer
    double m_priceBuffer[]; // Price buffer for Z-Score
    
    int m_atrHandle;        // ATR indicator handle
    int m_emaHandle;        // EMA indicator handle
    
    // Calculate weighted Z-Score
    double CalculateWeightedZScore(double &data[], int period) {
        if(ArraySize(data) < period) return 0;
        
        // Create weights (more recent data has higher weight)
        double weights[];
        ArrayResize(weights, period);
        for(int i = 0; i < period; i++) {
            weights[i] = period - i;
        }
        
        // Calculate weighted mean
        double weightSum = 0;
        double weightedSum = 0;
        
        for(int i = 0; i < period; i++) {
            weightedSum += data[i] * weights[i];
            weightSum += weights[i];
        }
        
        double weightedMean = weightedSum / weightSum;
        
        // Calculate weighted variance
        double weightedVar = 0;
        
        for(int i = 0; i < period; i++) {
            weightedVar += weights[i] * MathPow(data[i] - weightedMean, 2);
        }
        
        weightedVar /= weightSum;
        
        // Calculate weighted standard deviation
        double weightedStd = MathSqrt(weightedVar);
        
        // Calculate Z-Score
        if(weightedStd > 0) {
            return (data[0] - weightedMean) / weightedStd;
        } else {
            return 0;
        }
    }
    
    // Calculate standard Z-Score
    double CalculateStandardZScore(double &data[], int period) {
        if(ArraySize(data) < period) return 0;
        
        // Calculate mean
        double sum = 0;
        for(int i = 0; i < period; i++) {
            sum += data[i];
        }
        double mean = sum / period;
        
        // Calculate standard deviation
        double variance = 0;
        for(int i = 0; i < period; i++) {
            variance += MathPow(data[i] - mean, 2);
        }
        variance /= period;
        double std = MathSqrt(variance);
        
        // Calculate Z-Score
        if(std > 0) {
            return (data[0] - mean) / std;
        } else {
            return 0;
        }
    }

public:
    // Constructor
    COptimizedIndicators(int atrPeriod = 14, int emaPeriod = 50, int zScorePeriod = 20) {
        m_atrPeriod = atrPeriod;
        m_emaPeriod = emaPeriod;
        m_zScorePeriod = zScorePeriod;
        
        // Initialize buffers
        ArrayResize(m_atrBuffer, atrPeriod);
        ArrayResize(m_emaBuffer, 1);
        ArrayResize(m_priceBuffer, zScorePeriod);
        
        // Create indicator handles
        m_atrHandle = iATR(_Symbol, PERIOD_CURRENT, atrPeriod);
        m_emaHandle = iMA(_Symbol, PERIOD_CURRENT, emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
        
        // Check if handles are valid
        if(m_atrHandle == INVALID_HANDLE || m_emaHandle == INVALID_HANDLE) {
            Print("Failed to create indicator handles");
        }
    }
    
    // Destructor
    ~COptimizedIndicators() {
        // Release indicator handles
        if(m_atrHandle != INVALID_HANDLE) IndicatorRelease(m_atrHandle);
        if(m_emaHandle != INVALID_HANDLE) IndicatorRelease(m_emaHandle);
    }
    
    // Get ATR value
    double GetATR() {
        // Copy ATR values
        if(CopyBuffer(m_atrHandle, 0, 0, 1, m_atrBuffer) <= 0) {
            Print("Failed to copy ATR buffer");
            return 0;
        }
        
        return m_atrBuffer[0];
    }
    
    // Get EMA value
    double GetEMA() {
        // Copy EMA values
        if(CopyBuffer(m_emaHandle, 0, 0, 1, m_emaBuffer) <= 0) {
            Print("Failed to copy EMA buffer");
            return 0;
        }
        
        return m_emaBuffer[0];
    }
    
    // Get Z-Score
    double GetZScore(bool weighted = true) {
        // Copy price data
        if(CopyClose(_Symbol, PERIOD_CURRENT, 0, m_zScorePeriod, m_priceBuffer) <= 0) {
            Print("Failed to copy price data for Z-Score");
            return 0;
        }
        
        // Calculate Z-Score
        if(weighted) {
            return CalculateWeightedZScore(m_priceBuffer, m_zScorePeriod);
        } else {
            return CalculateStandardZScore(m_priceBuffer, m_zScorePeriod);
        }
    }
};

//+------------------------------------------------------------------+
//| Range Calculator Class                                            |
//+------------------------------------------------------------------+
class CRangeCalculator {
private:
    int m_lookbackPeriod;   // Lookback period for range calculation
    double m_rangeHigh;     // Range high
    double m_rangeLow;      // Range low
    
    double m_highBuffer[];  // High price buffer
    double m_lowBuffer[];   // Low price buffer

public:
    // Constructor
    CRangeCalculator(int lookbackPeriod = 10) {
        m_lookbackPeriod = lookbackPeriod;
        m_rangeHigh = 0;
        m_rangeLow = 0;
        
        // Initialize buffers
        ArrayResize(m_highBuffer, lookbackPeriod);
        ArrayResize(m_lowBuffer, lookbackPeriod);
    }
    
    // Update range calculation
    bool UpdateRange(bool useCurrentBar = true) {
        // Copy price data
        if(CopyHigh(_Symbol, PERIOD_CURRENT, 0, m_lookbackPeriod, m_highBuffer) <= 0) {
            Print("Failed to copy high price data for range calculation");
            return false;
        }
        
        if(CopyLow(_Symbol, PERIOD_CURRENT, 0, m_lookbackPeriod, m_lowBuffer) <= 0) {
            Print("Failed to copy low price data for range calculation");
            return false;
        }
        
        // Calculate range
        m_rangeHigh = m_highBuffer[ArrayMaximum(m_highBuffer)];
        m_rangeLow = m_lowBuffer[ArrayMinimum(m_lowBuffer)];
        
        return true;
    }
    
    // Get range high
    double GetRangeHigh() {
        return m_rangeHigh;
    }
    
    // Get range low
    double GetRangeLow() {
        return m_rangeLow;
    }
    
    // Get range size
    double GetRangeSize() {
        return m_rangeHigh - m_rangeLow;
    }
    
    // Get position in range (0-1)
    double GetPositionInRange(double price) {
        double rangeSize = GetRangeSize();
        
        if(rangeSize > 0) {
            return (price - m_rangeLow) / rangeSize;
        } else {
            return 0.5; // Default to middle if range size is zero
        }
    }
};

//+------------------------------------------------------------------+
//| Position Manager Class                                            |
//+------------------------------------------------------------------+
class CPositionManager {
private:
    CArrayDouble m_tickets;         // Array of position tickets
    CArrayDouble m_entryPrices;     // Array of entry prices
    CArrayDouble m_stopLosses;      // Array of stop losses
    CArrayDouble m_takeProfits;     // Array of take profits
    CArrayDouble m_sizes;           // Array of position sizes
    CArrayDouble m_openTimes;       // Array of open times
    CArrayDouble m_initialStopLosses;    // Array of initial stop losses
    CArrayDouble m_firstTargetHit;       // Array of first target hit flags
    CArrayDouble m_swingTargetHit;       // Array of swing target hit flags
    CArrayDouble m_trailReferencePrices; // Array of trail reference prices
    
    // Find index of ticket in array
    int FindTicketIndex(ulong ticket) {
        for(int i = 0; i < m_tickets.Total(); i++) {
            if((ulong)m_tickets.At(i) == ticket) {
                return i;
            }
        }
        
        return -1;
    }

public:
    // Constructor
    CPositionManager() {
        // Initialize arrays
        m_tickets.Clear();
        m_entryPrices.Clear();
        m_stopLosses.Clear();
        m_takeProfits.Clear();
        m_sizes.Clear();
        m_openTimes.Clear();
        m_initialStopLosses.Clear();
        m_firstTargetHit.Clear();
        m_swingTargetHit.Clear();
        m_trailReferencePrices.Clear();
    }
    
    // Add position tracker
    void AddTracker(ulong ticket) {
        // Skip if ticket already exists
        if(FindTicketIndex(ticket) >= 0) {
            return;
        }
        
        // Get position details
        if(!PositionSelectByTicket(ticket)) {
            Print("Failed to select position by ticket: ", ticket);
            return;
        }
        
        double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double stopLoss = PositionGetDouble(POSITION_SL);
        double takeProfit = PositionGetDouble(POSITION_TP);
        double size = PositionGetDouble(POSITION_VOLUME);
        datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
        
        // Add to arrays
        m_tickets.Add(ticket);
        m_entryPrices.Add(entryPrice);
        m_stopLosses.Add(stopLoss);
        m_takeProfits.Add(takeProfit);
        m_sizes.Add(size);
        m_openTimes.Add(openTime);
        m_initialStopLosses.Add(stopLoss);
        m_firstTargetHit.Add(0);
        m_swingTargetHit.Add(0);
        m_trailReferencePrices.Add(0);
    }
    
    // Update positions
    void UpdatePositions() {
        // Loop through all positions
        for(int i = m_tickets.Total() - 1; i >= 0; i--) {
            ulong ticket = (ulong)m_tickets.At(i);
            
            // Check if position still exists
            if(!PositionSelectByTicket(ticket)) {
                // Remove from arrays
                m_tickets.Delete(i);
                m_entryPrices.Delete(i);
                m_stopLosses.Delete(i);
                m_takeProfits.Delete(i);
                m_sizes.Delete(i);
                m_openTimes.Delete(i);
                m_initialStopLosses.Delete(i);
                m_firstTargetHit.Delete(i);
                m_swingTargetHit.Delete(i);
                m_trailReferencePrices.Delete(i);
                continue;
            }
            
            // Update position details
            double stopLoss = PositionGetDouble(POSITION_SL);
            double takeProfit = PositionGetDouble(POSITION_TP);
            double size = PositionGetDouble(POSITION_VOLUME);
            
            m_stopLosses.Update(i, stopLoss);
            m_takeProfits.Update(i, takeProfit);
            m_sizes.Update(i, size);
        }
    }
    
    // Get position count
    int GetPositionCount() {
        return m_tickets.Total();
    }
    
    // Get trail reference price
    double GetTrailReferencePrice(ulong ticket) {
        int index = FindTicketIndex(ticket);
        if(index >= 0) {
            return m_trailReferencePrices.At(index);
        }
        return 0;
    }
    
    // Set trail reference price
    void SetTrailReferencePrice(ulong ticket, double price) {
        int index = FindTicketIndex(ticket);
        if(index >= 0) {
            m_trailReferencePrices.Update(index, price);
        }
    }
    
    // Add new methods for two-stage targets
    bool IsFirstTargetHit(ulong ticket) {
        int index = FindTicketIndex(ticket);
        return index >= 0 ? m_firstTargetHit.At(index) > 0 : false;
    }
    
    void SetFirstTargetHit(ulong ticket, bool value) {
        int index = FindTicketIndex(ticket);
        if(index >= 0) m_firstTargetHit.Update(index, value ? 1 : 0);
    }
    
    bool IsSwingTargetHit(ulong ticket) {
        int index = FindTicketIndex(ticket);
        return index >= 0 ? m_swingTargetHit.At(index) > 0 : false;
    }
    
    void SetSwingTargetHit(ulong ticket, bool value) {
        int index = FindTicketIndex(ticket);
        if(index >= 0) m_swingTargetHit.Update(index, value ? 1 : 0);
    }

    // Add this method to get initial stop loss
    double GetInitialStopLoss(ulong ticket) {
        int index = FindTicketIndex(ticket);
        if(index >= 0) {
            return m_initialStopLosses.At(index);
        }
        return 0;
    }
};

//+------------------------------------------------------------------+
//| Pattern Detector Class                                            |
//+------------------------------------------------------------------+
class CPatternDetector {
private:
    double m_bodyRatio;         // Body/Total ratio for pattern detection
    double m_wickRatio;         // Wick/Total ratio for pattern detection
    double m_dojiRatio;         // Doji body/total ratio
    double m_pinbarRatio;       // Pinbar nose ratio
    double m_insideBarMaxSize;  // Inside bar maximum size ratio
    double m_outsideBarMinSize; // Outside bar minimum size ratio
    
    // Check if candle is bullish
    bool IsBullish(MqlRates &candle) {
        return candle.close > candle.open;
    }
    
    // Check if candle is bearish
    bool IsBearish(MqlRates &candle) {
        return candle.close < candle.open;
    }
    
    // Calculate body size
    double GetBodySize(MqlRates &candle) {
        return MathAbs(candle.close - candle.open);
    }
    
    // Calculate total size
    double GetTotalSize(MqlRates &candle) {
        return candle.high - candle.low;
    }
    
    // Calculate upper wick size
    double GetUpperWick(MqlRates &candle) {
        return candle.high - MathMax(candle.open, candle.close);
    }
    
    // Calculate lower wick size
    double GetLowerWick(MqlRates &candle) {
        return MathMin(candle.open, candle.close) - candle.low;
    }

public:
    // Pattern flags
    enum ENUM_PATTERN_FLAGS {
        PATTERN_NONE = 0,
        PATTERN_DOJI = 1,
        PATTERN_HAMMER = 2,
        PATTERN_HANGING_MAN = 4,
        PATTERN_SHOOTING_STAR = 8,
        PATTERN_BULLISH_ENGULFING = 16,
        PATTERN_BEARISH_ENGULFING = 32,
        PATTERN_MORNING_STAR = 64,
        PATTERN_EVENING_STAR = 128,
        PATTERN_PIERCING_LINE = 256,
        PATTERN_DARK_CLOUD_COVER = 512,
        PATTERN_BULLISH_HARAMI = 1024,
        PATTERN_BEARISH_HARAMI = 2048,
        PATTERN_BULLISH_KICKER = 4096,
        PATTERN_BEARISH_KICKER = 8192,
        PATTERN_THREE_WHITE_SOLDIERS = 16384,
        PATTERN_THREE_BLACK_CROWS = 32768,
        PATTERN_INSIDE_BAR = 65536,
        PATTERN_OUTSIDE_BAR = 131072
    };
    
    // Constructor
    CPatternDetector(double bodyRatio = 0.3, double wickRatio = 0.3, double dojiRatio = 0.1,
                    double pinbarRatio = 0.3, double insideBarMaxSize = 0.8, double outsideBarMinSize = 1.2) {
        m_bodyRatio = bodyRatio;
        m_wickRatio = wickRatio;
        m_dojiRatio = dojiRatio;
        m_pinbarRatio = pinbarRatio;
        m_insideBarMaxSize = insideBarMaxSize;
        m_outsideBarMinSize = outsideBarMinSize;
    }
    
    // Detect patterns in candles
    int DetectPatterns(MqlRates &rates[], int count) {
        // Check if we have enough candles
        if(count < 3) {
            return PATTERN_NONE;
        }
        
        int patterns = PATTERN_NONE;
        
        // Single candle patterns
        patterns |= DetectSingleCandlePatterns(rates[0]);
        
        // Two candle patterns
        patterns |= DetectTwoCandlePatterns(rates[0], rates[1]);
        
        // Three candle patterns
        patterns |= DetectThreeCandlePatterns(rates[0], rates[1], rates[2]);
        
        return patterns;
    }
    
    // Detect single candle patterns
    int DetectSingleCandlePatterns(MqlRates &candle) {
        int patterns = PATTERN_NONE;
        
        double bodySize = GetBodySize(candle);
        double totalSize = GetTotalSize(candle);
        double upperWick = GetUpperWick(candle);
        double lowerWick = GetLowerWick(candle);
        
        // Skip if total size is zero
        if(totalSize == 0) {
            return patterns;
        }
        
        // Doji
        if(bodySize / totalSize < m_dojiRatio) {
            patterns |= PATTERN_DOJI;
        }
        
        // Hammer (bullish)
        if(IsBullish(candle) && bodySize / totalSize < m_bodyRatio && 
           lowerWick / totalSize > m_wickRatio && upperWick / totalSize < m_wickRatio) {
            patterns |= PATTERN_HAMMER;
        }
        
        // Hanging Man (bearish)
        if(IsBearish(candle) && bodySize / totalSize < m_bodyRatio && 
           lowerWick / totalSize > m_wickRatio && upperWick / totalSize < m_wickRatio) {
            patterns |= PATTERN_HANGING_MAN;
        }
        
        // Shooting Star
        if(bodySize / totalSize < m_bodyRatio && 
           upperWick / totalSize > m_wickRatio && lowerWick / totalSize < m_wickRatio) {
            patterns |= PATTERN_SHOOTING_STAR;
        }
        
        return patterns;
    }
    
    // Detect two candle patterns
    int DetectTwoCandlePatterns(MqlRates &candle1, MqlRates &candle2) {
        int patterns = PATTERN_NONE;
        
        // Bullish Engulfing
        if(IsBearish(candle2) && IsBullish(candle1) && 
           candle1.close > candle2.open && candle1.open < candle2.close) {
            patterns |= PATTERN_BULLISH_ENGULFING;
        }
        
        // Bearish Engulfing
        if(IsBullish(candle2) && IsBearish(candle1) && 
           candle1.close < candle2.open && candle1.open > candle2.close) {
            patterns |= PATTERN_BEARISH_ENGULFING;
        }
        
        // Piercing Line
        if(IsBearish(candle2) && IsBullish(candle1) && 
           candle1.open < candle2.close && 
           candle1.close > candle2.close + (candle2.open - candle2.close) / 2) {
            patterns |= PATTERN_PIERCING_LINE;
        }
        
        // Dark Cloud Cover
        if(IsBullish(candle2) && IsBearish(candle1) && 
           candle1.open > candle2.close && 
           candle1.close < candle2.close - (candle2.close - candle2.open) / 2) {
            patterns |= PATTERN_DARK_CLOUD_COVER;
        }
        
        // Bullish Harami
        if(IsBearish(candle2) && IsBullish(candle1) && 
           candle1.high < candle2.open && candle1.low > candle2.close) {
            patterns |= PATTERN_BULLISH_HARAMI;
        }
        
        // Bearish Harami
        if(IsBullish(candle2) && IsBearish(candle1) && 
           candle1.high < candle2.close && candle1.low > candle2.open) {
            patterns |= PATTERN_BEARISH_HARAMI;
        }
        
        // Bullish Kicker
        if(IsBearish(candle2) && IsBullish(candle1) && 
           candle1.open > candle2.open) {
            patterns |= PATTERN_BULLISH_KICKER;
        }
        
        // Bearish Kicker
        if(IsBullish(candle2) && IsBearish(candle1) && 
           candle1.open < candle2.open) {
            patterns |= PATTERN_BEARISH_KICKER;
        }
        
        // Inside Bar
        if(candle1.high < candle2.high && candle1.low > candle2.low) {
            patterns |= PATTERN_INSIDE_BAR;
        }
        
        // Outside Bar
        if(candle1.high > candle2.high && candle1.low < candle2.low) {
            patterns |= PATTERN_OUTSIDE_BAR;
        }
        
        return patterns;
    }
    
    // Detect three candle patterns
    int DetectThreeCandlePatterns(MqlRates &candle1, MqlRates &candle2, MqlRates &candle3) {
        int patterns = PATTERN_NONE;
        
        // Morning Star
        if(IsBearish(candle3) && 
           GetBodySize(candle2) / GetTotalSize(candle2) < m_dojiRatio && 
           IsBullish(candle1) && 
           candle2.close < candle3.close && 
           candle1.close > candle2.open) {
            patterns |= PATTERN_MORNING_STAR;
        }
        
        // Evening Star
        if(IsBullish(candle3) && 
           GetBodySize(candle2) / GetTotalSize(candle2) < m_dojiRatio && 
           IsBearish(candle1) && 
           candle2.close > candle3.close && 
           candle1.close < candle2.open) {
            patterns |= PATTERN_EVENING_STAR;
        }
        
        // Three White Soldiers
        if(IsBullish(candle3) && IsBullish(candle2) && IsBullish(candle1) && 
           candle1.close > candle2.close && candle2.close > candle3.close && 
           candle1.open > candle2.open && candle2.open > candle3.open) {
            patterns |= PATTERN_THREE_WHITE_SOLDIERS;
        }
        
        // Three Black Crows
        if(IsBearish(candle3) && IsBearish(candle2) && IsBearish(candle1) && 
           candle1.close < candle2.close && candle2.close < candle3.close && 
           candle1.open < candle2.open && candle2.open < candle3.open) {
            patterns |= PATTERN_THREE_BLACK_CROWS;
        }
        
        return patterns;
    }
    
    // Calculate pattern strength (0-1)
    double CalculatePatternStrength(int patterns) {
        // Count number of patterns
        int patternCount = 0;
        
        for(int i = 0; i < 32; i++) {
            if((patterns & (1 << i)) != 0) {
                patternCount++;
            }
        }
        
        // Calculate strength based on pattern count and significance
        double strength = 0;
        
        // Add strength for each pattern
        if((patterns & PATTERN_DOJI) != 0) strength += 0.3;
        if((patterns & PATTERN_HAMMER) != 0) strength += 0.7;
        if((patterns & PATTERN_HANGING_MAN) != 0) strength += 0.7;
        if((patterns & PATTERN_SHOOTING_STAR) != 0) strength += 0.7;
        if((patterns & PATTERN_BULLISH_ENGULFING) != 0) strength += 0.8;
        if((patterns & PATTERN_BEARISH_ENGULFING) != 0) strength += 0.8;
        if((patterns & PATTERN_MORNING_STAR) != 0) strength += 0.9;
        if((patterns & PATTERN_EVENING_STAR) != 0) strength += 0.9;
        if((patterns & PATTERN_PIERCING_LINE) != 0) strength += 0.6;
        if((patterns & PATTERN_DARK_CLOUD_COVER) != 0) strength += 0.6;
        if((patterns & PATTERN_BULLISH_HARAMI) != 0) strength += 0.5;
        if((patterns & PATTERN_BEARISH_HARAMI) != 0) strength += 0.5;
        if((patterns & PATTERN_BULLISH_KICKER) != 0) strength += 0.8;
        if((patterns & PATTERN_BEARISH_KICKER) != 0) strength += 0.8;
        if((patterns & PATTERN_THREE_WHITE_SOLDIERS) != 0) strength += 0.9;
        if((patterns & PATTERN_THREE_BLACK_CROWS) != 0) strength += 0.9;
        if((patterns & PATTERN_INSIDE_BAR) != 0) strength += 0.4;
        if((patterns & PATTERN_OUTSIDE_BAR) != 0) strength += 0.6;
        
        // Normalize strength to 0-1 range
        if(patternCount > 0) {
            strength = MathMin(strength, 1.0);
        }
        
        return strength;
    }
    
    // Get pattern signal (-1 to 1)
    double GetPatternSignal(int patterns) {
        double bullishStrength = 0;
        double bearishStrength = 0;
        
        // Calculate bullish strength
        if((patterns & PATTERN_HAMMER) != 0) bullishStrength += 0.7;
        if((patterns & PATTERN_BULLISH_ENGULFING) != 0) bullishStrength += 0.8;
        if((patterns & PATTERN_MORNING_STAR) != 0) bullishStrength += 0.9;
        if((patterns & PATTERN_PIERCING_LINE) != 0) bullishStrength += 0.6;
        if((patterns & PATTERN_BULLISH_HARAMI) != 0) bullishStrength += 0.5;
        if((patterns & PATTERN_BULLISH_KICKER) != 0) bullishStrength += 0.8;
        if((patterns & PATTERN_THREE_WHITE_SOLDIERS) != 0) bullishStrength += 0.9;
        
        // Calculate bearish strength
        if((patterns & PATTERN_HANGING_MAN) != 0) bearishStrength += 0.7;
        if((patterns & PATTERN_SHOOTING_STAR) != 0) bearishStrength += 0.7;
        if((patterns & PATTERN_BEARISH_ENGULFING) != 0) bearishStrength += 0.8;
        if((patterns & PATTERN_EVENING_STAR) != 0) bearishStrength += 0.9;
        if((patterns & PATTERN_DARK_CLOUD_COVER) != 0) bearishStrength += 0.6;
        if((patterns & PATTERN_BEARISH_HARAMI) != 0) bearishStrength += 0.5;
        if((patterns & PATTERN_BEARISH_KICKER) != 0) bearishStrength += 0.8;
        if((patterns & PATTERN_THREE_BLACK_CROWS) != 0) bearishStrength += 0.9;
        
        // Doji is neutral
        if((patterns & PATTERN_DOJI) != 0) {
            bullishStrength += 0.15;
            bearishStrength += 0.15;
        }
        
        // Inside/Outside bars depend on context
        if((patterns & PATTERN_INSIDE_BAR) != 0) {
            // Use previous trend
            double prevTrend = 0;
            
            // Get previous candles
            MqlRates rates[];
            if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 5, rates) > 0) {
                // Calculate trend based on last 5 candles
                for(int i = 1; i < 5; i++) {
                    if(rates[i].close > rates[i].open) {
                        prevTrend += 0.1;
                    } else if(rates[i].close < rates[i].open) {
                        prevTrend -= 0.1;
                    }
                }
            }
            
            if(prevTrend > 0) {
                bullishStrength += 0.4;
            } else if(prevTrend < 0) {
                bearishStrength += 0.4;
            }
        }
        
        if((patterns & PATTERN_OUTSIDE_BAR) != 0) {
            // Check if bullish or bearish outside bar
            MqlRates rates[];
            if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 2, rates) > 0) {
                if(rates[0].close > rates[1].close) {
                    bullishStrength += 0.6;
                } else {
                    bearishStrength += 0.6;
                }
            }
        }
        
        // Calculate net signal (-1 to 1)
        return bullishStrength - bearishStrength;
    }
};

//+------------------------------------------------------------------+
//| ATFNet Predictor Class                                            |
//+------------------------------------------------------------------+
class CATFNetPredictor {
private:
    int m_sequenceLength;       // Input sequence length
    int m_predictionLength;     // Prediction length
    double m_timeDomainWeight;  // Weight for time domain
    double m_freqDomainWeight;  // Weight for frequency domain
    
    // Time domain prediction using EMA
    double PredictTimeDoamin(double &prices[]) {
        // Simple EMA-based prediction
        double alpha = 2.0 / (14.0 + 1.0);
        double ema = prices[0];
        
        for(int i = 1; i < m_sequenceLength; i++) {
            ema = alpha * prices[i] + (1.0 - alpha) * ema;
        }
        
        // Extrapolate trend
        double lastPrice = prices[m_sequenceLength - 1];
        double trend = ema - lastPrice;
        
        return lastPrice + trend;
    }
    
    // Frequency domain prediction using FFT
    double PredictFrequencyDomain(double &prices[]) {
        // This is a simplified version without actual FFT
        // In a real implementation, you would use FFT to analyze frequency components
        
        // For now, we'll use a simple linear regression
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = m_sequenceLength;
        
        for(int i = 0; i < n; i++) {
            sumX += i;
            sumY += prices[i];
            sumXY += i * prices[i];
            sumX2 += i * i;
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;
        
        // Predict next value
        return intercept + slope * n;
    }

public:
    // Constructor
    CATFNetPredictor(int sequenceLength = 60, int predictionLength = 20, 
                    double timeDomainWeight = 0.5, double freqDomainWeight = 0.5) {
        m_sequenceLength = sequenceLength;
        m_predictionLength = predictionLength;
        m_timeDomainWeight = timeDomainWeight;
        m_freqDomainWeight = freqDomainWeight;
        
        // Normalize weights
        double totalWeight = m_timeDomainWeight + m_freqDomainWeight;
        if(totalWeight > 0) {
            m_timeDomainWeight /= totalWeight;
            m_freqDomainWeight /= totalWeight;
        } else {
            m_timeDomainWeight = 0.5;
            m_freqDomainWeight = 0.5;
        }
    }
    
    // Predict next price
    double PredictNextPrice(double &prices[]) {
        // Check if we have enough data
        if(ArraySize(prices) < m_sequenceLength) {
            Print("Not enough data for prediction");
            return 0;
        }
        
        // Make time domain prediction
        double timePredict = PredictTimeDoamin(prices);
        
        // Make frequency domain prediction
        double freqPredict = PredictFrequencyDomain(prices);
        
        // Combine predictions
        double prediction = m_timeDomainWeight * timePredict + m_freqDomainWeight * freqPredict;
        
        return prediction;
    }
    
    // Predict multiple steps ahead
    void PredictMultipleSteps(double &prices[], double &predictions[]) {
        // Resize predictions array
        ArrayResize(predictions, m_predictionLength);
        
        // Create a copy of prices for iterative prediction
        double tempPrices[];
        ArrayResize(tempPrices, m_sequenceLength);
        ArrayCopy(tempPrices, prices);
        
        // Predict each step
        for(int i = 0; i < m_predictionLength; i++) {
            // Predict next price
            predictions[i] = PredictNextPrice(tempPrices);
            
            // Shift prices and add prediction
            for(int j = 0; j < m_sequenceLength - 1; j++) {
                tempPrices[j] = tempPrices[j + 1];
            }
            tempPrices[m_sequenceLength - 1] = predictions[i];
        }
    }
    
    // Set weights
    void SetWeights(double timeDomainWeight, double freqDomainWeight) {
        m_timeDomainWeight = timeDomainWeight;
        m_freqDomainWeight = freqDomainWeight;
        
        // Normalize weights
        double totalWeight = m_timeDomainWeight + m_freqDomainWeight;
        if(totalWeight > 0) {
            m_timeDomainWeight /= totalWeight;
            m_freqDomainWeight /= totalWeight;
        } else {
            m_timeDomainWeight = 0.5;
            m_freqDomainWeight = 0.5;
        }
    }
};
