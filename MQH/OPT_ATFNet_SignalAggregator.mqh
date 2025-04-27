//+------------------------------------------------------------------+
//|                             OPT_ATFNet_SignalAggregator.mqh      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

//+------------------------------------------------------------------+
//| Signal aggregation class for OPT_ATFNet                          |
//+------------------------------------------------------------------+
class CSignalAggregator {
private:
    // Signal sources
    CATFNetAttention* m_attention;
    CPriceForecaster* m_priceForecaster;
    CFrequencyFilter* m_freqFilter;
    
    // Signal weights
    double m_attentionWeight;
    double m_forecastWeight;
    double m_filterWeight;
    
    // Signal thresholds
    double m_entryThreshold;
    double m_exitThreshold;
    
    // Current signals
    double m_predictedReturn;
    ATFNetFeatures m_features;
    double m_attentionScore;
    
    // Trend filter
    bool m_useTrendFilter;
    double m_currentPrice;
    double m_emaValue;
    
public:
    // Constructor
    CSignalAggregator() {
        m_attention = NULL;
        m_priceForecaster = NULL;
        m_freqFilter = NULL;
        
        m_attentionWeight = 0.4;
        m_forecastWeight = 0.4;
        m_filterWeight = 0.2;
        
        m_entryThreshold = 0.6;
        m_exitThreshold = 0.3;
        
        m_predictedReturn = 0;
        m_attentionScore = 0;
        
        m_useTrendFilter = true;
        m_currentPrice = 0;
        m_emaValue = 0;
    }
    
    // Set components
    void SetAttention(CATFNetAttention* attention) { m_attention = attention; }
    void SetForecaster(CPriceForecaster* forecaster) { m_priceForecaster = forecaster; }
    void SetFrequencyFilter(CFrequencyFilter* filter) { m_freqFilter = filter; }
    
    // Set parameters
    void SetThreshold(double threshold) { m_entryThreshold = threshold; }
    void SetPredictedReturn(double predictedReturn) { m_predictedReturn = predictedReturn; }
    void SetFeatures(ATFNetFeatures &features) { m_features = features; }
    void SetTrendFilter(bool useTrendFilter, double currentPrice, double emaValue) {
        m_useTrendFilter = useTrendFilter;
        m_currentPrice = currentPrice;
        m_emaValue = emaValue;
    }
    
    // Signal generation
    double CalculateEntrySignal() {
        // Skip if components are not initialized
        if(m_attention == NULL || m_priceForecaster == NULL || m_freqFilter == NULL) {
            return 0;
        }
        
        // Calculate attention score
        m_attentionScore = m_attention.CalculateScore(m_features);
        
        // Check if frequency filter allows trading
        bool freqFilterPass = m_freqFilter.ShouldTrade(m_predictedReturn > 0);
        
        // Calculate combined signal
        double signal = m_attentionWeight * m_attentionScore + 
                       m_forecastWeight * MathAbs(m_predictedReturn) / PREDICTION_THRESHOLD + 
                       m_filterWeight * (freqFilterPass ? 1.0 : 0.0);
        
        return signal;
    }
    
    // Signal interpretation
    ENUM_ORDER_TYPE GetSignalDirection() {
        return (m_predictedReturn > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    }
    
    bool ShouldEnterTrade() {
        // Calculate entry signal
        double entrySignal = CalculateEntrySignal();
        
        // Check if signal exceeds threshold
        if(entrySignal < m_entryThreshold) {
            return false;
        }
        
        // Check trend filter if enabled
        if(m_useTrendFilter) {
            if(m_predictedReturn > 0 && m_currentPrice < m_emaValue) {
                return false;
            }
            else if(m_predictedReturn < 0 && m_currentPrice > m_emaValue) {
                return false;
            }
        }
        
        return true;
    }
    
    bool ShouldExitTrade(ulong ticket) {
        // This would check if a position should be exited based on signals
        // For now, we'll rely on the stop loss and take profit mechanisms
        return false;
    }
    
    // Weight adaptation
    void UpdateWeights(double profitLoss) {
        // Adjust weights based on trade result
        if(profitLoss > 0) {
            // Increase weight of components that contributed to profitable trade
            if(m_attentionScore > m_entryThreshold) {
                m_attentionWeight = MathMin(0.6, m_attentionWeight + 0.05);
            }
            
            if(MathAbs(m_predictedReturn) > PREDICTION_THRESHOLD) {
                m_forecastWeight = MathMin(0.6, m_forecastWeight + 0.05);
            }
        } else {
            // Decrease weight of components that contributed to losing trade
            if(m_attentionScore > m_entryThreshold) {
                m_attentionWeight = MathMax(0.2, m_attentionWeight - 0.05);
            }
            
            if(MathAbs(m_predictedReturn) > PREDICTION_THRESHOLD) {
                m_forecastWeight = MathMax(0.2, m_forecastWeight - 0.05);
            }
        }
        
        // Normalize weights
        double totalWeight = m_attentionWeight + m_forecastWeight + m_filterWeight;
        if(totalWeight > 0) {
            m_attentionWeight /= totalWeight;
            m_forecastWeight /= totalWeight;
            m_filterWeight /= totalWeight;
        }
    }
};
