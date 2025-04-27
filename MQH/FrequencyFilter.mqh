//+------------------------------------------------------------------+
//|                                     FrequencyFilter.mqh |
//|                Frequency Domain Filter for ATFNet Integration |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

#include "FrequencyFeatures.mqh"

//+------------------------------------------------------------------+
//| Frequency Domain Filter Class                                     |
//+------------------------------------------------------------------+
class CFrequencyFilter {
private:
    CFrequencyFeatures* freqFeatures;
    double entropyThreshold;
    double cyclicalityThreshold;
    
public:
    CFrequencyFilter(int lookbackPeriod, double _entropyThreshold = 0.6, double _cyclicalityThreshold = 0.5) {
        freqFeatures = new CFrequencyFeatures(lookbackPeriod);
        entropyThreshold = _entropyThreshold;
        cyclicalityThreshold = _cyclicalityThreshold;
    }
    
    ~CFrequencyFilter() {
        if(freqFeatures != NULL) delete freqFeatures;
    }
    
    bool ShouldTrade(bool isLong) {
        if(!freqFeatures.UpdateFeatures()) {
            return true; // Default to allowing trades if we can't update features
        }
        
        double features[];
        freqFeatures.GetFrequencyFeatures(features);
        
        double entropy = features[0];
        double trendStrength = features[1];
        double cyclicality = features[2];
        
        // High entropy (complex, noisy market) - reduce trading
        if(entropy > entropyThreshold) {
            return false;
        }
        
        // For trending markets (low cyclicality), allow trend trades
        if(cyclicality < cyclicalityThreshold) {
            // In trending markets, only allow trades in the direction of the trend
            double trend = CalculateTrendDirection();
            return (isLong && trend > 0) || (!isLong && trend < 0);
        }
        
        // For cyclical markets, allow counter-trend trades
        return true;
    }
    
    double CalculateTrendDirection() {
        // Use EMA to determine trend direction
        double emaFast[], emaSlow[];
        
        if(CopyBuffer(iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE), 0, 0, 2, emaFast) != 2 ||
           CopyBuffer(iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE), 0, 0, 2, emaSlow) != 2) {
            return 0;
        }
        
        // Positive for uptrend, negative for downtrend
        return emaFast[0] - emaSlow[0];
    }
};
