//+------------------------------------------------------------------+
//|                              OPT_ATFNet_MarketRegime.mqh         |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

#include "OPT_ATFNet_Types.mqh"

//+------------------------------------------------------------------+
//| Market regime detection class for OPT_ATFNet                     |
//+------------------------------------------------------------------+
class CMarketRegimeDetector {
private:
    ENUM_MARKET_REGIME m_currentRegime;
    int m_lookbackBars;
    double m_trendThreshold;
    double m_volatilityThreshold;
    
    // Cached values
    double m_trendStrength;
    double m_volatility;
    double m_cyclicality;
    
public:
    // Constructor
    CMarketRegimeDetector() {
        m_currentRegime = REGIME_RANGING;
        m_lookbackBars = 100;
        m_trendThreshold = 0.6;
        m_volatilityThreshold = 1.5;
        
        m_trendStrength = 0;
        m_volatility = 0;
        m_cyclicality = 0;
    }
    
    // Detect current market regime
    ENUM_MARKET_REGIME DetectRegime() {
        // Get price data
        double closeArray[];
        if(CopyClose(_Symbol, PERIOD_CURRENT, 0, m_lookbackBars, closeArray) != m_lookbackBars) {
            return m_currentRegime; // Return current regime if data fetch fails
        }
        
        // Calculate trend strength
        m_trendStrength = CalculateTrendStrength(closeArray);
        
        // Calculate volatility
        m_volatility = CalculateVolatility(closeArray);
        
        // Calculate cyclicality
        m_cyclicality = CalculateCyclicality(closeArray);
        
        // Determine regime
        if(m_trendStrength > m_trendThreshold) {
            if(m_volatility > m_volatilityThreshold) {
                m_currentRegime = REGIME_VOLATILE;
            } else {
                m_currentRegime = REGIME_TRENDING;
            }
        } else {
            if(m_volatility > m_volatilityThreshold) {
                m_currentRegime = REGIME_CHOPPY;
            } else {
                m_currentRegime = REGIME_RANGING;
            }
        }
        
        return m_currentRegime;
    }
    
    // Calculate trend strength (0-1)
    double CalculateTrendStrength(double &prices[]) {
        int n = ArraySize(prices);
        if(n < 2) return 0;
        
        // Linear regression
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for(int i = 0; i < n; i++) {
            sumX += i;
            sumY += prices[i];
            sumXY += i * prices[i];
            sumX2 += i * i;
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;
        
        // Calculate R-squared
        double meanY = sumY / n;
        double totalSS = 0, residualSS = 0;
        
        for(int i = 0; i < n; i++) {
            double predicted = intercept + slope * i;
            totalSS += MathPow(prices[i] - meanY, 2);
            residualSS += MathPow(prices[i] - predicted, 2);
        }
        
        double rSquared = 1 - (residualSS / totalSS);
        return rSquared;
    }
    
    // Calculate volatility
    double CalculateVolatility(double &prices[]) {
        int n = ArraySize(prices);
        if(n < 2) return 0;
        
        // Calculate returns
        double returns[];
        ArrayResize(returns, n - 1);
        
        for(int i = 0; i < n - 1; i++) {
            returns[i] = (prices[i] - prices[i + 1]) / prices[i + 1];
        }
        
        // Calculate standard deviation of returns
        double sum = 0;
        for(int i = 0; i < n - 1; i++) {
            sum += returns[i];
        }
        double mean = sum / (n - 1);
        
        double variance = 0;
        for(int i = 0; i < n - 1; i++) {
            variance += MathPow(returns[i] - mean, 2);
        }
        variance /= (n - 1);
        
        return MathSqrt(variance);
    }
    
    // Calculate cyclicality
    double CalculateCyclicality(double &prices[]) {
        int n = ArraySize(prices);
        if(n < 4) return 0;
        
        // Count direction changes
        int directionChanges = 0;
        int currentDirection = 0;
        
        for(int i = 1; i < n; i++) {
            int newDirection = (prices[i] > prices[i-1]) ? 1 : -1;
            
            if(currentDirection != 0 && newDirection != currentDirection) {
                directionChanges++;
            }
            
            currentDirection = newDirection;
        }
        
        // Calculate cyclicality as ratio of direction changes to total possible changes
        return (double)directionChanges / (n - 2);
    }
    
    // Adjust strategy parameters for current regime
    void OptimizeForRegime(ENUM_MARKET_REGIME regime) {
        switch(regime) {
            case REGIME_TRENDING:
                // Optimize for trending markets
                // - Increase trend following weight
                // - Reduce mean reversion weight
                // - Increase position holding time
                break;
                
            case REGIME_RANGING:
                // Optimize for ranging markets
                // - Increase mean reversion weight
                // - Reduce trend following weight
                // - Decrease position holding time
                break;
                
            case REGIME_VOLATILE:
                // Optimize for volatile markets
                // - Widen stop losses
                // - Reduce position sizes
                // - Increase profit targets
                break;
                
            case REGIME_CHOPPY:
                // Optimize for choppy markets
                // - Reduce overall exposure
                // - Increase signal thresholds
                // - Consider pausing trading
                break;
        }
    }
    
    // Getters
    ENUM_MARKET_REGIME GetCurrentRegime() { return m_currentRegime; }
    double GetTrendStrength() { return m_trendStrength; }
    double GetVolatility() { return m_volatility; }
    double GetCyclicality() { return m_cyclicality; }
};
