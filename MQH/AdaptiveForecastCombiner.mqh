//+------------------------------------------------------------------+
//|                                 AdaptiveForecastCombiner.mqh |
//|            Adaptive Forecast Combination for ATFNet Integration |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Adaptive Forecast Combiner Class                                  |
//+------------------------------------------------------------------+
class CAdaptiveForecastCombiner {
private:
    double timeWeight;
    double freqWeight;
    double adaptationRate;
    double lastTimeError;
    double lastFreqError;
    
public:
    CAdaptiveForecastCombiner(double initialTimeWeight = 0.5, double _adaptationRate = 0.05) {
        timeWeight = initialTimeWeight;
        freqWeight = 1.0 - timeWeight;
        adaptationRate = _adaptationRate;
        lastTimeError = 1.0;
        lastFreqError = 1.0;
    }
    
    double CombineForecasts(double timeForecast, double freqForecast) {
        return timeWeight * timeForecast + freqWeight * freqForecast;
    }
    
    void UpdateWeights(double timeForecast, double freqForecast, double actualValue) {
        // Calculate forecast errors
        double timeError = MathAbs(timeForecast - actualValue);
        double freqError = MathAbs(freqForecast - actualValue);
        
        // Update error history with exponential smoothing
        lastTimeError = 0.7 * lastTimeError + 0.3 * timeError;
        lastFreqError = 0.7 * lastFreqError + 0.3 * freqError;
        
        // Calculate total error
        double totalError = lastTimeError + lastFreqError;
        
        if(totalError > 0) {
            // Inverse error weighting
            double newTimeWeight = lastFreqError / totalError;
            double newFreqWeight = lastTimeError / totalError;
            
            // Apply adaptation rate
            timeWeight = timeWeight * (1 - adaptationRate) + newTimeWeight * adaptationRate;
            freqWeight = 1.0 - timeWeight;
        }
    }
    
    void GetWeights(double &time, double &freq) {
        time = timeWeight;
        freq = freqWeight;
    }
};
