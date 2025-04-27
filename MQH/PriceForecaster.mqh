//+------------------------------------------------------------------+
//|                                      PriceForecaster.mqh |
//|                Price Forecasting for ATFNet Integration |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

#include "FrequencyFeatures.mqh"
#include "AdaptiveForecastCombiner.mqh"

//+------------------------------------------------------------------+
//| Price Forecaster Class                                            |
//+------------------------------------------------------------------+
class CPriceForecaster {
private:
    int forecastHorizon;
    CFrequencyFeatures* freqFeatures;
    CAdaptiveForecastCombiner* combiner;
    
    // RNN parameters - fix array declarations
    double rnnWeights[];      // Weights for 10 input timesteps
    double rnnHiddenState;    // Single hidden state
    double rnnOutputWeight;   // Weight from hidden to output
    double rnnLearningRate;   // Learning rate for RNN updates
    double lastRNNPrediction; // Store last prediction

    double ForecastRNNMethod(const double &prices[]) {
        int n = ArraySize(prices);
        if(n < 10) return prices[0];
        
        double temp[];  // Temporary array for calculations
        ArrayResize(temp, 10);
        
        // Normalize prices
        for(int i = 0; i < 10; i++) {
            temp[i] = (prices[n - 10 + i] - prices[n - 10]) / prices[n - 10];
        }
        
        // Update hidden state
        double newHidden = 0.0;
        for(int i = 0; i < 10; i++) {
            newHidden += rnnWeights[i] * temp[i];
        }
        newHidden = MathTanh(newHidden);
        rnnHiddenState = 0.7 * rnnHiddenState + 0.3 * newHidden;
        
        // Make prediction
        lastRNNPrediction = prices[n - 1] * (1.0 + rnnOutputWeight * rnnHiddenState);
        return lastRNNPrediction;
    }

public:
    CPriceForecaster(int _forecastHorizon, int lookbackPeriod) {
        forecastHorizon = _forecastHorizon;
        freqFeatures = new CFrequencyFeatures(lookbackPeriod);
        combiner = new CAdaptiveForecastCombiner(0.5, 0.05);
        
        // Initialize RNN arrays properly
        ArrayResize(rnnWeights, 10);
        ArrayInitialize(rnnWeights, 0.1);
        rnnHiddenState = 0.0;
        rnnOutputWeight = 0.5;
        rnnLearningRate = 0.01;
        lastRNNPrediction = 0.0;
    }
    
    ~CPriceForecaster() {
        if(freqFeatures != NULL) delete freqFeatures;
        if(combiner != NULL) delete combiner;
    }
    
    double ForecastPrice() {
        // Get current price data
        double prices[];
        if(CopyClose(_Symbol, PERIOD_CURRENT, 0, 30, prices) != 30) {
            return 0;
        }
        
        // Time domain forecast (linear regression)
        double timeForecast = ForecastTimeMethod(prices);
        
        // Frequency domain forecast
        double freqForecast = ForecastFrequencyMethod(prices);
        
        // RNN forecast
        double rnnForecast = ForecastRNNMethod(prices);
        
        // Combine forecasts adaptively
        double combinedBase = combiner.CombineForecasts(timeForecast, freqForecast);
        return (combinedBase + rnnForecast) / 2.0;
    }
    
    double ForecastTimeMethod(const double &prices[]) {
        // Simple linear regression forecast
        int n = ArraySize(prices);
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for(int i = 0; i < n; i++) {
            sumX += i;
            sumY += prices[i];
            sumXY += i * prices[i];
            sumX2 += i * i;
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;
        
        // Forecast future price
        return intercept + slope * (n + forecastHorizon);
    }
    
    double ForecastFrequencyMethod(const double &prices[]) {
        if(!freqFeatures.UpdateFeatures()) {
            return prices[0]; // Return current price if update fails
        }
        
        // Get frequency features
        double features[];
        freqFeatures.GetFrequencyFeatures(features);
        
        double spectralEntropy = features[0];
        double trendStrength = features[1];
        double cyclicality = features[2];
        
        // Get FFT results
        double fftReal[], fftImag[];
        if(!freqFeatures.GetFFTResults(fftReal, fftImag)) {
            return prices[0];
        }
        
        // Get dominant frequencies and magnitudes
        double freqs[], mags[];
        freqFeatures.GetDominantFrequencies(freqs, mags);
        
        // Reconstruct signal with dominant frequencies
        double forecast = prices[0]; // Start with current price
        
        for(int i = 0; i < MathMin(3, ArraySize(freqs)); i++) {
            // Add contribution of each dominant frequency
            double freq = freqs[i];
            double mag = mags[i];
            
            // Calculate phase (approximate)
            int freqIndex = (int)(freq * ArraySize(fftReal));
            if(freqIndex >= ArraySize(fftReal)) freqIndex = ArraySize(fftReal) - 1;
            
            double phase = MathArctan2(fftImag[freqIndex], fftReal[freqIndex]);
            
            forecast += mag * MathCos(2 * M_PI * freq * forecastHorizon + phase);
        }
        
        return forecast;
    }
    
    void UpdateCombiner(double actualPrice) {
        // Get price data for previous forecast
        double oldPrices[];
        if(CopyClose(_Symbol, PERIOD_CURRENT, forecastHorizon, 30, oldPrices) != 30) {
            return;
        }
        
        // Calculate what would have been forecasted
        double oldTimeForecast = ForecastTimeMethod(oldPrices);
        double oldFreqForecast = ForecastFrequencyMethod(oldPrices);
        double oldRNNForecast = lastRNNPrediction;
        
        // Update RNN weights
        double rnnError = actualPrice - oldRNNForecast;
        double gradient = rnnError * rnnLearningRate;
        
        rnnOutputWeight += gradient * rnnHiddenState;
        rnnOutputWeight = MathMax(-1.0, MathMin(1.0, rnnOutputWeight));
        
        for(int i = 0; i < 10; i++) {
            double inputVal = (oldPrices[30 - 10 + i] - oldPrices[20]) / oldPrices[20];
            rnnWeights[i] += gradient * inputVal;
            rnnWeights[i] = MathMax(-0.5, MathMin(0.5, rnnWeights[i]));
        }
        
        // Update weights based on actual outcome
        combiner.UpdateWeights(oldTimeForecast, oldFreqForecast, actualPrice);
    }
    
    void GetCombinerWeights(double &timeWeight, double &freqWeight) {
        combiner.GetWeights(timeWeight, freqWeight);
    }
    
    // Get forecast direction (bullish/bearish)
    bool IsForecastBullish() {
        double forecastPrice = ForecastPrice();
        double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        return forecastPrice > currentPrice;
    }
};
