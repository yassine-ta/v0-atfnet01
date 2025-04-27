//+------------------------------------------------------------------+
//|                                          FrequencyFeatures.mqh |
//|                 Frequency Domain Analysis for ATFNet Integration |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

#include "FFTUtility.mqh"

//+------------------------------------------------------------------+
//| Frequency Domain Features Class                                   |
//+------------------------------------------------------------------+
class CFrequencyFeatures {
private:
    CFFTUtility* fftUtil;
    int lookbackPeriod;
    int fftSize;
    
    // Cached frequency domain features
    double dominantFreqs[];
    double dominantMags[];
    double fftReal[];
    double fftImag[];
    datetime lastUpdateTime;
    
public:
    CFrequencyFeatures(int _lookbackPeriod) {
        lookbackPeriod = _lookbackPeriod;
        lastUpdateTime = 0;
        
        // Initialize FFT utility with appropriate size
        fftUtil = new CFFTUtility(lookbackPeriod);
        fftSize = fftUtil.GetFFTSize();
    }
    
    ~CFrequencyFeatures() {
        if(fftUtil != NULL) delete fftUtil;
    }
    
    bool UpdateFeatures(bool forceUpdate = false) {
        datetime currentTime = TimeCurrent();
        
        // Only update if needed
        if(!forceUpdate && currentTime < lastUpdateTime + PeriodSeconds(PERIOD_CURRENT)) {
            return false;
        }
        
        // Get price data
        double priceData[];
        if(CopyClose(_Symbol, PERIOD_CURRENT, 0, lookbackPeriod, priceData) != lookbackPeriod) {
            return false;
        }
        
        // Perform FFT
        if(!fftUtil.PerformFFT(priceData, fftReal, fftImag)) {
            return false;
        }
        
        // Extract dominant frequencies
        fftUtil.ExtractDominantFrequencies(fftReal, fftImag, dominantFreqs, dominantMags, 5);
        
        lastUpdateTime = currentTime;
        return true;
    }
    
    // Calculate spectral entropy (measure of frequency distribution complexity)
    double CalculateSpectralEntropy() {
        double totalMag = 0;
        for(int i = 0; i < ArraySize(dominantMags); i++) {
            totalMag += dominantMags[i];
        }
        
        if(totalMag == 0) return 0;
        
        double entropy = 0;
        for(int i = 0; i < ArraySize(dominantMags); i++) {
            double p = dominantMags[i] / totalMag;
            if(p > 0) entropy -= p * MathLog(p);
        }
        
        // Normalize to [0,1]
        return entropy / MathLog(ArraySize(dominantMags));
    }
    
    // Calculate frequency domain trend strength
    double CalculateFrequencyTrendStrength() {
        if(ArraySize(dominantMags) == 0) return 0;
        
        // Calculate ratio of dominant frequency to total magnitude
        double totalMag = 0;
        for(int i = 0; i < ArraySize(dominantMags); i++) {
            totalMag += dominantMags[i];
        }
        
        if(totalMag == 0) return 0;
        
        return dominantMags[0] / totalMag;
    }
    
    // Calculate cyclicality score
    double CalculateCyclicalityScore() {
        if(ArraySize(dominantFreqs) == 0) return 0;
        
        // Lower frequencies indicate stronger cycles
        double lowFreqPower = 0;
        double totalPower = 0;
        
        for(int i = 0; i < ArraySize(dominantFreqs); i++) {
            if(dominantFreqs[i] < 0.2) { // Consider frequencies below 0.2 as "low"
                lowFreqPower += dominantMags[i];
            }
            totalPower += dominantMags[i];
        }
        
        if(totalPower == 0) return 0;
        
        return lowFreqPower / totalPower;
    }
    
    // Get frequency domain features as an array
    void GetFrequencyFeatures(double &features[]) {
        ArrayResize(features, 3);
        features[0] = CalculateSpectralEntropy();
        features[1] = CalculateFrequencyTrendStrength();
        features[2] = CalculateCyclicalityScore();
    }
    
    // Get FFT results for external use
    bool GetFFTResults(double &outReal[], double &outImag[]) {
        if(ArraySize(fftReal) == 0 || ArraySize(fftImag) == 0) {
            return false;
        }
        
        ArrayCopy(outReal, fftReal);
        ArrayCopy(outImag, fftImag);
        return true;
    }
    
    // Get dominant frequencies and magnitudes
    void GetDominantFrequencies(double &freqs[], double &mags[]) {
        ArrayCopy(freqs, dominantFreqs);
        ArrayCopy(mags, dominantMags);
    }
};
