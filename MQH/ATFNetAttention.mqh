//+------------------------------------------------------------------+
//|                                        ATFNetAttention.mqh |
//|                Enhanced Attention Mechanism with Dual-Domain Analysis |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property version   "1.00"
#property strict

#include "FrequencyFeatures.mqh"

// Extended Attention Features Structure
struct ATFNetFeatures {
    // Time domain features
    double price;           // Current price
    double atr;             // ATR value
    double zScore;          // Z-Score
    double trendStrength;   // EMA trend strength
    double rangePosition;   // Position within range
    double volumeProfile;   // Volume profile
    double patternStrength; // Pattern recognition strength
    
    // Frequency domain features
    double spectralEntropy;     // Complexity of frequency distribution
    double freqTrendStrength;   // Strength of dominant frequency
    double cyclicalityScore;    // Measure of price cyclicality
};

// Extended Attention Weights Structure
struct ATFNetWeights {
    // Time domain weights
    double priceWeight;
    double atrWeight;
    double zScoreWeight;
    double trendWeight;
    double rangeWeight;
    double volumeWeight;
    double patternWeight;
    
    // Frequency domain weights
    double spectralEntropyWeight;
    double freqTrendWeight;
    double cyclicalityWeight;
};

//+------------------------------------------------------------------+
//| Cached Normalizer Class                                           |
//+------------------------------------------------------------------+
class CCachedNormalizer {
private:
    double cachedValues[10]; // Cache for all features (7 time + 3 freq)
    double lastInputs[10];
    bool cacheValid[10];
    
public:
    CCachedNormalizer() {
        for(int i = 0; i < 10; i++) {
            cacheValid[i] = false;
        }
    }
    
    double NormalizeFeature(double value, int featureIndex) {
        // Check if we can use cached value (within small epsilon)
        if(cacheValid[featureIndex] && MathAbs(value - lastInputs[featureIndex]) < 0.0001) {
            return cachedValues[featureIndex];
        }
        
        // Calculate new value
        double normalized = 1.0 / (1.0 + MathExp(-value));
        
        // Update cache
        cachedValues[featureIndex] = normalized;
        lastInputs[featureIndex] = value;
        cacheValid[featureIndex] = true;
        
        return normalized;
    }
    
    void InvalidateCache() {
        for(int i = 0; i < 10; i++) {
            cacheValid[i] = false;
        }
    }
};

//+------------------------------------------------------------------+
//| ATFNet Enhanced Attention Mechanism Class                         |
//+------------------------------------------------------------------+
class CATFNetAttention {
private:
    ATFNetWeights weights;
    double learningRate;
    CCachedNormalizer normalizer;
    CFrequencyFeatures* freqFeatures;
    
    // Adaptive weighting between time and frequency domains
    double timeDomainWeight;
    double freqDomainWeight;
    
    void CalculateSoftmaxWeights(double &weights[], int size) {
        // Find maximum weight for numerical stability
        double maxWeight = weights[ArrayMaximum(weights)];
        double sumExp = 0;
        
        // Pre-allocate array for exponential values
        double expValues[];
        ArrayResize(expValues, size);
        
        // Calculate exponentials in one pass
        for(int i = 0; i < size; i++) {
            expValues[i] = MathExp(weights[i] - maxWeight);
            sumExp += expValues[i];
        }
        
        // Normalize in a separate pass
        for(int i = 0; i < size; i++) {
            weights[i] = expValues[i] / sumExp;
        }
    }

public:
    CATFNetAttention(double initial_learning_rate = 0.01, int lookbackPeriod = 20) {
        learningRate = initial_learning_rate;
        
        // Initialize frequency features
        freqFeatures = new CFrequencyFeatures(lookbackPeriod);
        
        // Initialize domain weights equally
        timeDomainWeight = 0.5;
        freqDomainWeight = 0.5;
        
        InitializeWeights();
    }
    
    ~CATFNetAttention() {
        if(freqFeatures != NULL) delete freqFeatures;
    }
    
    void InitializeWeights() {
        // Initialize time domain weights
        weights.priceWeight = 1.0;
        weights.atrWeight = 1.0;
        weights.zScoreWeight = 1.0;
        weights.trendWeight = 1.0;
        weights.rangeWeight = 1.0;
        weights.volumeWeight = 1.0;
        weights.patternWeight = 1.0;
        
        // Initialize frequency domain weights
        weights.spectralEntropyWeight = 1.0;
        weights.freqTrendWeight = 1.0;
        weights.cyclicalityWeight = 1.0;
        
        NormalizeWeights();
    }
    
    double CalculateScore(const ATFNetFeatures &features) {
        // Calculate time domain score
        double timeScore = 
            weights.priceWeight * normalizer.NormalizeFeature(features.price, 0) +
            weights.atrWeight * normalizer.NormalizeFeature(features.atr, 1) +
            weights.zScoreWeight * normalizer.NormalizeFeature(features.zScore, 2) +
            weights.trendWeight * normalizer.NormalizeFeature(features.trendStrength, 3) +
            weights.rangeWeight * normalizer.NormalizeFeature(features.rangePosition, 4) +
            weights.volumeWeight * normalizer.NormalizeFeature(features.volumeProfile, 5) +
            weights.patternWeight * normalizer.NormalizeFeature(features.patternStrength, 6);
        
        // Calculate frequency domain score
        double freqScore = 
            weights.spectralEntropyWeight * normalizer.NormalizeFeature(features.spectralEntropy, 7) +
            weights.freqTrendWeight * normalizer.NormalizeFeature(features.freqTrendStrength, 8) +
            weights.cyclicalityWeight * normalizer.NormalizeFeature(features.cyclicalityScore, 9);
        
        // Combine scores using adaptive weights
        return timeDomainWeight * timeScore + freqDomainWeight * freqScore;
    }
    
    void UpdateWeights(const ATFNetFeatures &features, double tradeResult) {
        // Adjust learning rate based on trade result magnitude
        double adaptiveLR = learningRate * (1.0 + 0.5 * MathAbs(tradeResult));
        
        // Cap learning rate to prevent instability
        adaptiveLR = MathMin(adaptiveLR, 0.05);
        
        double gradient = tradeResult > 0 ? adaptiveLR : -adaptiveLR;
        
        // Update time domain weights
        weights.priceWeight += gradient * normalizer.NormalizeFeature(features.price, 0);
        weights.atrWeight += gradient * normalizer.NormalizeFeature(features.atr, 1);
        weights.zScoreWeight += gradient * normalizer.NormalizeFeature(features.zScore, 2);
        weights.trendWeight += gradient * normalizer.NormalizeFeature(features.trendStrength, 3);
        weights.rangeWeight += gradient * normalizer.NormalizeFeature(features.rangePosition, 4);
        weights.volumeWeight += gradient * normalizer.NormalizeFeature(features.volumeProfile, 5);
        weights.patternWeight += gradient * normalizer.NormalizeFeature(features.patternStrength, 6);
        
        // Update frequency domain weights
        weights.spectralEntropyWeight += gradient * normalizer.NormalizeFeature(features.spectralEntropy, 7);
        weights.freqTrendWeight += gradient * normalizer.NormalizeFeature(features.freqTrendStrength, 8);
        weights.cyclicalityWeight += gradient * normalizer.NormalizeFeature(features.cyclicalityScore, 9);
        
        // Update domain weights based on performance
        double timePerformance = 
            weights.priceWeight * normalizer.NormalizeFeature(features.price, 0) +
            weights.atrWeight * normalizer.NormalizeFeature(features.atr, 1) +
            weights.zScoreWeight * normalizer.NormalizeFeature(features.zScore, 2) +
            weights.trendWeight * normalizer.NormalizeFeature(features.trendStrength, 3) +
            weights.rangeWeight * normalizer.NormalizeFeature(features.rangePosition, 4) +
            weights.volumeWeight * normalizer.NormalizeFeature(features.volumeProfile, 5) +
            weights.patternWeight * normalizer.NormalizeFeature(features.patternStrength, 6);
            
        double freqPerformance = 
            weights.spectralEntropyWeight * normalizer.NormalizeFeature(features.spectralEntropy, 7) +
            weights.freqTrendWeight * normalizer.NormalizeFeature(features.freqTrendStrength, 8) +
            weights.cyclicalityWeight * normalizer.NormalizeFeature(features.cyclicalityScore, 9);
            
        // Adjust domain weights based on which domain performed better
        if(tradeResult > 0) {
            if(timePerformance > freqPerformance) {
                timeDomainWeight = MathMin(timeDomainWeight + 0.01, 0.8);
                freqDomainWeight = 1.0 - timeDomainWeight;
            } else {
                freqDomainWeight = MathMin(freqDomainWeight + 0.01, 0.8);
                timeDomainWeight = 1.0 - freqDomainWeight;
            }
        }
        
        NormalizeWeights();
        
        // Periodically invalidate cache to prevent stale values
        if(MathMod(TimeCurrent(), 3600) < 10) {
            normalizer.InvalidateCache();
        }
    }
    
    void NormalizeWeights() {
        // Normalize time domain weights
        double timeWeights[] = {
            weights.priceWeight, weights.atrWeight, weights.zScoreWeight,
            weights.trendWeight, weights.rangeWeight, weights.volumeWeight,
            weights.patternWeight
        };
        
        CalculateSoftmaxWeights(timeWeights, ArraySize(timeWeights));
        
        weights.priceWeight = timeWeights[0];
        weights.atrWeight = timeWeights[1];
        weights.zScoreWeight = timeWeights[2];
        weights.trendWeight = timeWeights[3];
        weights.rangeWeight = timeWeights[4];
        weights.volumeWeight = timeWeights[5];
        weights.patternWeight = timeWeights[6];
        
        // Normalize frequency domain weights
        double freqWeights[] = {
            weights.spectralEntropyWeight, weights.freqTrendWeight, weights.cyclicalityWeight
        };
        
        CalculateSoftmaxWeights(freqWeights, ArraySize(freqWeights));
        
        weights.spectralEntropyWeight = freqWeights[0];
        weights.freqTrendWeight = freqWeights[1];
        weights.cyclicalityWeight = freqWeights[2];
    }
    
    ATFNetWeights GetWeights() {
        return weights;
    }
    
    void GetDomainWeights(double &timeWeight, double &freqWeight) {
        timeWeight = timeDomainWeight;
        freqWeight = freqDomainWeight;
    }
    
    // Update frequency features and get them
    void UpdateFrequencyFeatures(ATFNetFeatures &features) {
        freqFeatures.UpdateFeatures();
        
        double freqFeatureValues[];
        freqFeatures.GetFrequencyFeatures(freqFeatureValues);
        
        features.spectralEntropy = freqFeatureValues[0];
        features.freqTrendStrength = freqFeatureValues[1];
        features.cyclicalityScore = freqFeatureValues[2];
    }
};
