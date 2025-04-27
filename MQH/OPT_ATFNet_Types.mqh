//+------------------------------------------------------------------+
//|                                      OPT_ATFNet_Types.mqh        |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

// Market regime types
enum ENUM_MARKET_REGIME {
    REGIME_TRENDING,
    REGIME_RANGING,
    REGIME_VOLATILE,
    REGIME_CHOPPY
};

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
