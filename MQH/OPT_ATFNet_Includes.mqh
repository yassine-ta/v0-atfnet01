//+------------------------------------------------------------------+
//|                                      OPT_ATFNet_Includes.mqh     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

// Standard MT5 includes
#include <Trade\Trade.mqh>
#include <Arrays\ArrayDouble.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>

// Type definitions
#include "OPT_ATFNet_Types.mqh"

// Original ATFNet components
#include "FFTUtility.mqh"
#include "FrequencyFeatures.mqh"
#include "FrequencyFilter.mqh"
#include "AdaptiveForecastCombiner.mqh"
#include "PriceForecaster.mqh"
#include "ATFNetAttention.mqh"
#include "ATFNet_CRT_Includes.mqh"

// New optimized components
#include "OPT_ATFNet_RiskManager.mqh"
#include "OPT_ATFNet_SignalAggregator.mqh"
#include "OPT_ATFNet_MarketRegime.mqh"
#include "OPT_ATFNet_PerformanceAnalytics.mqh"
#include "OPT_ATFNet_Controller.mqh"
