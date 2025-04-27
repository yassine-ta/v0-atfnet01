//+------------------------------------------------------------------+
//|                                 OPT_ATFNet_RiskManager.mqh       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

//+------------------------------------------------------------------+
//| Risk management class for OPT_ATFNet                             |
//+------------------------------------------------------------------+
class CRiskManager {
private:
    double m_initialBalance;
    double m_maxDailyLoss;
    double m_maxDrawdown;
    double m_emergencyDrawdown;
    double m_baseRiskPerTrade;
    double m_maxRiskPerTrade;
    bool m_dynamicRiskEnabled;
    
    // Daily tracking
    datetime m_lastDayChecked;
    double m_dayHighBalance;
    
    // Dynamic risk adjustment
    double m_currentRiskLevel;
    double m_adaptationRate;
    
    // Performance tracking
    double m_winRate;
    double m_profitFactor;
    int m_consecutiveLosses;
    
public:
    // Constructor
    CRiskManager() {
        m_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        m_maxDailyLoss = 5.0; // Default 5%
        m_maxDrawdown = 10.0; // Default 10%
        m_emergencyDrawdown = 15.0; // Default 15%
        m_baseRiskPerTrade = 1.0; // Default 1%
        m_maxRiskPerTrade = 3.0; // Default 3%
        m_dynamicRiskEnabled = true;
        
        m_lastDayChecked = 0;
        m_dayHighBalance = 0;
        
        m_currentRiskLevel = m_baseRiskPerTrade;
        m_adaptationRate = 0.1;
        
        m_winRate = 0.5;
        m_profitFactor = 1.0;
        m_consecutiveLosses = 0;
    }
    
    // Position sizing
    double CalculatePositionSize(double entryPrice, double stopLoss, string symbol) {
        // Calculate risk amount
        double riskAmount = CalculateRiskAmount();
        
        // Calculate risk per unit
        double riskPerUnit = MathAbs(entryPrice - stopLoss);
        
        // Skip if stop loss is invalid
        if(riskPerUnit <= 0) {
            Print("Invalid stop loss calculation. Entry: ", entryPrice, " SL: ", stopLoss);
            return 0;
        }
        
        // Calculate position size
        double positionSize = riskAmount / riskPerUnit;
        
        // Apply min/max lot constraints
        positionSize = MathMax(positionSize, MIN_LOTS);
        positionSize = MathMin(positionSize, MAX_LOTS);
        
        // Normalize position size to lot step
        double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
        positionSize = MathFloor(positionSize / lotStep) * lotStep;
        
        // Skip if position size is too small
        if(positionSize < SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN)) {
            Print("Position size too small: ", positionSize);
            return 0;
        }
        
        return positionSize;
    }
    
    // Calculate risk amount based on account balance
    double CalculateRiskAmount() {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double currentRisk = m_currentRiskLevel;
        
        // Calculate current drawdown
        double currentDrawdown = GetCurrentDrawdown();
        
        // Scale risk based on drawdown if enabled
        if(SCALE_POSITION_SIZE && currentDrawdown > 0) {
            currentRisk *= MathMax(0.1, 1 - (currentDrawdown * DD_SCALE_FACTOR / MAX_FLOATING_DD));
        }
        
        // Adjust risk based on account performance if dynamic risk is enabled
        if(m_dynamicRiskEnabled) {
            double growthPercent = (balance - m_initialBalance) / m_initialBalance * 100;
            if(growthPercent > PROFIT_THRESHOLD) {
                int thresholdCrosses = (int)(growthPercent / PROFIT_THRESHOLD);
                currentRisk = MathMin(
                    m_baseRiskPerTrade + thresholdCrosses * RISK_INCREASE_STEP,
                    m_maxRiskPerTrade
                );
            }
        }
        
        return balance * (currentRisk / 100);
    }
    
    // Risk checks
    bool CheckDailyLossLimit() {
        // Get current time
        MqlDateTime dt;
        datetime currentTime = TimeCurrent(dt);
        
        // Check if it's a new day
        if(m_lastDayChecked == 0) {
            m_dayHighBalance = AccountInfoDouble(ACCOUNT_BALANCE);
            m_lastDayChecked = currentTime;
        }
        else {
            MqlDateTime lastDt;
            TimeToStruct(m_lastDayChecked, lastDt);
            
            // Compare day, month, and year
            if(dt.day != lastDt.day || 
               dt.mon != lastDt.mon || 
               dt.year != lastDt.year) {
                // Reset daily tracking
                m_dayHighBalance = AccountInfoDouble(ACCOUNT_BALANCE);
                m_lastDayChecked = currentTime;
            }
        }
        
        // Update day high balance
        double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(currentBalance > m_dayHighBalance) {
            m_dayHighBalance = currentBalance;
        }
        
        // Check if daily loss limit is reached
        double dailyDrawdownPercent = ((m_dayHighBalance - currentBalance) / m_dayHighBalance) * 100;
        return (dailyDrawdownPercent > m_maxDailyLoss);
    }
    
    bool CheckDrawdownLimit() {
        return (GetCurrentDrawdown() > m_maxDrawdown);
    }
    
    bool CheckEmergencyDrawdownLimit() {
        return (GetCurrentDrawdown() > m_emergencyDrawdown);
    }
    
    // Calculate drawdown percentage
    double GetCurrentDrawdown() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        
        // If initial balance is 0, use current balance (should only happen on first run)
        double baseBalance = (m_initialBalance > 0) ? m_initialBalance : AccountInfoDouble(ACCOUNT_BALANCE);
        
        return ((baseBalance - equity) / baseBalance) * 100;
    }
    
    // Risk adaptation
    void UpdateRiskParameters(double profitLoss) {
        // Update consecutive losses
        if(profitLoss < 0) {
            m_consecutiveLosses++;
        } else {
            m_consecutiveLosses = 0;
        }
        
        // Reduce risk after consecutive losses
        if(m_consecutiveLosses >= 3) {
            m_currentRiskLevel = MathMax(m_baseRiskPerTrade * 0.5, m_currentRiskLevel - 0.2);
        }
        
        // Gradually increase risk after profitable trades
        if(profitLoss > 0 && m_currentRiskLevel < m_baseRiskPerTrade) {
            m_currentRiskLevel = MathMin(m_baseRiskPerTrade, m_currentRiskLevel + 0.1);
        }
    }
    
    // Setters
    void SetBaseRisk(double risk) { m_baseRiskPerTrade = risk; m_currentRiskLevel = risk; }
    void SetMaxRisk(double risk) { m_maxRiskPerTrade = risk; }
    void SetMaxDailyLoss(double loss) { m_maxDailyLoss = loss; }
    void SetMaxDrawdown(double dd) { m_maxDrawdown = dd; }
    void SetEmergencyDrawdown(double dd) { m_emergencyDrawdown = dd; }
    void SetDynamicRiskEnabled(bool enabled) { m_dynamicRiskEnabled = enabled; }
};
