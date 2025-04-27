//+------------------------------------------------------------------+
//|                         OPT_ATFNet_PerformanceAnalytics.mqh      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property strict

//+------------------------------------------------------------------+
//| Performance analytics class for OPT_ATFNet                       |
//+------------------------------------------------------------------+
class CPerformanceAnalytics {
private:
    // Trade statistics
    int m_totalTrades;
    int m_winningTrades;
    double m_grossProfit;
    double m_grossLoss;
    
    // Component performance
    double m_attentionAccuracy;
    double m_forecastAccuracy;
    double m_filterEffectiveness;
    
    // Trade tracking
    struct TradeRecord {
        ulong ticket;
        ENUM_ORDER_TYPE orderType;
        datetime openTime;
        double openPrice;
        double stopLoss;
        double takeProfit;
        double volume;
        double profit;
        bool closed;
        datetime closeTime;
    };
    
    TradeRecord m_trades[];
    
public:
    // Constructor
    CPerformanceAnalytics() {
        m_totalTrades = 0;
        m_winningTrades = 0;
        m_grossProfit = 0;
        m_grossLoss = 0;
        
        m_attentionAccuracy = 0;
        m_forecastAccuracy = 0;
        m_filterEffectiveness = 0;
    }
    
    // Record trade open
    void RecordTradeOpen(ulong ticket, ENUM_ORDER_TYPE orderType, double openPrice, double stopLoss, double volume) {
        int index = ArraySize(m_trades);
        ArrayResize(m_trades, index + 1);
        
        m_trades[index].ticket = ticket;
        m_trades[index].orderType = orderType;
        m_trades[index].openTime = TimeCurrent();
        m_trades[index].openPrice = openPrice;
        m_trades[index].stopLoss = stopLoss;
        m_trades[index].volume = volume;
        m_trades[index].profit = 0;
        m_trades[index].closed = false;
        
        m_totalTrades++;
    }
    
    // Record trade close
    void RecordTradeClose(ulong ticket, double closePrice, double profit) {
        // Find trade in array
        for(int i = 0; i < ArraySize(m_trades); i++) {
            if(m_trades[i].ticket == ticket && !m_trades[i].closed) {
                m_trades[i].closed = true;
                m_trades[i].closeTime = TimeCurrent();
                m_trades[i].profit = profit;
                
                // Update statistics
                if(profit > 0) {
                    m_winningTrades++;
                    m_grossProfit += profit;
                } else {
                    m_grossLoss += MathAbs(profit);
                }
                
                break;
            }
        }
    }
    
    // Record component performance
    void RecordComponentPerformance(bool attentionCorrect, bool forecastCorrect, bool filterCorrect) {
        static int attentionCount = 0;
        static int forecastCount = 0;
        static int filterCount = 0;
        
        // Update attention accuracy
        attentionCount++;
        m_attentionAccuracy = ((attentionCount - 1) * m_attentionAccuracy + (attentionCorrect ? 1 : 0)) / attentionCount;
        
        // Update forecast accuracy
        forecastCount++;
        m_forecastAccuracy = ((forecastCount - 1) * m_forecastAccuracy + (forecastCorrect ? 1 : 0)) / forecastCount;
        
        // Update filter effectiveness
        filterCount++;
        m_filterEffectiveness = ((filterCount - 1) * m_filterEffectiveness + (filterCorrect ? 1 : 0)) / filterCount;
    }
    
    // Generate performance report
    string GeneratePerformanceReport() {
        string report = "=== OPT_ATFNet Performance Report ===\n";
        
        // Calculate metrics
        double winRate = (m_totalTrades > 0) ? (double)m_winningTrades / m_totalTrades * 100 : 0;
        double profitFactor = (m_grossLoss > 0) ? m_grossProfit / m_grossLoss : 0;
        double netProfit = m_grossProfit - m_grossLoss;
        
        // Add trade statistics
        report += "Total Trades: " + IntegerToString(m_totalTrades) + "\n";
        report += "Winning Trades: " + IntegerToString(m_winningTrades) + " (" + DoubleToString(winRate, 2) + "%)\n";
        report += "Gross Profit: " + DoubleToString(m_grossProfit, 2) + "\n";
        report += "Gross Loss: " + DoubleToString(m_grossLoss, 2) + "\n";
        report += "Net Profit: " + DoubleToString(netProfit, 2) + "\n";
        report += "Profit Factor: " + DoubleToString(profitFactor, 2) + "\n\n";
        
        // Add component performance
        report += "Component Performance:\n";
        report += "Attention Accuracy: " + DoubleToString(m_attentionAccuracy * 100, 2) + "%\n";
        report += "Forecast Accuracy: " + DoubleToString(m_forecastAccuracy * 100, 2) + "%\n";
        report += "Filter Effectiveness: " + DoubleToString(m_filterEffectiveness * 100, 2) + "%\n";
        
        return report;
    }
    
    // Export trade history to CSV
    void ExportToCSV(string filename) {
        int fileHandle = FileOpen(filename, FILE_WRITE | FILE_CSV);
        
        if(fileHandle != INVALID_HANDLE) {
            // Write header
            FileWrite(fileHandle, "Ticket", "Type", "Open Time", "Close Time", "Open Price", "Stop Loss", "Volume", "Profit");
            
            // Write trade data
            for(int i = 0; i < ArraySize(m_trades); i++) {
                if(m_trades[i].closed) {
                    FileWrite(fileHandle, 
                             m_trades[i].ticket,
                             (m_trades[i].orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL",
                             TimeToString(m_trades[i].openTime),
                             TimeToString(m_trades[i].closeTime),
                             DoubleToString(m_trades[i].openPrice, 5),
                             DoubleToString(m_trades[i].stopLoss, 5),
                             DoubleToString(m_trades[i].volume, 2),
                             DoubleToString(m_trades[i].profit, 2));
                }
            }
            
            FileClose(fileHandle);
            Print("Trade history exported to ", filename);
        } else {
            Print("Failed to open file for writing: ", filename);
        }
    }
    
    // Getters
    double GetWinRate() const { return (m_totalTrades > 0) ? (double)m_winningTrades / m_totalTrades : 0; }
    double GetProfitFactor() const { return (m_grossLoss > 0) ? m_grossProfit / m_grossLoss : 0; }
    double GetNetProfit() const { return m_grossProfit - m_grossLoss; }
};
