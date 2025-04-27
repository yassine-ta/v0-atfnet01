from core import (
    ATFNetAttention, ATFNetFeatures,
    PriceForecaster, MarketRegimeDetector,
    RiskManager, SignalAggregator,
    PerformanceAnalytics
)

class ATFNetStrategy:
    def __init__(self, mt5_api, data_manager, model_manager, config):
        self.mt5_api       = mt5_api
        self.data_manager  = data_manager
        self.model_manager = model_manager
        self.config        = config

        # core modules
        self.attention       = ATFNetAttention(config)
        self.features        = ATFNetFeatures(config)
        self.forecaster      = PriceForecaster(model_manager.price_model, config)
        self.regime_detector = MarketRegimeDetector(model_manager.regime_model, config)
        self.signal_agg      = SignalAggregator(config)
        self.risk_mgr        = RiskManager(config)
        self.perf_analytics  = PerformanceAnalytics(config)

    def backtest(self, symbol, timeframe, start_date, end_date, progress_callback=None):
        # fetch data
        df = self.data_manager.get_data(symbol, timeframe, start_date, end_date)
        # feature engineering
        feats = self.features.transform(df)
        # predictions
        attn   = self.attention.predict(feats)
        preds  = self.forecaster.predict(feats)
        regime = self.regime_detector.detect(feats)
        # aggregate signals
        signals = self.signal_agg.combine(attn, preds, regime)
        # simulate trades
        trades = self.risk_mgr.simulate(df, signals)
        # performance metrics
        metrics = self.perf_analytics.evaluate(trades)
        return metrics

    def train(self, symbol, timeframe, start_date, end_date, progress_callback=None):
        # load data & features
        df    = self.data_manager.get_data(symbol, timeframe, start_date, end_date)
        feats = self.features.transform(df)
        # train price model
        price_hist = self.model_manager.train_price_model(feats)
        # train regime model
        regime_hist = self.model_manager.train_regime_model(feats)
        # assemble results
        return {
            'price_training_history': price_hist,
            'regime_training_history': regime_hist
        }

    def initialize_live_trading(self, symbol, timeframe):
        # nothing special needed; just ensure data subscription
        return True

    def update_live_trading(self):
        # fetch latest bar
        df = self.data_manager.get_latest(symbol=None, timeframe=None)
        feats = self.features.transform(df)
        attn   = self.attention.predict(feats)
        preds  = self.forecaster.predict(feats)
        regime = self.regime_detector.detect(feats)
        signal = self.signal_agg.combine(attn, preds, regime).iloc[-1]
        # execute
        self.risk_mgr.execute_live(signal)
        # return status
        return {
            'signal': signal,
            'balance': self.risk_mgr.current_balance,
            'equity':  self.risk_mgr.current_equity
        }

    def optimize_parameters(self, symbol, timeframe, start_date, end_date, params, progress_callback=None):
        # grid search over params
        results = self.risk_mgr.optimize(
            self.data_manager, self.features,
            self.attention, self.forecaster,
            self.regime_detector, self.signal_agg,
            start_date, end_date, params, progress_callback
        )
        return results
