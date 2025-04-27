@echo off
echo Running ATFNet Parameter Optimization
cd %~dp0
python backtest/run_optimization.py --symbol EURUSD --timeframe H1 --start_date 2023-01-01 --end_date 2023-12-31 --max_combinations 50 --parallel
pause
