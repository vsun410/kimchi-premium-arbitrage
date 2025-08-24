"""
Optimize Strategy Parameters
Phase 3: 파라미터 최적화
"""

import os
import sys
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine, TradingCosts
from src.backtesting.strategy import SimpleThresholdStrategy
from src.utils.logger import logger


def load_data():
    """데이터 로드"""
    import glob
    
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    # 바이낸스 데이터
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    # 업비트 데이터
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 5분 리샘플링
    binance_5m = binance_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    upbit_5m = upbit_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # 컬럼명 변경
    binance_5m.columns = [f'binance_{col}' for col in binance_5m.columns]
    upbit_5m.columns = [f'upbit_{col}' for col in upbit_5m.columns]
    
    # 병합
    merged_df = pd.merge(
        binance_5m, 
        upbit_5m,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    return merged_df


def run_optimization(data, param_grid):
    """파라미터 그리드 서치"""
    
    # 거래 비용
    costs = TradingCosts(
        upbit_fee=0.0005,
        binance_fee=0.001,
        slippage=0.0001,
        funding_rate=0.0001
    )
    
    results = []
    total_combinations = len(list(product(*param_grid.values())))
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    print("-" * 60)
    
    for i, params in enumerate(product(*param_grid.values())):
        entry_threshold, exit_threshold = params
        
        # 진입 임계값이 청산 임계값보다 작으면 스킵
        if entry_threshold <= exit_threshold:
            continue
        
        # 백테스트 엔진
        engine = BacktestEngine(
            initial_capital=40_000_000,
            max_position_size=0.1,
            costs=costs
        )
        
        # 전략
        strategy = SimpleThresholdStrategy(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )
        
        # 백테스트 실행
        backtest_results = engine.run(
            data=data,
            strategy=strategy
        )
        
        # 결과 저장
        result = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'total_trades': backtest_results['total_trades'],
            'total_return': backtest_results['total_return'],
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'max_drawdown': backtest_results['max_drawdown'],
            'win_rate': backtest_results['win_rate'],
            'final_capital': backtest_results['final_capital']
        }
        
        results.append(result)
        
        # 진행 상황 출력
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{total_combinations} tested")
    
    return pd.DataFrame(results)


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1/3] Loading data...")
    data = load_data()
    
    # 테스트 데이터만 사용 (마지막 20%)
    test_start = int(len(data) * 0.8)
    test_data = data.iloc[test_start:]
    
    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"Data points: {len(test_data):,}")
    
    # 2. 파라미터 그리드 정의
    param_grid = {
        'entry_threshold': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # 진입 김프
        'exit_threshold': [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]  # 청산 김프
    }
    
    print("\n[2/3] Running optimization...")
    print(f"Entry thresholds: {param_grid['entry_threshold']}")
    print(f"Exit thresholds: {param_grid['exit_threshold']}")
    
    # 3. 최적화 실행
    results_df = run_optimization(test_data, param_grid)
    
    # 4. 결과 분석
    print("\n[3/3] Analyzing results...")
    print("\n" + "=" * 60)
    print("  TOP 10 PARAMETER COMBINATIONS")
    print("=" * 60)
    
    # 샤프 비율 기준 상위 10개
    top_results = results_df.nlargest(10, 'sharpe_ratio')
    
    print(f"\n{'Rank':<5} {'Entry':<7} {'Exit':<7} {'Trades':<8} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10} {'WinRate':<8}")
    print("-" * 75)
    
    for idx, row in enumerate(top_results.itertuples(), 1):
        print(f"{idx:<5} {row.entry_threshold:<7.1f} {row.exit_threshold:<7.1f} "
              f"{row.total_trades:<8.0f} {row.total_return:<9.2f}% "
              f"{row.sharpe_ratio:<8.2f} {row.max_drawdown:<9.2f}% "
              f"{row.win_rate:<7.1f}%")
    
    # 최적 파라미터
    best_params = top_results.iloc[0]
    
    print("\n" + "=" * 60)
    print("  OPTIMAL PARAMETERS")
    print("=" * 60)
    print(f"Entry Threshold: {best_params['entry_threshold']:.1f}%")
    print(f"Exit Threshold: {best_params['exit_threshold']:.1f}%")
    print(f"Expected Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Expected Return: {best_params['total_return']:.2f}%")
    print(f"Expected Trades: {best_params['total_trades']:.0f}")
    print(f"Expected Max Drawdown: {best_params['max_drawdown']:.2f}%")
    
    # 결과 저장
    results_file = f"results/optimization/param_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\nFull results saved to: {results_file}")
    
    # 추천사항
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    if best_params['sharpe_ratio'] > 1.5:
        print("[GOOD] Strategy shows strong risk-adjusted returns")
    elif best_params['sharpe_ratio'] > 1.0:
        print("[OK] Strategy shows acceptable returns")
    else:
        print("[WARNING] Strategy needs improvement")
    
    if best_params['total_trades'] < 10:
        print("[WARNING] Too few trades - consider shorter test period or lower thresholds")
    
    if best_params['max_drawdown'] < -15:
        print("[WARNING] High drawdown - implement better risk management")
    
    # 거래 빈도 분석
    avg_trades_per_day = best_params['total_trades'] / ((test_data.index[-1] - test_data.index[0]).days)
    print(f"\nAverage trades per day: {avg_trades_per_day:.2f}")
    
    if avg_trades_per_day < 0.1:
        print("-> Consider more aggressive entry thresholds for higher frequency")
    elif avg_trades_per_day > 1:
        print("-> Consider higher thresholds to reduce overtrading")


if __name__ == "__main__":
    main()