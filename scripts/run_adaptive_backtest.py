"""
Adaptive Backtest Based on Recent Market Conditions
Phase 3: 최근 시장 상황 기반 적응형 백테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine, TradingCosts
from src.backtesting.strategy import SimpleThresholdStrategy
from src.backtesting.metrics import PerformanceMetrics
from src.utils.logger import logger


def load_recent_data(days_back=60):
    """최근 데이터만 로드"""
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
    
    # 최근 N일만 필터링
    cutoff_date = binance_df.index[-1] - timedelta(days=days_back)
    binance_df = binance_df[binance_df.index >= cutoff_date]
    upbit_df = upbit_df[upbit_df.index >= cutoff_date]
    
    logger.info(f"Loading last {days_back} days of data from {cutoff_date}")
    
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
    
    # 김프 계산
    USD_KRW = 1330
    merged_df['kimchi_premium'] = ((merged_df['upbit_close'] - merged_df['binance_close'] * USD_KRW) / 
                                   (merged_df['binance_close'] * USD_KRW)) * 100
    
    return merged_df


def analyze_market_regime(data):
    """시장 체제 분석"""
    print("\n" + "=" * 60)
    print("  MARKET REGIME ANALYSIS")
    print("=" * 60)
    
    premium = data['kimchi_premium']
    
    # 기본 통계
    stats = {
        'mean': premium.mean(),
        'std': premium.std(),
        'min': premium.min(),
        'max': premium.max(),
        'p25': premium.quantile(0.25),
        'p50': premium.quantile(0.50),
        'p75': premium.quantile(0.75),
        'p90': premium.quantile(0.90),
        'p95': premium.quantile(0.95)
    }
    
    print(f"\n[Statistics]")
    print(f"Mean: {stats['mean']:.2f}%")
    print(f"Std: {stats['std']:.2f}%")
    print(f"Range: {stats['min']:.2f}% to {stats['max']:.2f}%")
    print(f"P25-P75: {stats['p25']:.2f}% to {stats['p75']:.2f}%")
    
    # 체제 판단
    if stats['std'] < 1.0:
        regime = 'LOW_VOLATILITY'
        print("\n>>> Market Regime: LOW VOLATILITY <<<")
        print("Very stable market, need tight thresholds")
    elif stats['std'] < 2.0:
        regime = 'MODERATE'
        print("\n>>> Market Regime: MODERATE <<<")
        print("Normal market conditions")
    else:
        regime = 'HIGH_VOLATILITY'
        print("\n>>> Market Regime: HIGH VOLATILITY <<<")
        print("Volatile market, wider thresholds needed")
    
    return stats, regime


def get_adaptive_parameters(stats, regime):
    """시장 상황에 맞는 파라미터 결정"""
    
    params = []
    
    if regime == 'LOW_VOLATILITY':
        # 낮은 변동성: 타이트한 임계값
        params = [
            {'entry': stats['p90'], 'exit': stats['p50'], 'name': 'Conservative'},
            {'entry': stats['p75'], 'exit': stats['p25'], 'name': 'Moderate'},
            {'entry': stats['mean'] + stats['std'], 'exit': stats['mean'], 'name': 'Aggressive'}
        ]
    elif regime == 'MODERATE':
        # 중간 변동성
        params = [
            {'entry': stats['p95'], 'exit': stats['p75'], 'name': 'Conservative'},
            {'entry': stats['p90'], 'exit': stats['p50'], 'name': 'Moderate'},
            {'entry': stats['p75'], 'exit': stats['p25'], 'name': 'Aggressive'}
        ]
    else:
        # 높은 변동성
        params = [
            {'entry': stats['max'] * 0.8, 'exit': stats['p75'], 'name': 'Conservative'},
            {'entry': stats['p95'], 'exit': stats['p50'], 'name': 'Moderate'},
            {'entry': stats['p90'], 'exit': stats['mean'], 'name': 'Aggressive'}
        ]
    
    # 현재 시장이 -1.5~1.5% 범위라면 특별 조정
    if abs(stats['mean']) < 2.0 and stats['std'] < 1.5:
        print("\n[!] Detected low premium environment (-1.5% to 1.5%)")
        params = [
            {'entry': 1.5, 'exit': 0.5, 'name': 'Fixed_Conservative'},
            {'entry': 1.2, 'exit': 0.3, 'name': 'Fixed_Moderate'},
            {'entry': 1.0, 'exit': 0.2, 'name': 'Fixed_Aggressive'}
        ]
    
    return params


def run_adaptive_backtest(data, params, split_ratio=0.7):
    """적응형 백테스트 실행"""
    
    # 데이터 분할
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nTrain: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} points)")
    print(f"Test: {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} points)")
    
    # 거래 비용 (현실적으로 설정)
    costs = TradingCosts(
        upbit_fee=0.0005,      # 0.05%
        binance_fee=0.001,     # 0.1%
        slippage=0.0002,       # 0.02% (낮은 변동성 시장)
        funding_rate=0.0001    # 0.01% per 8h
    )
    
    results = []
    
    print("\n" + "=" * 60)
    print("  BACKTESTING RESULTS")
    print("=" * 60)
    
    for param_set in params:
        print(f"\n[{param_set['name']} Strategy]")
        print(f"Entry: {param_set['entry']:.2f}%, Exit: {param_set['exit']:.2f}%")
        
        # 백테스트 엔진
        engine = BacktestEngine(
            initial_capital=40_000_000,
            max_position_size=0.1,
            costs=costs
        )
        
        # 전략
        strategy = SimpleThresholdStrategy(
            entry_threshold=abs(param_set['entry']),
            exit_threshold=abs(param_set['exit'])
        )
        
        # 백테스트 실행 (테스트 데이터)
        backtest_results = engine.run(
            data=test_data,
            strategy=strategy
        )
        
        # 메트릭 계산
        metrics = PerformanceMetrics(backtest_results)
        report = metrics.generate_report()
        
        # 결과 출력
        print(f"  Trades: {backtest_results['total_trades']}")
        print(f"  Return: {backtest_results['total_return']:.2f}%")
        print(f"  Sharpe: {report['returns']['sharpe_ratio']:.2f}")
        print(f"  Max DD: {backtest_results['max_drawdown']:.2f}%")
        print(f"  Win Rate: {backtest_results['win_rate']:.1f}%")
        
        results.append({
            'strategy': param_set['name'],
            'entry': param_set['entry'],
            'exit': param_set['exit'],
            'trades': backtest_results['total_trades'],
            'return': backtest_results['total_return'],
            'sharpe': report['returns']['sharpe_ratio'],
            'max_dd': backtest_results['max_drawdown'],
            'win_rate': backtest_results['win_rate']
        })
    
    return results


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  ADAPTIVE BACKTESTING SYSTEM")
    print("  Based on Recent Market Conditions")
    print("=" * 60)
    
    # 1. 최근 데이터 로드 (60일)
    print("\n[1/4] Loading recent data...")
    data = load_recent_data(days_back=60)
    print(f"Loaded {len(data)} data points")
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    
    # 2. 시장 체제 분석
    print("\n[2/4] Analyzing market regime...")
    stats, regime = analyze_market_regime(data)
    
    # 3. 적응형 파라미터 결정
    print("\n[3/4] Determining adaptive parameters...")
    params = get_adaptive_parameters(stats, regime)
    
    print("\n[Adaptive Parameters]")
    for p in params:
        print(f"  {p['name']}: Entry={p['entry']:.2f}%, Exit={p['exit']:.2f}%")
    
    # 4. 백테스트 실행
    print("\n[4/4] Running adaptive backtests...")
    results = run_adaptive_backtest(data, params)
    
    # 5. 결과 분석
    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    
    # 최고 성과 전략
    best_by_return = results_df.loc[results_df['return'].idxmax()]
    best_by_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    
    print(f"\n[Best by Return]")
    print(f"Strategy: {best_by_return['strategy']}")
    print(f"Return: {best_by_return['return']:.2f}%")
    print(f"Entry/Exit: {best_by_return['entry']:.2f}% / {best_by_return['exit']:.2f}%")
    
    print(f"\n[Best by Sharpe]")
    print(f"Strategy: {best_by_sharpe['strategy']}")
    print(f"Sharpe: {best_by_sharpe['sharpe']:.2f}")
    print(f"Entry/Exit: {best_by_sharpe['entry']:.2f}% / {best_by_sharpe['exit']:.2f}%")
    
    # 6. 실전 권장사항
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS FOR LIVE TRADING")
    print("=" * 60)
    
    avg_trades = results_df['trades'].mean()
    
    if avg_trades < 5:
        print("\n[WARNING] Too few trades detected!")
        print("Consider:")
        print("- Using shorter timeframes (1min instead of 5min)")
        print("- Lowering entry thresholds further")
        print("- Implementing mean-reversion strategy instead")
    
    if results_df['return'].max() < 0:
        print("\n[WARNING] All strategies showing negative returns!")
        print("Consider:")
        print("- Review trading costs (may be too high)")
        print("- Check for data quality issues")
        print("- Market may not be suitable for this strategy")
    else:
        print("\n[OK] Strategy shows positive returns")
        print(f"Recommended parameters:")
        print(f"- Entry: {best_by_sharpe['entry']:.2f}%")
        print(f"- Exit: {best_by_sharpe['exit']:.2f}%")
        print(f"- Expected monthly return: {best_by_sharpe['return'] * 30 / 60:.2f}%")
    
    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"results/adaptive/adaptive_backtest_{timestamp}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump({
            'market_stats': {k: float(v) for k, v in stats.items()},
            'regime': regime,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()