"""
Run Backtest with Correct Exchange Rates
실제 환율을 사용한 정확한 백테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine, TradingCosts
from src.backtesting.strategy import SimpleThresholdStrategy
from src.backtesting.metrics import PerformanceMetrics
from src.utils.logger import logger
from src.utils.exchange_rate_manager import get_exchange_rate_manager


def load_recent_data_with_correct_premium(days_back=30):
    """정확한 환율로 데이터 로드"""
    import glob
    
    # 환율 관리자
    rate_manager = get_exchange_rate_manager()
    
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
    
    # 정확한 김프 계산 (실제 환율 사용)
    merged_df['kimchi_premium'] = merged_df.apply(
        lambda row: rate_manager.calculate_kimchi_premium(
            row['upbit_close'],
            row['binance_close'],
            row.name  # timestamp
        ),
        axis=1
    )
    
    logger.info(f"Loaded {len(merged_df)} data points with CORRECT exchange rates")
    
    return merged_df


def analyze_corrected_premium(data):
    """수정된 김프 분석"""
    print("\n" + "=" * 60)
    print("  CORRECTED KIMCHI PREMIUM STATISTICS")
    print("=" * 60)
    
    premium = data['kimchi_premium']
    
    print(f"\n[With ACTUAL Exchange Rates]")
    print(f"Mean: {premium.mean():.3f}%")
    print(f"Std: {premium.std():.3f}%")
    print(f"Min: {premium.min():.3f}%")
    print(f"Max: {premium.max():.3f}%")
    
    print(f"\n[Percentiles]")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        print(f"P{p:02d}: {premium.quantile(p/100):.3f}%")
    
    # 임계값별 기회
    print(f"\n[Trading Opportunities]")
    thresholds = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    for threshold in thresholds:
        count = (premium.abs() > threshold).sum()
        percent = (count / len(data)) * 100
        print(f"  >{threshold:.1f}%: {count:4d} times ({percent:5.1f}%)")
    
    return premium.describe()


def run_realistic_backtest(data, entry_threshold, exit_threshold):
    """현실적인 백테스트"""
    
    # 거래 비용 (현실적)
    costs = TradingCosts(
        upbit_fee=0.0005,      # 0.05%
        binance_fee=0.001,     # 0.1%
        slippage=0.0002,       # 0.02%
        funding_rate=0.0001    # 0.01% per 8h
    )
    
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
    results = engine.run(
        data=data,
        strategy=strategy
    )
    
    return results


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  CORRECTED BACKTEST WITH REAL EXCHANGE RATES")
    print("=" * 60)
    
    # 환율 관리자 초기화
    print("\n[1/5] Initializing Exchange Rate Manager...")
    rate_manager = get_exchange_rate_manager()
    print(f"Current exchange rate: {rate_manager.current_rate:.2f} KRW/USD")
    
    # 데이터 로드
    print("\n[2/5] Loading data with correct exchange rates...")
    data = load_recent_data_with_correct_premium(days_back=30)
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    
    # 김프 분석
    print("\n[3/5] Analyzing corrected premium...")
    stats = analyze_corrected_premium(data)
    
    # 적응형 파라미터 결정
    print("\n[4/5] Determining realistic parameters...")
    
    # 실제 분포 기반 파라미터
    p90 = data['kimchi_premium'].quantile(0.90)
    p80 = data['kimchi_premium'].quantile(0.80)
    p70 = data['kimchi_premium'].quantile(0.70)
    mean = data['kimchi_premium'].mean()
    std = data['kimchi_premium'].std()
    
    strategies = [
        {'name': 'Conservative', 'entry': abs(p90), 'exit': abs(p90) * 0.3},
        {'name': 'Moderate', 'entry': abs(p80), 'exit': abs(p80) * 0.3},
        {'name': 'Aggressive', 'entry': abs(p70), 'exit': abs(p70) * 0.3},
        {'name': 'Mean Reversion', 'entry': mean + std, 'exit': mean},
        {'name': 'Scalping', 'entry': std * 2, 'exit': std * 0.5}
    ]
    
    print("\n[Strategies to Test]")
    for s in strategies:
        print(f"  {s['name']}: Entry={s['entry']:.3f}%, Exit={s['exit']:.3f}%")
    
    # 백테스트 실행
    print("\n[5/5] Running backtests...")
    
    # 데이터 분할
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nTrain: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(f"Test: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    
    best_return = -float('inf')
    best_strategy = None
    
    for strategy in strategies:
        print(f"\n[{strategy['name']}]")
        
        # 백테스트 실행
        results = run_realistic_backtest(
            test_data,
            strategy['entry'],
            strategy['exit']
        )
        
        # 성과 출력
        print(f"  Trades: {results['total_trades']}")
        print(f"  Return: {results['total_return']:.3f}%")
        print(f"  Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"  Max DD: {results['max_drawdown']:.3f}%")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        
        if results['total_return'] > best_return:
            best_return = results['total_return']
            best_strategy = strategy
    
    # 최종 추천
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    if best_strategy and best_return > 0:
        print(f"\n[BEST STRATEGY] {best_strategy['name']}")
        print(f"Entry: {best_strategy['entry']:.3f}%")
        print(f"Exit: {best_strategy['exit']:.3f}%")
        print(f"Expected Return: {best_return:.3f}%")
    else:
        print("\n[WARNING] No profitable strategy found")
        print("The market conditions may not be suitable for arbitrage")
        print("\nConsider:")
        print("1. Waiting for higher volatility")
        print("2. Reducing trading costs")
        print("3. Using different trading strategies")
    
    # 실제 거래 가능성 평가
    print("\n[Trading Feasibility]")
    
    avg_premium = data['kimchi_premium'].mean()
    volatility = data['kimchi_premium'].std()
    
    # 수수료 총합
    total_fees = 0.0005 + 0.001 + 0.0002  # 0.17%
    
    if volatility > total_fees * 2:
        print(f"[OK] Volatility ({volatility:.3f}%) exceeds fees ({total_fees*100:.2f}%)")
    else:
        print(f"[WARNING] Volatility ({volatility:.3f}%) too low for fees ({total_fees*100:.2f}%)")
    
    if abs(avg_premium) < 0.5:
        print(f"[INFO] Premium near zero ({avg_premium:.3f}%), market is efficient")
    else:
        print(f"[INFO] Persistent premium ({avg_premium:.3f}%), structural inefficiency")


if __name__ == "__main__":
    main()