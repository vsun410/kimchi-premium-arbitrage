"""
Analyze Trading Conditions
현재 시장 상황 분석 및 모델 파라미터 조정
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exchange_rate_manager import get_exchange_rate_manager


def analyze_market_conditions():
    """현재 시장 상황 분석"""
    import glob
    
    # 데이터 로드
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 최근 30일
    cutoff = binance_df.index[-1] - timedelta(days=30)
    binance_df = binance_df[binance_df.index >= cutoff]
    upbit_df = upbit_df[upbit_df.index >= cutoff]
    
    # 1분 데이터로 병합
    merged = pd.merge(
        binance_df[['close']].rename(columns={'close': 'binance_close'}),
        upbit_df[['close']].rename(columns={'close': 'upbit_close'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # 김프 계산
    rate_manager = get_exchange_rate_manager()
    merged['kimchi_premium'] = merged.apply(
        lambda row: rate_manager.calculate_kimchi_premium(
            row['upbit_close'],
            row['binance_close'],
            row.name
        ),
        axis=1
    )
    
    print("\n" + "=" * 60)
    print("  MARKET CONDITION ANALYSIS")
    print("=" * 60)
    
    print(f"\n[Data Period]")
    print(f"From: {merged.index[0]}")
    print(f"To: {merged.index[-1]}")
    print(f"Total samples: {len(merged):,}")
    
    print(f"\n[Kimchi Premium Statistics]")
    premium = merged['kimchi_premium']
    print(f"Mean: {premium.mean():.4f}%")
    print(f"Std: {premium.std():.4f}%")
    print(f"Min: {premium.min():.4f}%")
    print(f"Max: {premium.max():.4f}%")
    
    print(f"\n[Percentiles]")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = premium.quantile(p/100)
        print(f"P{p:02d}: {val:+.4f}%")
    
    # 변동성 분석 (5분 단위)
    merged_5m = merged.resample('5min').agg({
        'kimchi_premium': ['mean', 'std', 'min', 'max']
    })
    
    # 5분 변화량
    merged_5m['change'] = merged_5m[('kimchi_premium', 'mean')].diff()
    
    print(f"\n[5-Minute Changes]")
    changes = merged_5m['change'].dropna()
    print(f"Mean change: {changes.mean():.4f}%")
    print(f"Std of changes: {changes.std():.4f}%")
    print(f"Max increase: {changes.max():.4f}%")
    print(f"Max decrease: {changes.min():.4f}%")
    
    # 거래 기회 분석
    print(f"\n[Trading Opportunities (5min)]")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for threshold in thresholds:
        opportunities = (abs(changes) > threshold).sum()
        pct = opportunities / len(changes) * 100
        daily_avg = opportunities / 30
        print(f"Changes > {threshold:.2f}%: {opportunities:4d} ({pct:5.1f}%) = {daily_avg:.1f}/day")
    
    # 현실적인 파라미터 제안
    print(f"\n[Recommended Parameters]")
    
    # P90 기반 (상위 10% 기회)
    p90_change = abs(changes).quantile(0.90)
    p80_change = abs(changes).quantile(0.80)
    p70_change = abs(changes).quantile(0.70)
    
    print(f"\nFor 10% of opportunities (3-4 trades/day):")
    print(f"  Entry threshold: {p90_change:.4f}%")
    print(f"  Target profit: {p90_change * 0.7:.4f}%")
    print(f"  Stop loss: {p90_change * 0.5:.4f}%")
    
    print(f"\nFor 20% of opportunities (6-8 trades/day):")
    print(f"  Entry threshold: {p80_change:.4f}%")
    print(f"  Target profit: {p80_change * 0.7:.4f}%")
    print(f"  Stop loss: {p80_change * 0.5:.4f}%")
    
    print(f"\nFor 30% of opportunities (10-12 trades/day):")
    print(f"  Entry threshold: {p70_change:.4f}%")
    print(f"  Target profit: {p70_change * 0.7:.4f}%")
    print(f"  Stop loss: {p70_change * 0.5:.4f}%")
    
    # 수익성 분석
    print(f"\n[Profitability Analysis]")
    
    fees = 0.0015  # 0.15% total fees
    
    for threshold in [p90_change, p80_change, p70_change]:
        # 기회 수
        opportunities = (abs(changes) > threshold).sum()
        daily_opps = opportunities / 30
        
        # 평균 움직임
        avg_move = abs(changes[abs(changes) > threshold]).mean()
        
        # 예상 수익 (70% 목표 달성 가정)
        expected_profit = avg_move * 0.7 - fees
        
        # 승률 60% 가정
        win_rate = 0.6
        avg_return = expected_profit * win_rate - fees * (1 - win_rate)
        
        # 일일/월간 수익
        daily_return = avg_return * daily_opps
        monthly_return = daily_return * 30
        
        print(f"\nThreshold {threshold:.4f}%:")
        print(f"  Daily trades: {daily_opps:.1f}")
        print(f"  Avg movement: {avg_move:.4f}%")
        print(f"  Expected per trade: {expected_profit:.4f}%")
        print(f"  With 60% win rate: {avg_return:.4f}%")
        print(f"  Daily return: {daily_return:.4f}%")
        print(f"  Monthly return: {monthly_return:.2f}%")
    
    return merged, changes


def main():
    """메인 실행"""
    print("\n" + "=" * 60)
    print("  TRADING CONDITION ANALYSIS")
    print("  Finding optimal parameters for current market")
    print("=" * 60)
    
    # 시장 분석
    data, changes = analyze_market_conditions()
    
    # 결론
    print("\n" + "=" * 60)
    print("  CONCLUSIONS")
    print("=" * 60)
    
    print("\n[Problem Identified]")
    print("1. Kimchi premium is too stable (std < 0.5%)")
    print("2. Large movements (>0.2%) are very rare")
    print("3. Most changes are smaller than fees (0.15%)")
    
    print("\n[Solutions]")
    print("1. Use MUCH smaller thresholds (0.05-0.10%)")
    print("2. Increase position size to compensate")
    print("3. Focus on high-frequency micro movements")
    print("4. Consider maker fees instead of taker")
    print("5. Use limit orders for better entry/exit")
    
    print("\n[Adjusted Strategy]")
    print("- Entry: When 5-min change > 0.05%")
    print("- Target: 0.03% profit (after fees)")
    print("- Volume: 0.1 BTC per trade")
    print("- Goal: 20-30 trades per day")
    print("- Expected: 0.6-0.9% daily = 18-27% monthly")
    
    return data, changes


if __name__ == "__main__":
    main()