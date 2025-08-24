"""
Analyze Current Kimchi Premium Patterns
Phase 3: 현재 김프 패턴 상세 분석
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger


def load_1min_data(days_back=7):
    """최근 1분 데이터 로드"""
    import glob
    
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    # 데이터 로드
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 최근 N일만
    cutoff_date = binance_df.index[-1] - timedelta(days=days_back)
    binance_df = binance_df[binance_df.index >= cutoff_date]
    upbit_df = upbit_df[upbit_df.index >= cutoff_date]
    
    # 병합
    merged = pd.merge(
        binance_df[['close']].rename(columns={'close': 'binance_close'}),
        upbit_df[['close']].rename(columns={'close': 'upbit_close'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # 김프 계산
    USD_KRW = 1330
    merged['binance_krw'] = merged['binance_close'] * USD_KRW
    merged['kimchi_premium'] = ((merged['upbit_close'] - merged['binance_krw']) / merged['binance_krw']) * 100
    
    return merged


def analyze_micro_patterns(df):
    """마이크로 패턴 분석"""
    
    print("\n" + "=" * 60)
    print("  MICRO PATTERN ANALYSIS (1-MIN DATA)")
    print("=" * 60)
    
    premium = df['kimchi_premium']
    
    # 기본 통계
    print("\n[1-Minute Statistics]")
    print(f"Mean: {premium.mean():.3f}%")
    print(f"Std: {premium.std():.3f}%")
    print(f"Min: {premium.min():.3f}%")
    print(f"Max: {premium.max():.3f}%")
    
    # 백분위수
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n[Percentiles]")
    for p in percentiles:
        value = premium.quantile(p/100)
        print(f"P{p:02d}: {value:6.3f}%")
    
    # 임계값별 기회 (1분 데이터)
    print("\n[Trading Opportunities (1-min granularity)]")
    thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    print(f"{'Threshold':<10} {'Count':>8} {'Percent':>8} {'Avg Duration':>15} {'Max Duration':>15}")
    print("-" * 60)
    
    for threshold in thresholds:
        above_threshold = premium.abs() > threshold
        count = above_threshold.sum()
        percent = (count / len(df)) * 100
        
        # 연속 기간 계산
        changes = above_threshold.diff().fillna(0)
        starts = changes == 1
        
        if starts.sum() > 0:
            # 평균 지속 시간
            avg_duration = count / starts.sum()
            
            # 최대 지속 시간 찾기
            consecutive = []
            current_streak = 0
            for val in above_threshold:
                if val:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        consecutive.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                consecutive.append(current_streak)
            
            max_duration = max(consecutive) if consecutive else 0
        else:
            avg_duration = 0
            max_duration = 0
        
        print(f"{threshold:6.2f}%    {count:8d} {percent:7.2f}% {avg_duration:14.1f} min {max_duration:14d} min")
    
    return premium


def find_profitable_windows(premium):
    """수익 가능 시간대 찾기"""
    
    print("\n" + "=" * 60)
    print("  PROFITABLE TIME WINDOWS")
    print("=" * 60)
    
    # 시간대별 분석
    df = pd.DataFrame({'premium': premium})
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    
    # 시간별 평균
    hourly_stats = df.groupby('hour')['premium'].agg(['mean', 'std', 'count'])
    
    print("\n[Best Hours for Trading]")
    best_hours = hourly_stats.nlargest(5, 'mean')
    
    print(f"{'Hour':<6} {'Mean':>8} {'Std':>8} {'Count':>8}")
    print("-" * 32)
    for hour, row in best_hours.iterrows():
        print(f"{hour:2d}:00  {row['mean']:7.3f}% {row['std']:7.3f}% {row['count']:8.0f}")
    
    # 연속 높은 김프 기간
    high_premium_threshold = premium.quantile(0.75)
    high_periods = premium > high_premium_threshold
    
    # 연속 기간 찾기
    changes = high_periods.diff().fillna(0)
    starts = premium.index[changes == 1]
    ends = premium.index[changes == -1]
    
    if len(starts) > 0 and len(ends) > 0:
        periods = []
        for i in range(min(len(starts), len(ends))):
            duration = (ends[i] - starts[i]).total_seconds() / 60
            max_val = premium[starts[i]:ends[i]].max()
            avg_val = premium[starts[i]:ends[i]].mean()
            periods.append({
                'start': starts[i],
                'duration': duration,
                'max': max_val,
                'avg': avg_val
            })
        
        periods_df = pd.DataFrame(periods)
        if len(periods_df) > 0:
            periods_df = periods_df.sort_values('duration', ascending=False)
            
            print("\n[Longest High Premium Periods (>P75)]")
            print(f"{'Start Time':<20} {'Duration':>10} {'Max':>8} {'Avg':>8}")
            print("-" * 50)
            
            for _, row in periods_df.head(10).iterrows():
                print(f"{row['start'].strftime('%m-%d %H:%M'):<20} "
                      f"{row['duration']:9.0f}m {row['max']:7.3f}% {row['avg']:7.3f}%")


def suggest_realistic_strategy(premium):
    """현실적인 전략 제안"""
    
    print("\n" + "=" * 60)
    print("  REALISTIC STRATEGY FOR CURRENT MARKET")
    print("=" * 60)
    
    mean = premium.mean()
    std = premium.std()
    p90 = premium.quantile(0.90)
    p95 = premium.quantile(0.95)
    p99 = premium.quantile(0.99)
    
    print(f"\n[Current Market Characteristics]")
    print(f"Average premium: {mean:.3f}%")
    print(f"Volatility (std): {std:.3f}%")
    print(f"Realistic high (P90): {p90:.3f}%")
    print(f"Rare high (P95): {p95:.3f}%")
    print(f"Extreme (P99): {p99:.3f}%")
    
    # 전략 제안
    print("\n[Strategy Recommendations]")
    
    # 1. 평균 회귀 전략
    print("\n1. MEAN REVERSION STRATEGY")
    print(f"   - Long Entry: Premium < {mean - std:.3f}%")
    print(f"   - Short Entry: Premium > {mean + std:.3f}%")
    print(f"   - Exit: Premium returns to {mean:.3f}%")
    print(f"   - Stop Loss: {2 * std:.3f}% from entry")
    
    # 2. 모멘텀 전략
    print("\n2. MOMENTUM STRATEGY")
    print(f"   - Entry: Premium breaks above {p90:.3f}%")
    print(f"   - Add position: At {p95:.3f}%")
    print(f"   - Exit: Premium falls below {p90:.3f}%")
    print(f"   - Time limit: Max 30 minutes per trade")
    
    # 3. 스캘핑 전략
    print("\n3. SCALPING STRATEGY")
    print(f"   - Entry: Any move > {std:.3f}% from 5-min average")
    print(f"   - Target: {std * 0.3:.3f}% profit")
    print(f"   - Stop: {std * 0.5:.3f}% loss")
    print(f"   - Max trades: 10 per day")
    
    # 예상 수익성
    print("\n[Expected Performance]")
    
    # 평균 회귀 기회
    mean_reversion_signals = ((premium > mean + std) | (premium < mean - std)).sum()
    daily_signals = mean_reversion_signals / (len(premium) / (60 * 24))
    
    print(f"Mean reversion signals per day: {daily_signals:.1f}")
    
    # 모멘텀 기회
    momentum_signals = (premium > p90).sum()
    daily_momentum = momentum_signals / (len(premium) / (60 * 24))
    
    print(f"Momentum signals per day: {daily_momentum:.1f}")
    
    # 예상 수익 (보수적)
    avg_move = std * 0.3  # 평균 수익
    fee_cost = 0.15  # 왕복 수수료 0.15%
    net_per_trade = avg_move - fee_cost
    
    print(f"\nExpected per trade (after fees): {net_per_trade:.3f}%")
    print(f"Daily expected (5 trades): {net_per_trade * 5:.3f}%")
    print(f"Monthly expected: {net_per_trade * 5 * 20:.2f}%")


def plot_current_patterns(df):
    """현재 패턴 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    premium = df['kimchi_premium']
    
    # 1. 최근 24시간
    ax = axes[0, 0]
    recent_24h = premium[premium.index >= premium.index[-1] - timedelta(hours=24)]
    ax.plot(recent_24h.index, recent_24h.values, linewidth=0.5)
    ax.axhline(y=recent_24h.mean(), color='r', linestyle='--', label=f'Mean: {recent_24h.mean():.3f}%')
    ax.fill_between(recent_24h.index, -1.5, 1.5, alpha=0.2, color='green', label='Target Range')
    ax.set_title('Last 24 Hours (1-min)')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 분포
    ax = axes[0, 1]
    ax.hist(premium, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=premium.mean(), color='r', linestyle='--', label=f'Mean: {premium.mean():.3f}%')
    ax.axvline(x=-1.5, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=1.5, color='g', linestyle='--', alpha=0.5, label='±1.5%')
    ax.set_title('Premium Distribution (Last 7 Days)')
    ax.set_xlabel('Premium (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 자기상관
    ax = axes[1, 0]
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(premium.iloc[-1440:], ax=ax)  # 최근 24시간
    ax.set_title('Autocorrelation (Last 24h)')
    ax.set_xlabel('Lag (minutes)')
    ax.grid(True, alpha=0.3)
    
    # 4. 시간대별 박스플롯
    ax = axes[1, 1]
    df_temp = pd.DataFrame({'premium': premium, 'hour': premium.index.hour})
    df_temp.boxplot(column='premium', by='hour', ax=ax)
    ax.set_title('Hourly Distribution')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Premium (%)')
    plt.sca(ax)
    plt.xticks(rotation=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = f"results/analysis/current_premium_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  CURRENT KIMCHI PREMIUM ANALYSIS")
    print("  Focusing on -1.5% to 1.5% Range")
    print("=" * 60)
    
    # 최근 7일 1분 데이터 로드
    print("\nLoading last 7 days of 1-minute data...")
    df = load_1min_data(days_back=7)
    
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Data points: {len(df):,} (1-min candles)")
    
    # 마이크로 패턴 분석
    premium = analyze_micro_patterns(df)
    
    # 수익 시간대 찾기
    find_profitable_windows(premium)
    
    # 현실적 전략 제안
    suggest_realistic_strategy(premium)
    
    # 시각화
    print("\nGenerating visualizations...")
    plot_current_patterns(df)
    
    # 최종 결론
    print("\n" + "=" * 60)
    print("  FINAL CONCLUSIONS")
    print("=" * 60)
    
    if premium.std() < 1.0:
        print("\n[!] ULTRA-LOW VOLATILITY DETECTED")
        print("Current market is NOT suitable for traditional kimchi arbitrage.")
        print("\nConsider alternative strategies:")
        print("1. Market Making: Provide liquidity on both exchanges")
        print("2. Statistical Arbitrage: Trade mean reversion patterns")
        print("3. Cross-Exchange Arbitrage: Look for brief spikes")
        print("4. Wait for volatility to return before deploying capital")
    else:
        print("\n[OK] Market shows tradeable volatility")
        print("Proceed with adaptive strategy using current parameters")


if __name__ == "__main__":
    main()