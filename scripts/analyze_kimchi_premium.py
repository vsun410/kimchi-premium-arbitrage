"""
Analyze Kimchi Premium Statistics
Phase 3: 김치 프리미엄 분석
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger


def load_and_calculate_premium():
    """데이터 로드 및 김프 계산"""
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
    
    # 1분 데이터 병합
    merged_1m = pd.merge(
        binance_df[['close']].rename(columns={'close': 'binance_close'}),
        upbit_df[['close']].rename(columns={'close': 'upbit_close'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # 김치 프리미엄 계산 (USD/KRW = 1330)
    USD_KRW = 1330
    merged_1m['binance_krw'] = merged_1m['binance_close'] * USD_KRW
    merged_1m['kimchi_premium'] = ((merged_1m['upbit_close'] - merged_1m['binance_krw']) / merged_1m['binance_krw']) * 100
    
    return merged_1m


def analyze_premium_statistics(df):
    """김프 통계 분석"""
    
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM STATISTICS")
    print("=" * 60)
    
    # 기본 통계
    print("\n[Basic Statistics]")
    print(f"Mean: {df['kimchi_premium'].mean():.2f}%")
    print(f"Median: {df['kimchi_premium'].median():.2f}%")
    print(f"Std Dev: {df['kimchi_premium'].std():.2f}%")
    print(f"Min: {df['kimchi_premium'].min():.2f}%")
    print(f"Max: {df['kimchi_premium'].max():.2f}%")
    
    # 백분위수
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n[Percentiles]")
    for p in percentiles:
        value = df['kimchi_premium'].quantile(p/100)
        print(f"{p:3d}%: {value:6.2f}%")
    
    # 임계값별 기회
    print("\n[Trading Opportunities by Threshold]")
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    print(f"{'Threshold':<10} {'Count':>8} {'Percent':>8} {'Avg Duration':>12}")
    print("-" * 40)
    
    for threshold in thresholds:
        above_threshold = df['kimchi_premium'].abs() > threshold
        count = above_threshold.sum()
        percent = (count / len(df)) * 100
        
        # 평균 지속 시간 계산
        changes = above_threshold.diff().fillna(0)
        starts = changes == 1
        ends = changes == -1
        
        if starts.sum() > 0:
            avg_duration = count / starts.sum()
        else:
            avg_duration = 0
        
        print(f"{threshold:6.1f}%    {count:8d} {percent:7.2f}% {avg_duration:11.1f} min")
    
    # 시간대별 분석
    print("\n[Hourly Analysis]")
    df['hour'] = df.index.hour
    hourly_stats = df.groupby('hour')['kimchi_premium'].agg(['mean', 'std', 'count'])
    
    print(f"{'Hour':<6} {'Mean':>8} {'Std Dev':>8} {'Count':>8}")
    print("-" * 32)
    
    for hour, row in hourly_stats.iterrows():
        print(f"{hour:2d}:00  {row['mean']:7.2f}% {row['std']:7.2f}% {row['count']:8.0f}")
    
    return df


def plot_premium_analysis(df):
    """김프 시각화"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 김프 시계열
    ax = axes[0, 0]
    ax.plot(df.index, df['kimchi_premium'], linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='g', linestyle='--', alpha=0.3, label='Entry 2%')
    ax.axhline(y=-2, color='g', linestyle='--', alpha=0.3)
    ax.set_title('Kimchi Premium Time Series')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 히스토그램
    ax = axes[0, 1]
    ax.hist(df['kimchi_premium'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Premium Distribution')
    ax.set_xlabel('Premium (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 3. 일별 평균
    ax = axes[1, 0]
    daily_mean = df.resample('D')['kimchi_premium'].mean()
    ax.bar(daily_mean.index, daily_mean.values, alpha=0.7)
    ax.set_title('Daily Average Premium')
    ax.set_ylabel('Premium (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 4. 시간대별 평균
    ax = axes[1, 1]
    hourly_mean = df.groupby(df.index.hour)['kimchi_premium'].mean()
    ax.bar(hourly_mean.index, hourly_mean.values, alpha=0.7)
    ax.set_title('Hourly Average Premium')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Premium (%)')
    ax.grid(True, alpha=0.3)
    
    # 5. Rolling 평균/표준편차
    ax = axes[2, 0]
    rolling_mean = df['kimchi_premium'].rolling(window=60).mean()  # 1시간
    rolling_std = df['kimchi_premium'].rolling(window=60).std()
    ax.plot(df.index, rolling_mean, label='1H Mean', alpha=0.8)
    ax.fill_between(df.index, 
                     rolling_mean - rolling_std, 
                     rolling_mean + rolling_std, 
                     alpha=0.2, label='±1 Std')
    ax.set_title('Rolling Statistics (1 Hour Window)')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Q-Q Plot
    ax = axes[2, 1]
    from scipy import stats
    stats.probplot(df['kimchi_premium'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Test)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = f"results/analysis/kimchi_premium_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()  # Show 대신 close


def find_profitable_periods(df):
    """수익 가능 기간 찾기"""
    
    print("\n" + "=" * 60)
    print("  PROFITABLE PERIODS ANALYSIS")
    print("=" * 60)
    
    # 높은 김프 기간 찾기
    high_premium = df['kimchi_premium'] > 2.0
    
    # 연속된 기간 찾기
    changes = high_premium.diff().fillna(0)
    starts = df.index[changes == 1]
    ends = df.index[changes == -1]
    
    if len(starts) > 0:
        periods = []
        for i in range(min(len(starts), len(ends))):
            duration = (ends[i] - starts[i]).total_seconds() / 60  # minutes
            max_premium = df.loc[starts[i]:ends[i], 'kimchi_premium'].max()
            avg_premium = df.loc[starts[i]:ends[i], 'kimchi_premium'].mean()
            
            periods.append({
                'start': starts[i],
                'end': ends[i],
                'duration_min': duration,
                'max_premium': max_premium,
                'avg_premium': avg_premium
            })
        
        if len(periods) > 0:
            periods_df = pd.DataFrame(periods)
            periods_df = periods_df.sort_values('duration_min', ascending=False)
        else:
            print("No profitable periods found above 2% threshold")
            return
        
        print("\n[Top 10 Longest High Premium Periods (>2%)]")
        print(f"{'Start':<20} {'End':<20} {'Duration':>10} {'Max':>8} {'Avg':>8}")
        print("-" * 70)
        
        for _, row in periods_df.head(10).iterrows():
            print(f"{row['start'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['end'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['duration_min']:9.0f}m "
                  f"{row['max_premium']:7.2f}% "
                  f"{row['avg_premium']:7.2f}%")
        
        print(f"\nTotal periods: {len(periods_df)}")
        print(f"Average duration: {periods_df['duration_min'].mean():.1f} minutes")
        print(f"Total time above 2%: {periods_df['duration_min'].sum():.0f} minutes")
        print(f"Percentage of time above 2%: {(periods_df['duration_min'].sum() / (len(df))):.2%}")


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM DEEP ANALYSIS")
    print("=" * 60)
    
    # 데이터 로드
    print("\nLoading data...")
    df = load_and_calculate_premium()
    
    print(f"Data period: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # 통계 분석
    df = analyze_premium_statistics(df)
    
    # 수익 기간 분석
    find_profitable_periods(df)
    
    # 시각화
    print("\nGenerating visualizations...")
    plot_premium_analysis(df)
    
    # 결론
    print("\n" + "=" * 60)
    print("  CONCLUSIONS")
    print("=" * 60)
    
    mean_premium = df['kimchi_premium'].mean()
    std_premium = df['kimchi_premium'].std()
    
    if mean_premium > 1.0:
        print(f"[POSITIVE] Average premium {mean_premium:.2f}% suggests profit potential")
    else:
        print(f"[NEUTRAL] Average premium {mean_premium:.2f}% is relatively low")
    
    if std_premium > 2.0:
        print(f"[POSITIVE] High volatility {std_premium:.2f}% provides trading opportunities")
    else:
        print(f"[NEGATIVE] Low volatility {std_premium:.2f}% limits opportunities")
    
    # 최적 임계값 제안
    percentile_95 = df['kimchi_premium'].quantile(0.95)
    percentile_80 = df['kimchi_premium'].quantile(0.80)
    
    print(f"\n[SUGGESTED PARAMETERS]")
    print(f"Conservative: Entry={percentile_95:.1f}%, Exit={percentile_95/2:.1f}%")
    print(f"Moderate: Entry={percentile_80:.1f}%, Exit={percentile_80/2:.1f}%")
    print(f"Aggressive: Entry={mean_premium + std_premium:.1f}%, Exit={mean_premium:.1f}%")


if __name__ == "__main__":
    main()