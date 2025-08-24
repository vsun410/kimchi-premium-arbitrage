"""
Analyze Kimchi Premium Trends Over Time
Phase 3: 시간대별 김프 트렌드 분석
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

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


def analyze_by_period(df):
    """기간별 김프 분석"""
    
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM TREND ANALYSIS")
    print("=" * 60)
    
    # 월별 분석
    print("\n[Monthly Analysis]")
    df['year_month'] = df.index.to_period('M')
    monthly_stats = df.groupby('year_month')['kimchi_premium'].agg(['mean', 'std', 'min', 'max', 'count'])
    
    print(f"{'Month':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Days':>6}")
    print("-" * 50)
    
    for month, row in monthly_stats.iterrows():
        days = row['count'] / (60 * 24)  # Convert minutes to days
        print(f"{str(month):<10} {row['mean']:7.2f}% {row['std']:7.2f}% "
              f"{row['min']:7.2f}% {row['max']:7.2f}% {days:5.0f}")
    
    # 주별 분석 (최근 12주)
    print("\n[Weekly Analysis - Last 12 Weeks]")
    df['week'] = df.index.to_period('W')
    weekly_stats = df.groupby('week')['kimchi_premium'].agg(['mean', 'std', 'min', 'max']).tail(12)
    
    print(f"{'Week':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    
    for week, row in weekly_stats.iterrows():
        week_start = week.start_time.strftime('%Y-%m-%d')
        print(f"{week_start:<20} {row['mean']:7.2f}% {row['std']:7.2f}% "
              f"{row['min']:7.2f}% {row['max']:7.2f}%")
    
    # 최근 30일 vs 전체 기간 비교
    print("\n[Recent 30 Days vs Historical Average]")
    recent_30d = df[df.index >= df.index[-1] - timedelta(days=30)]
    recent_7d = df[df.index >= df.index[-1] - timedelta(days=7)]
    recent_1d = df[df.index >= df.index[-1] - timedelta(days=1)]
    
    periods = [
        ('Last 24 hours', recent_1d),
        ('Last 7 days', recent_7d),
        ('Last 30 days', recent_30d),
        ('All time', df)
    ]
    
    print(f"{'Period':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P25':>8} {'P75':>8}")
    print("-" * 70)
    
    for period_name, period_df in periods:
        if len(period_df) > 0:
            mean = period_df['kimchi_premium'].mean()
            std = period_df['kimchi_premium'].std()
            min_val = period_df['kimchi_premium'].min()
            max_val = period_df['kimchi_premium'].max()
            p25 = period_df['kimchi_premium'].quantile(0.25)
            p75 = period_df['kimchi_premium'].quantile(0.75)
            
            print(f"{period_name:<15} {mean:7.2f}% {std:7.2f}% {min_val:7.2f}% "
                  f"{max_val:7.2f}% {p25:7.2f}% {p75:7.2f}%")
    
    return df, recent_30d, recent_7d


def plot_trend_analysis(df, recent_30d, recent_7d):
    """트렌드 시각화"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 전체 기간 김프 with 이동평균
    ax = axes[0, 0]
    daily_mean = df.resample('D')['kimchi_premium'].mean()
    ax.plot(daily_mean.index, daily_mean.values, linewidth=1, alpha=0.7, label='Daily Mean')
    
    # 30일 이동평균
    ma30 = daily_mean.rolling(window=30, min_periods=1).mean()
    ax.plot(ma30.index, ma30.values, linewidth=2, color='red', label='30-Day MA')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Daily Average Premium with 30-Day MA')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 월별 박스플롯
    ax = axes[0, 1]
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # 최근 6개월만
    recent_6m = df[df.index >= df.index[-1] - timedelta(days=180)]
    recent_6m['year_month'] = recent_6m.index.strftime('%Y-%m')
    
    if len(recent_6m['year_month'].unique()) > 0:
        sns.boxplot(data=recent_6m, x='year_month', y='kimchi_premium', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Monthly Distribution (Last 6 Months)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Premium (%)')
    ax.grid(True, alpha=0.3)
    
    # 3. 최근 30일 상세
    ax = axes[1, 0]
    if len(recent_30d) > 0:
        ax.plot(recent_30d.index, recent_30d['kimchi_premium'], linewidth=0.5, alpha=0.7)
        ax.axhline(y=recent_30d['kimchi_premium'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {recent_30d['kimchi_premium'].mean():.2f}%")
        ax.fill_between(recent_30d.index, -1.5, 1.5, alpha=0.2, color='green', 
                        label='Current Range (-1.5% to 1.5%)')
    
    ax.set_title('Last 30 Days Detail')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 최근 7일 상세
    ax = axes[1, 1]
    if len(recent_7d) > 0:
        hourly_mean = recent_7d.resample('H')['kimchi_premium'].mean()
        ax.plot(hourly_mean.index, hourly_mean.values, linewidth=1, alpha=0.8)
        ax.axhline(y=recent_7d['kimchi_premium'].mean(), color='r', linestyle='--',
                   label=f"Mean: {recent_7d['kimchi_premium'].mean():.2f}%")
        ax.fill_between(hourly_mean.index, -1.5, 1.5, alpha=0.2, color='green',
                        label='Current Range')
    
    ax.set_title('Last 7 Days (Hourly Average)')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 트렌드 변화
    ax = axes[2, 0]
    weekly_mean = df.resample('W')['kimchi_premium'].mean()
    ax.plot(weekly_mean.index, weekly_mean.values, marker='o', markersize=3, linewidth=1)
    
    # 트렌드라인
    if len(weekly_mean) > 1:
        z = np.polyfit(range(len(weekly_mean)), weekly_mean.values, 1)
        p = np.poly1d(z)
        ax.plot(weekly_mean.index, p(range(len(weekly_mean))), 
                "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}% per week')
    
    ax.set_title('Weekly Average with Trend')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 변동성 변화
    ax = axes[2, 1]
    daily_std = df.resample('D')['kimchi_premium'].std()
    ma7_std = daily_std.rolling(window=7, min_periods=1).mean()
    
    ax.plot(daily_std.index, daily_std.values, linewidth=0.5, alpha=0.5, label='Daily Volatility')
    ax.plot(ma7_std.index, ma7_std.values, linewidth=2, color='red', label='7-Day MA')
    
    ax.set_title('Volatility Trend')
    ax.set_ylabel('Standard Deviation (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = f"results/analysis/premium_trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def suggest_adaptive_strategy(df, recent_30d):
    """적응형 전략 제안"""
    
    print("\n" + "=" * 60)
    print("  ADAPTIVE STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    
    historical_mean = df['kimchi_premium'].mean()
    historical_std = df['kimchi_premium'].std()
    
    recent_mean = recent_30d['kimchi_premium'].mean()
    recent_std = recent_30d['kimchi_premium'].std()
    
    print(f"\n[Market Regime Change Detection]")
    print(f"Historical: Mean={historical_mean:.2f}%, Std={historical_std:.2f}%")
    print(f"Recent 30d: Mean={recent_mean:.2f}%, Std={recent_std:.2f}%")
    
    change_ratio = recent_mean / historical_mean if historical_mean != 0 else 0
    print(f"Change ratio: {change_ratio:.2f}x")
    
    if change_ratio < 0.3:
        print("\n>>> SIGNIFICANT REGIME CHANGE DETECTED <<<")
        print("The market has fundamentally changed from historical patterns.")
        
    # 적응형 파라미터 제안
    print("\n[Adaptive Parameters Based on Recent Data]")
    
    # 최근 데이터 기반 백분위수
    p95_recent = recent_30d['kimchi_premium'].quantile(0.95)
    p90_recent = recent_30d['kimchi_premium'].quantile(0.90)
    p80_recent = recent_30d['kimchi_premium'].quantile(0.80)
    p70_recent = recent_30d['kimchi_premium'].quantile(0.70)
    
    print(f"\nRecent percentiles:")
    print(f"95th: {p95_recent:.2f}%")
    print(f"90th: {p90_recent:.2f}%")
    print(f"80th: {p80_recent:.2f}%")
    print(f"70th: {p70_recent:.2f}%")
    
    # 전략 제안
    print("\n[Strategy Recommendations]")
    
    if recent_std < 1.0:
        print("\n1. LOW VOLATILITY STRATEGY")
        print(f"   - Entry: {abs(p95_recent):.2f}%")
        print(f"   - Exit: {abs(p95_recent) * 0.3:.2f}%")
        print(f"   - Stop Loss: {abs(p95_recent) * 1.5:.2f}%")
        print("   - Position Size: Increase to 0.2 BTC (low risk)")
        
    elif recent_std < 2.0:
        print("\n1. MODERATE VOLATILITY STRATEGY")
        print(f"   - Entry: {abs(p90_recent):.2f}%")
        print(f"   - Exit: {abs(p90_recent) * 0.4:.2f}%")
        print(f"   - Stop Loss: {abs(p90_recent) * 1.3:.2f}%")
        print("   - Position Size: Standard 0.1 BTC")
        
    else:
        print("\n1. HIGH VOLATILITY STRATEGY")
        print(f"   - Entry: {abs(p80_recent):.2f}%")
        print(f"   - Exit: {abs(p80_recent) * 0.5:.2f}%")
        print(f"   - Stop Loss: {abs(p80_recent) * 1.2:.2f}%")
        print("   - Position Size: Reduce to 0.05 BTC (high risk)")
    
    # 양방향 전략
    if recent_30d['kimchi_premium'].min() < -0.5 and recent_30d['kimchi_premium'].max() > 0.5:
        print("\n2. BI-DIRECTIONAL STRATEGY")
        print("   - Long Entry (positive premium): >{:.2f}%".format(abs(p80_recent)))
        print("   - Short Entry (negative premium): <{:.2f}%".format(-abs(p80_recent)))
        print("   - Exit: When premium crosses 0%")
    
    # 거래 빈도 예측
    recent_above_1 = (recent_30d['kimchi_premium'].abs() > 1.0).sum()
    recent_above_15 = (recent_30d['kimchi_premium'].abs() > 1.5).sum()
    
    total_minutes = len(recent_30d)
    
    print("\n[Expected Trading Frequency]")
    print(f"Opportunities >1.0%: {recent_above_1} times ({recent_above_1/total_minutes*100:.1f}% of time)")
    print(f"Opportunities >1.5%: {recent_above_15} times ({recent_above_15/total_minutes*100:.1f}% of time)")
    print(f"Average per day: {recent_above_1 / 30:.1f} trades")
    
    return recent_mean, recent_std


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM TREND ANALYSIS")
    print("  Focus on Recent vs Historical Patterns")
    print("=" * 60)
    
    # 데이터 로드
    print("\nLoading data...")
    df = load_and_calculate_premium()
    
    print(f"Data period: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # 기간별 분석
    df, recent_30d, recent_7d = analyze_by_period(df)
    
    # 트렌드 시각화
    print("\nGenerating trend visualizations...")
    plot_trend_analysis(df, recent_30d, recent_7d)
    
    # 적응형 전략 제안
    recent_mean, recent_std = suggest_adaptive_strategy(df, recent_30d)
    
    # 결론
    print("\n" + "=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)
    
    if recent_mean < 2.0 and recent_std < 2.0:
        print("\n[!] MARKET HAS SIGNIFICANTLY CHANGED")
        print("Recent premium is much lower than historical average.")
        print("You should use RECENT data for backtesting, not full historical data.")
        print("\nRecommendations:")
        print("1. Use only last 30-60 days for parameter optimization")
        print("2. Lower entry thresholds to 1.0-1.5%")
        print("3. Consider bi-directional trading for negative premiums")
        print("4. Implement adaptive parameters that update weekly")
    else:
        print("\n[OK] Market conditions relatively stable")
        print("Historical data can be used with some adjustments")


if __name__ == "__main__":
    main()