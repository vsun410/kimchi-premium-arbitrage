"""
Scalping Strategy Analysis
목표: 4000만원 기준 거래당 10만원(0.25%) 수익
월 2-3% 수익 목표 (80-120회 거래)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


def load_1min_data_with_real_premium(days_back=30):
    """1분 데이터 로드 (실제 환율)"""
    import glob
    
    rate_manager = get_exchange_rate_manager()
    
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
    
    # 최근 N일
    cutoff_date = binance_df.index[-1] - timedelta(days=days_back)
    binance_df = binance_df[binance_df.index >= cutoff_date]
    upbit_df = upbit_df[upbit_df.index >= cutoff_date]
    
    # 병합 (1분 데이터 유지)
    merged = pd.merge(
        binance_df[['close', 'volume']].rename(columns={'close': 'binance_close', 'volume': 'binance_volume'}),
        upbit_df[['close', 'volume']].rename(columns={'close': 'upbit_close', 'volume': 'upbit_volume'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # 정확한 김프 계산
    current_rate = rate_manager.current_rate
    merged['binance_krw'] = merged['binance_close'] * current_rate
    merged['kimchi_premium'] = ((merged['upbit_close'] - merged['binance_krw']) / merged['binance_krw']) * 100
    
    # 기술적 지표 추가
    merged['premium_ma5'] = merged['kimchi_premium'].rolling(5).mean()
    merged['premium_ma15'] = merged['kimchi_premium'].rolling(15).mean()
    merged['premium_std5'] = merged['kimchi_premium'].rolling(5).std()
    merged['premium_change'] = merged['kimchi_premium'].diff()
    
    return merged.dropna()


def analyze_scalping_opportunities(df, target_profit=0.25, total_cost=0.15):
    """
    스캘핑 기회 분석
    
    Args:
        df: 데이터프레임
        target_profit: 목표 수익률 (0.25%)
        total_cost: 총 비용 (0.15%)
    """
    
    print("\n" + "=" * 60)
    print("  SCALPING OPPORTUNITY ANALYSIS")
    print("=" * 60)
    
    print(f"\n[Target]")
    print(f"Per Trade Profit: {target_profit}% (10만원 on 4000만원)")
    print(f"Total Cost (fees): {total_cost}%")
    print(f"Required Move: {target_profit + total_cost}% = {target_profit + total_cost:.2f}%")
    
    # 1. 단순 진입/청산 기회
    print("\n[1. Simple Entry-Exit Opportunities]")
    
    required_move = target_profit + total_cost  # 0.40%
    
    # 연속된 김프 변화 추적
    opportunities = []
    
    for i in range(1, len(df)):
        current_premium = df['kimchi_premium'].iloc[i]
        
        # 앞으로 N분 동안 required_move 이상 움직이는지 확인
        for j in range(1, min(31, len(df) - i)):  # 최대 30분
            future_premium = df['kimchi_premium'].iloc[i + j]
            move = abs(future_premium - current_premium)
            
            if move >= required_move:
                opportunities.append({
                    'entry_time': df.index[i],
                    'exit_time': df.index[i + j],
                    'entry_premium': current_premium,
                    'exit_premium': future_premium,
                    'move': move,
                    'duration': j,
                    'direction': 'long' if future_premium > current_premium else 'short'
                })
                break
    
    if opportunities:
        opp_df = pd.DataFrame(opportunities)
        
        print(f"Total opportunities: {len(opp_df)}")
        print(f"Daily average: {len(opp_df) / (len(df) / (60 * 24)):.1f}")
        print(f"Average duration: {opp_df['duration'].mean():.1f} minutes")
        print(f"Average move: {opp_df['move'].mean():.3f}%")
        
        # 방향별 분석
        long_opps = opp_df[opp_df['direction'] == 'long']
        short_opps = opp_df[opp_df['direction'] == 'short']
        
        print(f"\nLong opportunities: {len(long_opps)} ({len(long_opps)/len(opp_df)*100:.1f}%)")
        print(f"Short opportunities: {len(short_opps)} ({len(short_opps)/len(opp_df)*100:.1f}%)")
    
    # 2. 평균 회귀 전략
    print("\n[2. Mean Reversion Opportunities]")
    
    mean = df['kimchi_premium'].mean()
    std = df['kimchi_premium'].std()
    
    # 극단값에서 평균으로 회귀
    extreme_high = mean + std
    extreme_low = mean - std
    
    mean_reversion_signals = 0
    
    for i in range(len(df) - 1):
        current = df['kimchi_premium'].iloc[i]
        
        # 극단에서 진입
        if abs(current - mean) > std:
            # 다음 30분 내 평균 근처로 회귀하는지 확인
            for j in range(1, min(31, len(df) - i)):
                future = df['kimchi_premium'].iloc[i + j]
                if abs(future - mean) < std * 0.5:
                    move = abs(future - current)
                    if move >= required_move:
                        mean_reversion_signals += 1
                        break
    
    print(f"Mean reversion signals: {mean_reversion_signals}")
    print(f"Daily average: {mean_reversion_signals / (len(df) / (60 * 24)):.1f}")
    
    # 3. 볼린저 밴드 전략
    print("\n[3. Bollinger Band Strategy]")
    
    df['bb_upper'] = df['premium_ma15'] + 2 * df['premium_std5']
    df['bb_lower'] = df['premium_ma15'] - 2 * df['premium_std5']
    
    bb_signals = 0
    
    for i in range(15, len(df) - 1):
        current = df['kimchi_premium'].iloc[i]
        upper = df['bb_upper'].iloc[i]
        lower = df['bb_lower'].iloc[i]
        
        # 밴드 터치 시 반대 진입
        if current >= upper or current <= lower:
            # 다음 15분 내 중간선 도달 확인
            for j in range(1, min(16, len(df) - i)):
                future = df['kimchi_premium'].iloc[i + j]
                middle = df['premium_ma15'].iloc[i + j]
                
                if (current >= upper and future <= middle) or \
                   (current <= lower and future >= middle):
                    move = abs(future - current)
                    if move >= required_move:
                        bb_signals += 1
                        break
    
    print(f"Bollinger band signals: {bb_signals}")
    print(f"Daily average: {bb_signals / (len(df) / (60 * 24)):.1f}")
    
    # 4. 급격한 변화 포착
    print("\n[4. Momentum Scalping]")
    
    # 1분 내 큰 변화
    rapid_changes = df['premium_change'].abs() > required_move / 2
    momentum_signals = rapid_changes.sum()
    
    print(f"Rapid change signals (>{required_move/2:.3f}% per min): {momentum_signals}")
    print(f"Daily average: {momentum_signals / (len(df) / (60 * 24)):.1f}")
    
    return opp_df if opportunities else None


def simulate_scalping_strategy(df, opportunities_df):
    """스캘핑 전략 시뮬레이션"""
    
    print("\n" + "=" * 60)
    print("  SCALPING STRATEGY SIMULATION")
    print("=" * 60)
    
    initial_capital = 40_000_000
    position_size = 0.1  # BTC
    
    # 거래 제한
    max_daily_trades = 10
    min_interval_minutes = 5  # 최소 5분 간격
    
    # 시뮬레이션
    capital = initial_capital
    trades = []
    last_trade_time = None
    daily_trades = {}
    
    for _, opp in opportunities_df.iterrows():
        # 일일 거래 제한 확인
        trade_date = opp['entry_time'].date()
        if trade_date not in daily_trades:
            daily_trades[trade_date] = 0
        
        if daily_trades[trade_date] >= max_daily_trades:
            continue
        
        # 최소 간격 확인
        if last_trade_time and (opp['entry_time'] - last_trade_time).total_seconds() < min_interval_minutes * 60:
            continue
        
        # 거래 실행
        gross_profit = opp['move'] - 0.15  # 수수료 차감
        net_profit = capital * (gross_profit / 100)
        
        trades.append({
            'entry_time': opp['entry_time'],
            'exit_time': opp['exit_time'],
            'duration': opp['duration'],
            'gross_move': opp['move'],
            'net_profit_pct': gross_profit,
            'net_profit_krw': net_profit,
            'capital_after': capital + net_profit
        })
        
        capital += net_profit
        daily_trades[trade_date] += 1
        last_trade_time = opp['exit_time']
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        print(f"\n[Simulation Results]")
        print(f"Total trades: {len(trades_df)}")
        print(f"Trading days: {len(daily_trades)}")
        print(f"Average trades per day: {len(trades_df) / len(daily_trades):.1f}")
        
        print(f"\n[Performance]")
        total_return = (capital - initial_capital) / initial_capital * 100
        print(f"Initial capital: {initial_capital:,} KRW")
        print(f"Final capital: {capital:,.0f} KRW")
        print(f"Total return: {total_return:.2f}%")
        print(f"Average per trade: {trades_df['net_profit_pct'].mean():.3f}%")
        print(f"Win rate: {(trades_df['net_profit_pct'] > 0).mean() * 100:.1f}%")
        
        # 월간 수익 추정
        days_in_data = (df.index[-1] - df.index[0]).days
        monthly_return = total_return * 30 / days_in_data if days_in_data > 0 else 0
        print(f"\n[Monthly Projection]")
        print(f"Expected monthly return: {monthly_return:.2f}%")
        print(f"Expected monthly profit: {initial_capital * monthly_return / 100:,.0f} KRW")
        
        # 목표 달성 여부
        print(f"\n[Target Achievement]")
        target_monthly = 2.5  # 2-3% 중간값
        if monthly_return >= target_monthly:
            print(f"[SUCCESS] Monthly {monthly_return:.2f}% exceeds target {target_monthly}%")
        else:
            print(f"[NEEDS IMPROVEMENT] Monthly {monthly_return:.2f}% below target {target_monthly}%")
            print(f"Required improvement: {target_monthly - monthly_return:.2f}%")
        
        return trades_df
    
    return None


def plot_scalping_analysis(df, opportunities_df):
    """스캘핑 분석 시각화"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 김프 분포와 목표 수익
    ax = axes[0, 0]
    ax.hist(df['kimchi_premium'], bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.4, color='g', linestyle='--', label='Target +0.4%')
    ax.axvline(x=-0.4, color='g', linestyle='--', label='Target -0.4%')
    ax.axvline(x=df['kimchi_premium'].mean(), color='r', linestyle='-', label='Mean')
    ax.set_title('Premium Distribution vs Target')
    ax.set_xlabel('Premium (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 1분 변화율 분포
    ax = axes[0, 1]
    changes = df['premium_change'].dropna()
    ax.hist(changes, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.2, color='g', linestyle='--', label='Half target')
    ax.axvline(x=-0.2, color='g', linestyle='--')
    ax.set_title('1-Minute Changes Distribution')
    ax.set_xlabel('Change (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 기회 지속 시간
    if opportunities_df is not None and len(opportunities_df) > 0:
        ax = axes[1, 0]
        ax.hist(opportunities_df['duration'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=opportunities_df['duration'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {opportunities_df['duration'].mean():.1f}min")
        ax.set_title('Opportunity Duration')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 수익 크기 분포
        ax = axes[1, 1]
        ax.hist(opportunities_df['move'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0.4, color='g', linestyle='--', label='Min required')
        ax.axvline(x=opportunities_df['move'].mean(), color='r', linestyle='--',
                   label=f"Mean: {opportunities_df['move'].mean():.3f}%")
        ax.set_title('Move Size Distribution')
        ax.set_xlabel('Move (%)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. 시간대별 기회
    ax = axes[2, 0]
    hourly_volatility = df.groupby(df.index.hour)['premium_change'].std()
    ax.bar(hourly_volatility.index, hourly_volatility.values, alpha=0.7)
    ax.set_title('Volatility by Hour')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Std Dev (%)')
    ax.grid(True, alpha=0.3)
    
    # 6. 볼린저 밴드
    ax = axes[2, 1]
    recent_data = df.iloc[-500:]  # 최근 500분
    ax.plot(recent_data.index, recent_data['kimchi_premium'], linewidth=0.5, label='Premium')
    ax.plot(recent_data.index, recent_data['premium_ma15'], 'r-', linewidth=1, label='MA15')
    if 'bb_upper' in recent_data.columns:
        ax.plot(recent_data.index, recent_data['bb_upper'], 'g--', linewidth=0.5, label='BB Upper')
        ax.plot(recent_data.index, recent_data['bb_lower'], 'g--', linewidth=0.5, label='BB Lower')
    ax.set_title('Recent Premium with Bollinger Bands')
    ax.set_xlabel('Time')
    ax.set_ylabel('Premium (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = f"results/analysis/scalping_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  SCALPING STRATEGY FEASIBILITY ANALYSIS")
    print("  Target: 0.25% per trade (100k KRW on 40M KRW)")
    print("  Monthly Goal: 2-3% (80-120 trades)")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/4] Loading 1-minute data...")
    df = load_1min_data_with_real_premium(days_back=30)
    print(f"Loaded {len(df)} minutes of data")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # 기본 통계
    print(f"\n[2/4] Basic Statistics")
    print(f"Mean premium: {df['kimchi_premium'].mean():.3f}%")
    print(f"Std deviation: {df['kimchi_premium'].std():.3f}%")
    print(f"Min: {df['kimchi_premium'].min():.3f}%")
    print(f"Max: {df['kimchi_premium'].max():.3f}%")
    
    # 스캘핑 기회 분석
    print(f"\n[3/4] Analyzing scalping opportunities...")
    opportunities = analyze_scalping_opportunities(df, target_profit=0.25, total_cost=0.15)
    
    # 시뮬레이션
    if opportunities is not None and len(opportunities) > 0:
        print(f"\n[4/4] Running simulation...")
        trades = simulate_scalping_strategy(df, opportunities)
        
        # 시각화
        plot_scalping_analysis(df, opportunities)
    else:
        print("\n[WARNING] No scalping opportunities found with current parameters")
    
    # 최종 권장사항
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    volatility = df['kimchi_premium'].std()
    mean_abs_change = df['premium_change'].abs().mean()
    
    print(f"\n[Market Conditions]")
    print(f"Average 1-min move: {mean_abs_change:.4f}%")
    print(f"Moves needed for profit: {0.4 / mean_abs_change:.0f} minutes average")
    
    if volatility > 0.5:
        print("\n[FEASIBLE] Market has sufficient volatility for scalping")
        print("\nSuggested approach:")
        print("1. Use limit orders to reduce slippage")
        print("2. Trade during high volatility hours")
        print("3. Set strict stop-loss at -0.2%")
        print("4. Take profit quickly at +0.4%")
        print("5. Maximum 10 trades per day")
    else:
        print("\n[CHALLENGING] Low volatility makes scalping difficult")
        print("\nAlternative approaches:")
        print("1. Increase position size (but higher risk)")
        print("2. Look for other coin pairs")
        print("3. Use market making instead")
        print("4. Wait for volatility events")


if __name__ == "__main__":
    main()