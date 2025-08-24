"""
Proper Backtesting without Overfitting
과적합 없는 올바른 백테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


class ProperBacktest:
    """
    과적합 방지를 위한 엄격한 백테스트
    """
    
    def __init__(self):
        self.entry_threshold = 0.05  # 0.05% 변동시 진입
        self.target_profit = 0.03    # 0.03% 목표
        self.stop_loss = 0.02         # 0.02% 손절
        self.fees = 0.0015            # 0.15% 수수료
        self.slippage = 0.0005        # 0.05% 슬리피지
        
    def simulate_realistic_trading(self, data: pd.DataFrame):
        """
        현실적인 거래 시뮬레이션
        - 미래 데이터 참조 금지
        - 슬리피지 적용
        - 실패 거래 포함
        """
        
        trades = []
        capital = 10_000_000
        position = None
        
        # 단순 규칙 기반 전략 (ML 없이)
        for i in range(30, len(data)):
            current_premium = data['kimchi_premium'].iloc[i]
            
            # 과거 데이터만 사용
            past_5min = data['kimchi_premium'].iloc[i-5:i]
            ma5 = past_5min.mean()
            std5 = past_5min.std()
            
            # 1분 전 변화량 (과거 데이터)
            change_1min = current_premium - data['kimchi_premium'].iloc[i-1]
            
            # 포지션 없을 때
            if position is None:
                # 진입 조건: 평균에서 크게 벗어남
                if abs(current_premium - ma5) > self.entry_threshold and std5 > 0.03:
                    
                    # 50% 확률로 진입 (랜덤성 추가)
                    if np.random.random() > 0.5:
                        
                        # 슬리피지 적용
                        actual_entry = current_premium + np.random.uniform(-self.slippage, self.slippage)
                        
                        position = {
                            'entry_idx': i,
                            'entry_premium': actual_entry,
                            'direction': 'long' if current_premium < ma5 else 'short'
                        }
            
            # 포지션 있을 때
            else:
                holding_time = i - position['entry_idx']
                
                # 슬리피지 적용한 현재가
                actual_current = current_premium + np.random.uniform(-self.slippage, self.slippage)
                
                # 수익/손실 계산
                if position['direction'] == 'long':
                    pnl_pct = actual_current - position['entry_premium']
                else:
                    pnl_pct = position['entry_premium'] - actual_current
                
                # 청산 조건
                should_close = False
                reason = ''
                
                # 1. 목표 달성 (30% 확률로 미끄러짐)
                if pnl_pct >= self.target_profit:
                    if np.random.random() > 0.3:  # 70% 확률로 체결
                        should_close = True
                        reason = 'Target reached'
                    else:
                        # 미끄러져서 손실로 전환될 수 있음
                        pnl_pct *= 0.5
                
                # 2. 손절 (무조건 체결)
                elif pnl_pct <= -self.stop_loss:
                    should_close = True
                    reason = 'Stop loss'
                    pnl_pct = -self.stop_loss - self.slippage  # 손절시 추가 슬리피지
                
                # 3. 시간 초과
                elif holding_time >= 15:  # 15분
                    should_close = True
                    reason = 'Time limit'
                
                # 4. 랜덤 노이즈 (5% 확률로 예상치 못한 청산)
                elif np.random.random() < 0.05:
                    should_close = True
                    reason = 'Market order hit'
                    pnl_pct *= np.random.uniform(0.5, 1.5)  # 랜덤 결과
                
                if should_close:
                    # 수수료 차감
                    net_pnl = pnl_pct - self.fees
                    
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'pnl_pct': net_pnl,
                        'reason': reason,
                        'success': net_pnl > 0
                    })
                    
                    capital *= (1 + net_pnl / 100)
                    position = None
        
        return trades, capital
    
    def run_monte_carlo(self, data: pd.DataFrame, n_simulations: int = 100):
        """
        몬테카를로 시뮬레이션으로 결과 분포 확인
        """
        
        results = []
        
        for sim in range(n_simulations):
            trades, final_capital = self.simulate_realistic_trading(data)
            
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                win_rate = trades_df['success'].mean() * 100
                avg_pnl = trades_df['pnl_pct'].mean()
                total_return = (final_capital - 10_000_000) / 10_000_000 * 100
            else:
                win_rate = 0
                avg_pnl = 0
                total_return = 0
            
            results.append({
                'simulation': sim,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_return': total_return,
                'final_capital': final_capital
            })
        
        return pd.DataFrame(results)


def main():
    """
    올바른 백테스트 실행
    """
    
    print("\n" + "=" * 60)
    print("  PROPER BACKTESTING (WITHOUT OVERFITTING)")
    print("=" * 60)
    
    # 데이터 로드
    import glob
    
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    if not binance_files or not upbit_files:
        print("No data files found")
        return
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    # 데이터 로드
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
    
    # 병합
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
    
    print(f"\nData loaded: {len(merged)} points")
    print(f"Period: {merged.index[0]} to {merged.index[-1]}")
    
    # 데이터 분할
    split_idx = int(len(merged) * 0.7)
    train_data = merged.iloc[:split_idx]
    test_data = merged.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_data)} points")
    print(f"Test: {len(test_data)} points")
    
    # 백테스트 실행
    print("\n[1/3] Running single realistic backtest...")
    
    backtest = ProperBacktest()
    trades, final_capital = backtest.simulate_realistic_trading(test_data)
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        print(f"\n[Single Run Results]")
        print(f"Total trades: {len(trades)}")
        print(f"Win rate: {trades_df['success'].mean() * 100:.1f}%")
        print(f"Average PnL: {trades_df['pnl_pct'].mean():.4f}%")
        print(f"Total return: {(final_capital - 10_000_000) / 10_000_000 * 100:.2f}%")
        
        # 실패 이유 분석
        print(f"\n[Exit Reasons]")
        for reason, count in trades_df['reason'].value_counts().items():
            print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")
    
    # 몬테카를로 시뮬레이션
    print("\n[2/3] Running Monte Carlo simulation (100 runs)...")
    
    mc_results = backtest.run_monte_carlo(test_data, n_simulations=100)
    
    print(f"\n[Monte Carlo Results - Statistics]")
    print(f"Average trades: {mc_results['num_trades'].mean():.1f}")
    print(f"Average win rate: {mc_results['win_rate'].mean():.1f}% ± {mc_results['win_rate'].std():.1f}%")
    print(f"Average return: {mc_results['total_return'].mean():.2f}% ± {mc_results['total_return'].std():.2f}%")
    
    # 분포 분석
    print(f"\n[Return Distribution]")
    print(f"Best case (95th percentile): {mc_results['total_return'].quantile(0.95):.2f}%")
    print(f"Median: {mc_results['total_return'].median():.2f}%")
    print(f"Worst case (5th percentile): {mc_results['total_return'].quantile(0.05):.2f}%")
    
    # 손실 확률
    loss_probability = (mc_results['total_return'] < 0).mean() * 100
    print(f"\nProbability of loss: {loss_probability:.1f}%")
    
    # 현실적인 월 수익 계산
    test_days = (test_data.index[-1] - test_data.index[0]).days
    median_return = mc_results['total_return'].median()
    
    if test_days > 0:
        daily_return = median_return / test_days
        monthly_return = daily_return * 30
        
        print(f"\n[Realistic Monthly Projection]")
        print(f"Median daily return: {daily_return:.3f}%")
        print(f"Median monthly return: {monthly_return:.2f}%")
        print(f"On 40M KRW: {40_000_000 * monthly_return / 100:,.0f} KRW/month")
    
    # 경고
    print("\n" + "=" * 60)
    print("  REALITY CHECK")
    print("=" * 60)
    
    print("\n[Important Notes]")
    print("1. This includes slippage and random failures")
    print("2. Win rate is realistic (not 100%)")
    print("3. Some trades fail due to market conditions")
    print("4. Results vary significantly (see std deviation)")
    
    if median_return < 0:
        print("\n⚠️  WARNING: Strategy shows NEGATIVE median return!")
        print("   This strategy would likely lose money in real trading.")
    elif monthly_return < 2:
        print("\n⚠️  CAUTION: Returns below target (2-3% monthly)")
        print("   Consider adjusting parameters or strategy.")
    else:
        print("\n✓  Strategy shows positive returns")
        print("   But remember: Past performance ≠ Future results")
    
    return mc_results


if __name__ == "__main__":
    main()