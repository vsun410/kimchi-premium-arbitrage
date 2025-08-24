"""
Realistic Scalping Backtest
현실적인 스캘핑 백테스트 (수수료, 슬리피지 포함)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine, TradingCosts
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


class ScalpingStrategy:
    """
    스캘핑 전략
    - 0.4% 이상 움직임 포착
    - 빠른 진입/청산
    - 일일 거래 제한
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.2,  # 진입 신호 (평균에서 벗어남)
        target_profit: float = 0.25,   # 목표 수익 0.25%
        stop_loss: float = 0.15,       # 손절 0.15%
        max_holding_minutes: int = 30,  # 최대 보유 시간
        max_daily_trades: int = 10     # 일일 최대 거래
    ):
        self.entry_threshold = entry_threshold
        self.target_profit = target_profit
        self.stop_loss = stop_loss
        self.max_holding_minutes = max_holding_minutes
        self.max_daily_trades = max_daily_trades
        
        # 상태 추적
        self.position_open = False
        self.entry_time = None
        self.entry_premium = 0
        self.daily_trades = {}
        self.premium_history = []
        
    def generate_signal(self, timestamp, kimchi_premium, row):
        """거래 신호 생성"""
        
        # 김프 히스토리 업데이트 (최근 30분)
        self.premium_history.append(kimchi_premium)
        if len(self.premium_history) > 30:
            self.premium_history.pop(0)
        
        # 기본 신호
        signal = {
            'timestamp': timestamp,
            'action': 'HOLD',
            'confidence': 0.5,
            'kimchi_premium': kimchi_premium
        }
        
        # 일일 거래 제한 확인
        trade_date = timestamp.date()
        if trade_date not in self.daily_trades:
            self.daily_trades[trade_date] = 0
        
        if self.daily_trades[trade_date] >= self.max_daily_trades:
            return signal
        
        # 포지션이 없을 때
        if not self.position_open:
            if len(self.premium_history) >= 15:
                # 이동평균과 표준편차
                ma15 = np.mean(self.premium_history[-15:])
                std15 = np.std(self.premium_history[-15:])
                
                # 진입 조건: 평균에서 일정 이상 벗어남
                deviation = abs(kimchi_premium - ma15)
                
                if deviation > self.entry_threshold and std15 > 0.1:
                    signal['action'] = 'ENTER'
                    signal['confidence'] = min(deviation / (std15 * 2), 1.0)
                    signal['reason'] = f'Deviation {deviation:.3f}% from MA'
                    
                    self.position_open = True
                    self.entry_time = timestamp
                    self.entry_premium = kimchi_premium
                    self.daily_trades[trade_date] += 1
        
        # 포지션이 있을 때
        else:
            holding_time = (timestamp - self.entry_time).total_seconds() / 60
            premium_change = kimchi_premium - self.entry_premium
            
            # 청산 조건
            # 1. 목표 수익 달성 (0.25% 이상 움직임)
            if abs(premium_change) >= self.target_profit:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Target reached: {abs(premium_change):.3f}%'
            
            # 2. 손절 (반대로 0.15% 이상 움직임)
            elif premium_change * np.sign(self.entry_premium) < -self.stop_loss:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Stop loss: {premium_change:.3f}%'
            
            # 3. 시간 초과
            elif holding_time >= self.max_holding_minutes:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Time limit: {holding_time:.0f} minutes'
            
            # 4. 평균 회귀 완료
            elif len(self.premium_history) >= 5:
                current_ma = np.mean(self.premium_history[-5:])
                if abs(current_ma) < abs(self.entry_premium) * 0.5:
                    signal['action'] = 'EXIT'
                    signal['reason'] = 'Mean reversion complete'
            
            if signal['action'] == 'EXIT':
                self.position_open = False
                self.entry_time = None
                self.entry_premium = 0
        
        return signal


def prepare_1min_data():
    """1분 데이터 준비"""
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
    
    # 최근 30일
    cutoff_date = binance_df.index[-1] - timedelta(days=30)
    binance_df = binance_df[binance_df.index >= cutoff_date]
    upbit_df = upbit_df[upbit_df.index >= cutoff_date]
    
    # 병합 (1분 데이터)
    merged = pd.merge(
        binance_df[['close', 'volume']].rename(columns={'close': 'binance_close', 'volume': 'binance_volume'}),
        upbit_df[['close', 'volume']].rename(columns={'close': 'upbit_close', 'volume': 'upbit_volume'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    return merged


def run_scalping_backtest():
    """스캘핑 백테스트 실행"""
    
    print("\n" + "=" * 60)
    print("  REALISTIC SCALPING BACKTEST")
    print("=" * 60)
    
    # 데이터 준비
    print("\n[1/4] Loading 1-minute data...")
    data = prepare_1min_data()
    print(f"Loaded {len(data)} minutes")
    
    # 거래 비용 (현실적)
    costs = TradingCosts(
        upbit_fee=0.0005,      # 0.05%
        binance_fee=0.001,     # 0.1%
        slippage=0.0003,       # 0.03% (스캘핑은 슬리피지 높음)
        funding_rate=0.00005   # 0.005% per 8h (짧은 보유)
    )
    
    # 전략 파라미터 테스트
    strategies = [
        {'name': 'Conservative', 'entry': 0.3, 'target': 0.4, 'stop': 0.2},
        {'name': 'Moderate', 'entry': 0.2, 'target': 0.3, 'stop': 0.15},
        {'name': 'Aggressive', 'entry': 0.15, 'target': 0.25, 'stop': 0.1},
        {'name': 'Tight', 'entry': 0.1, 'target': 0.2, 'stop': 0.1}
    ]
    
    # 데이터 분할
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nTrain: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test: {test_data.index[0]} to {test_data.index[-1]}")
    
    print("\n[2/4] Testing strategies...")
    
    best_result = None
    best_monthly = -float('inf')
    
    for params in strategies:
        print(f"\n[{params['name']} Strategy]")
        print(f"Entry: >{params['entry']}% deviation")
        print(f"Target: {params['target']}% profit")
        print(f"Stop: {params['stop']}% loss")
        
        # 백테스트 엔진
        engine = BacktestEngine(
            initial_capital=40_000_000,
            max_position_size=0.05,  # 스캘핑은 작은 포지션
            costs=costs
        )
        
        # 전략
        strategy = ScalpingStrategy(
            entry_threshold=params['entry'],
            target_profit=params['target'],
            stop_loss=params['stop'],
            max_holding_minutes=30,
            max_daily_trades=10
        )
        
        # 백테스트 실행
        results = engine.run(
            data=test_data,
            strategy=strategy
        )
        
        # 결과 출력
        print(f"  Trades: {results['total_trades']}")
        print(f"  Return: {results['total_return']:.3f}%")
        print(f"  Win rate: {results['win_rate']:.1f}%")
        
        # 월간 수익 계산
        days = (test_data.index[-1] - test_data.index[0]).days
        if days > 0:
            monthly_return = results['total_return'] * 30 / days
            print(f"  Monthly (projected): {monthly_return:.2f}%")
            
            if monthly_return > best_monthly:
                best_monthly = monthly_return
                best_result = {
                    'strategy': params['name'],
                    'params': params,
                    'results': results,
                    'monthly': monthly_return
                }
    
    # 최적 전략 분석
    if best_result:
        print("\n[3/4] Best Strategy Analysis")
        print(f"Strategy: {best_result['strategy']}")
        print(f"Monthly return: {best_result['monthly']:.2f}%")
        print(f"Total trades: {best_result['results']['total_trades']}")
        
        # 거래당 수익
        if best_result['results']['total_trades'] > 0:
            per_trade = best_result['results']['total_return'] / best_result['results']['total_trades']
            print(f"Average per trade: {per_trade:.3f}%")
            print(f"In KRW: {40_000_000 * per_trade / 100:,.0f} KRW per trade")
    
    # 실전 권장사항
    print("\n[4/4] Trading Recommendations")
    
    if best_monthly >= 2.0:
        print("\n[SUCCESS] Target achievable!")
        print("\nImplementation steps:")
        print("1. Start with small position (0.01 BTC)")
        print("2. Use limit orders for entry")
        print("3. Set automatic stop-loss")
        print("4. Monitor for 1 week before scaling")
        print("5. Keep detailed trade log")
    else:
        print("\n[CHALLENGING] Target difficult to achieve")
        print("\nAdjustments needed:")
        print("1. Reduce target to 0.15-0.20% per trade")
        print("2. Increase daily trade limit to 15-20")
        print("3. Consider other pairs (ETH, XRP)")
        print("4. Optimize entry timing")
    
    return best_result


def calculate_required_performance():
    """목표 달성에 필요한 성과 계산"""
    
    print("\n" + "=" * 60)
    print("  TARGET CALCULATION")
    print("=" * 60)
    
    capital = 40_000_000
    target_monthly = 0.025  # 2.5%
    target_per_trade = 100_000  # 10만원
    
    print(f"\n[Goals]")
    print(f"Capital: {capital:,} KRW")
    print(f"Monthly target: {target_monthly*100}% = {capital * target_monthly:,.0f} KRW")
    print(f"Per trade target: {target_per_trade:,} KRW = {target_per_trade/capital*100:.3f}%")
    
    # 필요 거래 수
    trades_needed = (capital * target_monthly) / target_per_trade
    print(f"\n[Requirements]")
    print(f"Trades needed per month: {trades_needed:.0f}")
    print(f"Trades needed per day: {trades_needed/30:.1f}")
    
    # 수수료 고려
    total_fees = 0.0015  # 0.15%
    required_move = (target_per_trade / capital * 100) + total_fees
    print(f"\n[With Fees]")
    print(f"Total fees: {total_fees*100:.2f}%")
    print(f"Required price move: {required_move:.3f}%")
    print(f"Break-even move: {total_fees*100:.2f}%")
    
    # 실현 가능성
    print(f"\n[Feasibility Check]")
    print(f"If win rate = 60%:")
    print(f"  Need {trades_needed/0.6:.0f} total trades")
    print(f"  Average profit must be {required_move*100/60:.3f}% per winning trade")
    
    print(f"\nIf win rate = 70%:")
    print(f"  Need {trades_needed/0.7:.0f} total trades")
    print(f"  Average profit must be {required_move*100/70:.3f}% per winning trade")


def main():
    """메인 실행"""
    
    print("\n" + "=" * 60)
    print("  SCALPING STRATEGY - COMPLETE ANALYSIS")
    print("  Goal: 10만원 per trade, 2-3% monthly")
    print("=" * 60)
    
    # 목표 계산
    calculate_required_performance()
    
    # 백테스트 실행
    best_result = run_scalping_backtest()
    
    # 최종 결론
    print("\n" + "=" * 60)
    print("  FINAL VERDICT")
    print("=" * 60)
    
    if best_result and best_result['monthly'] >= 2.0:
        print("\n[FEASIBLE] 월 2-3% 목표 달성 가능!")
        print(f"예상 월 수익: {best_result['monthly']:.2f}%")
        print(f"예상 월 수익금: {40_000_000 * best_result['monthly'] / 100:,.0f} KRW")
        print("\n다음 단계:")
        print("1. Paper trading으로 1주일 테스트")
        print("2. 소액으로 실거래 시작 (1000만원)")
        print("3. 성과 검증 후 자금 증액")
    else:
        print("\n[DIFFICULT] 현재 시장에서는 목표 달성 어려움")
        print("대안:")
        print("1. 목표 수익률 하향 (월 1-1.5%)")
        print("2. 다른 거래 전략 병행")
        print("3. 시장 변동성 증가 대기")


if __name__ == "__main__":
    main()