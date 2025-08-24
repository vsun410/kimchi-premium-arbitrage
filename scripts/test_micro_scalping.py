"""
Test Micro Scalping Strategy
마이크로 스쾘핑 전략 테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.micro_scalping_model import MicroScalpingModel
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


def load_test_data(days_back: int = 30):
    """테스트 데이터 로드"""
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
    
    # 최근 N일
    cutoff_date = binance_df.index[-1] - timedelta(days=days_back)
    binance_df = binance_df[binance_df.index >= cutoff_date]
    upbit_df = upbit_df[upbit_df.index >= cutoff_date]
    
    # 1분 데이터로 병합
    merged = pd.merge(
        binance_df[['close', 'volume']].rename(columns={'close': 'binance_close', 'volume': 'binance_volume'}),
        upbit_df[['close', 'volume']].rename(columns={'close': 'upbit_close', 'volume': 'upbit_volume'}),
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
    
    return merged


def simulate_micro_trading(model: MicroScalpingModel, data: pd.DataFrame, 
                          initial_capital: float = 10_000_000):
    """
    마이크로 거래 시뮬레이션
    
    Args:
        model: 스쾘핑 모델
        data: 가격 데이터
        initial_capital: 초기 자본
        
    Returns:
        거래 결과
    """
    capital = initial_capital
    trades = []
    position = None
    
    # 수수료
    fees = 0.0015  # 0.15% 총 수수료
    
    print(f"\nStarting simulation with {len(data)} data points")
    print(f"Initial capital: {initial_capital:,.0f} KRW")
    print(f"Position size: {model.base_position_size} BTC")
    
    # 시뮬레이션 루프
    for idx in range(10, len(data)):
        timestamp = data.index[idx]
        current_premium = data['kimchi_premium'].iloc[idx]
        
        # 특징 계산
        features = model.calculate_micro_features(data, idx)
        if features is None:
            continue
        
        # 포지션이 없을 때
        if position is None:
            # 예측
            decision = model.predict_micro_movement(features)
            
            # 진입 신호
            if decision.action in ['BUY', 'SELL']:
                position = {
                    'entry_idx': idx,
                    'entry_time': timestamp,
                    'entry_premium': current_premium,
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'size': decision.position_size
                }
                
                # 학습용 가상 보상 설정
                if idx + 5 < len(data):
                    future_premium = data['kimchi_premium'].iloc[idx + 5]
                    premium_change = abs(future_premium - current_premium)
                    
                    # 성공/실패 판단
                    if premium_change > model.target_profit:
                        fake_reward = premium_change - fees
                    else:
                        fake_reward = -fees
                    
                    model.update_learning(features, decision.action, fake_reward, features)
        
        # 포지션이 있을 때
        else:
            holding_minutes = (timestamp - position['entry_time']).total_seconds() / 60
            
            # 청산 여부 확인
            should_close, reason = model.should_close_position(
                position['entry_premium'],
                current_premium,
                holding_minutes
            )
            
            if should_close:
                # PnL 계산
                premium_change = abs(current_premium - position['entry_premium'])
                
                # 단순화된 수익 계산
                if premium_change > fees:
                    pnl_pct = premium_change - fees
                    reward = 1
                else:
                    pnl_pct = -fees
                    reward = -1
                
                pnl_krw = capital * pnl_pct / 100
                capital += pnl_krw
                
                # 거래 기록
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'holding_minutes': holding_minutes,
                    'entry_premium': position['entry_premium'],
                    'exit_premium': current_premium,
                    'premium_change': premium_change,
                    'pnl_pct': pnl_pct,
                    'pnl_krw': pnl_krw,
                    'reason': reason
                })
                
                # 학습 업데이트
                model.update_learning(features, 'EXIT', reward, features)
                
                position = None
        
        # 주기적 학습 (100거래마다)
        if len(trades) > 0 and len(trades) % 100 == 0:
            model.train_models(min_samples=200)
            print(f"\n[{timestamp}] Trained models after {len(trades)} trades")
            stats = model.get_stats()
            print(f"  Win rate: {stats['win_rate']:.1f}%")
            print(f"  Threshold: {stats['current_threshold']:.4f}%")
    
    # 결과 분석
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'total_return': 0,
            'final_capital': capital
        }
    
    trades_df = pd.DataFrame(trades)
    
    return {
        'total_trades': len(trades),
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'final_capital': capital,
        'win_rate': (trades_df['pnl_pct'] > 0).mean() * 100,
        'avg_pnl': trades_df['pnl_pct'].mean(),
        'avg_holding': trades_df['holding_minutes'].mean(),
        'trades': trades
    }


def run_progressive_test():
    """
    점진적 테스트
    - 다양한 파라미터로 테스트
    - 최적 파라미터 찾기
    """
    print("\n" + "=" * 60)
    print("  MICRO SCALPING STRATEGY TEST")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/3] Loading data...")
    data = load_test_data(days_back=30)
    print(f"Loaded {len(data)} minutes of data")
    
    # 김프 통계
    print(f"\n[Kimchi Premium Stats]")
    print(f"Mean: {data['kimchi_premium'].mean():.4f}%")
    print(f"Std: {data['kimchi_premium'].std():.4f}%")
    print(f"Min: {data['kimchi_premium'].min():.4f}%")
    print(f"Max: {data['kimchi_premium'].max():.4f}%")
    
    # 테스트할 파라미터
    test_params = [
        {'name': 'Ultra Micro', 'entry': 0.03, 'target': 0.02, 'stop': 0.015},
        {'name': 'Micro', 'entry': 0.05, 'target': 0.03, 'stop': 0.02},
        {'name': 'Small', 'entry': 0.07, 'target': 0.04, 'stop': 0.03},
        {'name': 'Medium', 'entry': 0.10, 'target': 0.05, 'stop': 0.04}
    ]
    
    # 데이터 분할
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\n[Data Split]")
    print(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test: {test_data.index[0]} to {test_data.index[-1]}")
    
    print("\n[2/3] Testing different parameters...")
    
    results = []
    best_return = -float('inf')
    best_params = None
    
    for params in test_params:
        print(f"\n[{params['name']} Strategy]")
        print(f"  Entry: >{params['entry']}%")
        print(f"  Target: {params['target']}%")
        print(f"  Stop: {params['stop']}%")
        
        # 모델 생성
        model = MicroScalpingModel(
            entry_threshold=params['entry'],
            target_profit=params['target'],
            stop_loss=params['stop'],
            max_holding_minutes=15,
            min_confidence=0.55,
            position_size=0.1
        )
        
        # 학습 데이터로 초기 학습
        print("  Training on historical data...")
        for idx in range(10, len(train_data), 100):  # 100분 간격
            features = model.calculate_micro_features(train_data, idx)
            if features is None:
                continue
            
            # 가상 보상
            if idx + 5 < len(train_data):
                future_premium = train_data['kimchi_premium'].iloc[idx + 5]
                current_premium = train_data['kimchi_premium'].iloc[idx]
                premium_change = abs(future_premium - current_premium)
                
                if premium_change > params['entry']:
                    action = 'BUY' if future_premium > current_premium else 'SELL'
                    reward = premium_change - 0.0015 if premium_change > params['target'] else -0.0015
                else:
                    action = 'HOLD'
                    reward = 0
                
                model.update_learning(features, action, reward, features)
        
        # 모델 학습
        model.train_models(min_samples=100)
        
        # 테스트
        print("  Testing on recent data...")
        result = simulate_micro_trading(model, test_data)
        
        # 결과 저장
        result['strategy'] = params['name']
        result['params'] = params
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Trades: {result['total_trades']}")
        print(f"    Return: {result['total_return']:.3f}%")
        print(f"    Win rate: {result['win_rate']:.1f}%" if result['total_trades'] > 0 else "    No trades")
        
        if result['total_return'] > best_return:
            best_return = result['total_return']
            best_params = params
    
    # 최적 전략 분석
    print("\n[3/3] Best Strategy Analysis")
    
    if best_params:
        print(f"\nBest: {best_params['name']}")
        print(f"Entry threshold: {best_params['entry']}%")
        print(f"Target profit: {best_params['target']}%")
        print(f"Expected return: {best_return:.3f}%")
        
        # 일일/월간 추정
        test_days = (test_data.index[-1] - test_data.index[0]).days
        if test_days > 0:
            daily_return = best_return / test_days
            monthly_return = daily_return * 30
            
            print(f"\n[Projected Performance]")
            print(f"Daily return: {daily_return:.3f}%")
            print(f"Monthly return: {monthly_return:.2f}%")
            
            # 4000만원 기준
            monthly_krw = 40_000_000 * monthly_return / 100
            print(f"Monthly profit (40M KRW): {monthly_krw:,.0f} KRW")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"micro_scalping_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_file}")
    
    # 최종 권장사항
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    if best_return > 0:
        print("\n[POSITIVE] Strategy shows profit potential!")
        print("\nNext steps:")
        print("1. Test with live data feed (paper trading)")
        print("2. Implement limit order execution")
        print("3. Add market maker rebates")
        print("4. Monitor slippage in real trading")
    else:
        print("\n[NEEDS ADJUSTMENT]")
        print("\nSuggestions:")
        print("1. Further reduce thresholds")
        print("2. Increase position size")
        print("3. Focus on specific time windows")
        print("4. Consider other pairs with higher volatility")
    
    return results


def main():
    """메인 실행"""
    results = run_progressive_test()
    return results


if __name__ == "__main__":
    main()