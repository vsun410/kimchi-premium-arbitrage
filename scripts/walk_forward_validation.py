"""
Walk-Forward Validation for Adaptive Scalping Model
점진적 학습과 과적합 방지를 위한 검증 시스템
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.adaptive_scalping_model import AdaptiveScalpingModel, TradingDecision
from src.backtesting.engine import BacktestEngine, TradingCosts
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


class WalkForwardValidator:
    """
    Walk-Forward Validation 시스템
    - 시간 순차적 학습/검증
    - 과적합 방지
    - 점진적 성능 향상 추적
    """
    
    def __init__(
        self,
        train_window_days: int = 20,  # 학습 기간
        test_window_days: int = 10,   # 테스트 기간
        step_days: int = 5            # 이동 단위
    ):
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step_days = step_days
        
        # 결과 저장
        self.results = []
        self.model_states = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모델을 위한 특징 준비
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            특징이 추가된 데이터프레임
        """
        # 기본 특징은 모델 내부에서 계산
        # 여기서는 필요한 컬럼만 확인
        required_cols = ['upbit_close', 'binance_close', 'kimchi_premium']
        
        for col in required_cols:
            if col not in data.columns:
                if col == 'kimchi_premium':
                    # 김프 계산
                    rate_manager = get_exchange_rate_manager()
                    data['kimchi_premium'] = data.apply(
                        lambda row: rate_manager.calculate_kimchi_premium(
                            row['upbit_close'],
                            row['binance_close'],
                            row.name
                        ),
                        axis=1
                    )
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # 볼륨 정보 추가 (옵션)
        if 'upbit_volume' not in data.columns:
            data['upbit_volume'] = 0
        if 'binance_volume' not in data.columns:
            data['binance_volume'] = 0
            
        return data
    
    def simulate_trading(
        self,
        model: AdaptiveScalpingModel,
        data: pd.DataFrame,
        initial_capital: float = 10_000_000  # 1천만원으로 테스트
    ) -> Dict:
        """
        모델을 사용한 거래 시뮬레이션
        
        Args:
            model: 학습된 모델
            data: 테스트 데이터
            initial_capital: 초기 자본
            
        Returns:
            거래 결과
        """
        capital = initial_capital
        trades = []
        position_open = False
        entry_info = None
        
        for idx in range(30, len(data)):  # 최소 30분 데이터 필요
            # 특징 계산
            features = model.calculate_features(data, idx)
            if features is None:
                continue
            
            # 예측
            decision = model.predict(features)
            
            # 포지션 관리
            if not position_open and decision.action in ['ENTER_LONG', 'ENTER_SHORT']:
                # 진입
                position_open = True
                entry_info = {
                    'entry_idx': idx,
                    'entry_price': data['upbit_close'].iloc[idx],
                    'entry_premium': data['kimchi_premium'].iloc[idx],
                    'action': decision.action,
                    'confidence': decision.confidence
                }
                
            elif position_open and (decision.action == 'EXIT' or idx - entry_info['entry_idx'] > 30):
                # 청산 (EXIT 신호 또는 30분 초과)
                exit_price = data['upbit_close'].iloc[idx]
                exit_premium = data['kimchi_premium'].iloc[idx]
                
                # PnL 계산 (단순화)
                premium_change = abs(exit_premium) - abs(entry_info['entry_premium'])
                
                # 수수료 (0.15%)
                fees = 0.0015
                
                # 수익률
                if entry_info['action'] == 'ENTER_LONG':
                    returns = -premium_change - fees  # 김프 축소로 이익
                else:
                    returns = premium_change - fees   # 김프 확대로 이익
                
                # 자본 업데이트
                trade_pnl = capital * returns / 100
                capital += trade_pnl
                
                # 거래 기록
                trades.append({
                    'entry_idx': entry_info['entry_idx'],
                    'exit_idx': idx,
                    'returns': returns,
                    'pnl': trade_pnl,
                    'confidence': entry_info['confidence']
                })
                
                # 모델 업데이트 (온라인 학습)
                reward = 1 if returns > 0 else -1
                model.update_memory(
                    state=features,
                    action=decision.action,
                    reward=reward,
                    next_state=features  # 단순화
                )
                
                position_open = False
                entry_info = None
        
        # 결과 계산
        if len(trades) == 0:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'final_capital': capital
            }
        
        trades_df = pd.DataFrame(trades)
        
        return {
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'num_trades': len(trades),
            'win_rate': (trades_df['returns'] > 0).mean() * 100,
            'avg_return': trades_df['returns'].mean(),
            'final_capital': capital,
            'trades': trades
        }
    
    def run_validation(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict:
        """
        Walk-Forward Validation 실행
        
        Args:
            data: 전체 데이터
            verbose: 상세 출력 여부
            
        Returns:
            검증 결과
        """
        # 데이터 준비
        data = self.prepare_features(data)
        
        # 시작/종료 인덱스
        total_days = (data.index[-1] - data.index[0]).days
        
        if verbose:
            print("\n" + "=" * 60)
            print("  WALK-FORWARD VALIDATION")
            print("=" * 60)
            print(f"\n[Configuration]")
            print(f"Train window: {self.train_window} days")
            print(f"Test window: {self.test_window} days")
            print(f"Step size: {self.step_days} days")
            print(f"Total data: {total_days} days")
        
        # Walk-Forward 루프
        fold = 0
        current_start = 0
        
        while current_start + self.train_window + self.test_window <= total_days:
            fold += 1
            
            # 기간 설정
            train_start = data.index[0] + timedelta(days=current_start)
            train_end = train_start + timedelta(days=self.train_window)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window)
            
            # 데이터 분할
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) < 100 or len(test_data) < 50:
                current_start += self.step_days
                continue
            
            if verbose:
                print(f"\n[Fold {fold}]")
                print(f"Train: {train_start.date()} to {train_end.date()} ({len(train_data)} samples)")
                print(f"Test: {test_start.date()} to {test_end.date()} ({len(test_data)} samples)")
            
            # 모델 생성 (매 fold마다 새로운 모델)
            model = AdaptiveScalpingModel(
                initial_target=0.15,    # 1.5% 월 목표로 시작
                max_target=0.25,        # 2.5% 월 목표까지
                learning_rate=0.01,
                memory_size=500,        # 작은 메모리로 과적합 방지
                min_confidence=0.6
            )
            
            # 이전 fold의 경험 전달 (선택적)
            if fold > 1 and len(self.model_states) > 0:
                # 이전 모델의 일부 경험 전달
                prev_state = self.model_states[-1]
                for exp in prev_state['memory'][-50:]:  # 최근 50개만
                    model.memory.append(exp)
            
            # 학습 데이터로 초기 학습
            if verbose:
                print("  Training model...")
            
            # 학습 데이터로 메모리 구축
            for idx in range(30, len(train_data), 5):  # 5분 간격 샘플링
                features = model.calculate_features(train_data, idx)
                if features is None:
                    continue
                
                # 단순 레이블 (김프 변화 기반)
                if idx + 5 < len(train_data):
                    future_premium = train_data['kimchi_premium'].iloc[idx + 5]
                    current_premium = train_data['kimchi_premium'].iloc[idx]
                    premium_change = future_premium - current_premium
                    
                    # 가상 거래 결정
                    if abs(premium_change) > 0.2:
                        action = 'ENTER_LONG' if premium_change < 0 else 'ENTER_SHORT'
                        reward = abs(premium_change) - 0.15  # 수수료 차감
                    else:
                        action = 'HOLD'
                        reward = 0
                    
                    model.update_memory(
                        state=features,
                        action=action,
                        reward=reward,
                        next_state=features
                    )
            
            # 모델 학습
            model.train_online(min_samples=100)
            
            # 테스트
            if verbose:
                print("  Testing model...")
            
            test_results = self.simulate_trading(model, test_data)
            
            # 결과 저장
            fold_result = {
                'fold': fold,
                'train_period': f"{train_start.date()} to {train_end.date()}",
                'test_period': f"{test_start.date()} to {test_end.date()}",
                'total_return': test_results['total_return'],
                'num_trades': test_results['num_trades'],
                'win_rate': test_results['win_rate'],
                'model_target': model.current_target,
                'model_confidence': model.min_confidence
            }
            
            self.results.append(fold_result)
            
            # 모델 상태 저장
            self.model_states.append({
                'fold': fold,
                'memory': list(model.memory)[-100:],
                'target': model.current_target,
                'params': model.adaptive_params
            })
            
            if verbose:
                print(f"  Return: {test_results['total_return']:.2f}%")
                print(f"  Trades: {test_results['num_trades']}")
                print(f"  Win rate: {test_results['win_rate']:.1f}%")
                print(f"  Target: {model.current_target:.3f}%")
            
            # 다음 윈도우로 이동
            current_start += self.step_days
        
        # 전체 결과 분석
        return self.analyze_results(verbose)
    
    def analyze_results(self, verbose: bool = True) -> Dict:
        """
        전체 결과 분석
        
        Args:
            verbose: 상세 출력 여부
            
        Returns:
            분석 결과
        """
        if len(self.results) == 0:
            return {'error': 'No results to analyze'}
        
        results_df = pd.DataFrame(self.results)
        
        # 통계 계산
        avg_return = results_df['total_return'].mean()
        std_return = results_df['total_return'].std()
        avg_trades = results_df['num_trades'].mean()
        avg_win_rate = results_df['win_rate'].mean()
        
        # 성능 추이 (학습 효과)
        first_half = results_df.iloc[:len(results_df)//2]
        second_half = results_df.iloc[len(results_df)//2:]
        
        improvement = second_half['total_return'].mean() - first_half['total_return'].mean()
        
        if verbose:
            print("\n" + "=" * 60)
            print("  VALIDATION RESULTS")
            print("=" * 60)
            
            print(f"\n[Overall Performance]")
            print(f"Total folds: {len(results_df)}")
            print(f"Average return: {avg_return:.2f}% ± {std_return:.2f}%")
            print(f"Average trades: {avg_trades:.1f}")
            print(f"Average win rate: {avg_win_rate:.1f}%")
            
            print(f"\n[Learning Progress]")
            print(f"First half avg: {first_half['total_return'].mean():.2f}%")
            print(f"Second half avg: {second_half['total_return'].mean():.2f}%")
            print(f"Improvement: {improvement:+.2f}%")
            
            # 월간 수익 추정
            if avg_trades > 0:
                daily_trades = avg_trades / self.test_window
                monthly_trades = daily_trades * 30
                monthly_return = avg_return * (30 / self.test_window)
                
                print(f"\n[Monthly Projection]")
                print(f"Expected monthly return: {monthly_return:.2f}%")
                print(f"Expected monthly trades: {monthly_trades:.0f}")
                
                # 목표 달성 여부
                if monthly_return >= 1.5:
                    print(f"\n[SUCCESS] Target achieved! ({monthly_return:.2f}% >= 1.5%)")
                else:
                    print(f"\n[IN PROGRESS] Below target ({monthly_return:.2f}% < 1.5%)")
        
        return {
            'num_folds': len(results_df),
            'avg_return': avg_return,
            'std_return': std_return,
            'avg_trades': avg_trades,
            'avg_win_rate': avg_win_rate,
            'improvement': improvement,
            'results': self.results
        }


def load_data_for_validation():
    """
    검증용 데이터 로드
    """
    import glob
    
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    if not binance_files or not upbit_files:
        raise FileNotFoundError("Historical data not found")
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    # 데이터 로드
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 최근 60일 (학습 20일 + 테스트 10일 × 2회분)
    cutoff_date = binance_df.index[-1] - timedelta(days=60)
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
    
    # 병합
    merged = pd.merge(
        binance_5m[['close', 'volume']].rename(columns={'close': 'binance_close', 'volume': 'binance_volume'}),
        upbit_5m[['close', 'volume']].rename(columns={'close': 'upbit_close', 'volume': 'upbit_volume'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    print(f"Loaded {len(merged)} data points ({(merged.index[-1] - merged.index[0]).days} days)")
    
    return merged


def main():
    """
    메인 실행
    """
    print("\n" + "=" * 60)
    print("  ADAPTIVE SCALPING MODEL - WALK-FORWARD VALIDATION")
    print("  Goal: 1.5% monthly → 2.5% monthly (progressive)")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/3] Loading data...")
    data = load_data_for_validation()
    
    # Walk-Forward Validation
    print("\n[2/3] Running walk-forward validation...")
    validator = WalkForwardValidator(
        train_window_days=20,
        test_window_days=10,
        step_days=5
    )
    
    results = validator.run_validation(data, verbose=True)
    
    # 결과 저장
    print("\n[3/3] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"validation_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_file}")
    
    # 최종 권장사항
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    if results.get('avg_return', 0) > 0 and results.get('improvement', 0) > 0:
        print("\n[POSITIVE] Model shows learning capability!")
        print("\nNext steps:")
        print("1. Continue training with more recent data")
        print("2. Start paper trading with small position")
        print("3. Monitor for 1 week before increasing size")
        print("4. Gradually increase target as performance improves")
    else:
        print("\n[NEEDS IMPROVEMENT] Model needs adjustment")
        print("\nSuggestions:")
        print("1. Reduce initial target to 0.1% (1% monthly)")
        print("2. Increase training data window")
        print("3. Add more market features")
        print("4. Consider different time frames")
    
    return results


if __name__ == "__main__":
    main()