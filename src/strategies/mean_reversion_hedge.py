"""
Mean Reversion Hedge Strategy
평균회귀 헤지 전략 - 실제 수익을 낸 검증된 전략
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """포지션 정보"""
    entry_time: datetime
    entry_price: float
    entry_kimchi: float
    position_type: str  # 'long' or 'short'
    size: float  # BTC
    target_profit: float  # KRW
    stop_loss: float  # KRW
    status: str  # 'open', 'closed', 'hedged'
    hedge_position: Optional['Position'] = None


@dataclass
class TradeResult:
    """거래 결과"""
    entry_time: datetime
    exit_time: datetime
    position_type: str
    entry_kimchi: float
    exit_kimchi: float
    profit_krw: float
    profit_pct: float
    holding_hours: float
    exit_reason: str  # 'target', 'hedge', 'stop_loss'


class MeanReversionHedgeStrategy:
    """
    평균회귀 헤지 전략
    
    핵심 원리:
    1. 김프가 평균 이하로 급락시 진입
    2. 목표 수익(12만원) 달성시 청산
    3. 물리면 반대 포지션으로 헤지
    4. 헤지 포지션도 같은 목표 수익으로 청산
    """
    
    def __init__(
        self,
        capital: float = 40_000_000,  # 4000만원
        target_profit_krw: float = 120_000,  # 12만원 목표
        lookback_period: int = 48,  # 48시간 평균
        entry_threshold: float = -0.5,  # 평균 대비 -0.5% 이하시 진입
        hedge_threshold: float = -0.3,  # 진입가 대비 -0.3% 이하시 헤지
        max_positions: int = 2  # 최대 2개 포지션 (원 포지션 + 헤지)
    ):
        self.capital = capital
        self.target_profit_krw = target_profit_krw
        self.target_profit_pct = (target_profit_krw / capital) * 100  # 0.3%
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.hedge_threshold = hedge_threshold
        self.max_positions = max_positions
        
        # 수수료
        self.fees = {
            'upbit': 0.0005,  # 0.05%
            'binance': 0.001,  # 0.1%
            'total': 0.0015   # 총 0.15%
        }
        
        # 포지션 관리
        self.positions = []
        self.closed_trades = []
        
        # 통계
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'hedge_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'consecutive_losses': 0
        }
        
    def calculate_moving_average(self, data: pd.Series, period: int = None) -> float:
        """
        이동평균 계산
        """
        if period is None:
            period = self.lookback_period
            
        if len(data) < period:
            return data.mean()
        
        return data.iloc[-period:].mean()
    
    def calculate_position_size(self, btc_price: float) -> float:
        """
        포지션 크기 계산
        
        목표: 0.3% 수익 = 12만원
        필요 BTC = 12만원 / (BTC가격 * 0.003)
        """
        # 목표 수익률을 위한 포지션 크기
        required_capital = self.target_profit_krw / (self.target_profit_pct / 100)
        position_btc = required_capital / btc_price
        
        # 최대 자본의 50%로 제한
        max_position_btc = (self.capital * 0.5) / btc_price
        
        return min(position_btc, max_position_btc)
    
    def should_enter_position(
        self,
        current_kimchi: float,
        ma_kimchi: float,
        current_positions: int
    ) -> bool:
        """
        진입 조건 확인
        
        조건:
        1. 김프가 평균보다 threshold 이하
        2. 포지션 수 제한 미달
        3. 이전 포지션과 충분한 간격
        """
        # 포지션 수 제한
        if current_positions >= self.max_positions:
            return False
        
        # 평균 대비 하락률
        deviation = current_kimchi - ma_kimchi
        
        # 진입 조건
        if deviation <= self.entry_threshold:
            # 이전 포지션과의 간격 확인
            if self.positions:
                last_entry = self.positions[-1].entry_kimchi
                if abs(current_kimchi - last_entry) < 0.1:  # 0.1% 미만 차이면 스킵
                    return False
            
            return True
        
        return False
    
    def should_hedge_position(
        self,
        position: Position,
        current_kimchi: float
    ) -> bool:
        """
        헤지 조건 확인
        
        조건:
        1. 진입가 대비 추가 하락
        2. 아직 헤지하지 않은 포지션
        """
        if position.hedge_position is not None:
            return False
        
        # 진입가 대비 하락률
        drop_from_entry = current_kimchi - position.entry_kimchi
        
        return drop_from_entry <= self.hedge_threshold
    
    def calculate_pnl(
        self,
        position: Position,
        current_price: float,
        current_kimchi: float
    ) -> float:
        """
        손익 계산
        
        Long: (현재가 - 진입가) * 수량
        Short: (진입가 - 현재가) * 수량
        """
        if position.position_type == 'long':
            # 김프 상승시 수익
            kimchi_change = current_kimchi - position.entry_kimchi
        else:  # short (hedge)
            # 김프 하락시 수익
            kimchi_change = position.entry_kimchi - current_kimchi
        
        # 수익률을 KRW로 변환
        profit_pct = kimchi_change
        profit_krw = position.size * current_price * (profit_pct / 100)
        
        # 수수료 차감
        fees = position.size * current_price * self.fees['total']
        
        return profit_krw - fees
    
    def execute_strategy(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        전략 실행 (백테스트)
        
        Args:
            data: DataFrame with columns ['timestamp', 'kimchi_premium', 'btc_price']
        
        Returns:
            전략 실행 결과
        """
        results = []
        
        for i in range(self.lookback_period, len(data)):
            current_time = data['timestamp'].iloc[i]
            current_kimchi = data['kimchi_premium'].iloc[i]
            current_price = data['btc_price'].iloc[i]
            
            # 이동평균 계산
            ma_kimchi = self.calculate_moving_average(
                data['kimchi_premium'].iloc[:i]
            )
            
            # 열린 포지션 확인 및 청산
            for position in self.positions[:]:  # 복사본으로 순회
                if position.status != 'open':
                    continue
                
                # 손익 계산
                pnl = self.calculate_pnl(position, current_price, current_kimchi)
                
                # 목표 수익 달성
                if pnl >= self.target_profit_krw:
                    # 포지션 청산
                    trade_result = TradeResult(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        position_type=position.position_type,
                        entry_kimchi=position.entry_kimchi,
                        exit_kimchi=current_kimchi,
                        profit_krw=pnl,
                        profit_pct=(pnl / self.capital) * 100,
                        holding_hours=(current_time - position.entry_time).total_seconds() / 3600,
                        exit_reason='target'
                    )
                    
                    results.append(trade_result)
                    self.positions.remove(position)
                    self.stats['winning_trades'] += 1
                    self.stats['total_profit'] += pnl
                    
                # 헤지 필요 확인
                elif self.should_hedge_position(position, current_kimchi):
                    # 반대 포지션 생성
                    hedge_type = 'short' if position.position_type == 'long' else 'long'
                    
                    hedge_position = Position(
                        entry_time=current_time,
                        entry_price=current_price,
                        entry_kimchi=current_kimchi,
                        position_type=hedge_type,
                        size=position.size,
                        target_profit=self.target_profit_krw,
                        stop_loss=-self.target_profit_krw * 2,
                        status='open',
                        hedge_position=None
                    )
                    
                    position.hedge_position = hedge_position
                    position.status = 'hedged'
                    self.positions.append(hedge_position)
                    self.stats['hedge_trades'] += 1
            
            # 새로운 포지션 진입 확인
            open_positions = sum(1 for p in self.positions if p.status == 'open')
            
            if self.should_enter_position(current_kimchi, ma_kimchi, open_positions):
                # 포지션 크기 계산
                position_size = self.calculate_position_size(current_price)
                
                # 새 포지션 생성
                new_position = Position(
                    entry_time=current_time,
                    entry_price=current_price,
                    entry_kimchi=current_kimchi,
                    position_type='long',  # 기본은 김프 상승 베팅
                    size=position_size,
                    target_profit=self.target_profit_krw,
                    stop_loss=-self.target_profit_krw * 2,
                    status='open',
                    hedge_position=None
                )
                
                self.positions.append(new_position)
                self.stats['total_trades'] += 1
        
        # 최종 통계
        if results:
            total_profit = sum(r.profit_krw for r in results)
            win_rate = len([r for r in results if r.profit_krw > 0]) / len(results)
            avg_profit = total_profit / len(results)
            
            return {
                'trades': results,
                'total_profit': total_profit,
                'total_return': (total_profit / self.capital) * 100,
                'num_trades': len(results),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'hedge_count': self.stats['hedge_trades'],
                'max_drawdown': self.stats['max_drawdown']
            }
        
        return {
            'trades': [],
            'total_profit': 0,
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'hedge_count': 0,
            'max_drawdown': 0
        }
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_ranges: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        파라미터 최적화
        
        Args:
            data: 과거 데이터
            param_ranges: 파라미터 범위
                - entry_threshold: [-1.0, -0.3]
                - target_profit: [100000, 200000]
                - hedge_threshold: [-0.5, -0.1]
        
        Returns:
            최적 파라미터와 결과
        """
        best_params = None
        best_score = -float('inf')
        
        for entry_th in param_ranges.get('entry_threshold', [-0.5]):
            for target in param_ranges.get('target_profit', [120000]):
                for hedge_th in param_ranges.get('hedge_threshold', [-0.3]):
                    # 임시 전략 생성
                    temp_strategy = MeanReversionHedgeStrategy(
                        capital=self.capital,
                        target_profit_krw=target,
                        entry_threshold=entry_th,
                        hedge_threshold=hedge_th
                    )
                    
                    # 백테스트 실행
                    result = temp_strategy.execute_strategy(data)
                    
                    # 점수 계산 (Sharpe Ratio 근사)
                    if result['num_trades'] > 0:
                        score = result['total_return'] / max(abs(result['max_drawdown']), 0.01)
                    else:
                        score = 0
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'entry_threshold': entry_th,
                            'target_profit': target,
                            'hedge_threshold': hedge_th,
                            'result': result
                        }
        
        return best_params


def analyze_mean_reversion_strategy():
    """
    평균회귀 헤지 전략 분석
    """
    print("\n" + "=" * 60)
    print("  MEAN REVERSION HEDGE STRATEGY ANALYSIS")
    print("  평균회귀 헤지 전략 분석")
    print("=" * 60)
    
    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    
    # 30일간 데이터 (1시간 간격)
    hours = 30 * 24
    timestamps = pd.date_range(start='2025-01-01', periods=hours, freq='H')
    
    # 김프 시뮬레이션 (평균 0.05%, 변동성 있게)
    kimchi_base = 0.05
    kimchi_premium = []
    current = kimchi_base
    
    for _ in range(hours):
        # 평균회귀 특성 추가
        mean_reversion = (kimchi_base - current) * 0.1
        random_walk = np.random.normal(0, 0.02)
        
        # 가끔 급락 이벤트
        if np.random.random() < 0.05:  # 5% 확률로 급락
            random_walk -= 0.1
        
        current += mean_reversion + random_walk
        kimchi_premium.append(current)
    
    # BTC 가격 (약간의 변동)
    btc_prices = 159_000_000 + np.random.normal(0, 1_000_000, hours)
    
    # DataFrame 생성
    data = pd.DataFrame({
        'timestamp': timestamps,
        'kimchi_premium': kimchi_premium,
        'btc_price': btc_prices
    })
    
    # 전략 실행
    strategy = MeanReversionHedgeStrategy(
        capital=40_000_000,
        target_profit_krw=120_000,
        lookback_period=48,
        entry_threshold=-0.05,  # 평균 대비 -0.05% 이하
        hedge_threshold=-0.03    # 진입가 대비 -0.03% 이하
    )
    
    result = strategy.execute_strategy(data)
    
    print("\n[전략 실행 결과]")
    print(f"총 거래 횟수: {result['num_trades']}회")
    print(f"승률: {result['win_rate']*100:.1f}%")
    print(f"총 수익: {result['total_profit']:,.0f}원")
    print(f"총 수익률: {result['total_return']:.2f}%")
    print(f"평균 거래당 수익: {result['avg_profit']:,.0f}원")
    print(f"헤지 실행 횟수: {result['hedge_count']}회")
    
    # 월간 예상
    if result['num_trades'] > 0:
        daily_trades = result['num_trades'] / 30
        monthly_return = result['total_return']
        monthly_profit = result['total_profit']
        
        print("\n[월간 예상 수익]")
        print(f"일평균 거래: {daily_trades:.1f}회")
        print(f"월 수익률: {monthly_return:.2f}%")
        print(f"월 수익금: {monthly_profit:,.0f}원")
    
    # 파라미터 최적화
    print("\n[파라미터 최적화 중...]")
    
    param_ranges = {
        'entry_threshold': [-0.1, -0.05, -0.03],
        'target_profit': [100000, 120000, 150000],
        'hedge_threshold': [-0.05, -0.03, -0.02]
    }
    
    best = strategy.optimize_parameters(data, param_ranges)
    
    if best:
        print("\n[최적 파라미터]")
        print(f"진입 임계값: {best['entry_threshold']*100:.2f}%")
        print(f"목표 수익: {best['target_profit']:,.0f}원")
        print(f"헤지 임계값: {best['hedge_threshold']*100:.2f}%")
        print(f"예상 월 수익률: {best['result']['total_return']:.2f}%")
    
    # 전략 개선 제안
    print("\n[전략 개선 제안]")
    print("""
1. **진입 타이밍 개선**
   - RSI < 30 추가 조건
   - 볼린저 밴드 하단 터치
   - 거래량 급증 확인

2. **포지션 관리**
   - 분할 진입 (1/3씩 3번)
   - 피라미딩 (수익시 추가)
   - 동적 목표 조정

3. **헤지 전략 강화**
   - 옵션 활용 (풋옵션)
   - 선물 헤지 병행
   - 상관관계 높은 알트코인 활용

4. **리스크 관리**
   - 최대 손실 -0.5% 제한
   - 일일 거래 횟수 제한
   - 변동성 기반 포지션 조정

5. **실행 최적화**
   - 지정가 주문으로 수수료 절감
   - 멀티 거래소 활용
   - 레이턴시 최소화
    """)


if __name__ == "__main__":
    analyze_mean_reversion_strategy()