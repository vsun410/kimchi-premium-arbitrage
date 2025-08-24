"""
Strategy Simulator for Dynamic Hedge
Dynamic Hedge 전략 시뮬레이션
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Dynamic Hedge 모듈 임포트
from dynamic_hedge import (
    TrendAnalysisEngine,
    DynamicPositionManager, 
    TrianglePatternDetector,
    ReversePremiumHandler
)
from dynamic_hedge.position_manager import ExitCondition

from .backtest_engine import BacktestEngine, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """거래 신호"""
    timestamp: datetime
    action: str  # 'open_hedge', 'close_all', 'close_long', 'close_short'
    reason: str
    confidence: float
    data: Dict[str, Any]


class StrategySimulator:
    """
    Dynamic Hedge 전략 시뮬레이터
    - 진입/청산 신호 생성
    - 포지션 사이징
    - 리스크 관리
    """
    
    def __init__(self, backtest_engine: BacktestEngine,
                 position_size_pct: float = 0.02):
        """
        Args:
            backtest_engine: 백테스팅 엔진
            position_size_pct: 포지션 크기 (자본금 대비 %)
        """
        self.engine = backtest_engine
        self.position_size_pct = position_size_pct
        
        # Dynamic Hedge 컴포넌트
        self.trend_engine = TrendAnalysisEngine(window_size=100)
        self.position_manager = DynamicPositionManager(capital_per_exchange=20000000)
        self.pattern_detector = TrianglePatternDetector()
        self.reverse_handler = ReversePremiumHandler()
        
        # 상태
        self.is_hedged = False
        self.entry_premium = None
        self.signals: List[Signal] = []
        
        # 파라미터
        self.entry_threshold = 0.01  # 1% 이상 김프에서 진입
        self.exit_threshold = 0.005  # 0.5% 이하에서 청산
        self.max_premium_threshold = 0.0014  # 0.14% (14만원) 이상에서 전체 청산
        
    def generate_signals(self, timestamp: datetime, data: Dict) -> List[Signal]:
        """
        거래 신호 생성
        
        Args:
            timestamp: 현재 시간
            data: {
                'upbit_price': float,
                'binance_price': float,
                'kimchi_premium': float,
                'upbit_ohlcv': pd.DataFrame,
                'binance_ohlcv': pd.DataFrame
            }
            
        Returns:
            신호 리스트
        """
        signals = []
        
        # 현재 프리미엄
        current_premium = data.get('kimchi_premium', 0) / 100  # 퍼센트를 비율로
        
        # 역프리미엄 업데이트
        reverse_result = self.reverse_handler.update(
            current_premium,
            {
                'upbit_price': data['upbit_price'],
                'binance_price': data['binance_price'],
                'volume': data.get('volume', 100),
                'trend': 'sideways'
            }
        )
        
        # 추세 분석 (충분한 데이터가 있을 때만)
        trend_analysis = {}
        if 'upbit_ohlcv' in data and len(data['upbit_ohlcv']) >= 100:
            trend_analysis = self.trend_engine.analyze(data['upbit_ohlcv'])
        
        # 포지션 상태에 따른 신호 생성
        if not self.is_hedged:
            # 진입 신호 확인
            if self._check_entry_conditions(current_premium, trend_analysis):
                signal = Signal(
                    timestamp=timestamp,
                    action='open_hedge',
                    reason=f'Premium {current_premium:.2%} above threshold',
                    confidence=0.8,
                    data={'premium': current_premium}
                )
                signals.append(signal)
                
        else:
            # 청산 신호 확인
            exit_signals = self._check_exit_conditions(
                current_premium, 
                trend_analysis,
                reverse_result
            )
            signals.extend(exit_signals)
        
        self.signals.extend(signals)
        return signals
    
    def _check_entry_conditions(self, premium: float, 
                               trend_analysis: Dict) -> bool:
        """진입 조건 확인"""
        # 기본 조건: 프리미엄이 임계값 이상
        if premium < self.entry_threshold:
            return False
        
        # 추가 조건: 변동성이 적절한 수준
        # (너무 높으면 리스크 증가)
        if trend_analysis:
            # 삼각수렴 패턴이 있으면 진입 연기
            if trend_analysis.get('triangle_patterns'):
                for pattern in trend_analysis['triangle_patterns']:
                    if pattern.volatility_compression > 0.7:
                        logger.info("Triangle pattern detected, delaying entry")
                        return False
        
        return True
    
    def _check_exit_conditions(self, premium: float,
                              trend_analysis: Dict,
                              reverse_result: Dict) -> List[Signal]:
        """청산 조건 확인"""
        signals = []
        timestamp = datetime.now()  # 실제로는 백테스트 시간 사용
        
        # 1. 김프가 너무 높아진 경우 (14만원 이상)
        if premium >= self.max_premium_threshold:
            signal = Signal(
                timestamp=timestamp,
                action='close_all',
                reason=f'Premium {premium:.2%} exceeded max threshold',
                confidence=0.9,
                data={'premium': premium}
            )
            signals.append(signal)
            return signals
        
        # 2. 프리미엄이 임계값 이하로 떨어진 경우
        if premium <= self.exit_threshold:
            signal = Signal(
                timestamp=timestamp,
                action='close_all',
                reason=f'Premium {premium:.2%} below exit threshold',
                confidence=0.7,
                data={'premium': premium}
            )
            signals.append(signal)
            return signals
        
        # 3. 추세 돌파 신호
        if trend_analysis.get('breakout_signals'):
            for breakout in trend_analysis['breakout_signals']:
                if breakout.direction == 'up' and breakout.strength > 0.02:
                    signal = Signal(
                        timestamp=timestamp,
                        action='close_short',
                        reason=f'Upward breakout detected, strength={breakout.strength:.2%}',
                        confidence=breakout.strength,
                        data={'breakout': breakout}
                    )
                    signals.append(signal)
                elif breakout.direction == 'down' and breakout.strength > 0.02:
                    signal = Signal(
                        timestamp=timestamp,
                        action='close_long',
                        reason=f'Downward breakout detected, strength={breakout.strength:.2%}',
                        confidence=breakout.strength,
                        data={'breakout': breakout}
                    )
                    signals.append(signal)
        
        # 4. 역프리미엄 대응
        if reverse_result.get('is_reverse') and reverse_result.get('action'):
            action = reverse_result['action']
            if action.get('type') == 'immediate_exit':
                signal = Signal(
                    timestamp=timestamp,
                    action='close_all',
                    reason='Reverse premium detected, immediate exit required',
                    confidence=0.8,
                    data={'reverse_premium': premium}
                )
                signals.append(signal)
        
        return signals
    
    def execute_signal(self, signal: Signal, upbit_price: float, 
                       binance_price: float) -> bool:
        """
        신호 실행
        
        Args:
            signal: 거래 신호
            upbit_price: 업비트 가격
            binance_price: 바이낸스 가격
            
        Returns:
            실행 성공 여부
        """
        success = True
        
        if signal.action == 'open_hedge':
            # 포지션 크기 계산
            total_capital = self.engine.get_portfolio_value()
            position_value = total_capital * self.position_size_pct
            
            # BTC 수량 계산 (업비트 기준)
            btc_amount = position_value / upbit_price
            
            # 업비트 롱 포지션
            if self.engine.open_position('upbit', 'BTC', PositionSide.LONG, 
                                        btc_amount, upbit_price):
                # 바이낸스 숏 포지션
                if self.engine.open_position('binance', 'BTC', PositionSide.SHORT,
                                           btc_amount, binance_price):
                    self.is_hedged = True
                    self.entry_premium = signal.data.get('premium')
                    logger.info(f"Hedge opened: {btc_amount:.4f} BTC @ premium {self.entry_premium:.2%}")
                else:
                    # 바이낸스 실패 시 업비트도 롤백
                    self.engine.close_position('upbit', 'BTC')
                    success = False
            else:
                success = False
                
        elif signal.action == 'close_all':
            # 모든 포지션 청산
            pnl_upbit = self.engine.close_position('upbit', 'BTC', price=upbit_price)
            pnl_binance = self.engine.close_position('binance', 'BTC', price=binance_price)
            
            total_pnl = pnl_upbit + pnl_binance
            self.is_hedged = False
            
            logger.info(f"All positions closed: PnL={total_pnl:,.0f}")
            
        elif signal.action == 'close_long':
            # 업비트 롱 포지션만 청산
            pnl = self.engine.close_position('upbit', 'BTC', price=upbit_price)
            logger.info(f"Long position closed: PnL={pnl:,.0f}")
            
        elif signal.action == 'close_short':
            # 바이낸스 숏 포지션만 청산
            pnl = self.engine.close_position('binance', 'BTC', price=binance_price)
            logger.info(f"Short position closed: PnL={pnl:,.0f}")
        
        return success
    
    def calculate_position_size(self, capital: float, risk_pct: float = 0.02) -> float:
        """
        Kelly Criterion 기반 포지션 사이징
        
        Args:
            capital: 가용 자본
            risk_pct: 리스크 비율
            
        Returns:
            포지션 크기
        """
        # 과거 승률 계산 (신호 기반)
        if len(self.signals) >= 10:
            winning_signals = sum(1 for s in self.signals[-10:] 
                                if 'profit' in s.data and s.data['profit'] > 0)
            win_rate = winning_signals / 10
        else:
            win_rate = 0.6  # 기본 승률
        
        # Kelly 비율 계산 (보수적으로 25% 적용)
        kelly_ratio = (win_rate * 1.5 - (1 - win_rate)) / 1.5
        conservative_ratio = kelly_ratio * 0.25
        
        # 최종 포지션 크기
        position_ratio = min(conservative_ratio, risk_pct)
        return capital * position_ratio
    
    def get_strategy_stats(self) -> Dict:
        """전략 통계 반환"""
        stats = {
            'total_signals': len(self.signals),
            'open_signals': sum(1 for s in self.signals if s.action == 'open_hedge'),
            'close_signals': sum(1 for s in self.signals if 'close' in s.action),
            'avg_confidence': np.mean([s.confidence for s in self.signals]) if self.signals else 0,
            'is_hedged': self.is_hedged
        }
        
        # 신호별 통계
        action_counts = {}
        for signal in self.signals:
            action_counts[signal.action] = action_counts.get(signal.action, 0) + 1
        stats['action_counts'] = action_counts
        
        return stats