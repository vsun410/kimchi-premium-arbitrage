"""
임계값 기반 전략 (Threshold Strategy)
김치 프리미엄이 특정 임계값을 넘으면 진입/청산
"""

from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta

from .base_strategy import (
    BaseStrategy,
    MarketData,
    TradingSignal,
    SignalType
)

logger = logging.getLogger(__name__)


class ThresholdStrategy(BaseStrategy):
    """
    임계값 기반 전략
    
    김프가 특정 임계값을 넘으면 진입, 
    낮은 임계값 이하로 떨어지면 청산
    """
    
    def __init__(
        self,
        name: str = "ThresholdStrategy",
        config: Optional[Dict[str, Any]] = None,
        initial_capital: float = 1_000_000
    ):
        """
        초기화
        
        Args:
            name: 전략 이름
            config: 전략 설정
            initial_capital: 초기 자본금
        """
        # 기본 설정
        default_config = {
            'entry_threshold': 3.0,      # 진입 임계값 (%)
            'exit_threshold': 1.5,       # 청산 임계값 (%)
            'stop_loss': -2.0,           # 손절 임계값 (%)
            'position_size_pct': 0.1,    # 포지션 크기 (10%)
            'min_hold_time': 300,        # 최소 보유 시간 (5분)
            'cooldown_period': 600,      # 재진입 쿨다운 (10분)
            'volume_check': True,        # 거래량 체크 여부
            'min_volume_krw': 100_000_000,  # 최소 거래량 (1억원)
            'buffer_size': 60            # 데이터 버퍼 크기
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config, initial_capital)
        
        # 전략 특화 변수
        self.last_exit_time = None  # 마지막 청산 시간
        self.consecutive_signals = 0  # 연속 신호 카운터
        
        logger.info(
            f"ThresholdStrategy initialized - "
            f"Entry: {self.config['entry_threshold']}%, "
            f"Exit: {self.config['exit_threshold']}%"
        )
    
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        시장 데이터 분석 및 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            거래 신호 또는 None
        """
        # 김프 체크
        kimchi_premium = market_data.kimchi_premium
        
        # 거래량 체크
        if self.config['volume_check']:
            volume_krw = market_data.volume_upbit * market_data.upbit_price
            if volume_krw < self.config['min_volume_krw']:
                logger.debug(
                    f"Volume too low: {volume_krw:,.0f} < {self.config['min_volume_krw']:,.0f}"
                )
                return None
        
        # 쿨다운 중인지 체크
        if self._is_in_cooldown(market_data.timestamp):
            return None
        
        # 포지션이 없는 경우 - 진입 신호 검토
        if self.position == 0:
            if kimchi_premium >= self.config['entry_threshold']:
                # 진입 조건 만족
                confidence = self._calculate_confidence(kimchi_premium, 'entry')
                
                signal = TradingSignal(
                    timestamp=market_data.timestamp,
                    strategy_name=self.name,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    suggested_amount=0,  # calculate_position_size에서 계산
                    reason=f"김프 {kimchi_premium:.2f}% > 임계값 {self.config['entry_threshold']}%",
                    metadata={
                        'kimchi_premium': kimchi_premium,
                        'upbit_price': market_data.upbit_price,
                        'binance_price': market_data.binance_price,
                        'exchange_rate': market_data.exchange_rate
                    }
                )
                
                logger.info(f"Entry signal generated: {signal}")
                return signal
        
        return None
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            
        Returns:
            포지션 크기 (BTC)
        """
        # 포지션 크기 = 자본금의 X%
        position_value_krw = self.current_capital * self.config['position_size_pct']
        
        # BTC 가격으로 변환 (메타데이터에서 가격 참조)
        if 'upbit_price' in signal.metadata:
            btc_price = signal.metadata['upbit_price']
            position_size = position_value_krw / btc_price
            
            # 신뢰도에 따른 조정
            position_size *= signal.confidence
            
            # 최소 거래 크기 체크 (0.0001 BTC)
            if position_size < 0.0001:
                return 0
            
            return round(position_size, 4)
        
        return 0
    
    def should_close_position(self, market_data: MarketData) -> bool:
        """
        포지션 청산 여부 결정
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            청산 여부
        """
        if self.position == 0:
            return False
        
        kimchi_premium = market_data.kimchi_premium
        
        # 최소 보유 시간 체크
        if self.entry_time:
            hold_time = (market_data.timestamp - self.entry_time).total_seconds()
            if hold_time < self.config['min_hold_time']:
                return False
        
        # 청산 조건 체크
        # 1. 이익 청산: 김프가 exit_threshold 이하
        if kimchi_premium <= self.config['exit_threshold']:
            logger.info(
                f"Exit condition met: 김프 {kimchi_premium:.2f}% <= {self.config['exit_threshold']}%"
            )
            return True
        
        # 2. 손절: 김프가 음수 또는 stop_loss 이하
        if kimchi_premium <= self.config['stop_loss']:
            logger.warning(
                f"Stop loss triggered: 김프 {kimchi_premium:.2f}% <= {self.config['stop_loss']}%"
            )
            return True
        
        # 3. 김프 역전 (음수)
        if kimchi_premium < 0:
            logger.warning(f"Kimchi premium reversed: {kimchi_premium:.2f}%")
            return True
        
        return False
    
    def _calculate_confidence(self, kimchi_premium: float, signal_type: str) -> float:
        """
        신호 신뢰도 계산
        
        Args:
            kimchi_premium: 현재 김프
            signal_type: 'entry' 또는 'exit'
            
        Returns:
            신뢰도 (0~1)
        """
        if signal_type == 'entry':
            # 진입 신호의 경우
            threshold = self.config['entry_threshold']
            
            # 임계값 대비 초과 비율로 신뢰도 계산
            if kimchi_premium >= threshold * 1.5:  # 50% 초과
                confidence = 0.9
            elif kimchi_premium >= threshold * 1.2:  # 20% 초과
                confidence = 0.7
            else:
                confidence = 0.5
        else:
            # 기본 신뢰도
            confidence = 0.6
        
        # 버퍼 데이터가 충분히 쌓였으면 보너스
        if len(self.data_buffer) >= self.config['buffer_size']:
            confidence = min(1.0, confidence + 0.1)
        
        return confidence
    
    def _is_in_cooldown(self, current_time: datetime) -> bool:
        """
        쿨다운 중인지 체크
        
        Args:
            current_time: 현재 시간
            
        Returns:
            쿨다운 중 여부
        """
        if self.last_exit_time is None:
            return False
        
        cooldown_seconds = self.config['cooldown_period']
        time_since_exit = (current_time - self.last_exit_time).total_seconds()
        
        if time_since_exit < cooldown_seconds:
            remaining = cooldown_seconds - time_since_exit
            logger.debug(f"In cooldown period: {remaining:.0f} seconds remaining")
            return True
        
        return False
    
    def execute_trade(self, signal: TradingSignal, execution_price: float) -> bool:
        """
        거래 실행 (오버라이드)
        
        Args:
            signal: 거래 신호
            execution_price: 체결 가격
            
        Returns:
            실행 성공 여부
        """
        result = super().execute_trade(signal, execution_price)
        
        # 청산/매도 시 마지막 청산 시간 업데이트
        if result and signal.signal_type in [SignalType.SELL, SignalType.CLOSE]:
            self.last_exit_time = signal.timestamp
        
        return result
    
    def get_strategy_params(self) -> Dict:
        """
        전략 파라미터 조회
        
        Returns:
            전략 파라미터
        """
        return {
            'name': self.name,
            'type': 'Threshold',
            'entry_threshold': self.config['entry_threshold'],
            'exit_threshold': self.config['exit_threshold'],
            'stop_loss': self.config['stop_loss'],
            'position_size_pct': self.config['position_size_pct'],
            'min_hold_time': self.config['min_hold_time'],
            'cooldown_period': self.config['cooldown_period'],
            'in_cooldown': self._is_in_cooldown(datetime.now())
        }