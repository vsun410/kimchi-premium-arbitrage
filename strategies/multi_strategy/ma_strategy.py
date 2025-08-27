"""
이동평균 기반 전략 (Moving Average Strategy)
단기/장기 이동평균을 활용한 모멘텀 기반 거래
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
import numpy as np

from .base_strategy import (
    BaseStrategy,
    MarketData,
    TradingSignal,
    SignalType
)

logger = logging.getLogger(__name__)


class MovingAverageStrategy(BaseStrategy):
    """
    이동평균 기반 전략
    
    김프의 단기/장기 이동평균을 비교하여
    모멘텀 기반으로 진입/청산 결정
    """
    
    def __init__(
        self,
        name: str = "MovingAverageStrategy",
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
            'short_window': 10,          # 단기 이동평균 (10분)
            'long_window': 30,           # 장기 이동평균 (30분)
            'entry_ma_spread': 0.5,      # MA 스프레드 진입 임계값 (%)
            'exit_ma_spread': -0.2,      # MA 스프레드 청산 임계값 (%)
            'min_premium': 2.0,          # 최소 김프 요구 수준 (%)
            'position_size_pct': 0.15,   # 포지션 크기 (15%)
            'stop_loss': -1.5,           # 손절 (%)
            'take_profit': 3.0,          # 익절 (%)
            'volume_check': True,        # 거래량 체크 여부
            'min_volume_krw': 100_000_000,  # 최소 거래량
            'buffer_size': 100,          # 데이터 버퍼 크기
            'ma_type': 'sma'            # 이동평균 유형 (sma, ema, wma)
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config, initial_capital)
        
        # 전략 특화 변수
        self.ma_short_values: List[float] = []  # 단기 MA 값들
        self.ma_long_values: List[float] = []   # 장기 MA 값들
        self.premium_history: List[float] = []  # 김프 이력
        
        logger.info(
            f"MovingAverageStrategy initialized - "
            f"Short: {self.config['short_window']}, Long: {self.config['long_window']}"
        )
    
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        시장 데이터 분석 및 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            거래 신호 또는 None
        """
        # 김프 이력 업데이트
        self._update_premium_history(market_data.kimchi_premium)
        
        # 데이터가 충분하지 않으면 패스
        if len(self.premium_history) < self.config['long_window']:
            logger.debug(
                f"Not enough data: {len(self.premium_history)}/{self.config['long_window']}"
            )
            return None
        
        # 이동평균 계산
        ma_short = self._calculate_ma(
            self.premium_history,
            self.config['short_window'],
            self.config['ma_type']
        )
        ma_long = self._calculate_ma(
            self.premium_history,
            self.config['long_window'],
            self.config['ma_type']
        )
        
        # MA 스프레드 계산
        ma_spread = ma_short - ma_long
        
        # 이동평균 값 저장
        self.ma_short_values.append(ma_short)
        self.ma_long_values.append(ma_long)
        
        # 버퍼 크기 유지
        if len(self.ma_short_values) > self.config['buffer_size']:
            self.ma_short_values = self.ma_short_values[-self.config['buffer_size']:]
            self.ma_long_values = self.ma_long_values[-self.config['buffer_size']:]
        
        # 거래량 체크
        if self.config['volume_check']:
            volume_krw = market_data.volume_upbit * market_data.upbit_price
            if volume_krw < self.config['min_volume_krw']:
                return None
        
        # 포지션이 없는 경우 - 진입 신호 검토
        if self.position == 0:
            # 진입 조건:
            # 1. 단기 MA > 장기 MA (상승 모멘텀)
            # 2. MA 스프레드가 임계값 이상
            # 3. 현재 김프가 최소 수준 이상
            if (ma_spread >= self.config['entry_ma_spread'] and
                market_data.kimchi_premium >= self.config['min_premium']):
                
                confidence = self._calculate_confidence(
                    ma_spread,
                    market_data.kimchi_premium,
                    'entry'
                )
                
                signal = TradingSignal(
                    timestamp=market_data.timestamp,
                    strategy_name=self.name,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    suggested_amount=0,
                    reason=(
                        f"MA 크로스오버: 단기({ma_short:.2f}) > 장기({ma_long:.2f}), "
                        f"스프레드: {ma_spread:.2f}%"
                    ),
                    metadata={
                        'ma_short': ma_short,
                        'ma_long': ma_long,
                        'ma_spread': ma_spread,
                        'kimchi_premium': market_data.kimchi_premium,
                        'upbit_price': market_data.upbit_price,
                        'binance_price': market_data.binance_price
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
        # 기본 포지션 크기
        position_value_krw = self.current_capital * self.config['position_size_pct']
        
        # BTC 가격으로 변환
        if 'upbit_price' in signal.metadata:
            btc_price = signal.metadata['upbit_price']
            position_size = position_value_krw / btc_price
            
            # 신뢰도에 따른 조정
            position_size *= signal.confidence
            
            # MA 스프레드가 클수록 포지션 증가
            if 'ma_spread' in signal.metadata:
                spread_multiplier = min(1.5, 1 + signal.metadata['ma_spread'] / 10)
                position_size *= spread_multiplier
            
            # 최소 거래 크기 체크
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
        
        # 이동평균이 계산되지 않았으면 유지
        if len(self.ma_short_values) < 2 or len(self.ma_long_values) < 2:
            return False
        
        ma_short = self.ma_short_values[-1]
        ma_long = self.ma_long_values[-1]
        ma_spread = ma_short - ma_long
        
        # 수익률 계산
        current_price = market_data.upbit_price
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # 청산 조건
        # 1. MA 크로스언더 (하락 전환)
        if ma_spread <= self.config['exit_ma_spread']:
            logger.info(
                f"MA crossunder exit: spread {ma_spread:.2f}% <= {self.config['exit_ma_spread']}%"
            )
            return True
        
        # 2. 익절
        if pnl_pct >= self.config['take_profit']:
            logger.info(f"Take profit triggered: {pnl_pct:.2f}%")
            return True
        
        # 3. 손절
        if pnl_pct <= self.config['stop_loss']:
            logger.warning(f"Stop loss triggered: {pnl_pct:.2f}%")
            return True
        
        # 4. 김프 역전
        if market_data.kimchi_premium < 0:
            logger.warning(f"Kimchi premium reversed: {market_data.kimchi_premium:.2f}%")
            return True
        
        return False
    
    def _update_premium_history(self, premium: float):
        """
        김프 이력 업데이트
        
        Args:
            premium: 현재 김치 프리미엄
        """
        self.premium_history.append(premium)
        
        # 최대 버퍼 크기 유지
        max_size = max(self.config['long_window'] * 2, self.config['buffer_size'])
        if len(self.premium_history) > max_size:
            self.premium_history = self.premium_history[-max_size:]
    
    def _calculate_ma(
        self,
        data: List[float],
        window: int,
        ma_type: str = 'sma'
    ) -> float:
        """
        이동평균 계산
        
        Args:
            data: 데이터 리스트
            window: 이동평균 기간
            ma_type: 이동평균 유형 (sma, ema, wma)
            
        Returns:
            이동평균 값
        """
        if len(data) < window:
            return 0
        
        recent_data = data[-window:]
        
        if ma_type == 'sma':
            # Simple Moving Average
            return np.mean(recent_data)
        
        elif ma_type == 'ema':
            # Exponential Moving Average
            alpha = 2 / (window + 1)
            ema = recent_data[0]
            for value in recent_data[1:]:
                ema = alpha * value + (1 - alpha) * ema
            return ema
        
        elif ma_type == 'wma':
            # Weighted Moving Average
            weights = np.arange(1, window + 1)
            return np.average(recent_data, weights=weights)
        
        else:
            return np.mean(recent_data)
    
    def _calculate_confidence(
        self,
        ma_spread: float,
        kimchi_premium: float,
        signal_type: str
    ) -> float:
        """
        신호 신뢰도 계산
        
        Args:
            ma_spread: MA 스프레드
            kimchi_premium: 현재 김프
            signal_type: 신호 유형
            
        Returns:
            신뢰도 (0~1)
        """
        if signal_type == 'entry':
            # 기본 신뢰도
            confidence = 0.5
            
            # MA 스프레드에 따른 조정
            if ma_spread >= self.config['entry_ma_spread'] * 2:
                confidence += 0.2
            elif ma_spread >= self.config['entry_ma_spread'] * 1.5:
                confidence += 0.1
            
            # 김프 수준에 따른 조정
            if kimchi_premium >= self.config['min_premium'] * 1.5:
                confidence += 0.2
            elif kimchi_premium >= self.config['min_premium'] * 1.2:
                confidence += 0.1
            
            # 트렌드 강도 체크 (최근 MA 기울기)
            if len(self.ma_short_values) >= 3:
                recent_trend = self.ma_short_values[-1] - self.ma_short_values[-3]
                if recent_trend > 0.5:
                    confidence += 0.1
            
            return min(1.0, confidence)
        
        return 0.6
    
    def get_strategy_params(self) -> Dict:
        """
        전략 파라미터 조회
        
        Returns:
            전략 파라미터
        """
        return {
            'name': self.name,
            'type': 'MovingAverage',
            'short_window': self.config['short_window'],
            'long_window': self.config['long_window'],
            'ma_type': self.config['ma_type'],
            'entry_ma_spread': self.config['entry_ma_spread'],
            'exit_ma_spread': self.config['exit_ma_spread'],
            'min_premium': self.config['min_premium'],
            'position_size_pct': self.config['position_size_pct'],
            'current_ma_short': self.ma_short_values[-1] if self.ma_short_values else 0,
            'current_ma_long': self.ma_long_values[-1] if self.ma_long_values else 0
        }