"""
볼린저 밴드 기반 전략 (Bollinger Bands Strategy)
변동성 기반 평균회귀 전략
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import numpy as np

from .base_strategy import (
    BaseStrategy,
    MarketData,
    TradingSignal,
    SignalType
)

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    볼린저 밴드 기반 전략
    
    김프의 볼린저 밴드를 계산하여
    상단/하단 밴드 터치 시 평균회귀 거래
    """
    
    def __init__(
        self,
        name: str = "BollingerBandsStrategy",
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
            'bb_period': 20,             # 볼린저 밴드 기간
            'bb_std': 2.0,               # 표준편차 배수
            'entry_bb_pct': 0.9,         # 상단밴드 진입 위치 (90%)
            'exit_bb_pct': 0.5,          # 중간선 청산 위치 (50%)
            'min_band_width': 0.5,       # 최소 밴드 폭 (%)
            'max_band_width': 5.0,       # 최대 밴드 폭 (%)
            'position_size_pct': 0.12,   # 포지션 크기 (12%)
            'stop_loss': -2.0,           # 손절 (%)
            'take_profit': 4.0,          # 익절 (%)
            'volume_check': True,        # 거래량 체크
            'min_volume_krw': 100_000_000,  # 최소 거래량
            'buffer_size': 100,          # 버퍼 크기
            'squeeze_detection': True,    # 스퀴즈 감지 여부
            'min_premium': 1.5           # 최소 김프 요구 수준
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config, initial_capital)
        
        # 전략 특화 변수
        self.premium_history: List[float] = []
        self.bb_upper: List[float] = []  # 상단 밴드
        self.bb_middle: List[float] = []  # 중간선 (SMA)
        self.bb_lower: List[float] = []   # 하단 밴드
        self.band_width: List[float] = []  # 밴드 폭
        self.bb_pct: List[float] = []     # BB %B 지표
        
        logger.info(
            f"BollingerBandsStrategy initialized - "
            f"Period: {self.config['bb_period']}, Std: {self.config['bb_std']}"
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
        if len(self.premium_history) < self.config['bb_period']:
            logger.debug(
                f"Not enough data: {len(self.premium_history)}/{self.config['bb_period']}"
            )
            return None
        
        # 볼린저 밴드 계산
        bb_data = self._calculate_bollinger_bands()
        
        if bb_data is None:
            return None
        
        upper, middle, lower, width, pct_b = bb_data
        
        # 볼린저 밴드 값 저장
        self.bb_upper.append(upper)
        self.bb_middle.append(middle)
        self.bb_lower.append(lower)
        self.band_width.append(width)
        self.bb_pct.append(pct_b)
        
        # 버퍼 크기 유지
        if len(self.bb_upper) > self.config['buffer_size']:
            self.bb_upper = self.bb_upper[-self.config['buffer_size']:]
            self.bb_middle = self.bb_middle[-self.config['buffer_size']:]
            self.bb_lower = self.bb_lower[-self.config['buffer_size']:]
            self.band_width = self.band_width[-self.config['buffer_size']:]
            self.bb_pct = self.bb_pct[-self.config['buffer_size']:]
        
        # 거래량 체크
        if self.config['volume_check']:
            volume_krw = market_data.volume_upbit * market_data.upbit_price
            if volume_krw < self.config['min_volume_krw']:
                return None
        
        # 스퀴즈 감지 (밴드 폭이 너무 좁으면 거래 안함)
        if self.config['squeeze_detection']:
            if width < self.config['min_band_width']:
                logger.debug(f"Band squeeze detected: width {width:.2f}% too narrow")
                return None
            if width > self.config['max_band_width']:
                logger.debug(f"Band too wide: width {width:.2f}% exceeds max")
                return None
        
        # 포지션이 없는 경우 - 진입 신호 검토
        if self.position == 0:
            # 진입 조건:
            # 1. 김프가 상단 밴드 근처 (overbought)
            # 2. 밴드 폭이 적절함
            # 3. 최소 김프 수준 충족
            if (pct_b >= self.config['entry_bb_pct'] and
                market_data.kimchi_premium >= self.config['min_premium']):
                
                confidence = self._calculate_confidence(
                    pct_b,
                    width,
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
                        f"BB 상단 터치: %B={pct_b:.2f}, "
                        f"Upper={upper:.2f}%, Current={market_data.kimchi_premium:.2f}%"
                    ),
                    metadata={
                        'bb_upper': upper,
                        'bb_middle': middle,
                        'bb_lower': lower,
                        'bb_pct': pct_b,
                        'band_width': width,
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
            
            # 밴드 폭에 따른 조정 (폭이 넓을수록 변동성 크므로 포지션 줄임)
            if 'band_width' in signal.metadata:
                width = signal.metadata['band_width']
                optimal_width = (self.config['min_band_width'] + self.config['max_band_width']) / 2
                width_multiplier = min(1.2, optimal_width / max(width, 0.1))
                position_size *= width_multiplier
            
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
        
        # 볼린저 밴드가 계산되지 않았으면 유지
        if not self.bb_pct:
            return False
        
        current_pct_b = self.bb_pct[-1]
        current_middle = self.bb_middle[-1]
        
        # 수익률 계산
        current_price = market_data.upbit_price
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # 청산 조건
        # 1. 평균회귀 완료 (중간선 도달)
        if current_pct_b <= self.config['exit_bb_pct']:
            logger.info(
                f"Mean reversion exit: %B {current_pct_b:.2f} <= {self.config['exit_bb_pct']}"
            )
            return True
        
        # 2. 하단 밴드 돌파 (oversold)
        if current_pct_b <= 0.1:
            logger.info(f"Lower band breach: %B {current_pct_b:.2f}")
            return True
        
        # 3. 익절
        if pnl_pct >= self.config['take_profit']:
            logger.info(f"Take profit triggered: {pnl_pct:.2f}%")
            return True
        
        # 4. 손절
        if pnl_pct <= self.config['stop_loss']:
            logger.warning(f"Stop loss triggered: {pnl_pct:.2f}%")
            return True
        
        # 5. 김프 역전
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
        max_size = max(self.config['bb_period'] * 2, self.config['buffer_size'])
        if len(self.premium_history) > max_size:
            self.premium_history = self.premium_history[-max_size:]
    
    def _calculate_bollinger_bands(self) -> Optional[tuple]:
        """
        볼린저 밴드 계산
        
        Returns:
            (upper, middle, lower, width, pct_b) 튜플 또는 None
        """
        if len(self.premium_history) < self.config['bb_period']:
            return None
        
        # 최근 데이터
        recent_data = self.premium_history[-self.config['bb_period']:]
        current_value = self.premium_history[-1]
        
        # 중간선 (SMA)
        middle = np.mean(recent_data)
        
        # 표준편차
        std = np.std(recent_data)
        
        # 상단/하단 밴드
        upper = middle + (std * self.config['bb_std'])
        lower = middle - (std * self.config['bb_std'])
        
        # 밴드 폭
        width = upper - lower
        
        # %B 계산 (현재 값이 밴드 내 어디에 있는지)
        if width > 0:
            pct_b = (current_value - lower) / width
        else:
            pct_b = 0.5
        
        return upper, middle, lower, width, pct_b
    
    def _calculate_confidence(
        self,
        pct_b: float,
        band_width: float,
        kimchi_premium: float,
        signal_type: str
    ) -> float:
        """
        신호 신뢰도 계산
        
        Args:
            pct_b: 볼린저 밴드 %B
            band_width: 밴드 폭
            kimchi_premium: 현재 김프
            signal_type: 신호 유형
            
        Returns:
            신뢰도 (0~1)
        """
        if signal_type == 'entry':
            # 기본 신뢰도
            confidence = 0.5
            
            # %B 위치에 따른 조정
            if pct_b >= 1.0:  # 상단 밴드 돌파
                confidence += 0.2
            elif pct_b >= 0.95:
                confidence += 0.1
            
            # 밴드 폭에 따른 조정 (적정 폭일수록 신뢰도 증가)
            optimal_width = (self.config['min_band_width'] + self.config['max_band_width']) / 2
            if abs(band_width - optimal_width) < 0.5:
                confidence += 0.15
            
            # 김프 수준에 따른 조정
            if kimchi_premium >= self.config['min_premium'] * 1.5:
                confidence += 0.15
            
            # 볼린저 밴드 수렴/발산 패턴 체크
            if len(self.band_width) >= 5:
                recent_widths = self.band_width[-5:]
                if all(recent_widths[i] < recent_widths[i+1] for i in range(len(recent_widths)-1)):
                    # 밴드 확장 중 (변동성 증가)
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
            'type': 'BollingerBands',
            'bb_period': self.config['bb_period'],
            'bb_std': self.config['bb_std'],
            'entry_bb_pct': self.config['entry_bb_pct'],
            'exit_bb_pct': self.config['exit_bb_pct'],
            'min_band_width': self.config['min_band_width'],
            'max_band_width': self.config['max_band_width'],
            'position_size_pct': self.config['position_size_pct'],
            'current_bb_upper': self.bb_upper[-1] if self.bb_upper else 0,
            'current_bb_middle': self.bb_middle[-1] if self.bb_middle else 0,
            'current_bb_lower': self.bb_lower[-1] if self.bb_lower else 0,
            'current_bb_pct': self.bb_pct[-1] if self.bb_pct else 0,
            'current_band_width': self.band_width[-1] if self.band_width else 0
        }