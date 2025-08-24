"""
Mean Reversion Trading Strategy
평균회귀 거래 전략 (김치 프리미엄 기반)
"""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy, Signal, Position

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    평균회귀 전략
    
    Strategy:
    - 김치 프리미엄이 이동평균보다 낮을 때 진입
    - 목표 수익 달성 시 청산
    - 지정가 주문으로 수수료 최소화
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="MeanReversion", config=config)
        
        # 전략 파라미터
        self.lookback_period = config.get('lookback_period', 48)  # 48시간 MA
        self.entry_threshold = config.get('entry_threshold', -0.02)  # MA - 0.02%
        self.target_profit_percent = config.get('target_profit_percent', 0.2)  # 0.2%
        self.stop_loss_percent = config.get('stop_loss_percent', -0.1)  # -0.1%
        self.use_maker_only = config.get('use_maker_only', True)
        
        # 데이터 저장
        self.kimchi_history = deque(maxlen=self.lookback_period * 60)  # 분봉 기준
        self.ma_value = 0.0
        self.current_kimchi = 0.0
        
        # 포지션 관리
        self.max_positions = config.get('max_positions', 2)
        self.position_size_percent = config.get('position_size_percent', 30)  # 자본의 30%
        
        # 리스크 관리
        self.daily_max_trades = config.get('daily_max_trades', 3)
        self.daily_max_loss = config.get('daily_max_loss', -100000)  # -10만원
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        시장 데이터 분석 및 신호 생성
        
        Args:
            market_data: {
                'upbit_price': float,
                'binance_price': float,
                'kimchi_premium': float,
                'upbit_spread': float,
                'binance_spread': float,
                'timestamp': datetime
            }
        """
        try:
            # 일일 카운터 리셋
            self._check_daily_reset()
            
            # 리스크 체크
            if not self._check_risk_limits():
                return None
            
            # 김프 히스토리 업데이트
            self.current_kimchi = market_data['kimchi_premium']
            self.kimchi_history.append(self.current_kimchi)
            
            # MA 계산
            if len(self.kimchi_history) >= 10:  # 최소 10개 데이터
                self.ma_value = np.mean(list(self.kimchi_history))
            else:
                return None
            
            # 진입 조건 확인
            deviation = self.current_kimchi - self.ma_value
            
            if deviation <= self.entry_threshold:
                # 추가 조건 확인
                if not self._check_entry_conditions(market_data):
                    return None
                
                # 신호 생성
                signal = Signal(
                    timestamp=datetime.now(),
                    action='buy',
                    symbol='BTC/KRW',
                    exchange='upbit',
                    amount=0,  # calculate_position_size에서 계산
                    price=market_data['upbit_price'] * 1.0001 if self.use_maker_only else None,
                    order_type='limit' if self.use_maker_only else 'market',
                    confidence=min(abs(deviation) / abs(self.entry_threshold), 1.0),
                    reason=f"Kimchi {self.current_kimchi:.3f}% < MA {self.ma_value:.3f}% - {self.entry_threshold:.3f}%",
                    metadata={
                        'kimchi_premium': self.current_kimchi,
                        'ma_value': self.ma_value,
                        'deviation': deviation,
                        'spread': market_data['upbit_spread']
                    }
                )
                
                if self.validate_signal(signal):
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return None
    
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            capital: 가용 자본
            
        Returns:
            BTC 수량
        """
        try:
            # 사용 가능 자본 계산
            available_capital = capital * (self.position_size_percent / 100)
            
            # 현재 포지션 수 확인
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Max positions reached: {len(self.positions)}/{self.max_positions}")
                return 0
            
            # BTC 수량 계산
            btc_price = signal.price or signal.metadata.get('current_price', 0)
            if btc_price <= 0:
                return 0
            
            btc_amount = available_capital / btc_price
            
            # 최소/최대 제한
            btc_amount = max(0.001, min(btc_amount, 0.1))  # 0.001 ~ 0.1 BTC
            
            return round(btc_amount, 4)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def should_close_position(self, position: Position, market_data: Dict) -> bool:
        """
        포지션 청산 여부 결정
        
        Args:
            position: 현재 포지션
            market_data: 시장 데이터
            
        Returns:
            청산 여부
        """
        try:
            # PnL 계산
            current_price = market_data['upbit_price']
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # 목표 수익 도달
            if pnl_percent >= self.target_profit_percent:
                logger.info(f"Target profit reached: {pnl_percent:.3f}%")
                return True
            
            # 손절
            if pnl_percent <= self.stop_loss_percent:
                logger.info(f"Stop loss triggered: {pnl_percent:.3f}%")
                return True
            
            # 시간 기반 청산 (24시간)
            if (datetime.now() - position.opened_at).total_seconds() > 86400:
                logger.info(f"Position timeout: held for >24 hours")
                return True
            
            # 김프 역전 (양수 전환)
            if self.current_kimchi > self.ma_value + 0.5:  # MA + 0.5% 이상
                logger.info(f"Kimchi premium reversed: {self.current_kimchi:.3f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking close conditions: {e}")
            return False
    
    def _check_entry_conditions(self, market_data: Dict) -> bool:
        """추가 진입 조건 확인"""
        # 스프레드 체크
        if market_data['upbit_spread'] > 0.002:  # 0.2% 이상
            logger.debug(f"Spread too high: {market_data['upbit_spread']*100:.3f}%")
            return False
        
        # 최근 진입 체크 (1시간 내 진입 금지)
        if self.positions:
            last_position = self.positions[-1]
            if (datetime.now() - last_position.opened_at).total_seconds() < 3600:
                logger.debug("Recent entry exists (within 1 hour)")
                return False
        
        return True
    
    def _check_risk_limits(self) -> bool:
        """리스크 한도 체크"""
        # 일일 거래 횟수
        if self.daily_trades >= self.daily_max_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trades}/{self.daily_max_trades}")
            return False
        
        # 일일 손실
        if self.daily_pnl <= self.daily_max_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.0f}")
            return False
        
        return True
    
    def _check_daily_reset(self):
        """일일 카운터 리셋"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("Daily counters reset")
    
    async def on_position_opened(self, position: Position):
        """포지션 오픈 시 처리"""
        await super().on_position_opened(position)
        self.daily_trades += 1
        logger.info(f"Daily trades: {self.daily_trades}/{self.daily_max_trades}")
    
    async def on_position_closed(self, position: Position):
        """포지션 청산 시 처리"""
        await super().on_position_closed(position)
        self.daily_pnl += position.pnl
        logger.info(f"Daily PnL: {self.daily_pnl:.0f}")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """전략 상태 조회"""
        return {
            'name': self.name,
            'is_running': self.is_running,
            'current_kimchi': self.current_kimchi,
            'ma_value': self.ma_value,
            'deviation': self.current_kimchi - self.ma_value,
            'active_positions': len(self.positions),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'performance': self.get_performance()
        }