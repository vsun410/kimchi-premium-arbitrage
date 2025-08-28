"""
Execution Engine for Realtime Trading
실시간 거래 실행 엔진
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from dynamic_hedge import DynamicPositionManager, TrendAnalysisEngine
from backtesting.strategy_simulator import StrategySimulator, Signal

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """거래 모드"""
    PAPER = "paper"      # 모의 거래
    LIVE = "live"        # 실거래


@dataclass
class TradingConfig:
    """거래 설정"""
    mode: TradingMode
    initial_capital_krw: float = 20000000
    initial_capital_usd: float = 15000
    position_size_pct: float = 0.02
    entry_threshold: float = 0.02  # 2% 김프
    exit_threshold: float = 0.01   # 1% 김프
    max_daily_trades: int = 10
    enable_risk_management: bool = True


class ExecutionEngine:
    """
    실시간 거래 실행 엔진
    - 시장 데이터 수신
    - 신호 생성
    - 주문 실행
    - 리스크 관리
    """
    
    def __init__(self, config: TradingConfig):
        """
        Args:
            config: 거래 설정
        """
        self.config = config
        self.is_running = False
        
        # 전략 컴포넌트
        self.position_manager = DynamicPositionManager(
            capital_per_exchange=config.initial_capital_krw
        )
        self.trend_engine = TrendAnalysisEngine(window_size=100)
        
        # 상태 추적
        self.current_positions = {}
        self.pending_orders = {}
        self.daily_trade_count = 0
        self.last_signal_time = None
        
        # 콜백 함수들
        self.signal_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        logger.info(f"ExecutionEngine initialized in {config.mode.value} mode")
    
    async def start(self):
        """엔진 시작"""
        if self.is_running:
            logger.warning("Engine already running")
            return
        
        self.is_running = True
        logger.info("ExecutionEngine started")
        
        # 메인 루프 시작
        asyncio.create_task(self._main_loop())
    
    async def stop(self):
        """엔진 중지"""
        self.is_running = False
        
        # 모든 포지션 청산
        if self.current_positions:
            logger.info("Closing all positions...")
            await self._close_all_positions()
        
        logger.info("ExecutionEngine stopped")
    
    async def _main_loop(self):
        """메인 실행 루프"""
        while self.is_running:
            try:
                # 1초마다 상태 체크
                await asyncio.sleep(1)
                
                # 일일 거래 횟수 리셋 (자정)
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.daily_trade_count = 0
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self._handle_error(e)
    
    async def on_market_data(self, data: Dict):
        """
        시장 데이터 수신 처리
        
        Args:
            data: {
                'timestamp': datetime,
                'upbit_price': float,
                'binance_price': float,
                'upbit_volume': float,
                'binance_volume': float,
                'kimchi_premium': float
            }
        """
        try:
            # 리스크 체크
            if not await self._check_risk_limits():
                return
            
            # 신호 생성
            signals = await self._generate_signals(data)
            
            # 신호 실행
            for signal in signals:
                await self._execute_signal(signal, data)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            await self._handle_error(e)
    
    async def _generate_signals(self, data: Dict) -> List[Signal]:
        """거래 신호 생성"""
        signals = []
        
        # 김치 프리미엄 확인
        premium = data.get('kimchi_premium', 0) / 100  # 퍼센트를 비율로
        
        # 포지션 상태 확인
        has_position = len(self.current_positions) > 0
        
        if not has_position:
            # 진입 신호 확인
            if premium >= self.config.entry_threshold:
                if self.daily_trade_count < self.config.max_daily_trades:
                    signal = Signal(
                        timestamp=data['timestamp'],
                        action='open_hedge',
                        reason=f'Premium {premium:.2%} above threshold',
                        confidence=min(0.9, premium / self.config.entry_threshold),
                        data={'premium': premium}
                    )
                    signals.append(signal)
        else:
            # 청산 신호 확인
            if premium <= self.config.exit_threshold:
                signal = Signal(
                    timestamp=data['timestamp'],
                    action='close_all',
                    reason=f'Premium {premium:.2%} below exit threshold',
                    confidence=0.8,
                    data={'premium': premium}
                )
                signals.append(signal)
            
            # 최대 김프 도달 시 청산
            elif premium >= 0.05:  # 5% 이상
                signal = Signal(
                    timestamp=data['timestamp'],
                    action='close_all',
                    reason=f'Premium {premium:.2%} at extreme level',
                    confidence=0.95,
                    data={'premium': premium}
                )
                signals.append(signal)
        
        # 신호 콜백 실행
        for signal in signals:
            await self._notify_signal(signal)
        
        return signals
    
    async def _execute_signal(self, signal: Signal, market_data: Dict):
        """신호 실행"""
        logger.info(f"Executing signal: {signal.action} - {signal.reason}")
        
        if signal.action == 'open_hedge':
            await self._open_hedge_position(market_data)
            self.daily_trade_count += 1
            
        elif signal.action == 'close_all':
            await self._close_all_positions()
            
        elif signal.action == 'close_long':
            await self._close_position('upbit', 'long')
            
        elif signal.action == 'close_short':
            await self._close_position('binance', 'short')
        
        self.last_signal_time = datetime.now()
    
    async def _open_hedge_position(self, market_data: Dict):
        """헤지 포지션 오픈"""
        # 포지션 크기 계산
        upbit_price = market_data['upbit_price']
        binance_price = market_data['binance_price']
        
        position_value_krw = self.config.initial_capital_krw * self.config.position_size_pct
        btc_amount = position_value_krw / upbit_price
        
        # 포지션 기록
        self.current_positions['upbit'] = {
            'side': 'long',
            'amount': btc_amount,
            'entry_price': upbit_price,
            'entry_time': datetime.now()
        }
        
        self.current_positions['binance'] = {
            'side': 'short',
            'amount': btc_amount,
            'entry_price': binance_price,
            'entry_time': datetime.now()
        }
        
        # 거래 콜백 실행
        await self._notify_trade({
            'action': 'open_hedge',
            'upbit_position': self.current_positions['upbit'],
            'binance_position': self.current_positions['binance'],
            'premium': market_data.get('kimchi_premium')
        })
        
        logger.info(f"Opened hedge: {btc_amount:.4f} BTC")
    
    async def _close_all_positions(self):
        """모든 포지션 청산"""
        if not self.current_positions:
            return
        
        closed_positions = self.current_positions.copy()
        self.current_positions.clear()
        
        # 거래 콜백 실행
        await self._notify_trade({
            'action': 'close_all',
            'closed_positions': closed_positions,
            'close_time': datetime.now()
        })
        
        logger.info("All positions closed")
    
    async def _close_position(self, exchange: str, side: str):
        """특정 포지션 청산"""
        if exchange in self.current_positions:
            closed = self.current_positions.pop(exchange)
            
            await self._notify_trade({
                'action': f'close_{side}',
                'exchange': exchange,
                'position': closed,
                'close_time': datetime.now()
            })
            
            logger.info(f"Closed {exchange} {side} position")
    
    async def _check_risk_limits(self) -> bool:
        """리스크 한도 체크"""
        if not self.config.enable_risk_management:
            return True
        
        # 일일 거래 횟수 체크
        if self.daily_trade_count >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trade_count}")
            return False
        
        # 최소 대기 시간 체크 (1분)
        if self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time).seconds
            if time_since_last < 60:
                return False
        
        return True
    
    async def _notify_signal(self, signal: Signal):
        """신호 알림"""
        for callback in self.signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
    
    async def _notify_trade(self, trade_info: Dict):
        """거래 알림"""
        for callback in self.trade_callbacks:
            try:
                await callback(trade_info)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    async def _handle_error(self, error: Exception):
        """에러 처리"""
        for callback in self.error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def register_signal_callback(self, callback: Callable):
        """신호 콜백 등록"""
        self.signal_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        """거래 콜백 등록"""
        self.trade_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable):
        """에러 콜백 등록"""
        self.error_callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'is_running': self.is_running,
            'mode': self.config.mode.value,
            'positions': self.current_positions,
            'daily_trades': self.daily_trade_count,
            'last_signal': self.last_signal_time
        }