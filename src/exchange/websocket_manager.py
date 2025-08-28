"""
WebSocket Manager with Auto-Reconnection
실시간 데이터 수신 및 자동 재연결 관리
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import ccxt.pro as ccxtpro
from src.utils.logger import LoggerManager

logger = LoggerManager(__name__)


class WebSocketManager:
    """
    WebSocket 연결 관리자
    - 자동 재연결 메커니즘
    - 멀티 거래소 지원
    - 실시간 데이터 스트리밍
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: WebSocket 설정
                - exchanges: 거래소 설정
                - reconnect_delay: 재연결 지연 시간
                - max_reconnect_attempts: 최대 재연결 시도 횟수
        """
        self.config = config
        self.exchanges = {}
        self.is_running = False
        self.reconnect_delay = config.get('reconnect_delay', 5)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.callbacks = {
            'ticker': [],
            'orderbook': [],
            'trade': [],
            'error': []
        }
        self.connection_status = {}
        self._reconnect_counts = {}
        
    async def initialize(self):
        """거래소 객체 초기화"""
        try:
            # Upbit 초기화
            if 'upbit' in self.config['exchanges']:
                self.exchanges['upbit'] = ccxtpro.upbit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'watchOrderBook': {'limit': 20}
                    }
                })
                self.connection_status['upbit'] = 'disconnected'
                self._reconnect_counts['upbit'] = 0
                logger.info("Upbit WebSocket initialized")
            
            # Binance 초기화
            if 'binance' in self.config['exchanges']:
                self.exchanges['binance'] = ccxtpro.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'watchOrderBook': {'limit': 20}
                    }
                })
                self.connection_status['binance'] = 'disconnected'
                self._reconnect_counts['binance'] = 0
                logger.info("Binance WebSocket initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            await self._handle_error('initialization', e)
            raise
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        콜백 함수 등록
        
        Args:
            event_type: 이벤트 타입 (ticker, orderbook, trade, error)
            callback: 콜백 함수
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"Callback registered for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def start(self, symbols: List[str]):
        """
        WebSocket 연결 시작
        
        Args:
            symbols: 구독할 심볼 리스트
        """
        self.is_running = True
        logger.info(f"Starting WebSocket connections for symbols: {symbols}")
        
        tasks = []
        for exchange_id, exchange in self.exchanges.items():
            # 각 거래소별로 태스크 생성
            tasks.append(self._watch_ticker(exchange_id, exchange, symbols))
            tasks.append(self._watch_orderbook(exchange_id, exchange, symbols))
            tasks.append(self._watch_trades(exchange_id, exchange, symbols))
        
        # 모든 태스크 동시 실행
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """WebSocket 연결 종료"""
        self.is_running = False
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.connection_status[exchange_id] = 'disconnected'
                logger.info(f"Closed {exchange_id} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")
    
    async def _watch_ticker(self, exchange_id: str, exchange: Any, symbols: List[str]):
        """
        Ticker 데이터 구독
        
        Args:
            exchange_id: 거래소 ID
            exchange: 거래소 객체
            symbols: 심볼 리스트
        """
        while self.is_running:
            try:
                for symbol in symbols:
                    if not self.is_running:
                        break
                        
                    ticker = await exchange.watch_ticker(symbol)
                    self.connection_status[exchange_id] = 'connected'
                    self._reconnect_counts[exchange_id] = 0  # 재연결 카운트 리셋
                    
                    # 콜백 실행
                    for callback in self.callbacks['ticker']:
                        await callback(exchange_id, symbol, ticker)
                        
            except Exception as e:
                await self._handle_reconnection(exchange_id, exchange, 'ticker', symbols, e)
    
    async def _watch_orderbook(self, exchange_id: str, exchange: Any, symbols: List[str]):
        """
        오더북 데이터 구독
        
        Args:
            exchange_id: 거래소 ID
            exchange: 거래소 객체
            symbols: 심볼 리스트
        """
        while self.is_running:
            try:
                for symbol in symbols:
                    if not self.is_running:
                        break
                        
                    orderbook = await exchange.watch_order_book(symbol)
                    self.connection_status[exchange_id] = 'connected'
                    self._reconnect_counts[exchange_id] = 0
                    
                    # 콜백 실행
                    for callback in self.callbacks['orderbook']:
                        await callback(exchange_id, symbol, orderbook)
                        
            except Exception as e:
                await self._handle_reconnection(exchange_id, exchange, 'orderbook', symbols, e)
    
    async def _watch_trades(self, exchange_id: str, exchange: Any, symbols: List[str]):
        """
        실시간 거래 데이터 구독
        
        Args:
            exchange_id: 거래소 ID
            exchange: 거래소 객체
            symbols: 심볼 리스트
        """
        while self.is_running:
            try:
                for symbol in symbols:
                    if not self.is_running:
                        break
                        
                    trades = await exchange.watch_trades(symbol)
                    self.connection_status[exchange_id] = 'connected'
                    self._reconnect_counts[exchange_id] = 0
                    
                    # 콜백 실행
                    for callback in self.callbacks['trade']:
                        await callback(exchange_id, symbol, trades)
                        
            except Exception as e:
                await self._handle_reconnection(exchange_id, exchange, 'trades', symbols, e)
    
    async def _handle_reconnection(self, exchange_id: str, exchange: Any, 
                                  data_type: str, symbols: List[str], error: Exception):
        """
        재연결 처리
        
        Args:
            exchange_id: 거래소 ID
            exchange: 거래소 객체
            data_type: 데이터 타입
            symbols: 심볼 리스트
            error: 발생한 에러
        """
        self.connection_status[exchange_id] = 'reconnecting'
        self._reconnect_counts[exchange_id] += 1
        
        logger.warning(f"{exchange_id} {data_type} connection lost: {error}")
        logger.info(f"Attempting reconnection {self._reconnect_counts[exchange_id]}/{self.max_reconnect_attempts}")
        
        # 에러 콜백 실행
        await self._handle_error(f"{exchange_id}_{data_type}", error)
        
        # 최대 재연결 시도 횟수 초과 확인
        if self._reconnect_counts[exchange_id] >= self.max_reconnect_attempts:
            logger.error(f"{exchange_id} max reconnection attempts reached. Stopping.")
            self.connection_status[exchange_id] = 'failed'
            return
        
        # Exponential backoff
        delay = self.reconnect_delay * (2 ** min(self._reconnect_counts[exchange_id] - 1, 5))
        logger.info(f"Waiting {delay} seconds before reconnection...")
        await asyncio.sleep(delay)
        
        # 거래소 객체 재생성
        try:
            await exchange.close()
            
            # 새 객체 생성
            if exchange_id == 'upbit':
                self.exchanges[exchange_id] = ccxtpro.upbit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'watchOrderBook': {'limit': 20}
                    }
                })
            elif exchange_id == 'binance':
                self.exchanges[exchange_id] = ccxtpro.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'watchOrderBook': {'limit': 20}
                    }
                })
                
            logger.info(f"{exchange_id} WebSocket recreated successfully")
            
        except Exception as e:
            logger.error(f"Failed to recreate {exchange_id} connection: {e}")
            await asyncio.sleep(self.reconnect_delay)
    
    async def _handle_error(self, source: str, error: Exception):
        """
        에러 처리
        
        Args:
            source: 에러 발생 위치
            error: 에러 객체
        """
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'error': str(error),
            'type': type(error).__name__
        }
        
        # 에러 콜백 실행
        for callback in self.callbacks['error']:
            try:
                await callback(error_data)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def get_connection_status(self) -> Dict[str, str]:
        """
        연결 상태 조회
        
        Returns:
            거래소별 연결 상태
        """
        return self.connection_status.copy()
    
    def get_reconnect_stats(self) -> Dict[str, int]:
        """
        재연결 통계 조회
        
        Returns:
            거래소별 재연결 시도 횟수
        """
        return self._reconnect_counts.copy()
    
    def is_connected(self, exchange_id: Optional[str] = None) -> bool:
        """
        연결 상태 확인
        
        Args:
            exchange_id: 거래소 ID (None이면 전체 확인)
            
        Returns:
            연결 여부
        """
        if exchange_id:
            return self.connection_status.get(exchange_id) == 'connected'
        else:
            return all(status == 'connected' 
                      for status in self.connection_status.values())


async def create_websocket_manager(config: Dict[str, Any]) -> WebSocketManager:
    """
    WebSocket Manager 생성 헬퍼 함수
    
    Args:
        config: WebSocket 설정
        
    Returns:
        초기화된 WebSocketManager 인스턴스
    """
    manager = WebSocketManager(config)
    await manager.initialize()
    return manager