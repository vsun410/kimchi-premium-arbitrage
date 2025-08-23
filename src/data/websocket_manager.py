"""
WebSocket 연결 관리자
CCXT Pro를 사용한 실시간 데이터 수집
"""

import asyncio
import ccxt.pro as ccxt_pro
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from collections import deque
import json

from src.utils.logger import logger
from src.utils.metrics import metrics_collector, MetricsTimer
from src.models.schemas import Exchange, Symbol, PriceData, OrderBookData


class WebSocketManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        """초기화"""
        self.exchanges = {}
        self.callbacks = {
            'ticker': [],
            'orderbook': [],
            'trades': [],
            'ohlcv': []
        }
        self.is_running = False
        self.reconnect_attempts = {}
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # 초
        
        # 데이터 버퍼 (연결 끊김 시 데이터 보존)
        self.data_buffer = {
            'upbit': deque(maxlen=1000),
            'binance': deque(maxlen=1000)
        }
        
        # 연결 상태
        self.connection_status = {
            'upbit': False,
            'binance': False
        }
        
        # 하트비트
        self.last_heartbeat = {
            'upbit': None,
            'binance': None
        }
        
        logger.info("WebSocket manager initialized")
    
    async def initialize_exchanges(self):
        """거래소 초기화"""
        try:
            # Upbit 초기화
            self.exchanges['upbit'] = ccxt_pro.upbit({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'watchOrderBook': {
                        'depth': 20
                    }
                }
            })
            
            # Binance 초기화
            self.exchanges['binance'] = ccxt_pro.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 선물
                    'watchOrderBook': {
                        'depth': 20
                    }
                }
            })
            
            logger.info("Exchanges initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            raise
    
    def register_callback(self, data_type: str, callback: Callable):
        """콜백 함수 등록"""
        if data_type in self.callbacks:
            self.callbacks[data_type].append(callback)
            logger.debug(f"Callback registered for {data_type}")
        else:
            logger.warning(f"Unknown data type: {data_type}")
    
    async def watch_ticker(self, exchange_id: str, symbol: str):
        """실시간 티커 데이터 구독"""
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange not found: {exchange_id}")
            return
        
        while self.is_running:
            try:
                with MetricsTimer(exchange_id, "watch_ticker"):
                    ticker = await exchange.watch_ticker(symbol)
                
                # 연결 상태 업데이트
                self.connection_status[exchange_id] = True
                self.last_heartbeat[exchange_id] = datetime.now()
                self.reconnect_attempts[exchange_id] = 0
                
                # 데이터 처리
                processed_data = self._process_ticker(exchange_id, ticker)
                
                # 콜백 실행
                for callback in self.callbacks['ticker']:
                    await callback(processed_data)
                
                # 메트릭 업데이트
                metrics_collector.record_api_latency(
                    exchange_id, 
                    "ticker", 
                    ticker.get('info', {}).get('latency', 0)
                )
                
            except Exception as e:
                logger.error(f"Error watching ticker on {exchange_id}: {e}")
                await self._handle_reconnection(exchange_id, 'ticker', symbol)
    
    async def watch_orderbook(self, exchange_id: str, symbol: str, limit: int = 20):
        """실시간 오더북 데이터 구독"""
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange not found: {exchange_id}")
            return
        
        while self.is_running:
            try:
                with MetricsTimer(exchange_id, "watch_orderbook"):
                    orderbook = await exchange.watch_order_book(symbol, limit)
                
                # 연결 상태 업데이트
                self.connection_status[exchange_id] = True
                self.last_heartbeat[exchange_id] = datetime.now()
                
                # 데이터 처리
                processed_data = self._process_orderbook(exchange_id, orderbook)
                
                # 콜백 실행
                for callback in self.callbacks['orderbook']:
                    await callback(processed_data)
                
                # 유동성 메트릭 계산
                liquidity_score = self._calculate_liquidity_score(orderbook)
                metrics_collector.update_balance(
                    exchange_id, 
                    "liquidity", 
                    liquidity_score
                )
                
            except Exception as e:
                logger.error(f"Error watching orderbook on {exchange_id}: {e}")
                await self._handle_reconnection(exchange_id, 'orderbook', symbol, limit)
    
    async def watch_ohlcv(self, exchange_id: str, symbol: str, timeframe: str = None):
        """실시간 OHLCV (캔들) 데이터 구독"""
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange not found: {exchange_id}")
            return
        
        # Upbit는 1분 캔들을 지원하지 않으므로 3분 또는 5분 사용
        if exchange_id == 'upbit':
            timeframe = timeframe or '3m'  # Upbit 기본값: 3분
        else:
            timeframe = timeframe or '1m'  # 기타 거래소 기본값: 1분
        
        while self.is_running:
            try:
                with MetricsTimer(exchange_id, "watch_ohlcv"):
                    ohlcv = await exchange.watch_ohlcv(symbol, timeframe)
                
                # 연결 상태 업데이트
                self.connection_status[exchange_id] = True
                self.last_heartbeat[exchange_id] = datetime.now()
                
                # 최신 캔들 처리
                if ohlcv:
                    latest_candle = ohlcv[-1]
                    processed_data = self._process_ohlcv(exchange_id, latest_candle, symbol)
                    
                    # 콜백 실행
                    for callback in self.callbacks['ohlcv']:
                        await callback(processed_data)
                
            except Exception as e:
                logger.error(f"Error watching OHLCV on {exchange_id}: {e}")
                await self._handle_reconnection(exchange_id, 'ohlcv', symbol, timeframe)
    
    async def watch_trades(self, exchange_id: str, symbol: str):
        """실시간 체결 데이터 구독"""
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange not found: {exchange_id}")
            return
        
        while self.is_running:
            try:
                trades = await exchange.watch_trades(symbol)
                
                # 연결 상태 업데이트
                self.connection_status[exchange_id] = True
                self.last_heartbeat[exchange_id] = datetime.now()
                
                # 데이터 처리
                for trade in trades:
                    processed_data = self._process_trade(exchange_id, trade)
                    
                    # 콜백 실행
                    for callback in self.callbacks['trades']:
                        await callback(processed_data)
                
            except Exception as e:
                logger.error(f"Error watching trades on {exchange_id}: {e}")
                await self._handle_reconnection(exchange_id, 'trades', symbol)
    
    def _process_ticker(self, exchange_id: str, ticker: Dict) -> Dict[str, Any]:
        """티커 데이터 처리"""
        return {
            'exchange': exchange_id,
            'symbol': ticker['symbol'],
            'timestamp': ticker['timestamp'],
            'datetime': ticker['datetime'],
            'bid': ticker.get('bid'),
            'ask': ticker.get('ask'),
            'last': ticker.get('last'),
            'close': ticker.get('close'),
            'volume': ticker.get('baseVolume'),
            'quote_volume': ticker.get('quoteVolume'),
            'change': ticker.get('change'),
            'percentage': ticker.get('percentage')
        }
    
    def _process_orderbook(self, exchange_id: str, orderbook: Dict) -> OrderBookData:
        """오더북 데이터 처리"""
        # 상위 20개 호가만 추출
        bids = orderbook['bids'][:20] if orderbook['bids'] else []
        asks = orderbook['asks'][:20] if orderbook['asks'] else []
        
        # 총 물량 계산
        bid_volume = sum(bid[1] for bid in bids)
        ask_volume = sum(ask[1] for ask in asks)
        
        # 스프레드 계산
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        spread_percentage = (spread / bids[0][0] * 100) if bids and bids[0][0] > 0 else 0
        
        return OrderBookData(
            timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000),
            exchange=Exchange(exchange_id),
            symbol=orderbook['symbol'],  # str로 직접 사용
            bids=[(price, amount) for price, amount in bids],
            asks=[(price, amount) for price, amount in asks],
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            spread=spread,
            spread_percentage=spread_percentage,
            liquidity_score=self._calculate_liquidity_score(orderbook)
        )
    
    def _process_ohlcv(self, exchange_id: str, candle: List, symbol: str) -> PriceData:
        """OHLCV 데이터 처리"""
        timestamp, open_price, high, low, close, volume = candle
        
        return PriceData(
            timestamp=datetime.fromtimestamp(timestamp / 1000),
            exchange=Exchange(exchange_id),
            symbol=symbol,  # str로 직접 사용
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
    
    def _process_trade(self, exchange_id: str, trade: Dict) -> Dict[str, Any]:
        """체결 데이터 처리"""
        return {
            'exchange': exchange_id,
            'symbol': trade['symbol'],
            'timestamp': trade['timestamp'],
            'datetime': trade['datetime'],
            'id': trade['id'],
            'price': trade['price'],
            'amount': trade['amount'],
            'side': trade['side'],
            'cost': trade.get('cost', trade['price'] * trade['amount'])
        }
    
    def _calculate_liquidity_score(self, orderbook: Dict) -> float:
        """유동성 점수 계산 (0-100)"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0
            
            # 상위 10개 호가의 물량
            bid_liquidity = sum(bid[1] for bid in bids[:10])
            ask_liquidity = sum(ask[1] for ask in asks[:10])
            
            # 스프레드
            spread = (asks[0][0] - bids[0][0]) / bids[0][0] if bids[0][0] > 0 else float('inf')
            
            # 점수 계산 (물량이 많고 스프레드가 좁을수록 높은 점수)
            volume_score = min((bid_liquidity + ask_liquidity) / 100, 50)  # 최대 50점
            spread_score = max(0, 50 - (spread * 1000))  # 최대 50점
            
            return min(volume_score + spread_score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0
    
    async def _handle_reconnection(self, exchange_id: str, data_type: str, *args):
        """재연결 처리"""
        self.connection_status[exchange_id] = False
        
        if exchange_id not in self.reconnect_attempts:
            self.reconnect_attempts[exchange_id] = 0
        
        self.reconnect_attempts[exchange_id] += 1
        
        if self.reconnect_attempts[exchange_id] > self.max_reconnect_attempts:
            logger.critical(f"Max reconnection attempts reached for {exchange_id}")
            return
        
        delay = self.reconnect_delay * (2 ** min(self.reconnect_attempts[exchange_id] - 1, 5))
        logger.warning(f"Reconnecting {exchange_id} {data_type} in {delay} seconds... "
                      f"(attempt {self.reconnect_attempts[exchange_id]})")
        
        await asyncio.sleep(delay)
        
        # 재연결 시도
        logger.info(f"Attempting to reconnect {exchange_id} {data_type}")
    
    async def start(self, symbols: Dict[str, List[str]]):
        """
        WebSocket 연결 시작
        
        Args:
            symbols: {'upbit': ['BTC/KRW'], 'binance': ['BTC/USDT']}
        """
        if self.is_running:
            logger.warning("WebSocket manager already running")
            return
        
        self.is_running = True
        await self.initialize_exchanges()
        
        tasks = []
        
        # 각 거래소별로 구독 태스크 생성
        for exchange_id, symbol_list in symbols.items():
            for symbol in symbol_list:
                # 티커 구독
                tasks.append(
                    asyncio.create_task(
                        self.watch_ticker(exchange_id, symbol)
                    )
                )
                
                # 오더북 구독
                tasks.append(
                    asyncio.create_task(
                        self.watch_orderbook(exchange_id, symbol)
                    )
                )
                
                # OHLCV 구독
                tasks.append(
                    asyncio.create_task(
                        self.watch_ohlcv(exchange_id, symbol)
                    )
                )
                
                logger.info(f"Started watching {symbol} on {exchange_id}")
        
        # 모든 태스크 실행
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"WebSocket manager error: {e}")
            await self.stop()
    
    async def stop(self):
        """WebSocket 연결 중지"""
        logger.info("Stopping WebSocket manager...")
        self.is_running = False
        
        # 거래소 연결 종료
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed {exchange_id} connection")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")
        
        self.exchanges.clear()
        logger.info("WebSocket manager stopped")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 조회"""
        status = {}
        for exchange_id in self.connection_status:
            status[exchange_id] = {
                'connected': self.connection_status[exchange_id],
                'last_heartbeat': self.last_heartbeat.get(exchange_id),
                'reconnect_attempts': self.reconnect_attempts.get(exchange_id, 0)
            }
        return status
    
    async def health_check(self):
        """연결 상태 체크 (하트비트)"""
        while self.is_running:
            await asyncio.sleep(30)  # 30초마다 체크
            
            for exchange_id in self.connection_status:
                if self.last_heartbeat.get(exchange_id):
                    time_since_heartbeat = (
                        datetime.now() - self.last_heartbeat[exchange_id]
                    ).total_seconds()
                    
                    if time_since_heartbeat > 60:  # 60초 이상 응답 없음
                        logger.warning(f"No heartbeat from {exchange_id} for {time_since_heartbeat:.0f}s")
                        self.connection_status[exchange_id] = False


# 전역 WebSocket 관리자
ws_manager = WebSocketManager()


if __name__ == "__main__":
    # WebSocket 테스트
    async def test_callback(data):
        """테스트 콜백 함수"""
        print(f"Received data: {data}")
    
    async def main():
        # 콜백 등록
        ws_manager.register_callback('ticker', test_callback)
        ws_manager.register_callback('orderbook', test_callback)
        
        # WebSocket 시작
        symbols = {
            'upbit': ['BTC/KRW'],
            'binance': ['BTC/USDT']
        }
        
        try:
            # 10초 동안 실행
            await asyncio.wait_for(
                ws_manager.start(symbols),
                timeout=10
            )
        except asyncio.TimeoutError:
            print("Test completed")
        finally:
            await ws_manager.stop()
    
    print("WebSocket Manager Test")
    print("-" * 40)
    asyncio.run(main())