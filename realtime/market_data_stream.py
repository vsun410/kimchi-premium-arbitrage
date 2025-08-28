"""
Market Data Stream for Realtime Trading
실시간 시장 데이터 스트림
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
import json

import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataStream:
    """
    실시간 시장 데이터 스트리밍
    - WebSocket을 통한 실시간 가격 수신
    - 김치 프리미엄 계산
    - 데이터 정규화 및 전달
    """
    
    def __init__(self, 
                 upbit_api_key: Optional[str] = None,
                 upbit_secret: Optional[str] = None,
                 binance_api_key: Optional[str] = None,
                 binance_secret: Optional[str] = None,
                 exchange_rate: float = 1350.0):
        """
        Args:
            upbit_api_key: 업비트 API 키 (옵션)
            upbit_secret: 업비트 시크릿 키 (옵션)
            binance_api_key: 바이낸스 API 키 (옵션)
            binance_secret: 바이낸스 시크릿 키 (옵션)
            exchange_rate: USD/KRW 환율
        """
        self.exchange_rate = exchange_rate
        
        # 거래소 초기화
        self.upbit = ccxtpro.upbit({
            'apiKey': upbit_api_key,
            'secret': upbit_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        self.binance = ccxtpro.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # 상태
        self.is_running = False
        self.latest_data = {}
        
        # 콜백 함수들
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 버퍼 (최근 데이터 저장)
        self.price_buffer = {
            'upbit': [],
            'binance': [],
            'premium': []
        }
        self.buffer_size = 100
        
        logger.info("MarketDataStream initialized")
    
    async def start(self, symbols: List[str] = ['BTC/KRW', 'BTC/USDT']):
        """
        스트리밍 시작
        
        Args:
            symbols: 구독할 심볼 리스트
        """
        if self.is_running:
            logger.warning("Stream already running")
            return
        
        self.is_running = True
        logger.info(f"Starting market data stream for {symbols}")
        
        # 업비트, 바이낸스 동시 구독
        tasks = [
            self._stream_upbit('BTC/KRW'),
            self._stream_binance('BTC/USDT')
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """스트리밍 중지"""
        self.is_running = False
        
        # 거래소 연결 종료
        await self.upbit.close()
        await self.binance.close()
        
        logger.info("MarketDataStream stopped")
    
    async def _stream_upbit(self, symbol: str):
        """업비트 데이터 스트리밍"""
        while self.is_running:
            try:
                # 오더북 구독
                orderbook = await self.upbit.watch_order_book(symbol)
                
                # 최신 데이터 업데이트
                self.latest_data['upbit'] = {
                    'symbol': symbol,
                    'bid': orderbook['bids'][0][0] if orderbook['bids'] else 0,
                    'ask': orderbook['asks'][0][0] if orderbook['asks'] else 0,
                    'mid': (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2 
                           if orderbook['bids'] and orderbook['asks'] else 0,
                    'bid_volume': orderbook['bids'][0][1] if orderbook['bids'] else 0,
                    'ask_volume': orderbook['asks'][0][1] if orderbook['asks'] else 0,
                    'timestamp': datetime.now()
                }
                
                # 버퍼에 저장
                self._update_buffer('upbit', self.latest_data['upbit']['mid'])
                
                # 김프 계산 및 콜백 실행
                await self._process_data()
                
            except Exception as e:
                logger.error(f"Error in Upbit stream: {e}")
                await self._handle_error(e)
                await asyncio.sleep(1)
    
    async def _stream_binance(self, symbol: str):
        """바이낸스 데이터 스트리밍"""
        while self.is_running:
            try:
                # 오더북 구독
                orderbook = await self.binance.watch_order_book(symbol)
                
                # 최신 데이터 업데이트
                self.latest_data['binance'] = {
                    'symbol': symbol,
                    'bid': orderbook['bids'][0][0] if orderbook['bids'] else 0,
                    'ask': orderbook['asks'][0][0] if orderbook['asks'] else 0,
                    'mid': (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
                           if orderbook['bids'] and orderbook['asks'] else 0,
                    'bid_volume': orderbook['bids'][0][1] if orderbook['bids'] else 0,
                    'ask_volume': orderbook['asks'][0][1] if orderbook['asks'] else 0,
                    'timestamp': datetime.now()
                }
                
                # 버퍼에 저장
                self._update_buffer('binance', self.latest_data['binance']['mid'])
                
                # 김프 계산 및 콜백 실행
                await self._process_data()
                
            except Exception as e:
                logger.error(f"Error in Binance stream: {e}")
                await self._handle_error(e)
                await asyncio.sleep(1)
    
    async def _process_data(self):
        """데이터 처리 및 콜백 실행"""
        # 두 거래소 데이터가 모두 있는지 확인
        if 'upbit' not in self.latest_data or 'binance' not in self.latest_data:
            return
        
        upbit_data = self.latest_data['upbit']
        binance_data = self.latest_data['binance']
        
        # 김치 프리미엄 계산
        upbit_price = upbit_data['mid']
        binance_price = binance_data['mid']
        binance_krw = binance_price * self.exchange_rate
        
        if binance_krw > 0:
            kimchi_premium = ((upbit_price - binance_krw) / binance_krw) * 100
        else:
            kimchi_premium = 0
        
        # 버퍼에 저장
        self._update_buffer('premium', kimchi_premium)
        
        # 통합 데이터 생성
        market_data = {
            'timestamp': datetime.now(),
            'upbit_price': upbit_price,
            'upbit_bid': upbit_data['bid'],
            'upbit_ask': upbit_data['ask'],
            'upbit_volume': upbit_data['bid_volume'] + upbit_data['ask_volume'],
            'binance_price': binance_price,
            'binance_bid': binance_data['bid'],
            'binance_ask': binance_data['ask'],
            'binance_volume': binance_data['bid_volume'] + binance_data['ask_volume'],
            'binance_krw': binance_krw,
            'kimchi_premium': kimchi_premium,
            'exchange_rate': self.exchange_rate
        }
        
        # 콜백 실행
        await self._notify_data(market_data)
    
    def _update_buffer(self, exchange: str, value: float):
        """버퍼 업데이트"""
        buffer = self.price_buffer[exchange]
        buffer.append(value)
        
        # 버퍼 크기 제한
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        stats = {}
        
        for exchange, buffer in self.price_buffer.items():
            if buffer:
                stats[exchange] = {
                    'mean': np.mean(buffer),
                    'std': np.std(buffer),
                    'min': np.min(buffer),
                    'max': np.max(buffer),
                    'latest': buffer[-1] if buffer else None,
                    'count': len(buffer)
                }
            else:
                stats[exchange] = None
        
        return stats
    
    def get_price_dataframe(self) -> pd.DataFrame:
        """가격 데이터프레임 반환"""
        # 버퍼 데이터를 DataFrame으로 변환
        min_len = min(
            len(self.price_buffer['upbit']),
            len(self.price_buffer['binance']),
            len(self.price_buffer['premium'])
        )
        
        if min_len == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'upbit': self.price_buffer['upbit'][-min_len:],
            'binance': self.price_buffer['binance'][-min_len:],
            'premium': self.price_buffer['premium'][-min_len:]
        })
        
        return df
    
    async def _notify_data(self, data: Dict):
        """데이터 콜백 실행"""
        for callback in self.data_callbacks:
            try:
                # 비동기 콜백 지원
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    async def _handle_error(self, error: Exception):
        """에러 처리"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def register_data_callback(self, callback: Callable):
        """데이터 콜백 등록"""
        self.data_callbacks.append(callback)
        logger.info(f"Registered data callback: {callback.__name__}")
    
    def register_error_callback(self, callback: Callable):
        """에러 콜백 등록"""
        self.error_callbacks.append(callback)
    
    def update_exchange_rate(self, rate: float):
        """환율 업데이트"""
        self.exchange_rate = rate
        logger.info(f"Exchange rate updated to {rate}")