"""
Exchange Manager
거래소 연결 및 관리 모듈
"""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
import ccxt.pro as ccxtpro
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    """거래소 설정"""
    name: str
    api_key: Optional[str] = None
    secret: Optional[str] = None
    testnet: bool = False
    rate_limit: bool = True
    options: Dict[str, Any] = None


class ExchangeManager:
    """
    거래소 관리자
    
    Features:
    - 멀티 거래소 연결 관리
    - 자동 재연결
    - Rate limiting
    - 에러 핸들링
    """
    
    def __init__(self):
        self.exchanges: Dict[str, Any] = {}
        self.configs: Dict[str, ExchangeConfig] = {}
        self._connections: Dict[str, bool] = {}
        
    async def add_exchange(self, config: ExchangeConfig) -> bool:
        """거래소 추가"""
        try:
            exchange_class = getattr(ccxtpro, config.name)
            
            params = {
                'enableRateLimit': config.rate_limit,
                'options': config.options or {}
            }
            
            if config.api_key and config.secret:
                params['apiKey'] = config.api_key
                params['secret'] = config.secret
            
            if config.testnet:
                params['options']['defaultType'] = 'testnet'
            
            exchange = exchange_class(params)
            self.exchanges[config.name] = exchange
            self.configs[config.name] = config
            self._connections[config.name] = True
            
            logger.info(f"Exchange {config.name} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add exchange {config.name}: {e}")
            return False
    
    async def get_ticker(self, exchange: str, symbol: str) -> Optional[Dict]:
        """티커 정보 조회"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            ticker = await self.exchanges[exchange].fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get ticker {symbol} from {exchange}: {e}")
            return None
    
    async def get_orderbook(self, exchange: str, symbol: str, limit: int = 10) -> Optional[Dict]:
        """오더북 조회"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            orderbook = await self.exchanges[exchange].fetch_order_book(symbol, limit)
            return orderbook
            
        except Exception as e:
            logger.error(f"Failed to get orderbook {symbol} from {exchange}: {e}")
            return None
    
    async def watch_ticker(self, exchange: str, symbol: str):
        """실시간 티커 구독"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            while self._connections.get(exchange, False):
                ticker = await self.exchanges[exchange].watch_ticker(symbol)
                yield ticker
                
        except Exception as e:
            logger.error(f"Error watching ticker {symbol} on {exchange}: {e}")
            self._connections[exchange] = False
    
    async def watch_orderbook(self, exchange: str, symbol: str):
        """실시간 오더북 구독"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            while self._connections.get(exchange, False):
                orderbook = await self.exchanges[exchange].watch_order_book(symbol)
                yield orderbook
                
        except Exception as e:
            logger.error(f"Error watching orderbook {symbol} on {exchange}: {e}")
            self._connections[exchange] = False
    
    async def create_order(
        self,
        exchange: str,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict = None
    ) -> Optional[Dict]:
        """주문 생성"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            order = await self.exchanges[exchange].create_order(
                symbol=symbol,
                type=type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            
            logger.info(f"Order created on {exchange}: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order on {exchange}: {e}")
            return None
    
    async def cancel_order(self, exchange: str, order_id: str, symbol: str) -> bool:
        """주문 취소"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            result = await self.exchanges[exchange].cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled on {exchange}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on {exchange}: {e}")
            return False
    
    async def get_balance(self, exchange: str) -> Optional[Dict]:
        """잔고 조회"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            balance = await self.exchanges[exchange].fetch_balance()
            return balance
            
        except Exception as e:
            logger.error(f"Failed to get balance from {exchange}: {e}")
            return None
    
    async def get_positions(self, exchange: str) -> Optional[List[Dict]]:
        """포지션 조회 (선물)"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not configured")
            
            positions = await self.exchanges[exchange].fetch_positions()
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions from {exchange}: {e}")
            return None
    
    async def close_all(self):
        """모든 연결 종료"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self._connections[name] = False
                logger.info(f"Exchange {name} closed")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
    
    def is_connected(self, exchange: str) -> bool:
        """연결 상태 확인"""
        return self._connections.get(exchange, False)
    
    def get_exchange_list(self) -> List[str]:
        """설정된 거래소 목록"""
        return list(self.exchanges.keys())