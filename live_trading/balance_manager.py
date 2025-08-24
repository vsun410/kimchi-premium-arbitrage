"""
잔고 관리자 (Balance Manager)
실시간 잔고 추적 및 자금 관리

목적: 정확한 잔고 추적과 자금 안전성 확보
결과: 실시간 잔고 동기화, 최소 잔고 유지
평가: 잔고 정확도, 동기화 시간
"""

import asyncio
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json

from .exchanges.base_exchange import BaseExchangeAPI, Balance
from .exchanges.upbit_live import UpbitLiveAPI
from .exchanges.binance_live import BinanceLiveAPI
from .exchange_rate_fetcher import get_current_exchange_rate

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """포지션 정보"""
    exchange: str
    currency: str
    amount: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def pnl_pct(self) -> float:
        """수익률 (%)"""
        if self.avg_price == 0:
            return 0
        return (self.current_price - self.avg_price) / self.avg_price * 100


@dataclass
class BalanceSnapshot:
    """잔고 스냅샷"""
    timestamp: datetime
    upbit_krw: float
    upbit_btc: float
    binance_usdt: float
    binance_btc: float
    total_value_krw: float
    total_value_usdt: float
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'upbit': {
                'KRW': self.upbit_krw,
                'BTC': self.upbit_btc
            },
            'binance': {
                'USDT': self.binance_usdt,
                'BTC': self.binance_btc
            },
            'total_value': {
                'KRW': self.total_value_krw,
                'USDT': self.total_value_usdt
            }
        }


class BalanceManager:
    """
    잔고 관리자
    
    핵심 기능:
    1. 실시간 잔고 추적
    2. 포지션 관리
    3. 자금 안전성 체크
    4. 잔고 이력 관리
    5. 손익 계산
    """
    
    # 최소 유지 잔고
    MIN_BALANCE_KRW = 100_000      # 10만원
    MIN_BALANCE_USDT = 100         # 100 USDT
    MIN_BALANCE_BTC = 0.001        # 0.001 BTC
    
    def __init__(
        self,
        upbit_api: Optional[UpbitLiveAPI] = None,
        binance_api: Optional[BinanceLiveAPI] = None
    ):
        """
        초기화
        
        Args:
            upbit_api: 업비트 API
            binance_api: 바이낸스 API
        """
        self.exchanges: Dict[str, BaseExchangeAPI] = {}
        
        if upbit_api:
            self.exchanges['upbit'] = upbit_api
        if binance_api:
            self.exchanges['binance'] = binance_api
        
        # 현재 잔고
        self.current_balances: Dict[str, Dict[str, Balance]] = {}
        
        # 포지션 정보
        self.positions: Dict[str, PositionInfo] = {}
        
        # 잔고 이력
        self.balance_history: List[BalanceSnapshot] = []
        self.max_history = 1000
        
        # 동기화 설정
        self.sync_interval = 5  # 5초마다 동기화
        self.last_sync_time: Dict[str, datetime] = {}
        
        # 통계
        self.stats = {
            'total_deposits_krw': 0,
            'total_deposits_usdt': 0,
            'total_withdrawals_krw': 0,
            'total_withdrawals_usdt': 0,
            'total_realized_pnl_krw': 0,
            'total_realized_pnl_usdt': 0,
            'sync_count': 0,
            'sync_errors': 0
        }
        
        logger.info(f"BalanceManager initialized with exchanges: {list(self.exchanges.keys())}")
    
    async def sync_all_balances(self) -> Dict[str, Dict[str, Balance]]:
        """
        모든 거래소 잔고 동기화
        
        Returns:
            거래소별 잔고
        """
        all_balances = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                balances = await exchange.get_balance()
                all_balances[exchange_name] = balances
                self.current_balances[exchange_name] = balances
                self.last_sync_time[exchange_name] = datetime.now()
                
                logger.info(f"Synced {exchange_name} balances: {len(balances)} currencies")
                
            except Exception as e:
                logger.error(f"Failed to sync {exchange_name} balances: {e}")
                self.stats['sync_errors'] += 1
        
        self.stats['sync_count'] += 1
        
        # 스냅샷 저장
        await self._save_snapshot()
        
        return all_balances
    
    async def get_balance(
        self,
        exchange: str,
        currency: str,
        force_sync: bool = False
    ) -> Optional[Balance]:
        """
        특정 잔고 조회
        
        Args:
            exchange: 거래소 이름
            currency: 통화
            force_sync: 강제 동기화 여부
            
        Returns:
            잔고 정보
        """
        # 동기화 필요 여부 확인
        need_sync = (
            force_sync or
            exchange not in self.current_balances or
            exchange not in self.last_sync_time or
            (datetime.now() - self.last_sync_time[exchange]).total_seconds() > self.sync_interval
        )
        
        if need_sync:
            await self.sync_exchange_balance(exchange)
        
        # 잔고 반환
        if exchange in self.current_balances:
            return self.current_balances[exchange].get(currency)
        
        return None
    
    async def sync_exchange_balance(self, exchange_name: str) -> Dict[str, Balance]:
        """
        특정 거래소 잔고 동기화
        
        Args:
            exchange_name: 거래소 이름
            
        Returns:
            잔고 딕셔너리
        """
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange not found: {exchange_name}")
        
        try:
            exchange = self.exchanges[exchange_name]
            balances = await exchange.get_balance()
            
            self.current_balances[exchange_name] = balances
            self.last_sync_time[exchange_name] = datetime.now()
            
            logger.debug(f"Synced {exchange_name} balances")
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to sync {exchange_name}: {e}")
            self.stats['sync_errors'] += 1
            raise
    
    def check_sufficient_balance(
        self,
        exchange: str,
        currency: str,
        required_amount: float
    ) -> Tuple[bool, float]:
        """
        잔고 충분 여부 확인
        
        Args:
            exchange: 거래소 이름
            currency: 통화
            required_amount: 필요 금액
            
        Returns:
            (충분 여부, 사용 가능 잔고)
        """
        if exchange not in self.current_balances:
            return False, 0
        
        balance = self.current_balances[exchange].get(currency)
        if not balance:
            return False, 0
        
        available = balance.available
        
        # 최소 잔고 확보
        min_balance = self._get_min_balance(currency)
        usable = max(0, available - min_balance)
        
        return usable >= required_amount, usable
    
    def _get_min_balance(self, currency: str) -> float:
        """
        최소 유지 잔고 조회
        
        Args:
            currency: 통화
            
        Returns:
            최소 잔고
        """
        if currency == 'KRW':
            return self.MIN_BALANCE_KRW
        elif currency in ['USDT', 'USD']:
            return self.MIN_BALANCE_USDT
        elif currency == 'BTC':
            return self.MIN_BALANCE_BTC
        else:
            return 0
    
    async def _save_snapshot(self):
        """잔고 스냅샷 저장"""
        try:
            # 현재 환율 조회
            exchange_rate = await get_current_exchange_rate()
            
            # 업비트 잔고
            upbit_krw = 0
            upbit_btc = 0
            if 'upbit' in self.current_balances:
                upbit_krw = self.current_balances['upbit'].get('KRW', Balance('KRW', 0, 0, 0)).total
                upbit_btc = self.current_balances['upbit'].get('BTC', Balance('BTC', 0, 0, 0)).total
            
            # 바이낸스 잔고
            binance_usdt = 0
            binance_btc = 0
            if 'binance' in self.current_balances:
                binance_usdt = self.current_balances['binance'].get('USDT', Balance('USDT', 0, 0, 0)).total
                binance_btc = self.current_balances['binance'].get('BTC', Balance('BTC', 0, 0, 0)).total
            
            # BTC 가격 (임시 - 실제로는 API에서 조회)
            btc_price_krw = 140_000_000  # TODO: 실제 가격 조회
            btc_price_usdt = btc_price_krw / exchange_rate
            
            # 총 가치 계산
            total_krw = (
                upbit_krw +
                upbit_btc * btc_price_krw +
                binance_usdt * exchange_rate +
                binance_btc * btc_price_krw
            )
            
            total_usdt = total_krw / exchange_rate
            
            # 스냅샷 생성
            snapshot = BalanceSnapshot(
                timestamp=datetime.now(),
                upbit_krw=upbit_krw,
                upbit_btc=upbit_btc,
                binance_usdt=binance_usdt,
                binance_btc=binance_btc,
                total_value_krw=total_krw,
                total_value_usdt=total_usdt
            )
            
            # 이력에 추가
            self.balance_history.append(snapshot)
            
            # 최대 개수 유지
            if len(self.balance_history) > self.max_history:
                self.balance_history = self.balance_history[-self.max_history:]
            
            logger.debug(f"Balance snapshot saved: Total value = {total_krw:,.0f} KRW")
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def update_position(
        self,
        exchange: str,
        currency: str,
        amount: float,
        price: float,
        is_buy: bool
    ):
        """
        포지션 업데이트
        
        Args:
            exchange: 거래소
            currency: 통화
            amount: 수량
            price: 가격
            is_buy: 매수 여부
        """
        position_key = f"{exchange}_{currency}"
        
        if position_key not in self.positions:
            # 새 포지션
            self.positions[position_key] = PositionInfo(
                exchange=exchange,
                currency=currency,
                amount=amount if is_buy else -amount,
                avg_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        else:
            # 기존 포지션 업데이트
            position = self.positions[position_key]
            
            if is_buy:
                # 매수: 평균가 재계산
                total_value = position.amount * position.avg_price + amount * price
                position.amount += amount
                position.avg_price = total_value / position.amount if position.amount > 0 else price
            else:
                # 매도: 실현 손익 계산
                if position.amount > 0:
                    sell_amount = min(amount, position.amount)
                    realized_pnl = sell_amount * (price - position.avg_price)
                    position.realized_pnl += realized_pnl
                    position.amount -= sell_amount
                    
                    # 통계 업데이트
                    if 'KRW' in currency:
                        self.stats['total_realized_pnl_krw'] += realized_pnl
                    else:
                        self.stats['total_realized_pnl_usdt'] += realized_pnl
    
    async def update_position_prices(self):
        """포지션 현재가 업데이트"""
        for position_key, position in self.positions.items():
            try:
                exchange = self.exchanges.get(position.exchange)
                if not exchange:
                    continue
                
                # 현재가 조회
                symbol = f"{position.currency}/KRW" if position.exchange == 'upbit' else f"{position.currency}/USDT"
                ticker = await exchange.get_ticker(symbol)
                
                position.current_price = ticker.last
                position.unrealized_pnl = position.amount * (ticker.last - position.avg_price)
                
            except Exception as e:
                logger.error(f"Failed to update position price for {position_key}: {e}")
    
    def get_total_value(self) -> Dict[str, float]:
        """
        총 자산 가치 조회
        
        Returns:
            {'KRW': 원화 가치, 'USDT': 달러 가치}
        """
        if not self.balance_history:
            return {'KRW': 0, 'USDT': 0}
        
        latest = self.balance_history[-1]
        return {
            'KRW': latest.total_value_krw,
            'USDT': latest.total_value_usdt
        }
    
    def get_pnl_summary(self) -> Dict:
        """
        손익 요약 조회
        
        Returns:
            손익 정보
        """
        unrealized_pnl_krw = 0
        unrealized_pnl_usdt = 0
        
        for position in self.positions.values():
            if 'KRW' in position.currency or position.exchange == 'upbit':
                unrealized_pnl_krw += position.unrealized_pnl
            else:
                unrealized_pnl_usdt += position.unrealized_pnl
        
        return {
            'realized': {
                'KRW': self.stats['total_realized_pnl_krw'],
                'USDT': self.stats['total_realized_pnl_usdt']
            },
            'unrealized': {
                'KRW': unrealized_pnl_krw,
                'USDT': unrealized_pnl_usdt
            },
            'total': {
                'KRW': self.stats['total_realized_pnl_krw'] + unrealized_pnl_krw,
                'USDT': self.stats['total_realized_pnl_usdt'] + unrealized_pnl_usdt
            }
        }
    
    def get_balance_history(self, limit: int = 100) -> List[Dict]:
        """
        잔고 이력 조회
        
        Args:
            limit: 조회 개수
            
        Returns:
            잔고 이력
        """
        history = self.balance_history[-limit:] if len(self.balance_history) > limit else self.balance_history
        return [snapshot.to_dict() for snapshot in history]
    
    async def continuous_sync(self, interval_seconds: int = 5):
        """
        지속적인 잔고 동기화
        
        Args:
            interval_seconds: 동기화 간격
        """
        logger.info(f"Starting continuous balance sync every {interval_seconds} seconds")
        
        while True:
            try:
                await self.sync_all_balances()
                await self.update_position_prices()
                
                # 로그
                total_value = self.get_total_value()
                logger.info(f"Balance sync: {total_value['KRW']:,.0f} KRW / {total_value['USDT']:,.2f} USDT")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Balance sync error: {e}")
                await asyncio.sleep(interval_seconds)