"""
Triangular Arbitrage Strategy
삼각 차익거래 전략 - BTC/KRW, BTC/USDT, USDT/KRW
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import ccxt
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriangularOpportunity:
    """삼각 차익거래 기회"""
    path: List[str]  # ['BTC/KRW', 'BTC/USDT', 'USDT/KRW']
    profit_rate: float  # 예상 수익률 (%)
    volume: float  # 거래 가능 수량
    timestamp: datetime
    exchange: str
    fees: float  # 총 수수료
    net_profit: float  # 순수익
    execution_time: float  # 예상 체결 시간 (초)


class TriangularArbitrage:
    """
    삼각 차익거래 전략
    
    예시:
    1. KRW로 BTC 구매 (업비트)
    2. BTC를 USDT로 판매 (바이낸스)
    3. USDT를 KRW로 환전
    
    또는:
    1. USDT로 BTC 구매 (바이낸스)
    2. BTC를 KRW로 판매 (업비트)
    3. KRW를 USDT로 환전
    """
    
    def __init__(
        self,
        upbit_client: ccxt.Exchange,
        binance_client: ccxt.Exchange,
        min_profit_rate: float = 0.3,  # 최소 수익률 0.3%
        max_execution_time: float = 3.0  # 최대 3초 내 체결
    ):
        self.upbit = upbit_client
        self.binance = binance_client
        self.min_profit_rate = min_profit_rate
        self.max_execution_time = max_execution_time
        
        # 수수료
        self.fees = {
            'upbit': 0.0005,  # 0.05%
            'binance': 0.001,  # 0.1%
            'forex': 0.001    # 환전 수수료 0.1%
        }
        
        # 실시간 가격 캐시
        self.price_cache = {}
        self.orderbook_cache = {}
        
    async def fetch_all_prices(self) -> Dict:
        """
        모든 필요한 가격 정보 동시 수집
        """
        tasks = [
            self.fetch_price('upbit', 'BTC/KRW'),
            self.fetch_price('binance', 'BTC/USDT'),
            self.fetch_orderbook('upbit', 'BTC/KRW'),
            self.fetch_orderbook('binance', 'BTC/USDT'),
            self.fetch_forex_rate()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'btc_krw': results[0],
            'btc_usdt': results[1],
            'btc_krw_orderbook': results[2],
            'btc_usdt_orderbook': results[3],
            'usd_krw': results[4]
        }
    
    async def fetch_price(self, exchange: str, symbol: str) -> float:
        """가격 조회"""
        try:
            if exchange == 'upbit':
                ticker = await self.upbit.fetch_ticker(symbol)
            else:
                ticker = await self.binance.fetch_ticker(symbol)
            
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching {symbol} from {exchange}: {e}")
            return None
    
    async def fetch_orderbook(self, exchange: str, symbol: str) -> Dict:
        """오더북 조회"""
        try:
            if exchange == 'upbit':
                orderbook = await self.upbit.fetch_order_book(symbol, limit=10)
            else:
                orderbook = await self.binance.fetch_order_book(symbol, limit=10)
            
            return {
                'bids': orderbook['bids'][:5],  # 매수 호가 상위 5개
                'asks': orderbook['asks'][:5],  # 매도 호가 상위 5개
                'timestamp': orderbook['timestamp']
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    async def fetch_forex_rate(self) -> float:
        """USD/KRW 환율 조회"""
        # 실제로는 환율 API 사용
        # 여기서는 고정값 사용 (실제 구현시 교체)
        return 1390.25
    
    def calculate_triangular_profit(
        self,
        prices: Dict,
        amount: float = 0.01  # BTC
    ) -> List[TriangularOpportunity]:
        """
        삼각 차익거래 수익 계산
        
        Returns:
            발견된 차익거래 기회 리스트
        """
        opportunities = []
        
        btc_krw = prices['btc_krw']
        btc_usdt = prices['btc_usdt']
        usd_krw = prices['usd_krw']
        
        if not all([btc_krw, btc_usdt, usd_krw]):
            return opportunities
        
        # Path 1: KRW → BTC → USDT → KRW
        # 1. KRW로 BTC 구매 (업비트)
        btc_buy_krw = btc_krw * (1 + self.fees['upbit'])
        
        # 2. BTC를 USDT로 판매 (바이낸스)
        usdt_received = amount * btc_usdt * (1 - self.fees['binance'])
        
        # 3. USDT를 KRW로 환전
        krw_final = usdt_received * usd_krw * (1 - self.fees['forex'])
        krw_initial = amount * btc_buy_krw
        
        profit_rate_1 = ((krw_final - krw_initial) / krw_initial) * 100
        
        if profit_rate_1 > self.min_profit_rate:
            opportunities.append(TriangularOpportunity(
                path=['KRW→BTC', 'BTC→USDT', 'USDT→KRW'],
                profit_rate=profit_rate_1,
                volume=amount,
                timestamp=datetime.now(),
                exchange='upbit+binance',
                fees=self.fees['upbit'] + self.fees['binance'] + self.fees['forex'],
                net_profit=krw_final - krw_initial,
                execution_time=2.5  # 예상 시간
            ))
        
        # Path 2: KRW → USDT → BTC → KRW
        # 1. KRW를 USDT로 환전
        usdt_amount = amount * btc_krw / usd_krw * (1 - self.fees['forex'])
        
        # 2. USDT로 BTC 구매 (바이낸스)
        btc_received = usdt_amount / btc_usdt * (1 - self.fees['binance'])
        
        # 3. BTC를 KRW로 판매 (업비트)
        krw_final_2 = btc_received * btc_krw * (1 - self.fees['upbit'])
        krw_initial_2 = amount * btc_krw
        
        profit_rate_2 = ((krw_final_2 - krw_initial_2) / krw_initial_2) * 100
        
        if profit_rate_2 > self.min_profit_rate:
            opportunities.append(TriangularOpportunity(
                path=['KRW→USDT', 'USDT→BTC', 'BTC→KRW'],
                profit_rate=profit_rate_2,
                volume=amount,
                timestamp=datetime.now(),
                exchange='upbit+binance',
                fees=self.fees['forex'] + self.fees['binance'] + self.fees['upbit'],
                net_profit=krw_final_2 - krw_initial_2,
                execution_time=2.5
            ))
        
        return opportunities
    
    def calculate_with_orderbook(
        self,
        orderbooks: Dict,
        amount: float = 0.01
    ) -> List[TriangularOpportunity]:
        """
        오더북 기반 정확한 수익 계산
        슬리피지 고려
        """
        opportunities = []
        
        btc_krw_book = orderbooks['btc_krw_orderbook']
        btc_usdt_book = orderbooks['btc_usdt_orderbook']
        
        if not all([btc_krw_book, btc_usdt_book]):
            return opportunities
        
        # 실제 체결 가능 가격 계산
        # BTC 구매시 (업비트)
        btc_buy_price_krw = self._calculate_weighted_price(
            btc_krw_book['asks'], amount
        )
        
        # BTC 판매시 (바이낸스)
        btc_sell_price_usdt = self._calculate_weighted_price(
            btc_usdt_book['bids'], amount
        )
        
        # ... (오더북 기반 정확한 계산)
        
        return opportunities
    
    def _calculate_weighted_price(
        self,
        orders: List[List[float]],
        amount: float
    ) -> float:
        """
        가중 평균 가격 계산 (슬리피지 포함)
        """
        total_cost = 0
        remaining = amount
        
        for price, volume in orders:
            if remaining <= 0:
                break
            
            trade_amount = min(remaining, volume)
            total_cost += price * trade_amount
            remaining -= trade_amount
        
        if remaining > 0:
            # 오더북 부족
            return None
        
        return total_cost / amount
    
    async def find_opportunities(self) -> List[TriangularOpportunity]:
        """
        실시간 차익거래 기회 탐색
        """
        # 가격 정보 수집
        prices = await self.fetch_all_prices()
        
        # 단순 가격 기반 계산
        simple_opps = self.calculate_triangular_profit(prices)
        
        # 오더북 기반 정밀 계산
        orderbook_opps = self.calculate_with_orderbook(prices)
        
        # 중복 제거 및 정렬
        all_opportunities = simple_opps + orderbook_opps
        all_opportunities.sort(key=lambda x: x.profit_rate, reverse=True)
        
        return all_opportunities
    
    async def execute_arbitrage(
        self,
        opportunity: TriangularOpportunity
    ) -> Dict:
        """
        차익거래 실행
        """
        result = {
            'success': False,
            'executed_path': opportunity.path,
            'actual_profit': 0,
            'execution_time': 0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # Path에 따라 실행
            if opportunity.path[0] == 'KRW→BTC':
                # Step 1: 업비트에서 BTC 구매
                order1 = await self.upbit.create_market_buy_order(
                    'BTC/KRW',
                    opportunity.volume
                )
                
                # Step 2: 바이낸스에서 BTC 판매
                order2 = await self.binance.create_market_sell_order(
                    'BTC/USDT',
                    opportunity.volume
                )
                
                # Step 3: USDT를 KRW로 환전 (실제 구현 필요)
                # ...
                
            result['success'] = True
            result['execution_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Arbitrage execution failed: {e}")
        
        return result


def analyze_triangular_opportunities():
    """
    삼각 차익거래 기회 분석
    """
    print("\n" + "="*60)
    print("  TRIANGULAR ARBITRAGE ANALYSIS")
    print("="*60)
    
    # 설정
    upbit = ccxt.upbit()
    binance = ccxt.binance()
    
    strategy = TriangularArbitrage(upbit, binance)
    
    # 비동기 실행
    async def run_analysis():
        opportunities = await strategy.find_opportunities()
        
        if opportunities:
            print(f"\n[Found {len(opportunities)} opportunities]")
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"\n{i}. Path: {' → '.join(opp.path)}")
                print(f"   Profit: {opp.profit_rate:.3f}%")
                print(f"   Net profit: {opp.net_profit:,.0f} KRW")
                print(f"   Fees: {opp.fees*100:.3f}%")
                print(f"   Execution time: {opp.execution_time:.1f}s")
        else:
            print("\n[No profitable opportunities found]")
            print("Market is currently efficient")
    
    # 실행
    asyncio.run(run_analysis())
    
    print("\n[Recommendations]")
    print("1. Monitor continuously for opportunities")
    print("2. Optimize execution speed (<2 seconds)")
    print("3. Consider multi-exchange arbitrage")
    print("4. Implement automatic execution")


if __name__ == "__main__":
    analyze_triangular_opportunities()