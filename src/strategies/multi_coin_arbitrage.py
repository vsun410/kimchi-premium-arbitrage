"""
Multi-Coin Arbitrage Strategy
멀티코인 차익거래 전략 - BTC, ETH, XRP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import ccxt
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """차익거래 기회"""
    coin: str  # BTC, ETH, XRP
    strategy: str  # 'kimchi', 'triangular', 'cross_exchange'
    entry_exchange: str
    exit_exchange: str
    profit_rate: float  # 예상 수익률 (%)
    volume: float  # 거래 가능 수량
    timestamp: datetime
    kimchi_premium: float
    fees: float
    net_profit: float
    risk_score: float  # 0-1, lower is better


class MultiCoinArbitrage:
    """
    멀티코인 차익거래 전략
    
    지원 코인:
    - BTC: 가장 유동성 높음, 낮은 스프레드
    - ETH: 중간 유동성, 가끔 높은 김프
    - XRP: 빠른 전송, 변동성 높음
    """
    
    def __init__(
        self,
        upbit_client: ccxt.Exchange,
        binance_client: ccxt.Exchange,
        min_profit_rate: float = 0.2,  # 최소 수익률 0.2%
        max_position_size: Dict[str, float] = None
    ):
        self.upbit = upbit_client
        self.binance = binance_client
        self.min_profit_rate = min_profit_rate
        
        # 코인별 최대 포지션 크기
        self.max_position_size = max_position_size or {
            'BTC': 0.01,   # 0.01 BTC
            'ETH': 0.5,    # 0.5 ETH
            'XRP': 10000   # 10,000 XRP
        }
        
        # 코인별 특성
        self.coin_characteristics = {
            'BTC': {
                'transfer_time': 30,  # minutes
                'transfer_fee': 0.0005,  # BTC
                'liquidity_score': 1.0,
                'volatility': 0.02
            },
            'ETH': {
                'transfer_time': 5,  # minutes
                'transfer_fee': 0.01,  # ETH
                'liquidity_score': 0.8,
                'volatility': 0.03
            },
            'XRP': {
                'transfer_time': 1,  # minutes
                'transfer_fee': 0.25,  # XRP
                'liquidity_score': 0.6,
                'volatility': 0.05
            }
        }
        
        # 수수료
        self.fees = {
            'upbit': 0.0005,  # 0.05%
            'binance_spot': 0.001,  # 0.1%
            'binance_futures': 0.0002,  # 0.02% (maker)
        }
        
        # 가격 캐시
        self.price_cache = {}
        self.orderbook_cache = {}
        
    async def fetch_all_prices(self) -> Dict:
        """
        모든 코인의 가격 정보 동시 수집
        """
        tasks = []
        
        # 각 코인별로 가격 수집
        for coin in ['BTC', 'ETH', 'XRP']:
            tasks.append(self.fetch_coin_prices(coin))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        all_prices = {}
        for i, coin in enumerate(['BTC', 'ETH', 'XRP']):
            if not isinstance(results[i], Exception):
                all_prices[coin] = results[i]
            else:
                logger.error(f"Error fetching {coin} prices: {results[i]}")
                all_prices[coin] = None
        
        return all_prices
    
    async def fetch_coin_prices(self, coin: str) -> Dict:
        """
        특정 코인의 가격 정보 수집
        """
        try:
            # 업비트
            upbit_symbol = f"{coin}/KRW"
            upbit_ticker = await self.upbit.fetch_ticker(upbit_symbol)
            upbit_orderbook = await self.upbit.fetch_order_book(upbit_symbol, limit=5)
            
            # 바이낸스
            binance_symbol = f"{coin}/USDT"
            binance_ticker = await self.binance.fetch_ticker(binance_symbol)
            binance_orderbook = await self.binance.fetch_order_book(binance_symbol, limit=5)
            
            return {
                'upbit_price': upbit_ticker['last'],
                'upbit_bid': upbit_orderbook['bids'][0][0] if upbit_orderbook['bids'] else upbit_ticker['bid'],
                'upbit_ask': upbit_orderbook['asks'][0][0] if upbit_orderbook['asks'] else upbit_ticker['ask'],
                'upbit_volume': upbit_ticker['baseVolume'],
                'binance_price': binance_ticker['last'],
                'binance_bid': binance_orderbook['bids'][0][0] if binance_orderbook['bids'] else binance_ticker['bid'],
                'binance_ask': binance_orderbook['asks'][0][0] if binance_orderbook['asks'] else binance_ticker['ask'],
                'binance_volume': binance_ticker['baseVolume'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching {coin} prices: {e}")
            return None
    
    def calculate_kimchi_premium(
        self,
        coin: str,
        prices: Dict,
        usd_krw_rate: float = 1390.25
    ) -> float:
        """
        김치 프리미엄 계산
        """
        if not prices:
            return 0.0
        
        upbit_price = prices['upbit_price']
        binance_price = prices['binance_price']
        
        # 바이낸스 가격을 KRW로 변환
        binance_krw = binance_price * usd_krw_rate
        
        # 김프 계산
        premium = ((upbit_price - binance_krw) / binance_krw) * 100
        
        return premium
    
    def find_opportunities(
        self,
        all_prices: Dict,
        usd_krw_rate: float = 1390.25
    ) -> List[ArbitrageOpportunity]:
        """
        모든 코인에서 차익거래 기회 찾기
        """
        opportunities = []
        
        for coin, prices in all_prices.items():
            if not prices:
                continue
            
            # 김치 프리미엄 계산
            kimchi_premium = self.calculate_kimchi_premium(coin, prices, usd_krw_rate)
            
            # 1. 김프 차익거래 (프리미엄이 높을 때)
            if kimchi_premium > self.min_profit_rate:
                # 업비트에서 매도, 바이낸스에서 매수
                opportunity = self.calculate_arbitrage_profit(
                    coin=coin,
                    strategy='kimchi_sell',
                    prices=prices,
                    kimchi_premium=kimchi_premium,
                    usd_krw_rate=usd_krw_rate
                )
                if opportunity and opportunity.profit_rate > self.min_profit_rate:
                    opportunities.append(opportunity)
            
            # 2. 역김프 차익거래 (프리미엄이 낮을 때)
            elif kimchi_premium < -self.min_profit_rate:
                # 업비트에서 매수, 바이낸스에서 매도
                opportunity = self.calculate_arbitrage_profit(
                    coin=coin,
                    strategy='kimchi_buy',
                    prices=prices,
                    kimchi_premium=kimchi_premium,
                    usd_krw_rate=usd_krw_rate
                )
                if opportunity and opportunity.profit_rate > self.min_profit_rate:
                    opportunities.append(opportunity)
            
            # 3. 크로스 익스체인지 차익거래 (순간적인 가격 차이)
            instant_profit = self.calculate_instant_arbitrage(
                coin=coin,
                prices=prices,
                usd_krw_rate=usd_krw_rate
            )
            if instant_profit and instant_profit.profit_rate > self.min_profit_rate:
                opportunities.append(instant_profit)
        
        # 수익률 기준으로 정렬
        opportunities.sort(key=lambda x: x.profit_rate, reverse=True)
        
        return opportunities
    
    def calculate_arbitrage_profit(
        self,
        coin: str,
        strategy: str,
        prices: Dict,
        kimchi_premium: float,
        usd_krw_rate: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        차익거래 수익 계산
        """
        try:
            # 거래량 결정
            volume = min(
                self.max_position_size[coin],
                prices['upbit_volume'] * 0.01,  # 시장 거래량의 1%
                prices['binance_volume'] * 0.01
            )
            
            # 수수료 계산
            total_fees = self.fees['upbit'] + self.fees['binance_spot']
            
            if strategy == 'kimchi_sell':
                # 업비트 매도, 바이낸스 매수
                upbit_sell = prices['upbit_bid'] * volume
                binance_buy = prices['binance_ask'] * volume * usd_krw_rate
                
                gross_profit = upbit_sell - binance_buy
                net_profit = gross_profit - (upbit_sell * total_fees)
                profit_rate = (net_profit / binance_buy) * 100
                
            elif strategy == 'kimchi_buy':
                # 업비트 매수, 바이낸스 매도
                upbit_buy = prices['upbit_ask'] * volume
                binance_sell = prices['binance_bid'] * volume * usd_krw_rate
                
                gross_profit = binance_sell - upbit_buy
                net_profit = gross_profit - (upbit_buy * total_fees)
                profit_rate = (net_profit / upbit_buy) * 100
                
            else:
                return None
            
            # 리스크 점수 계산
            risk_score = self.calculate_risk_score(coin, volume, kimchi_premium)
            
            if profit_rate > self.min_profit_rate:
                return ArbitrageOpportunity(
                    coin=coin,
                    strategy=strategy,
                    entry_exchange='upbit' if 'sell' in strategy else 'binance',
                    exit_exchange='binance' if 'sell' in strategy else 'upbit',
                    profit_rate=profit_rate,
                    volume=volume,
                    timestamp=datetime.now(),
                    kimchi_premium=kimchi_premium,
                    fees=total_fees * 100,
                    net_profit=net_profit,
                    risk_score=risk_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage profit: {e}")
            return None
    
    def calculate_instant_arbitrage(
        self,
        coin: str,
        prices: Dict,
        usd_krw_rate: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        즉시 실행 가능한 차익거래 기회 계산
        """
        try:
            # 업비트 매수가 vs 바이낸스 매도가
            upbit_buy = prices['upbit_ask']
            binance_sell_krw = prices['binance_bid'] * usd_krw_rate
            
            # 업비트 매도가 vs 바이낸스 매수가
            upbit_sell = prices['upbit_bid']
            binance_buy_krw = prices['binance_ask'] * usd_krw_rate
            
            # Case 1: 업비트가 저렴한 경우
            if upbit_buy < binance_sell_krw * (1 - self.min_profit_rate / 100):
                volume = self.max_position_size[coin] * 0.5  # 안전하게 절반만
                gross_profit = (binance_sell_krw - upbit_buy) * volume
                total_fees = self.fees['upbit'] + self.fees['binance_spot']
                net_profit = gross_profit - (upbit_buy * volume * total_fees)
                profit_rate = (net_profit / (upbit_buy * volume)) * 100
                
                if profit_rate > self.min_profit_rate:
                    return ArbitrageOpportunity(
                        coin=coin,
                        strategy='instant_buy_upbit',
                        entry_exchange='upbit',
                        exit_exchange='binance',
                        profit_rate=profit_rate,
                        volume=volume,
                        timestamp=datetime.now(),
                        kimchi_premium=0,
                        fees=total_fees * 100,
                        net_profit=net_profit,
                        risk_score=0.3  # 즉시 실행 가능하므로 낮은 리스크
                    )
            
            # Case 2: 바이낸스가 저렴한 경우
            if binance_buy_krw < upbit_sell * (1 - self.min_profit_rate / 100):
                volume = self.max_position_size[coin] * 0.5
                gross_profit = (upbit_sell - binance_buy_krw) * volume
                total_fees = self.fees['upbit'] + self.fees['binance_spot']
                net_profit = gross_profit - (binance_buy_krw * volume * total_fees)
                profit_rate = (net_profit / (binance_buy_krw * volume)) * 100
                
                if profit_rate > self.min_profit_rate:
                    return ArbitrageOpportunity(
                        coin=coin,
                        strategy='instant_buy_binance',
                        entry_exchange='binance',
                        exit_exchange='upbit',
                        profit_rate=profit_rate,
                        volume=volume,
                        timestamp=datetime.now(),
                        kimchi_premium=0,
                        fees=total_fees * 100,
                        net_profit=net_profit,
                        risk_score=0.3
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating instant arbitrage: {e}")
            return None
    
    def calculate_risk_score(
        self,
        coin: str,
        volume: float,
        kimchi_premium: float
    ) -> float:
        """
        리스크 점수 계산 (0-1, 낮을수록 좋음)
        """
        risk_factors = []
        
        # 1. 전송 시간 리스크
        transfer_time = self.coin_characteristics[coin]['transfer_time']
        time_risk = min(transfer_time / 60, 1.0)  # 60분 이상이면 최대 리스크
        risk_factors.append(time_risk * 0.3)
        
        # 2. 변동성 리스크
        volatility = self.coin_characteristics[coin]['volatility']
        vol_risk = min(volatility / 0.1, 1.0)  # 10% 이상이면 최대 리스크
        risk_factors.append(vol_risk * 0.3)
        
        # 3. 유동성 리스크
        liquidity_score = self.coin_characteristics[coin]['liquidity_score']
        liq_risk = 1.0 - liquidity_score
        risk_factors.append(liq_risk * 0.2)
        
        # 4. 김프 극단성 리스크
        extreme_risk = min(abs(kimchi_premium) / 5, 1.0)  # 5% 이상이면 의심
        risk_factors.append(extreme_risk * 0.2)
        
        return sum(risk_factors)
    
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Dict:
        """
        차익거래 실행
        """
        result = {
            'success': False,
            'executed': False,
            'coin': opportunity.coin,
            'strategy': opportunity.strategy,
            'actual_profit': 0,
            'execution_time': 0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # 실행 전 최종 가격 확인
            current_prices = await self.fetch_coin_prices(opportunity.coin)
            
            if not current_prices:
                result['errors'].append("Failed to fetch current prices")
                return result
            
            # 가격 변동 체크 (1% 이상 변동시 취소)
            price_change = abs(current_prices['upbit_price'] - opportunity.kimchi_premium) / opportunity.kimchi_premium
            if price_change > 0.01:
                result['errors'].append(f"Price changed too much: {price_change*100:.2f}%")
                return result
            
            # 리스크 점수 체크
            if opportunity.risk_score > 0.7:
                result['errors'].append(f"Risk too high: {opportunity.risk_score:.2f}")
                return result
            
            # 실제 거래 실행 (시뮬레이션)
            logger.info(f"Would execute: {opportunity.strategy} {opportunity.volume} {opportunity.coin}")
            logger.info(f"Expected profit: {opportunity.profit_rate:.3f}%")
            
            result['success'] = True
            result['executed'] = True
            result['actual_profit'] = opportunity.net_profit * 0.8  # 실제는 예상의 80%
            result['execution_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Arbitrage execution failed: {e}")
        
        return result


def analyze_multi_coin_opportunities():
    """
    멀티코인 차익거래 기회 분석
    """
    print("\n" + "=" * 60)
    print("  MULTI-COIN ARBITRAGE ANALYSIS")
    print("=" * 60)
    
    # 설정
    upbit = ccxt.upbit()
    binance = ccxt.binance()
    
    strategy = MultiCoinArbitrage(upbit, binance)
    
    # 비동기 실행
    async def run_analysis():
        # 가격 수집
        all_prices = await strategy.fetch_all_prices()
        
        print("\n[Current Prices]")
        for coin, prices in all_prices.items():
            if prices:
                kimchi_premium = strategy.calculate_kimchi_premium(coin, prices)
                print(f"\n{coin}:")
                print(f"  Upbit: {prices['upbit_price']:,.0f} KRW")
                print(f"  Binance: {prices['binance_price']:,.2f} USDT")
                print(f"  Kimchi Premium: {kimchi_premium:.2f}%")
                print(f"  Volume (Upbit): {prices['upbit_volume']:.2f} {coin}")
                print(f"  Volume (Binance): {prices['binance_volume']:.2f} {coin}")
        
        # 기회 찾기
        opportunities = strategy.find_opportunities(all_prices)
        
        if opportunities:
            print(f"\n[Found {len(opportunities)} opportunities]")
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"\n{i}. {opp.coin} - {opp.strategy}")
                print(f"   Profit: {opp.profit_rate:.3f}%")
                print(f"   Volume: {opp.volume:.4f} {opp.coin}")
                print(f"   Net profit: {opp.net_profit:,.0f} KRW")
                print(f"   Risk score: {opp.risk_score:.2f}")
                print(f"   Route: {opp.entry_exchange} → {opp.exit_exchange}")
        else:
            print("\n[No profitable opportunities found]")
        
        # 코인별 특성 분석
        print("\n[Coin Characteristics]")
        for coin in ['BTC', 'ETH', 'XRP']:
            chars = strategy.coin_characteristics[coin]
            print(f"\n{coin}:")
            print(f"  Transfer time: {chars['transfer_time']} minutes")
            print(f"  Liquidity score: {chars['liquidity_score']}")
            print(f"  Volatility: {chars['volatility']*100:.1f}%")
    
    # 실행
    asyncio.run(run_analysis())
    
    print("\n[Recommendations]")
    print("1. XRP has fastest transfer time - good for quick arbitrage")
    print("2. BTC has highest liquidity - lowest slippage risk")
    print("3. ETH often shows higher kimchi premium spikes")
    print("4. Monitor all three simultaneously for best opportunities")


if __name__ == "__main__":
    analyze_multi_coin_opportunities()