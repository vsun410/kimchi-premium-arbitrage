"""
Maker Order Strategy - 지정가 주문 전용 전략
메이커 주문으로 수수료 절감 (0.15% → 0.02%)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class MakerOrderConfig:
    """메이커 주문 설정"""
    capital: float = 40_000_000  # 4000만원
    target_profit_krw: float = 80_000  # 8만원 (0.2%)
    
    # 수수료 (극적인 차이!)
    taker_fee: float = 0.0015  # 0.15% (시장가)
    maker_fee: float = 0.0002  # 0.02% (지정가) - 업비트 메이커
    
    # 주문 설정
    order_spread: float = 0.001  # 0.1% 스프레드
    order_timeout: int = 30  # 30초 대기
    retry_count: int = 3  # 3회 재시도
    
    # 슬리피지 관리
    max_slippage: float = 0.0005  # 0.05% 최대 슬리피지
    urgent_mode_threshold: float = 0.003  # 0.3% 이상 움직임시 긴급모드


class MakerOrderStrategy:
    """
    완전 지정가 주문 전략
    
    핵심:
    1. 모든 주문은 지정가로만
    2. 수수료 0.02%로 7.5배 절감
    3. 체결 안되면 가격 조정 후 재주문
    """
    
    def __init__(self, config: MakerOrderConfig = None):
        self.config = config or MakerOrderConfig()
        
        # 실제 수익 계산
        self.calculate_real_profits()
        
    def calculate_real_profits(self):
        """
        실제 수익 계산 (4000만원 기준)
        """
        capital = self.config.capital  # 4000만원
        target = self.config.target_profit_krw  # 8만원
        
        print("\n" + "=" * 60)
        print("  수수료별 실제 수익 비교 (4000만원 기준)")
        print("=" * 60)
        
        # BTC 가격
        btc_price = 159_000_000
        
        # 목표: 8만원 수익
        # 필요 수익률 = 8만원 / 4000만원 = 0.2%
        required_return = target / capital
        
        print(f"\n목표 수익: {target:,}원 ({required_return*100:.2f}%)")
        print(f"투자 자본: {capital:,}원")
        print(f"BTC 가격: {btc_price:,}원")
        
        # 시나리오 1: 테이커 (시장가)
        print("\n[시장가 주문 - Taker]")
        taker_fee = self.config.taker_fee
        
        # 왕복 수수료
        total_taker_fee = taker_fee * 2  # 진입 + 청산
        
        # 실제 필요 수익률 (수수료 포함)
        required_with_taker = required_return + total_taker_fee
        
        print(f"  수수료: {taker_fee*100:.3f}% (편도)")
        print(f"  왕복 수수료: {total_taker_fee*100:.3f}%")
        print(f"  필요 김프 변동: {required_with_taker*100:.3f}%")
        print(f"  실제 순수익: {target - capital*total_taker_fee:,.0f}원")
        
        # 시나리오 2: 메이커 (지정가)
        print("\n[지정가 주문 - Maker]")
        maker_fee = self.config.maker_fee
        
        # 왕복 수수료
        total_maker_fee = maker_fee * 2
        
        # 실제 필요 수익률
        required_with_maker = required_return + total_maker_fee
        
        print(f"  수수료: {maker_fee*100:.3f}% (편도)")
        print(f"  왕복 수수료: {total_maker_fee*100:.3f}%")
        print(f"  필요 김프 변동: {required_with_maker*100:.3f}%")
        print(f"  실제 순수익: {target - capital*total_maker_fee:,.0f}원")
        
        # 차이 분석
        print("\n[수수료 절감 효과]")
        fee_saved = total_taker_fee - total_maker_fee
        money_saved = capital * fee_saved
        
        print(f"  절감된 수수료: {fee_saved*100:.3f}%")
        print(f"  절감 금액: {money_saved:,.0f}원")
        print(f"  수익 증가율: {(money_saved/target)*100:.1f}%")
        
        # 월간 효과
        print("\n[월간 누적 효과]")
        monthly_trades = 6  # 월 6회 거래
        
        monthly_saved = money_saved * monthly_trades
        monthly_return_boost = fee_saved * monthly_trades * 100
        
        print(f"  월 {monthly_trades}회 거래시:")
        print(f"  월간 절감액: {monthly_saved:,.0f}원")
        print(f"  추가 수익률: +{monthly_return_boost:.2f}%")
        
        return {
            'target_profit': target,
            'taker_cost': capital * total_taker_fee,
            'maker_cost': capital * total_maker_fee,
            'saved_per_trade': money_saved,
            'monthly_saved': monthly_saved
        }
    
    async def place_maker_order(
        self,
        exchange,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount: float,
        reference_price: float
    ) -> Dict:
        """
        지정가 주문 실행
        
        전략:
        1. 현재가보다 유리한 가격에 주문
        2. 일정 시간 대기
        3. 체결 안되면 가격 조정
        4. 긴급시에만 시장가 사용
        """
        
        attempt = 0
        order_id = None
        
        while attempt < self.config.retry_count:
            try:
                # 주문 가격 계산
                if side == 'buy':
                    # 매수: 현재가보다 낮게
                    order_price = reference_price * (1 - self.config.order_spread * (1 + attempt * 0.5))
                else:
                    # 매도: 현재가보다 높게
                    order_price = reference_price * (1 + self.config.order_spread * (1 - attempt * 0.3))
                
                # 지정가 주문
                order = await exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=order_price
                )
                
                order_id = order['id']
                logger.info(f"Maker order placed: {side} {amount} @ {order_price}")
                
                # 체결 대기
                start_time = datetime.now()
                
                while (datetime.now() - start_time).seconds < self.config.order_timeout:
                    # 주문 상태 확인
                    order_status = await exchange.fetch_order(order_id, symbol)
                    
                    if order_status['status'] == 'closed':
                        logger.info(f"Maker order filled: {order_id}")
                        return {
                            'success': True,
                            'order_id': order_id,
                            'price': order_status['price'],
                            'amount': order_status['filled'],
                            'fee': order_status['fee'],
                            'is_maker': True
                        }
                    
                    await asyncio.sleep(1)
                
                # 시간 초과 - 주문 취소
                await exchange.cancel_order(order_id, symbol)
                logger.warning(f"Order timeout, cancelling: {order_id}")
                
                attempt += 1
                
            except Exception as e:
                logger.error(f"Order error: {e}")
                attempt += 1
        
        # 모든 시도 실패 - 긴급 모드
        logger.warning("All maker attempts failed, using emergency taker order")
        
        # 마지막 수단: 시장가 (이익이 충분할 때만)
        current_price = await self.get_current_price(exchange, symbol)
        expected_profit = abs(current_price - reference_price) / reference_price
        
        if expected_profit > self.config.urgent_mode_threshold:
            # 이익이 충분하면 시장가 실행
            market_order = await exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount
            )
            
            return {
                'success': True,
                'order_id': market_order['id'],
                'price': market_order['price'],
                'amount': market_order['filled'],
                'fee': market_order['fee'],
                'is_maker': False  # 테이커
            }
        
        return {
            'success': False,
            'reason': 'Could not fill order at acceptable price'
        }
    
    async def get_current_price(self, exchange, symbol: str) -> float:
        """현재가 조회"""
        ticker = await exchange.fetch_ticker(symbol)
        return ticker['last']
    
    def calculate_position_size(self, btc_price: float) -> float:
        """
        포지션 크기 계산
        
        목표: 8만원 수익
        4000만원 중 얼마를 투자할 것인가?
        """
        # 목표 수익률 0.2%
        target_return = self.config.target_profit_krw / self.config.capital
        
        # 안전 마진 고려 (자본의 30% 사용)
        safe_capital = self.config.capital * 0.3
        
        # BTC 수량
        btc_amount = safe_capital / btc_price
        
        print(f"\n[포지션 사이징]")
        print(f"  사용 자본: {safe_capital:,.0f}원 (30%)")
        print(f"  BTC 수량: {btc_amount:.4f} BTC")
        print(f"  목표 수익: {self.config.target_profit_krw:,.0f}원")
        
        return btc_amount


def analyze_maker_strategy():
    """
    메이커 전략 분석
    """
    print("\n" + "=" * 60)
    print("  MAKER ORDER STRATEGY ANALYSIS")
    print("  지정가 주문 전략 분석")
    print("=" * 60)
    
    # 전략 초기화
    strategy = MakerOrderStrategy()
    
    # 수익 계산
    profits = strategy.calculate_real_profits()
    
    # 성공 시나리오
    print("\n" + "=" * 60)
    print("  월 2% 달성 시나리오")
    print("=" * 60)
    
    scenarios = [
        {
            'name': '보수적',
            'monthly_trades': 5,
            'win_rate': 0.7,
            'avg_profit': 80_000
        },
        {
            'name': '현실적',
            'monthly_trades': 8,
            'win_rate': 0.65,
            'avg_profit': 80_000
        },
        {
            'name': '공격적',
            'monthly_trades': 12,
            'win_rate': 0.6,
            'avg_profit': 80_000
        }
    ]
    
    for scenario in scenarios:
        monthly_profit = (
            scenario['monthly_trades'] * 
            scenario['win_rate'] * 
            scenario['avg_profit']
        )
        
        # 손실 거래
        losing_trades = scenario['monthly_trades'] * (1 - scenario['win_rate'])
        monthly_loss = losing_trades * 40_000  # 손실시 5만원
        
        net_profit = monthly_profit - monthly_loss
        net_return = (net_profit / 40_000_000) * 100
        
        print(f"\n[{scenario['name']} 시나리오]")
        print(f"  월 거래: {scenario['monthly_trades']}회")
        print(f"  승률: {scenario['win_rate']*100:.0f}%")
        print(f"  월 총수익: {monthly_profit:,.0f}원")
        print(f"  월 총손실: {monthly_loss:,.0f}원")
        print(f"  월 순수익: {net_profit:,.0f}원")
        print(f"  월 수익률: {net_return:.2f}%")
    
    # 실행 가이드
    print("\n" + "=" * 60)
    print("  실전 실행 가이드")
    print("=" * 60)
    
    print("""
1. 지정가 주문 실행 순서:
   ① 현재 스프레드 확인
   ② 중간값보다 0.01% 유리한 가격에 주문
   ③ 30초 대기
   ④ 미체결시 0.005% 조정 후 재주문
   ⑤ 3회 실패시 포기 (다음 기회 대기)

2. 진입 조건 (모두 충족):
   □ 김프 < 48시간 평균 - 0.02%
   □ RSI < 35
   □ 스프레드 < 0.05%
   □ 최근 1시간 내 진입 없음

3. 청산 조건:
   □ 수익 8만원 달성 (0.2%)
   □ 손실 4만원 도달 (0.1%)
   □ 24시간 경과

4. 리스크 관리:
   □ 자본의 30%만 사용 (1200만원)
   □ 일 최대 3회 거래
   □ 연속 2회 손실시 당일 중단
    """)
    
    # 주의사항
    print("\n[중요 주의사항]")
    print("""
1. 업비트 메이커 수수료: 0.02% (반드시 확인!)
2. 바이낸스 메이커 수수료: 0.02% (VIP 레벨 확인)
3. 지정가 주문 실패시 무리하지 말 것
4. 급변동시에는 거래 자제
5. 체결률 모니터링 필수
    """)


if __name__ == "__main__":
    analyze_maker_strategy()