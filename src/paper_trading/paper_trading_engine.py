"""
Paper Trading Engine
실거래 전 시뮬레이션 엔진
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import ccxt.pro as ccxtpro
import logging
from pathlib import Path

from src.strategies.mean_reversion_hedge import MeanReversionHedgeStrategy
from src.strategies.maker_order_strategy import MakerOrderConfig
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger

logging.basicConfig(level=logging.INFO)


@dataclass
class PaperPosition:
    """Paper Trading 포지션"""
    id: str
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    entry_price: float
    size: float  # BTC
    target_profit: float
    stop_loss: float
    status: str  # 'pending', 'filled', 'closed'
    entry_kimchi: float
    exit_price: Optional[float] = None
    exit_kimchi: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class PaperTradingEngine:
    """
    Paper Trading 실행 엔진
    
    실제 거래 시뮬레이션:
    - 실시간 가격 수신
    - 지정가 주문 시뮬레이션
    - 체결 확률 계산
    - 손익 추적
    """
    
    def __init__(
        self,
        capital: float = 40_000_000,
        use_capital_ratio: float = 0.3,  # 30% 사용
        target_profit: float = 80_000,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0015   # 0.15%
    ):
        self.capital = capital
        self.available_capital = capital * use_capital_ratio
        self.target_profit = target_profit
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # 전략 초기화
        self.strategy = MeanReversionHedgeStrategy(
            capital=capital,
            target_profit_krw=target_profit,
            lookback_period=48,
            entry_threshold=-0.02,
            hedge_threshold=-0.03
        )
        
        # 거래소 클라이언트
        self.upbit = None
        self.binance = None
        
        # 상태 관리
        self.positions = []
        self.closed_positions = []
        self.price_history = []
        self.kimchi_history = []
        
        # 통계
        self.stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'total_fees_saved': 0,
            'signals_generated': 0,
            'orders_filled': 0
        }
        
        # 로그 디렉토리
        self.log_dir = Path("paper_trading_logs")
        self.log_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """거래소 연결 초기화"""
        try:
            self.upbit = ccxtpro.upbit({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            self.binance = ccxtpro.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            logger.info("Exchange connections initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            return False
    
    async def fetch_realtime_data(self) -> Dict:
        """실시간 데이터 수집"""
        try:
            # 업비트 BTC/KRW
            upbit_ticker = await self.upbit.fetch_ticker('BTC/KRW')
            upbit_orderbook = await self.upbit.fetch_order_book('BTC/KRW', limit=5)
            
            # 바이낸스 BTC/USDT  
            binance_ticker = await self.binance.fetch_ticker('BTC/USDT')
            binance_orderbook = await self.binance.fetch_order_book('BTC/USDT', limit=5)
            
            # 환율
            rate_manager = get_exchange_rate_manager()
            usd_krw = rate_manager.current_rate
            
            # 김프 계산
            kimchi_premium = rate_manager.calculate_kimchi_premium(
                upbit_ticker['last'],
                binance_ticker['last'],
                datetime.now()
            )
            
            return {
                'timestamp': datetime.now(),
                'upbit_price': upbit_ticker['last'],
                'upbit_bid': upbit_orderbook['bids'][0][0] if upbit_orderbook['bids'] else upbit_ticker['bid'],
                'upbit_ask': upbit_orderbook['asks'][0][0] if upbit_orderbook['asks'] else upbit_ticker['ask'],
                'upbit_spread': (upbit_orderbook['asks'][0][0] - upbit_orderbook['bids'][0][0]) / upbit_orderbook['bids'][0][0] if upbit_orderbook['bids'] and upbit_orderbook['asks'] else 0,
                'binance_price': binance_ticker['last'],
                'usd_krw': usd_krw,
                'kimchi_premium': kimchi_premium
            }
            
        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
            return None
    
    def calculate_ma(self, window: int = 48) -> float:
        """이동평균 계산"""
        if len(self.kimchi_history) < window:
            return np.mean(self.kimchi_history) if self.kimchi_history else 0
        
        return np.mean(self.kimchi_history[-window:])
    
    def should_enter(self, current_kimchi: float, ma_kimchi: float) -> bool:
        """진입 조건 확인"""
        # 기본 조건: 김프가 MA보다 threshold 이하
        deviation = current_kimchi - ma_kimchi
        
        if deviation <= self.strategy.entry_threshold:
            # 추가 조건들
            # 1. 최근 1시간 내 진입 없음
            if self.positions:
                last_entry = self.positions[-1].timestamp
                if (datetime.now() - last_entry).seconds < 3600:
                    return False
            
            # 2. 최대 포지션 수 제한
            open_positions = sum(1 for p in self.positions if p.status == 'filled')
            if open_positions >= 2:
                return False
            
            return True
        
        return False
    
    def simulate_maker_order(
        self,
        side: str,
        price: float,
        spread: float,
        size: float
    ) -> Dict:
        """
        지정가 주문 시뮬레이션
        
        체결 확률 계산:
        - 스프레드가 좁을수록 체결 확률 높음
        - 주문가격이 유리할수록 체결 확률 낮음
        """
        
        # 체결 확률 계산
        base_fill_prob = 0.7  # 기본 70%
        
        # 스프레드 영향 (좁을수록 좋음)
        if spread < 0.0005:  # 0.05% 미만
            spread_factor = 1.1
        elif spread < 0.001:  # 0.1% 미만
            spread_factor = 1.0
        else:
            spread_factor = 0.8
        
        # 주문 공격성 (중간값에서 얼마나 벗어났나)
        order_aggressiveness = 0.0001  # 0.01% 유리하게
        aggressiveness_factor = 1.0 - order_aggressiveness * 10
        
        fill_probability = base_fill_prob * spread_factor * aggressiveness_factor
        fill_probability = min(max(fill_probability, 0.3), 0.9)  # 30-90% 제한
        
        # 체결 시뮬레이션
        is_filled = np.random.random() < fill_probability
        
        # 체결 시간 (체결시)
        if is_filled:
            fill_time = np.random.randint(5, 30)  # 5-30초
        else:
            fill_time = None
        
        return {
            'filled': is_filled,
            'fill_probability': fill_probability,
            'fill_time': fill_time,
            'price': price,
            'size': size,
            'fee': size * price * self.maker_fee if is_filled else 0
        }
    
    async def execute_trade(self, data: Dict) -> Optional[PaperPosition]:
        """거래 실행 시뮬레이션"""
        
        # 포지션 크기 계산
        btc_price = data['upbit_price']
        position_size = self.available_capital / btc_price
        position_size = min(position_size, 0.1)  # 최대 0.1 BTC
        
        # 지정가 주문 시뮬레이션
        order_price = data['upbit_bid'] * 1.0001  # 매수호가보다 약간 높게
        
        order_result = self.simulate_maker_order(
            side='buy',
            price=order_price,
            spread=data['upbit_spread'],
            size=position_size
        )
        
        if order_result['filled']:
            # 포지션 생성
            position = PaperPosition(
                id=f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                side='buy',
                entry_price=order_result['price'],
                size=order_result['size'],
                target_profit=self.target_profit,
                stop_loss=-self.target_profit/2,
                status='filled',
                entry_kimchi=data['kimchi_premium']
            )
            
            self.positions.append(position)
            self.stats['orders_filled'] += 1
            
            logger.info(f"Position opened: {position.id} @ {position.entry_price:,.0f}")
            return position
        else:
            logger.info(f"Order not filled (probability: {order_result['fill_probability']:.1%})")
            return None
    
    async def check_exit_conditions(self, position: PaperPosition, data: Dict) -> bool:
        """청산 조건 확인"""
        
        if position.status != 'filled':
            return False
        
        # 손익 계산
        current_price = data['upbit_price']
        position_value = position.size * position.entry_price
        current_value = position.size * current_price
        
        if position.side == 'buy':
            pnl = current_value - position_value
        else:
            pnl = position_value - current_value
        
        # 수수료 차감
        pnl -= position_value * self.maker_fee * 2  # 진입 + 청산
        
        # 청산 조건
        should_exit = False
        exit_reason = None
        
        # 1. 목표 수익 달성
        if pnl >= self.target_profit:
            should_exit = True
            exit_reason = 'target_reached'
        
        # 2. 손절
        elif pnl <= -self.target_profit/2:
            should_exit = True
            exit_reason = 'stop_loss'
        
        # 3. 시간 초과 (24시간)
        elif (datetime.now() - position.timestamp).seconds > 86400:
            should_exit = True
            exit_reason = 'timeout'
        
        if should_exit:
            # 청산 실행
            position.exit_price = current_price
            position.exit_kimchi = data['kimchi_premium']
            position.pnl = pnl
            position.exit_reason = exit_reason
            position.status = 'closed'
            
            self.closed_positions.append(position)
            self.positions.remove(position)
            
            self.stats['total_pnl'] += pnl
            if pnl > 0:
                self.stats['successful_trades'] += 1
            
            logger.info(f"Position closed: {position.id} PnL: {pnl:,.0f} ({exit_reason})")
            return True
        
        return False
    
    async def run_paper_trading(self, duration_hours: int = 24):
        """Paper Trading 실행"""
        
        logger.info(f"Starting paper trading for {duration_hours} hours")
        logger.info(f"Capital: {self.capital:,} KRW")
        logger.info(f"Target profit per trade: {self.target_profit:,} KRW")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # 실시간 데이터 수집
                data = await self.fetch_realtime_data()
                
                if data:
                    # 히스토리 저장
                    self.price_history.append(data['upbit_price'])
                    self.kimchi_history.append(data['kimchi_premium'])
                    
                    # MA 계산
                    ma_kimchi = self.calculate_ma()
                    
                    # 진입 신호 확인
                    if self.should_enter(data['kimchi_premium'], ma_kimchi):
                        self.stats['signals_generated'] += 1
                        logger.info(f"Entry signal: Kimchi {data['kimchi_premium']:.3f}% < MA {ma_kimchi:.3f}%")
                        
                        # 거래 실행
                        position = await self.execute_trade(data)
                        if position:
                            self.stats['total_trades'] += 1
                    
                    # 기존 포지션 관리
                    for position in self.positions[:]:
                        await self.check_exit_conditions(position, data)
                    
                    # 상태 출력 (10분마다)
                    if len(self.price_history) % 10 == 0:
                        await self.print_status()
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in paper trading loop: {e}")
                await asyncio.sleep(60)
        
        # 최종 리포트
        await self.generate_report()
    
    async def print_status(self):
        """현재 상태 출력"""
        current_kimchi = self.kimchi_history[-1] if self.kimchi_history else 0
        ma_kimchi = self.calculate_ma()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update")
        print(f"Kimchi Premium: {current_kimchi:.3f}% (MA: {ma_kimchi:.3f}%)")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total PnL: {self.stats['total_pnl']:,.0f} KRW")
        print(f"Signals: {self.stats['signals_generated']} | Filled: {self.stats['orders_filled']}")
    
    async def generate_report(self):
        """최종 리포트 생성"""
        
        # 통계 계산
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = self.stats['successful_trades'] / self.stats['total_trades']
            self.stats['fill_rate'] = self.stats['orders_filled'] / self.stats['signals_generated'] if self.stats['signals_generated'] > 0 else 0
        
        # 리포트 생성
        report = {
            'summary': self.stats,
            'positions': [p.to_dict() for p in self.closed_positions],
            'config': {
                'capital': self.capital,
                'target_profit': self.target_profit,
                'entry_threshold': self.strategy.entry_threshold,
                'maker_fee': self.maker_fee
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 파일 저장
        report_file = self.log_dir / f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("  PAPER TRADING REPORT")
        print("="*60)
        print(f"\nTotal Trades: {self.stats['total_trades']}")
        print(f"Win Rate: {self.stats.get('win_rate', 0)*100:.1f}%")
        print(f"Fill Rate: {self.stats.get('fill_rate', 0)*100:.1f}%")
        print(f"Total PnL: {self.stats['total_pnl']:,.0f} KRW")
        print(f"Report saved: {report_file}")
    
    async def cleanup(self):
        """리소스 정리"""
        if self.upbit:
            await self.upbit.close()
        if self.binance:
            await self.binance.close()


async def main():
    """Paper Trading 메인 실행"""
    
    engine = PaperTradingEngine(
        capital=40_000_000,
        target_profit=80_000
    )
    
    # 초기화
    if not await engine.initialize():
        print("Failed to initialize")
        return
    
    try:
        # Paper Trading 실행 (24시간)
        await engine.run_paper_trading(duration_hours=24)
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())