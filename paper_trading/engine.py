"""
Paper Trading Engine (Modular Version)
모듈화된 Paper Trading 엔진
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.exchange_manager import ExchangeManager, ExchangeConfig
from strategies.base_strategy import BaseStrategy, Signal, Position
from strategies.mean_reversion.strategy import MeanReversionStrategy
from src.utils.exchange_rate_manager import get_exchange_rate_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """Paper Trading 주문"""
    id: str
    timestamp: datetime
    symbol: str
    exchange: str
    side: str
    order_type: str
    price: float
    amount: float
    status: str  # 'pending', 'filled', 'cancelled'
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    fee: float = 0.0
    

class PaperTradingEngine:
    """
    모듈화된 Paper Trading 엔진
    
    Features:
    - 전략 독립 실행
    - 실시간 데이터 시뮬레이션
    - 주문 체결 시뮬레이션
    - 성과 분석
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        capital: float = 40_000_000,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0015
    ):
        self.strategy = strategy
        self.capital = capital
        self.available_capital = capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # 거래소 관리자
        self.exchange_manager = ExchangeManager()
        
        # 주문 및 포지션
        self.orders: List[PaperOrder] = []
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # 통계
        self.stats = {
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_fees': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # 로그 디렉토리
        self.log_dir = Path("paper_trading_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 세션 ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def initialize(self) -> bool:
        """초기화"""
        try:
            # Upbit 추가
            upbit_config = ExchangeConfig(
                name='upbit',
                options={'defaultType': 'spot'}
            )
            await self.exchange_manager.add_exchange(upbit_config)
            
            # Binance 추가
            binance_config = ExchangeConfig(
                name='binance',
                options={'defaultType': 'spot'}
            )
            await self.exchange_manager.add_exchange(binance_config)
            
            # 전략 시작
            await self.strategy.start()
            
            logger.info("Paper Trading Engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def fetch_market_data(self) -> Optional[Dict]:
        """실시간 시장 데이터 수집"""
        try:
            # Upbit 데이터
            upbit_ticker = await self.exchange_manager.get_ticker('upbit', 'BTC/KRW')
            upbit_orderbook = await self.exchange_manager.get_orderbook('upbit', 'BTC/KRW', 5)
            
            # Binance 데이터
            binance_ticker = await self.exchange_manager.get_ticker('binance', 'BTC/USDT')
            binance_orderbook = await self.exchange_manager.get_orderbook('binance', 'BTC/USDT', 5)
            
            # 환율
            rate_manager = get_exchange_rate_manager()
            usd_krw = rate_manager.current_rate
            
            # 김프 계산
            kimchi = rate_manager.calculate_kimchi_premium(
                upbit_ticker['last'],
                binance_ticker['last'],
                datetime.now()
            )
            
            # 스프레드 계산
            upbit_spread = 0
            if upbit_orderbook and upbit_orderbook['bids'] and upbit_orderbook['asks']:
                upbit_spread = (upbit_orderbook['asks'][0][0] - upbit_orderbook['bids'][0][0]) / upbit_orderbook['bids'][0][0]
            
            binance_spread = 0
            if binance_orderbook and binance_orderbook['bids'] and binance_orderbook['asks']:
                binance_spread = (binance_orderbook['asks'][0][0] - binance_orderbook['bids'][0][0]) / binance_orderbook['bids'][0][0]
            
            return {
                'timestamp': datetime.now(),
                'upbit_price': upbit_ticker['last'],
                'binance_price': binance_ticker['last'],
                'usd_krw': usd_krw,
                'kimchi_premium': kimchi,
                'upbit_spread': upbit_spread,
                'binance_spread': binance_spread,
                'upbit_bid': upbit_orderbook['bids'][0][0] if upbit_orderbook['bids'] else upbit_ticker['bid'],
                'upbit_ask': upbit_orderbook['asks'][0][0] if upbit_orderbook['asks'] else upbit_ticker['ask']
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    async def process_signal(self, signal: Signal, market_data: Dict):
        """신호 처리 및 주문 생성"""
        try:
            # 포지션 크기 계산
            position_size = await self.strategy.calculate_position_size(
                signal, self.available_capital
            )
            
            if position_size <= 0:
                logger.warning("Position size is 0, skipping order")
                return
            
            signal.amount = position_size
            
            # 주문 생성
            order = PaperOrder(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                symbol=signal.symbol,
                exchange=signal.exchange,
                side=signal.action,
                order_type=signal.order_type,
                price=signal.price or market_data['upbit_price'],
                amount=signal.amount,
                status='pending'
            )
            
            self.orders.append(order)
            self.stats['orders_placed'] += 1
            
            # 주문 체결 시뮬레이션
            await self.simulate_order_fill(order, market_data)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def simulate_order_fill(self, order: PaperOrder, market_data: Dict):
        """주문 체결 시뮬레이션"""
        try:
            # 체결 확률 계산
            fill_probability = self.calculate_fill_probability(order, market_data)
            
            # 랜덤 체결
            if np.random.random() < fill_probability:
                # 체결
                order.status = 'filled'
                order.filled_at = datetime.now()
                order.fill_price = order.price
                
                # 수수료 계산
                if order.order_type == 'limit':
                    order.fee = order.amount * order.price * self.maker_fee
                else:
                    order.fee = order.amount * order.price * self.taker_fee
                
                self.stats['orders_filled'] += 1
                self.stats['total_fees'] += order.fee
                
                # 포지션 생성
                position = Position(
                    id=order.id,
                    symbol=order.symbol,
                    exchange=order.exchange,
                    side='long' if order.side == 'buy' else 'short',
                    amount=order.amount,
                    entry_price=order.fill_price,
                    current_price=order.fill_price,
                    pnl=0,
                    pnl_percent=0,
                    opened_at=order.filled_at,
                    metadata={'order_id': order.id}
                )
                
                self.positions.append(position)
                await self.strategy.on_position_opened(position)
                
                logger.info(f"Order filled: {order.id} @ {order.fill_price:,.0f}")
            else:
                # 미체결
                order.status = 'cancelled'
                self.stats['orders_cancelled'] += 1
                logger.info(f"Order not filled: {order.id} (prob: {fill_probability:.1%})")
                
        except Exception as e:
            logger.error(f"Error simulating order fill: {e}")
    
    def calculate_fill_probability(self, order: PaperOrder, market_data: Dict) -> float:
        """체결 확률 계산"""
        base_prob = 0.7
        
        # 지정가 주문
        if order.order_type == 'limit':
            # 스프레드 영향
            spread = market_data['upbit_spread']
            if spread < 0.0005:
                spread_factor = 1.1
            elif spread < 0.001:
                spread_factor = 1.0
            else:
                spread_factor = 0.8
            
            # 가격 차이
            current_price = market_data['upbit_price']
            price_diff = abs(order.price - current_price) / current_price
            if price_diff < 0.0001:
                price_factor = 1.1
            elif price_diff < 0.0005:
                price_factor = 1.0
            else:
                price_factor = 0.7
            
            return min(base_prob * spread_factor * price_factor, 0.95)
        
        # 시장가 주문
        return 0.99
    
    async def check_positions(self, market_data: Dict):
        """포지션 관리"""
        for position in self.positions[:]:
            # 현재가 업데이트
            position.current_price = market_data['upbit_price']
            
            # PnL 계산
            if position.side == 'long':
                position.pnl = (position.current_price - position.entry_price) * position.amount
            else:
                position.pnl = (position.entry_price - position.current_price) * position.amount
            
            position.pnl_percent = (position.pnl / (position.entry_price * position.amount)) * 100
            
            # 청산 확인
            should_close = await self.strategy.should_close_position(position, market_data)
            
            if should_close:
                # 청산
                self.positions.remove(position)
                self.closed_positions.append(position)
                self.stats['total_pnl'] += position.pnl
                
                # 자본 업데이트
                self.available_capital += position.pnl
                
                await self.strategy.on_position_closed(position)
                logger.info(f"Position closed: {position.id}, PnL: {position.pnl:,.0f} KRW")
    
    async def run(self, duration_hours: int = 1):
        """Paper Trading 실행"""
        logger.info(f"Starting Paper Trading for {duration_hours} hours")
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"Capital: {self.capital:,} KRW")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        iteration = 0
        while datetime.now() < end_time:
            try:
                iteration += 1
                
                # 시장 데이터 수집
                market_data = await self.fetch_market_data()
                
                if market_data:
                    # 전략 신호 생성
                    signal = await self.strategy.analyze(market_data)
                    
                    if signal:
                        self.stats['signals_generated'] += 1
                        logger.info(f"Signal generated: {signal.reason}")
                        await self.process_signal(signal, market_data)
                    
                    # 포지션 관리
                    await self.check_positions(market_data)
                    
                    # 상태 출력 (10회마다)
                    if iteration % 10 == 0:
                        await self.print_status(market_data)
                
                # 대기
                await asyncio.sleep(60)  # 1분
                
            except KeyboardInterrupt:
                logger.info("Paper Trading interrupted")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
        
        # 최종 리포트
        await self.generate_report()
    
    async def print_status(self, market_data: Dict):
        """상태 출력"""
        strategy_state = self.strategy.get_strategy_state()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update")
        print(f"Kimchi: {market_data['kimchi_premium']:.3f}% | MA: {strategy_state.get('ma_value', 0):.3f}%")
        print(f"Positions: {len(self.positions)} | PnL: {self.stats['total_pnl']:,.0f} KRW")
        print(f"Signals: {self.stats['signals_generated']} | Filled: {self.stats['orders_filled']}/{self.stats['orders_placed']}")
    
    async def generate_report(self):
        """리포트 생성"""
        report = {
            'session_id': self.session_id,
            'strategy': self.strategy.name,
            'duration': str(datetime.now() - datetime.strptime(self.session_id, "%Y%m%d_%H%M%S")),
            'statistics': self.stats,
            'performance': self.strategy.get_performance(),
            'closed_positions': [
                {
                    'id': p.id,
                    'symbol': p.symbol,
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'exit_price': p.current_price,
                    'pnl': p.pnl,
                    'pnl_percent': p.pnl_percent
                }
                for p in self.closed_positions
            ]
        }
        
        # 파일 저장
        report_file = self.log_dir / f"report_{self.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("  PAPER TRADING REPORT")
        print("="*60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Duration: {report['duration']}")
        print(f"Signals: {self.stats['signals_generated']}")
        print(f"Orders: {self.stats['orders_filled']}/{self.stats['orders_placed']}")
        print(f"Total PnL: {self.stats['total_pnl']:,.0f} KRW")
        print(f"Total Fees: {self.stats['total_fees']:,.0f} KRW")
        print(f"Win Rate: {self.strategy.get_performance()['win_rate']*100:.1f}%")
        print(f"Report: {report_file}")
    
    async def cleanup(self):
        """정리"""
        await self.strategy.stop()
        await self.exchange_manager.close_all()
        logger.info("Paper Trading Engine cleaned up")