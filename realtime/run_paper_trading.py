"""
Run Paper Trading
모의 거래 실행 스크립트
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime import (
    ExecutionEngine,
    MarketDataStream,
    PaperTrader
)
from realtime.execution_engine import TradingConfig, TradingMode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/paper_trading.log')
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingSystem:
    """모의 거래 시스템"""
    
    def __init__(self):
        """초기화"""
        # 설정
        self.config = TradingConfig(
            mode=TradingMode.PAPER,
            initial_capital_krw=20000000,
            initial_capital_usd=15000,
            position_size_pct=0.02,
            entry_threshold=0.025,  # 2.5% 김프
            exit_threshold=0.01,     # 1% 김프
            max_daily_trades=10
        )
        
        # 컴포넌트 초기화
        self.engine = ExecutionEngine(self.config)
        self.market_stream = MarketDataStream(exchange_rate=1350.0)
        self.paper_trader = PaperTrader(
            initial_balance={'KRW': 20000000, 'USD': 15000},
            fee_rate=0.001
        )
        
        # 콜백 등록
        self._register_callbacks()
        
        # 상태
        self.is_running = False
        self.start_time = None
        
        logger.info("PaperTradingSystem initialized")
    
    def _register_callbacks(self):
        """콜백 함수 등록"""
        # 시장 데이터 -> 실행 엔진
        self.market_stream.register_data_callback(self._on_market_data)
        
        # 실행 엔진 -> 모의 거래
        self.engine.register_trade_callback(self._on_trade_signal)
        self.engine.register_signal_callback(self._on_signal)
        
        # 에러 처리
        self.market_stream.register_error_callback(self._on_error)
        self.engine.register_error_callback(self._on_error)
    
    async def _on_market_data(self, data: dict):
        """시장 데이터 수신 처리"""
        # 실행 엔진에 전달
        await self.engine.on_market_data(data)
        
        # 포지션 가격 업데이트
        prices = {
            'upbit_BTC': data['upbit_price'],
            'binance_BTC': data['binance_price']
        }
        self.paper_trader.update_positions(prices)
        
        # 상태 출력 (10초마다)
        if datetime.now().second % 10 == 0:
            self._print_status(data)
    
    async def _on_trade_signal(self, trade_info: dict):
        """거래 신호 처리"""
        action = trade_info['action']
        
        if action == 'open_hedge':
            # 업비트 롱 포지션
            upbit_pos = trade_info['upbit_position']
            upbit_order = self.paper_trader.place_order(
                exchange='upbit',
                symbol='BTC/KRW',
                side='buy',
                order_type='market',
                amount=upbit_pos['amount']
            )
            self.paper_trader.execute_order(upbit_order, upbit_pos['entry_price'])
            self.paper_trader.open_position(
                exchange='upbit',
                symbol='BTC/KRW',
                side='long',
                amount=upbit_pos['amount'],
                price=upbit_pos['entry_price']
            )
            
            # 바이낸스 숏 포지션
            binance_pos = trade_info['binance_position']
            binance_order = self.paper_trader.place_order(
                exchange='binance',
                symbol='BTC/USDT',
                side='sell',
                order_type='market',
                amount=binance_pos['amount']
            )
            self.paper_trader.execute_order(binance_order, binance_pos['entry_price'])
            self.paper_trader.open_position(
                exchange='binance',
                symbol='BTC/USDT',
                side='short',
                amount=binance_pos['amount'],
                price=binance_pos['entry_price']
            )
            
            logger.info(f"📈 Hedge opened - Premium: {trade_info.get('premium'):.2f}%")
            
        elif action == 'close_all':
            # 모든 포지션 청산
            for exchange in ['upbit', 'binance']:
                position = self.paper_trader.get_position(exchange, 'BTC/KRW' if exchange == 'upbit' else 'BTC/USDT')
                if position:
                    pnl = self.paper_trader.close_position(
                        exchange=exchange,
                        symbol=position.symbol,
                        price=position.current_price
                    )
                    
            logger.info("📉 All positions closed")
    
    async def _on_signal(self, signal):
        """신호 알림"""
        logger.info(f"🔔 Signal: {signal.action} - {signal.reason} (confidence: {signal.confidence:.2%})")
    
    async def _on_error(self, error):
        """에러 처리"""
        logger.error(f"❌ Error: {error}")
    
    def _print_status(self, market_data: dict):
        """상태 출력"""
        stats = self.paper_trader.get_statistics()
        engine_status = self.engine.get_status()
        
        print("\n" + "=" * 60)
        print(f"📊 Paper Trading Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 시장 정보
        print(f"💹 Market Data:")
        print(f"  Upbit: ₩{market_data['upbit_price']:,.0f}")
        print(f"  Binance: ${market_data['binance_price']:,.2f}")
        print(f"  Premium: {market_data['kimchi_premium']:.2f}%")
        
        # 포지션 정보
        print(f"\n📈 Positions: {stats['positions']}")
        for key, position in self.paper_trader.positions.items():
            print(f"  {key}: {position.side} {position.amount:.4f} @ {position.entry_price:,.0f}")
            print(f"    PnL: {position.pnl:,.0f} ({position.pnl_pct:.2f}%)")
        
        # 통계
        print(f"\n📊 Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Total PnL: ₩{stats['total_pnl']:,.0f}")
        print(f"  Total Fees: ₩{stats['total_fees']:,.0f}")
        print(f"  Net PnL: ₩{stats['net_pnl']:,.0f}")
        
        # 잔고
        print(f"\n💰 Balances:")
        for currency, balance in self.paper_trader.balances.items():
            print(f"  {currency}: {balance.total:,.2f}")
        
        print("=" * 60)
    
    async def start(self):
        """시스템 시작"""
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("Starting Paper Trading System...")
        
        # 컴포넌트 시작
        await self.engine.start()
        
        # 시장 데이터 스트림 시작 (별도 태스크)
        asyncio.create_task(self.market_stream.start())
        
        logger.info("Paper Trading System started successfully")
        
        # 메인 루프
        while self.is_running:
            await asyncio.sleep(1)
    
    async def stop(self):
        """시스템 중지"""
        logger.info("Stopping Paper Trading System...")
        
        self.is_running = False
        
        # 컴포넌트 중지
        await self.engine.stop()
        await self.market_stream.stop()
        
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.paper_trader.save_history(f'reports/paper_trading/session_{timestamp}.json')
        
        # 최종 통계 출력
        self._print_final_report()
        
        logger.info("Paper Trading System stopped")
    
    def _print_final_report(self):
        """최종 리포트 출력"""
        stats = self.paper_trader.get_statistics()
        duration = datetime.now() - self.start_time if self.start_time else None
        
        print("\n" + "=" * 60)
        print("📊 FINAL REPORT")
        print("=" * 60)
        
        if duration:
            print(f"Duration: {duration}")
        
        print(f"\nPerformance:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Total PnL: ₩{stats['total_pnl']:,.0f}")
        print(f"  Total Fees: ₩{stats['total_fees']:,.0f}")
        print(f"  Net PnL: ₩{stats['net_pnl']:,.0f}")
        
        print(f"\nFinal Balances:")
        for currency, balance in self.paper_trader.balances.items():
            print(f"  {currency}: {balance.total:,.2f}")
        
        print("=" * 60)


async def main():
    """메인 함수"""
    # 로그 디렉토리 생성
    Path('logs').mkdir(exist_ok=True)
    Path('reports/paper_trading').mkdir(parents=True, exist_ok=True)
    
    # 시스템 생성
    system = PaperTradingSystem()
    
    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        asyncio.create_task(system.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 시스템 시작
        await system.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await system.stop()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║          Kimchi Premium Paper Trading System              ║
║                                                            ║
║  - Mode: Paper Trading (Simulation)                       ║
║  - Capital: KRW 20,000,000 + USD 15,000                  ║
║  - Strategy: Dynamic Hedge                                ║
║  - Entry: 2.5% Premium                                   ║
║  - Exit: 1.0% Premium                                    ║
║                                                            ║
║  Press Ctrl+C to stop                                    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())