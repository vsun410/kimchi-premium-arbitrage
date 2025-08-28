"""
Test Realtime Infrastructure
실시간 인프라 테스트
"""

import asyncio
import logging
from datetime import datetime
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_execution_engine():
    """실행 엔진 테스트"""
    from realtime import ExecutionEngine
    from realtime.execution_engine import TradingConfig, TradingMode
    
    logger.info("Testing ExecutionEngine...")
    
    config = TradingConfig(
        mode=TradingMode.PAPER,
        entry_threshold=0.025,
        exit_threshold=0.01
    )
    
    engine = ExecutionEngine(config)
    
    # 콜백 등록
    signal_received = []
    trade_received = []
    
    async def on_signal(signal):
        signal_received.append(signal)
        logger.info(f"Signal received: {signal.action}")
    
    async def on_trade(trade):
        trade_received.append(trade)
        logger.info(f"Trade executed: {trade['action']}")
    
    engine.register_signal_callback(on_signal)
    engine.register_trade_callback(on_trade)
    
    # 엔진 시작
    await engine.start()
    
    # 테스트 데이터 전송
    test_data = {
        'timestamp': datetime.now(),
        'upbit_price': 50000000,
        'binance_price': 36000,
        'kimchi_premium': 3.0  # 3% 프리미엄 (진입 신호)
    }
    
    await engine.on_market_data(test_data)
    await asyncio.sleep(1)
    
    # 청산 신호 테스트
    test_data['kimchi_premium'] = 0.5  # 0.5% (청산 신호)
    await engine.on_market_data(test_data)
    await asyncio.sleep(1)
    
    # 엔진 중지
    await engine.stop()
    
    # 결과 확인
    assert len(signal_received) > 0, "No signals generated"
    assert len(trade_received) > 0, "No trades executed"
    
    logger.info(f"ExecutionEngine test passed - {len(signal_received)} signals, {len(trade_received)} trades")


async def test_paper_trader():
    """모의 거래 시스템 테스트"""
    from realtime import PaperTrader
    
    logger.info("Testing PaperTrader...")
    
    trader = PaperTrader(
        initial_balance={'KRW': 20000000, 'USD': 15000},
        fee_rate=0.001
    )
    
    # 주문 생성 및 체결
    order = trader.place_order(
        exchange='upbit',
        symbol='BTC/KRW',
        side='buy',
        order_type='market',
        amount=0.1
    )
    
    success = trader.execute_order(order, 50000000)
    assert success, "Order execution failed"
    
    # 포지션 오픈
    position = trader.open_position(
        exchange='upbit',
        symbol='BTC/KRW',
        side='long',
        amount=0.1,
        price=50000000
    )
    assert position is not None, "Position creation failed"
    
    # 포지션 업데이트
    trader.update_positions({'upbit_BTC': 51000000})
    assert position.pnl > 0, "PnL calculation failed"
    
    # 포지션 청산
    pnl = trader.close_position('upbit', 'BTC/KRW', 51000000)
    assert pnl is not None, "Position close failed"
    
    # 통계 확인
    stats = trader.get_statistics()
    assert stats['total_trades'] == 1, "Trade count mismatch"
    
    logger.info(f"PaperTrader test passed - PnL: {pnl:,.0f}")


async def test_market_data_simulation():
    """시장 데이터 시뮬레이션 테스트"""
    logger.info("Testing market data simulation...")
    
    # 시뮬레이션 데이터 생성
    data_points = []
    
    for i in range(10):
        upbit_price = 50000000 + random.randint(-500000, 500000)
        binance_price = 36000 + random.randint(-200, 200)
        binance_krw = binance_price * 1350
        premium = ((upbit_price - binance_krw) / binance_krw) * 100
        
        data = {
            'timestamp': datetime.now(),
            'upbit_price': upbit_price,
            'binance_price': binance_price,
            'kimchi_premium': premium
        }
        
        data_points.append(data)
        await asyncio.sleep(0.1)
    
    assert len(data_points) == 10, "Data generation failed"
    
    # 프리미엄 범위 확인
    premiums = [d['kimchi_premium'] for d in data_points]
    logger.info(f"Premium range: {min(premiums):.2f}% ~ {max(premiums):.2f}%")
    
    logger.info(f"Market data simulation test passed - {len(data_points)} data points")


async def test_integration():
    """통합 테스트"""
    from realtime import ExecutionEngine, PaperTrader
    from realtime.execution_engine import TradingConfig, TradingMode
    
    logger.info("Testing integration...")
    
    # 컴포넌트 초기화
    config = TradingConfig(
        mode=TradingMode.PAPER,
        entry_threshold=0.02,
        exit_threshold=0.01
    )
    
    engine = ExecutionEngine(config)
    trader = PaperTrader(
        initial_balance={'KRW': 20000000, 'USD': 15000}
    )
    
    trade_count = 0
    
    async def on_trade(trade_info):
        nonlocal trade_count
        trade_count += 1
        
        if trade_info['action'] == 'open_hedge':
            # 모의 거래 실행
            trader.open_position(
                exchange='upbit',
                symbol='BTC/KRW',
                side='long',
                amount=0.01,
                price=50000000
            )
            trader.open_position(
                exchange='binance',
                symbol='BTC/USDT',
                side='short',
                amount=0.01,
                price=36000
            )
    
    engine.register_trade_callback(on_trade)
    await engine.start()
    
    # 시뮬레이션 실행
    for i in range(5):
        premium = 2.5 if i == 0 else random.uniform(0.5, 3.0)
        
        data = {
            'timestamp': datetime.now(),
            'upbit_price': 50000000,
            'binance_price': 36000,
            'kimchi_premium': premium
        }
        
        await engine.on_market_data(data)
        await asyncio.sleep(0.5)
    
    await engine.stop()
    
    # 결과 확인
    stats = trader.get_statistics()
    logger.info(f"Integration test results: {trade_count} trades, {stats['positions']} positions")
    
    logger.info(f"Integration test passed")


async def main():
    """메인 테스트 함수"""
    print("""
=====================================
Realtime Infrastructure Test Suite
=====================================
    """)
    
    tests = [
        ("Execution Engine", test_execution_engine),
        ("Paper Trader", test_paper_trader),
        ("Market Data Simulation", test_market_data_simulation),
        ("Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[TEST] Running: {name}")
            await test_func()
            passed += 1
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            failed += 1
    
    print(f"""
=====================================
           Test Results              
=====================================
  Passed: {passed}
  Failed: {failed}
  
  Status: {'SUCCESS' if failed == 0 else 'FAILURE'}
=====================================
    """)


if __name__ == "__main__":
    asyncio.run(main())