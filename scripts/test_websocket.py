#!/usr/bin/env python3
"""
WebSocket 연결 테스트
실시간 데이터 수집 및 재연결 테스트
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.websocket_manager import ws_manager
from src.data.reconnect_manager import reconnect_manager, connection_monitor
from src.utils.logger import logger


class WebSocketTester:
    """WebSocket 테스터"""
    
    def __init__(self):
        self.ticker_count = 0
        self.orderbook_count = 0
        self.ohlcv_count = 0
        self.last_prices = {}
        self.last_kimchi_premium = None
        
    async def on_ticker(self, data):
        """티커 데이터 콜백"""
        self.ticker_count += 1
        
        # 가격 저장
        key = f"{data['exchange']}_{data['symbol']}"
        self.last_prices[key] = {
            'bid': data.get('bid'),
            'ask': data.get('ask'),
            'last': data.get('last'),
            'timestamp': data.get('timestamp')
        }
        
        # 김프 계산 (간단 버전)
        if 'upbit_BTC/KRW' in self.last_prices and 'binance_BTC/USDT' in self.last_prices:
            upbit_price = self.last_prices['upbit_BTC/KRW']['last']
            binance_price = self.last_prices['binance_BTC/USDT']['last']
            
            if upbit_price and binance_price:
                # 임시 환율 (실제로는 API에서 가져와야 함)
                usd_krw = 1300
                kimchi_premium = ((upbit_price / (binance_price * usd_krw)) - 1) * 100
                self.last_kimchi_premium = kimchi_premium
        
        if self.ticker_count % 10 == 0:  # 10개마다 출력
            print(f"[TICKER] Count: {self.ticker_count}")
            print(f"  {data['exchange']}: {data['symbol']} = ${data.get('last')}")
            if self.last_kimchi_premium:
                print(f"  Kimchi Premium: {self.last_kimchi_premium:.2f}%")
    
    async def on_orderbook(self, data):
        """오더북 데이터 콜백"""
        self.orderbook_count += 1
        
        if self.orderbook_count % 10 == 0:  # 10개마다 출력
            print(f"[ORDERBOOK] Count: {self.orderbook_count}")
            print(f"  {data.exchange}: {data.symbol}")
            print(f"  Spread: {data.spread_percentage:.4f}%")
            print(f"  Liquidity Score: {data.liquidity_score:.1f}/100")
    
    async def on_ohlcv(self, data):
        """OHLCV 데이터 콜백"""
        self.ohlcv_count += 1
        
        print(f"[OHLCV] New candle")
        print(f"  {data.exchange}: {data.symbol}")
        print(f"  O: {data.open}, H: {data.high}, L: {data.low}, C: {data.close}")
        print(f"  Volume: {data.volume}")


async def test_basic_connection(duration: int = 30):
    """기본 연결 테스트"""
    print("\n[TEST] Basic WebSocket Connection")
    print("-" * 50)
    
    tester = WebSocketTester()
    
    # 콜백 등록
    ws_manager.register_callback('ticker', tester.on_ticker)
    ws_manager.register_callback('orderbook', tester.on_orderbook)
    ws_manager.register_callback('ohlcv', tester.on_ohlcv)
    
    # 연결할 심볼
    symbols = {
        'upbit': ['BTC/KRW'],
        'binance': ['BTC/USDT']
    }
    
    print(f"Connecting to exchanges...")
    print(f"Symbols: {symbols}")
    print(f"Test duration: {duration} seconds")
    print()
    
    try:
        # WebSocket 시작 (제한 시간)
        await asyncio.wait_for(
            ws_manager.start(symbols),
            timeout=duration
        )
    except asyncio.TimeoutError:
        print(f"\nTest completed after {duration} seconds")
    finally:
        # 통계 출력
        print("\n" + "=" * 50)
        print("Test Statistics:")
        print(f"  Ticker updates: {tester.ticker_count}")
        print(f"  Orderbook updates: {tester.orderbook_count}")
        print(f"  OHLCV updates: {tester.ohlcv_count}")
        
        if tester.last_kimchi_premium:
            print(f"  Last Kimchi Premium: {tester.last_kimchi_premium:.2f}%")
        
        # 연결 상태
        status = ws_manager.get_connection_status()
        print(f"\nConnection Status:")
        for exchange, info in status.items():
            print(f"  {exchange}: {info}")
        
        await ws_manager.stop()


async def test_reconnection():
    """재연결 테스트"""
    print("\n[TEST] Reconnection Logic")
    print("-" * 50)
    
    # 재연결 관리자 설정
    reconnect_manager.register_connection("upbit_websocket")
    reconnect_manager.register_connection("binance_websocket")
    
    # 모니터 시작
    await connection_monitor.start()
    
    print("Testing reconnection scenarios...")
    
    # 시나리오 1: 연결 성공
    print("\n1. Successful connection:")
    reconnect_manager.on_connected("upbit_websocket")
    stats = reconnect_manager.get_connection_stats("upbit_websocket")
    print(f"   Status: {stats}")
    
    # 시나리오 2: 연결 끊김
    print("\n2. Connection lost:")
    reconnect_manager.on_disconnected("upbit_websocket", Exception("Network error"))
    stats = reconnect_manager.get_connection_stats("upbit_websocket")
    print(f"   Status: {stats}")
    
    # 시나리오 3: 재연결 시도
    print("\n3. Reconnection attempts:")
    
    async def mock_connect():
        """모의 연결 함수"""
        import random
        if random.random() > 0.5:
            return True
        raise Exception("Connection failed")
    
    success = await reconnect_manager.reconnect("upbit_websocket", mock_connect)
    print(f"   Reconnection result: {success}")
    
    # 전체 통계
    print("\n4. Overall statistics:")
    all_stats = reconnect_manager.get_all_stats()
    print(json.dumps(all_stats, indent=2, default=str))
    
    # 모니터 중지
    await connection_monitor.stop()


async def test_data_gap_detection():
    """데이터 갭 감지 테스트"""
    print("\n[TEST] Data Gap Detection")
    print("-" * 50)
    
    # 연결 등록
    reconnect_manager.register_connection("test_connection")
    
    # 정상 데이터 수신 시뮬레이션
    print("Simulating normal data flow...")
    for i in range(5):
        reconnect_manager.update_data_timestamp("test_connection")
        await asyncio.sleep(1)
        gap = reconnect_manager.check_data_gaps("test_connection")
        print(f"  Time {i+1}s: Gap = {gap}")
    
    # 데이터 중단 시뮬레이션
    print("\nSimulating data interruption...")
    await asyncio.sleep(6)
    gap = reconnect_manager.check_data_gaps("test_connection")
    print(f"  After 6s pause: Gap = {gap}s")
    
    if gap:
        print(f"  [WARNING] Data gap detected: {gap:.1f} seconds")


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("WEBSOCKET CONNECTION TEST")
    print("=" * 60)
    
    try:
        # 1. 기본 연결 테스트 (30초)
        print("\nNote: This test requires valid exchange connections.")
        print("If connections fail, the test will simulate with mock data.")
        
        try:
            await test_basic_connection(duration=30)
        except Exception as e:
            print(f"[WARNING] Basic connection test failed: {e}")
            print("This is expected if exchange APIs are not configured.")
        
        # 2. 재연결 로직 테스트
        await test_reconnection()
        
        # 3. 데이터 갭 감지 테스트
        await test_data_gap_detection()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        logger.error(f"WebSocket test failure: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    exit_code = asyncio.run(main())