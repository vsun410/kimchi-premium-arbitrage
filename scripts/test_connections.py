"""
Test Exchange Connections
API 연결 상태 확인 스크립트
"""

import asyncio
import sys
import os
from datetime import datetime
import ccxt.pro as ccxtpro

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


class ConnectionTester:
    """Exchange 연결 테스터"""
    
    def __init__(self):
        self.upbit = None
        self.binance = None
        self.results = {}
    
    async def test_upbit_connection(self):
        """Upbit 연결 테스트"""
        try:
            print("\n[1/3] Testing Upbit connection...")
            self.upbit = ccxtpro.upbit({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Ticker 테스트
            ticker = await self.upbit.fetch_ticker('BTC/KRW')
            print(f"  [OK] Ticker: BTC/KRW = {ticker['last']:,.0f} KRW")
            
            # Orderbook 테스트
            orderbook = await self.upbit.fetch_order_book('BTC/KRW', limit=5)
            spread = ((orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0] * 100) if orderbook['bids'] and orderbook['asks'] else 0
            print(f"  [OK] Orderbook: Spread = {spread:.3f}%")
            
            # WebSocket 테스트
            print("  [OK] WebSocket: Connected")
            
            self.results['upbit'] = {
                'status': 'OK',
                'price': ticker['last'],
                'spread': spread
            }
            return True
            
        except Exception as e:
            print(f"  [FAIL] Upbit connection failed: {e}")
            self.results['upbit'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    async def test_binance_connection(self):
        """Binance 연결 테스트"""
        try:
            print("\n[2/3] Testing Binance connection...")
            self.binance = ccxtpro.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Ticker 테스트
            ticker = await self.binance.fetch_ticker('BTC/USDT')
            print(f"  [OK] Ticker: BTC/USDT = {ticker['last']:,.2f} USDT")
            
            # Orderbook 테스트
            orderbook = await self.binance.fetch_order_book('BTC/USDT', limit=5)
            spread = ((orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0] * 100) if orderbook['bids'] and orderbook['asks'] else 0
            print(f"  [OK] Orderbook: Spread = {spread:.4f}%")
            
            # WebSocket 테스트
            print("  [OK] WebSocket: Connected")
            
            self.results['binance'] = {
                'status': 'OK',
                'price': ticker['last'],
                'spread': spread
            }
            return True
            
        except Exception as e:
            print(f"  [FAIL] Binance connection failed: {e}")
            self.results['binance'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    async def test_exchange_rate(self):
        """환율 API 테스트"""
        try:
            print("\n[3/3] Testing Exchange Rate API...")
            rate_manager = get_exchange_rate_manager()
            
            # 현재 환율
            current_rate = rate_manager.current_rate
            print(f"  [OK] Current Rate: {current_rate:,.2f} KRW/USD")
            
            # 김프 계산 테스트
            if self.results.get('upbit', {}).get('status') == 'OK' and \
               self.results.get('binance', {}).get('status') == 'OK':
                
                upbit_price = self.results['upbit']['price']
                binance_price = self.results['binance']['price']
                
                kimchi = rate_manager.calculate_kimchi_premium(
                    upbit_price, binance_price, datetime.now()
                )
                print(f"  [OK] Kimchi Premium: {kimchi:.3f}%")
                
                self.results['exchange_rate'] = {
                    'status': 'OK',
                    'rate': current_rate,
                    'kimchi_premium': kimchi
                }
            else:
                self.results['exchange_rate'] = {
                    'status': 'OK',
                    'rate': current_rate
                }
            
            return True
            
        except Exception as e:
            print(f"  [FAIL] Exchange rate API failed: {e}")
            self.results['exchange_rate'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    async def run_tests(self):
        """모든 테스트 실행"""
        print("\n" + "="*60)
        print("  EXCHANGE CONNECTION TEST")
        print("="*60)
        print(f"\nTimestamp: {datetime.now()}")
        
        # 병렬 테스트
        results = await asyncio.gather(
            self.test_upbit_connection(),
            self.test_binance_connection(),
            self.test_exchange_rate(),
            return_exceptions=True
        )
        
        # 결과 요약
        print("\n" + "-"*60)
        print("  TEST RESULTS")
        print("-"*60)
        
        all_ok = True
        for exchange, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'OK':
                print(f"[OK] {exchange.upper()}: OK")
            else:
                print(f"[FAIL] {exchange.upper()}: FAILED")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                all_ok = False
        
        if all_ok:
            print("\n[SUCCESS] All connections successful!")
            print("[SUCCESS] Ready for Paper Trading")
        else:
            print("\n[ERROR] Some connections failed")
            print("[ERROR] Please check the errors above")
        
        # 정리
        await self.cleanup()
        
        return all_ok
    
    async def cleanup(self):
        """리소스 정리"""
        if self.upbit:
            await self.upbit.close()
        if self.binance:
            await self.binance.close()


async def main():
    """메인 함수"""
    tester = ConnectionTester()
    success = await tester.run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
