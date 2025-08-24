"""
Start Paper Trading
Paper Trading 시작 스크립트
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_trading.paper_trading_engine import PaperTradingEngine


async def start_paper_trading():
    """Paper Trading 시작"""
    
    print("\n" + "="*60)
    print("  PAPER TRADING - MEAN REVERSION STRATEGY")
    print("="*60)
    print(f"\nStart Time: {datetime.now()}")
    print("\nConfiguration:")
    print("  Capital: 40,000,000 KRW")
    print("  Active Capital: 12,000,000 KRW (30%)")
    print("  Target Profit: 80,000 KRW per trade")
    print("  Entry Threshold: Kimchi < MA48 - 0.02%")
    print("  Order Type: Maker only (0.02% fee)")
    print("\n" + "-"*60)
    
    # Paper Trading 엔진 생성
    engine = PaperTradingEngine(
        capital=40_000_000,
        use_capital_ratio=0.3,
        target_profit=80_000,
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0015   # 0.15%
    )
    
    # 초기화
    print("\n[1/3] Initializing exchange connections...")
    if not await engine.initialize():
        print("ERROR: Failed to initialize exchanges")
        return
    print("SUCCESS: Exchanges connected")
    
    # 테스트 데이터 수집
    print("\n[2/3] Testing data collection...")
    test_data = await engine.fetch_realtime_data()
    if test_data:
        print(f"  BTC/KRW: {test_data['upbit_price']:,.0f} KRW")
        print(f"  BTC/USDT: {test_data['binance_price']:,.2f} USDT")
        print(f"  USD/KRW: {test_data['usd_krw']:,.2f}")
        print(f"  Kimchi Premium: {test_data['kimchi_premium']:.3f}%")
        print(f"  Spread: {test_data['upbit_spread']*100:.3f}%")
        print("SUCCESS: Data collection working")
    else:
        print("ERROR: Failed to fetch data")
        return
    
    # Paper Trading 실행
    print("\n[3/3] Starting paper trading...")
    print("Duration: 1 hour (for testing)")
    print("\nMonitoring started. Updates every 10 minutes.")
    print("-"*60 + "\n")
    
    try:
        # 1시간 테스트 (실제는 24시간)
        await engine.run_paper_trading(duration_hours=1)
    except KeyboardInterrupt:
        print("\n\nPaper trading interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        # 정리
        print("\nCleaning up...")
        await engine.cleanup()
        print("Paper trading completed")


def main():
    """메인 함수"""
    
    # 사용자 확인
    print("\n" + "="*60)
    print("  PAPER TRADING LAUNCHER")
    print("="*60)
    print("\nThis will start paper trading with the following strategy:")
    print("  - Mean Reversion with 48h MA")
    print("  - Target: 80,000 KRW per trade")
    print("  - Maker orders only (0.02% fee)")
    print("  - 30% of capital (12M KRW)")
    
    response = input("\nStart paper trading? (y/n): ")
    
    if response.lower() == 'y':
        print("\nStarting paper trading...")
        asyncio.run(start_paper_trading())
    else:
        print("Paper trading cancelled")


if __name__ == "__main__":
    main()