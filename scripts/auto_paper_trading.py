"""
Auto Paper Trading (No Input Required)
자동 Paper Trading 실행 스크립트
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading.engine import PaperTradingEngine
from strategies.mean_reversion.strategy import MeanReversionStrategy


async def run_paper_trading():
    """Paper Trading 실행"""
    
    print("\n" + "="*60)
    print("  AUTO PAPER TRADING SYSTEM")
    print("="*60)
    print(f"\nTimestamp: {datetime.now()}")
    print("\nConfiguration:")
    print("  Strategy: Mean Reversion (48h MA)")
    print("  Capital: 40,000,000 KRW")
    print("  Position Size: 30% of capital")
    print("  Target Profit: 0.2% (80,000 KRW)")
    print("  Order Type: Maker only (0.02% fee)")
    print("\n" + "-"*60)
    
    # 전략 설정
    strategy_config = {
        'lookback_period': 48,           # 48시간 MA
        'entry_threshold': -0.02,        # MA - 0.02%
        'target_profit_percent': 0.2,    # 0.2% 목표
        'stop_loss_percent': -0.1,       # -0.1% 손절
        'use_maker_only': True,          # 지정가만
        'max_positions': 2,              # 최대 2개 포지션
        'position_size_percent': 30,     # 자본의 30%
        'daily_max_trades': 3,           # 일 최대 3회
        'daily_max_loss': -100000        # 일 최대 손실 10만원
    }
    
    # 전략 생성
    print("\n[1/3] Creating strategy...")
    strategy = MeanReversionStrategy(config=strategy_config)
    print("SUCCESS: Mean Reversion strategy created")
    
    # Paper Trading 엔진 생성
    print("\n[2/3] Initializing Paper Trading engine...")
    engine = PaperTradingEngine(
        strategy=strategy,
        capital=40_000_000,
        maker_fee=0.0002,  # 0.02%
        taker_fee=0.0015   # 0.15%
    )
    
    if not await engine.initialize():
        print("ERROR: Failed to initialize engine")
        return
    print("SUCCESS: Engine initialized")
    
    # 테스트 데이터 수집
    print("\n[3/3] Testing market data collection...")
    test_data = await engine.fetch_market_data()
    if test_data:
        print(f"  BTC/KRW: {test_data['upbit_price']:,.0f} KRW")
        print(f"  BTC/USDT: {test_data['binance_price']:,.2f} USDT")
        print(f"  USD/KRW: {test_data['usd_krw']:,.2f}")
        print(f"  Kimchi Premium: {test_data['kimchi_premium']:.3f}%")
        print(f"  Upbit Spread: {test_data['upbit_spread']*100:.3f}%")
        print("SUCCESS: Market data collection working")
    else:
        print("ERROR: Failed to fetch market data")
        return
    
    print("\n" + "="*60)
    print("  STARTING PAPER TRADING")
    print("="*60)
    print("\nMonitoring every 10 minutes...")
    print("Duration: 10 minutes (test mode)")
    print("-"*60 + "\n")
    
    try:
        # Paper Trading 실행 (10분 테스트)
        await engine.run(duration_hours=0.17)  # 10분 = 0.17시간
    except KeyboardInterrupt:
        print("\n\nPaper Trading stopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.cleanup()
        print("\nPaper Trading completed")


if __name__ == "__main__":
    print("\n*** AUTO PAPER TRADING - NO USER INPUT REQUIRED ***")
    print("Starting in 3 seconds...")
    import time
    time.sleep(3)
    
    asyncio.run(run_paper_trading())