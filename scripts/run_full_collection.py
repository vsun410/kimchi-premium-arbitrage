#!/usr/bin/env python3
"""
Automated Full Data Collection Script
1년치 BTC 데이터 자동 수집 (무인 실행용)
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collectors.historical_collector import HistoricalDataCollector
from src.utils.logger import logger


def main():
    """자동 데이터 수집 실행"""
    print("\n" + "=" * 60)
    print("  AUTOMATED FULL DATA COLLECTION")
    print("  Starting 1-year BTC data collection")
    print("=" * 60)
    print(f"\nStart time: {datetime.now()}")
    print("This will take 30-60 minutes...")
    print("\nYou can check progress in:")
    print("  - Console output")
    print("  - logs/kimchi_arbitrage.log")
    print("\n" + "=" * 60)
    
    results = {}
    
    # Binance 데이터 수집
    print("\n[1/2] Collecting Binance BTC/USDT (365 days)...")
    print("      This will take approximately 20-30 minutes")
    try:
        binance_collector = HistoricalDataCollector('binance')
        binance_data = binance_collector.collect_historical_data(
            symbol='BTC/USDT',
            days=365,
            timeframe='1m',  # 1분 캔들 (PRD 요구사항)
            save_dir='data/historical/full'
        )
        results['binance'] = binance_data
        print(f"  [SUCCESS] Collected {len(binance_data):,} candles")
    except Exception as e:
        print(f"  [ERROR] Binance collection failed: {e}")
        results['binance'] = None
    
    # Upbit 데이터 수집
    print("\n[2/2] Collecting Upbit BTC/KRW (365 days)...")
    print("      This will take approximately 20-30 minutes")
    try:
        upbit_collector = HistoricalDataCollector('upbit')
        upbit_data = upbit_collector.collect_historical_data(
            symbol='BTC/KRW',
            days=365,
            timeframe='1m',  # 1분 캔들
            save_dir='data/historical/full'
        )
        results['upbit'] = upbit_data
        print(f"  [SUCCESS] SUCCESS: Collected {len(upbit_data):,} candles")
    except Exception as e:
        print(f"  [FAILED] ERROR: Upbit collection failed: {e}")
        results['upbit'] = None
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    
    success_count = 0
    for exchange, data in results.items():
        if data is not None:
            success_count += 1
            print(f"\n{exchange.upper()}:")
            print(f"  [SUCCESS] Total candles: {len(data):,}")
            print(f"  [Period]: {data.index[0]} to {data.index[-1]}")
            print(f"  [File] File size: ~{len(data) * 50 / 1024 / 1024:.1f} MB")
            
            # 간단한 통계
            print(f"  [Stats] Price stats:")
            print(f"     Min: ${data['low'].min():,.2f}")
            print(f"     Max: ${data['high'].max():,.2f}")
            print(f"     Avg: ${data['close'].mean():,.2f}")
        else:
            print(f"\n{exchange.upper()}: [FAILED] Failed")
    
    print(f"\nEnd time: {datetime.now()}")
    
    if success_count == 2:
        print("\n[SUCCESS] SUCCESS: All data collected successfully!")
        print("\n[Next] Next steps:")
        print("1. Check data files in: data/historical/full/")
        print("2. Train ML models with collected data")
        print("3. Run backtesting")
        return 0
    elif success_count == 1:
        print("\n[WARNING]  WARNING: Partial success. One exchange failed.")
        return 1
    else:
        print("\n[FAILED] FAILED: No data collected")
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)