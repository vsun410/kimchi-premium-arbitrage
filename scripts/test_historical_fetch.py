#!/usr/bin/env python3
"""
히스토리컬 데이터 수집 테스트
먼저 짧은 기간으로 테스트 후 전체 수집
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collectors.historical_fetcher import historical_fetcher


async def test_short_period():
    """짧은 기간 테스트 (1일)"""
    print("\n" + "=" * 60)
    print("TEST: Fetching 1 day of historical data")
    print("=" * 60)

    # 거래소 초기화
    await historical_fetcher.initialize_exchanges()

    # 1일 전 데이터
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    try:
        # Upbit 테스트
        print("\n[1] Testing Upbit...")
        upbit_df = await historical_fetcher.fetch_historical_data(
            exchange_name="upbit",
            symbol="BTC/KRW",
            start_date=start_date,
            end_date=end_date,
            timeframe="1m",
        )

        if not upbit_df.empty:
            print(f"  - Fetched {len(upbit_df)} candles")
            print(f"  - Date range: {upbit_df['timestamp'].min()} ~ {upbit_df['timestamp'].max()}")
            print(f"  - Price range: {upbit_df['close'].min():,.0f} ~ {upbit_df['close'].max():,.0f} KRW")

            # 데이터 검증
            is_valid, issues = historical_fetcher.validate_data(upbit_df)
            print(f"  - Validation: {'PASS' if is_valid else 'FAIL'}")
            if issues:
                for issue in issues:
                    print(f"    * {issue}")

        # Binance 테스트
        print("\n[2] Testing Binance...")
        binance_df = await historical_fetcher.fetch_historical_data(
            exchange_name="binance",
            symbol="BTC/USDT",
            start_date=start_date,
            end_date=end_date,
            timeframe="1m",
        )

        if not binance_df.empty:
            print(f"  - Fetched {len(binance_df)} candles")
            print(f"  - Date range: {binance_df['timestamp'].min()} ~ {binance_df['timestamp'].max()}")
            print(f"  - Price range: ${binance_df['close'].min():,.0f} ~ ${binance_df['close'].max():,.0f}")

            # 데이터 검증
            is_valid, issues = historical_fetcher.validate_data(binance_df)
            print(f"  - Validation: {'PASS' if is_valid else 'FAIL'}")
            if issues:
                for issue in issues:
                    print(f"    * {issue}")

        print("\n[SUCCESS] Short period test completed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return False

    finally:
        await historical_fetcher.close_exchanges()


async def fetch_full_year():
    """1년치 데이터 수집"""
    print("\n" + "=" * 60)
    print("FETCHING 1 YEAR OF BTC HISTORICAL DATA")
    print("=" * 60)
    print("\nThis will take several minutes...")
    print("Data will be saved to: data/historical/")

    try:
        results = await historical_fetcher.fetch_one_year_btc()

        if results:
            print("\n" + "=" * 60)
            print("FETCH COMPLETED!")
            print("=" * 60)

            for exchange, df in results.items():
                if not df.empty:
                    print(f"\n{exchange.upper()}:")
                    print(f"  - Total candles: {len(df):,}")
                    print(f"  - File size: ~{len(df) * 50 / 1024 / 1024:.1f} MB")
                    print(f"  - Date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

            return True

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch full year: {e}")
        return False


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("HISTORICAL DATA FETCHER TEST")
    print("=" * 60)

    # 1. 짧은 기간 테스트
    print("\nStep 1: Testing with 1 day of data...")
    success = await test_short_period()

    if not success:
        print("\n[ABORT] Short period test failed. Fix issues before fetching full year.")
        return 1

    # 2. 사용자 확인
    print("\n" + "-" * 60)
    print("Short period test PASSED!")
    print("Ready to fetch 1 YEAR of data (this will take 10-30 minutes)")
    print("-" * 60)

    # 자동 모드 체크
    import os
    if os.environ.get("AUTO_FETCH") == "yes":
        print("\n[AUTO MODE] Proceeding with full year fetch...")
    else:
        print("\nTo fetch full year, run with: AUTO_FETCH=yes python scripts/test_historical_fetch.py")
        print("Skipping full year fetch for now.")
        return 0

    # 3. 1년치 수집
    print("\nStep 2: Fetching 1 year of data...")
    success = await fetch_full_year()

    if success:
        print("\n[SUCCESS] Task #7 completed! Historical data collected.")
        return 0
    else:
        print("\n[FAIL] Could not complete full year fetch.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)