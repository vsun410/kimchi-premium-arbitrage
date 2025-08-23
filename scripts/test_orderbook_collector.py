#!/usr/bin/env python3
"""
오더북 수집 파이프라인 테스트
15초 간격 수집 및 유동성 분석 테스트
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collectors.orderbook_collector import OrderbookCollector


async def test_single_snapshot():
    """단일 스냅샷 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Single Orderbook Snapshot")
    print("=" * 60)
    
    collector = OrderbookCollector()
    
    try:
        await collector.initialize_exchanges()
        
        # Upbit 스냅샷
        print("\n[Upbit BTC/KRW]")
        upbit_snapshot = await collector.collect_snapshot("upbit", "BTC/KRW")
        
        if upbit_snapshot:
            print(f"  Timestamp: {upbit_snapshot.timestamp}")
            print(f"  Best Bid: {upbit_snapshot.bids[0][0]:,.0f} KRW")
            print(f"  Best Ask: {upbit_snapshot.asks[0][0]:,.0f} KRW")
            print(f"  Spread: {upbit_snapshot.bid_ask_spread:.3f}%")
            print(f"  Mid Price: {upbit_snapshot.mid_price:,.0f} KRW")
            print(f"  Imbalance: {upbit_snapshot.imbalance:.3f}")
            print(f"  Liquidity Score: {upbit_snapshot.liquidity_score:.1f}/100")
            print(f"  10% Depth - Bid: {upbit_snapshot.depth_10_pct['bid']:.4f} BTC")
            print(f"  10% Depth - Ask: {upbit_snapshot.depth_10_pct['ask']:.4f} BTC")
        
        # Binance 스냅샷
        print("\n[Binance BTC/USDT]")
        binance_snapshot = await collector.collect_snapshot("binance", "BTC/USDT")
        
        if binance_snapshot:
            print(f"  Timestamp: {binance_snapshot.timestamp}")
            print(f"  Best Bid: ${binance_snapshot.bids[0][0]:,.2f}")
            print(f"  Best Ask: ${binance_snapshot.asks[0][0]:,.2f}")
            print(f"  Spread: {binance_snapshot.bid_ask_spread:.3f}%")
            print(f"  Mid Price: ${binance_snapshot.mid_price:,.2f}")
            print(f"  Imbalance: {binance_snapshot.imbalance:.3f}")
            print(f"  Liquidity Score: {binance_snapshot.liquidity_score:.1f}/100")
            print(f"  10% Depth - Bid: {binance_snapshot.depth_10_pct['bid']:.4f} BTC")
            print(f"  10% Depth - Ask: {binance_snapshot.depth_10_pct['ask']:.4f} BTC")
        
        return upbit_snapshot is not None and binance_snapshot is not None
        
    except Exception as e:
        print(f"[ERROR] Snapshot test failed: {e}")
        return False
        
    finally:
        await collector.close_exchanges()


async def test_continuous_collection():
    """연속 수집 테스트 (30초)"""
    print("\n" + "=" * 60)
    print("TEST 2: Continuous Collection (30 seconds)")
    print("=" * 60)
    
    # 10초 간격으로 설정 (테스트용)
    collector = OrderbookCollector(interval_seconds=10)
    
    try:
        print("\nStarting collection (10-second intervals)...")
        await collector.start_collection()
        
        # 30초 동안 수집
        for i in range(3):
            await asyncio.sleep(10)
            
            # 최신 유동성 정보 출력
            liquidity = collector.get_latest_liquidity()
            
            print(f"\n[Update {i+1}/3]")
            for key, info in liquidity.items():
                exchange, symbol = key.split("_", 1)
                print(f"  {exchange} {symbol}:")
                print(f"    Liquidity: {info['liquidity_score']:.1f}/100")
                print(f"    Spread: {info['spread']:.3f}%")
                print(f"    Imbalance: {info['imbalance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Continuous collection failed: {e}")
        return False
        
    finally:
        await collector.stop_collection()


async def test_liquidity_analysis():
    """유동성 분석 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Liquidity Analysis")
    print("=" * 60)
    
    collector = OrderbookCollector()
    
    # 테스트용 가짜 오더북 데이터
    test_cases = [
        {
            "name": "High liquidity, tight spread",
            "bids": [[100000, 1.0], [99990, 2.0], [99980, 3.0], [99970, 2.5], [99960, 1.5]],
            "asks": [[100010, 1.0], [100020, 2.0], [100030, 3.0], [100040, 2.5], [100050, 1.5]],
        },
        {
            "name": "Low liquidity, wide spread",
            "bids": [[99000, 0.1], [98000, 0.2], [97000, 0.15], [96000, 0.1], [95000, 0.05]],
            "asks": [[101000, 0.1], [102000, 0.2], [103000, 0.15], [104000, 0.1], [105000, 0.05]],
        },
        {
            "name": "Imbalanced orderbook (more bids)",
            "bids": [[100000, 5.0], [99990, 4.0], [99980, 3.0], [99970, 2.0], [99960, 1.0]],
            "asks": [[100010, 0.5], [100020, 0.4], [100030, 0.3], [100040, 0.2], [100050, 0.1]],
        },
    ]
    
    for test_case in test_cases:
        print(f"\n[{test_case['name']}]")
        
        metrics = collector.calculate_liquidity_metrics(
            test_case["bids"], test_case["asks"]
        )
        
        print(f"  Spread: {metrics['bid_ask_spread']:.3f}%")
        print(f"  Mid Price: {metrics['mid_price']:,.2f}")
        print(f"  Imbalance: {metrics['imbalance']:.3f}")
        print(f"  Liquidity Score: {metrics['liquidity_score']:.1f}/100")
        print(f"  10% Depth:")
        print(f"    Bid: {metrics['depth_10_pct']['bid']:.4f}")
        print(f"    Ask: {metrics['depth_10_pct']['ask']:.4f}")
    
    return True


async def test_data_persistence():
    """데이터 저장 및 로드 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: Data Persistence")
    print("=" * 60)
    
    collector = OrderbookCollector()
    
    try:
        await collector.initialize_exchanges()
        
        # 스냅샷 수집 및 저장
        print("\nCollecting and saving snapshot...")
        snapshot = await collector.collect_snapshot("upbit", "BTC/KRW")
        
        if snapshot:
            collector.save_snapshot(snapshot)
            print(f"  Saved snapshot at {snapshot.timestamp}")
            
            # 저장된 파일 확인
            date_str = snapshot.timestamp.strftime("%Y%m%d")
            hour_str = snapshot.timestamp.strftime("%H")
            expected_file = (
                collector.data_dir / date_str / 
                f"upbit_BTC_KRW_{hour_str}.jsonl.gz"
            )
            
            if expected_file.exists():
                file_size = expected_file.stat().st_size
                print(f"  File created: {expected_file.name}")
                print(f"  File size: {file_size} bytes (compressed)")
                
                # 데이터 로드 테스트
                print("\nLoading saved snapshots...")
                loaded = collector.load_snapshots(
                    "upbit", "BTC/KRW",
                    snapshot.timestamp - timedelta(minutes=1),
                    snapshot.timestamp + timedelta(minutes=1)
                )
                
                if loaded:
                    print(f"  Loaded {len(loaded)} snapshot(s)")
                    print(f"  First timestamp: {loaded[0].timestamp}")
                    print(f"  Liquidity score: {loaded[0].liquidity_score:.1f}")
                    return True
                else:
                    print("  [WARN] No snapshots loaded")
            else:
                print(f"  [ERROR] Expected file not found: {expected_file}")
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Persistence test failed: {e}")
        return False
        
    finally:
        await collector.close_exchanges()


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("ORDERBOOK COLLECTOR TEST SUITE")
    print("=" * 60)
    print("\nTesting Task #8: Orderbook Data Collection Pipeline")
    
    tests = [
        ("Single Snapshot", test_single_snapshot),
        ("Continuous Collection", test_continuous_collection),
        ("Liquidity Analysis", test_liquidity_analysis),
        ("Data Persistence", test_data_persistence),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name:25} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] Task #8 COMPLETED! Orderbook pipeline ready.")
        print("\nKey features implemented:")
        print("  1. Real-time orderbook collection (15-second intervals)")
        print("  2. Liquidity score calculation (0-100)")
        print("  3. Order imbalance detection")
        print("  4. 10% depth liquidity analysis")
        print("  5. Compressed storage (GZIP)")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)