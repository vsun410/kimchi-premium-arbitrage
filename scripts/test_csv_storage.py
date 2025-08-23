#!/usr/bin/env python3
"""
CSV 데이터 저장 시스템 테스트
Task #12: CSV 기반 데이터 저장 시스템
"""

import asyncio
import gzip
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_storage.csv_storage import (
    CSVStorage,
    KimchiPremiumRecord,
    OrderbookRecord,
    PriceRecord,
)


def test_price_data_storage():
    """가격 데이터 저장 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Price Data Storage")
    print("=" * 60)
    
    storage = CSVStorage(base_dir="data/test_csv", compress=True)
    
    try:
        # 테스트 데이터 생성
        records = []
        base_time = datetime.now()
        
        for i in range(100):
            record = PriceRecord(
                timestamp=base_time + timedelta(minutes=i),
                exchange="upbit",
                symbol="BTC/KRW",
                open=160000000 + i * 1000,
                high=160100000 + i * 1000,
                low=159900000 + i * 1000,
                close=160050000 + i * 1000,
                volume=10.5 + i * 0.1,
            )
            records.append(record)
        
        # 저장
        print(f"\nSaving {len(records)} price records...")
        start_time = time.time()
        storage.save_price_data(records, "upbit")
        save_time = time.time() - start_time
        
        print(f"  Save time: {save_time:.3f}s")
        print(f"  Records/sec: {len(records)/save_time:.0f}")
        
        # 로드
        print("\nLoading saved data...")
        start_time = time.time()
        df = storage.load_price_data(
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=2),
            "upbit"
        )
        load_time = time.time() - start_time
        
        print(f"  Load time: {load_time:.3f}s")
        print(f"  Records loaded: {len(df)}")
        
        if len(df) > 0:
            print(f"  First timestamp: {df.iloc[0]['timestamp']}")
            print(f"  Last timestamp: {df.iloc[-1]['timestamp']}")
            print(f"  Price range: {df['close'].min():,.0f} - {df['close'].max():,.0f}")
        
        # 검증
        if len(df) == len(records):
            print("  [OK] All records saved and loaded correctly")
            return True
        else:
            print(f"  [FAIL] Record count mismatch: {len(df)} != {len(records)}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Price data test failed: {e}")
        return False


def test_orderbook_data_storage():
    """오더북 데이터 저장 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Orderbook Data Storage")
    print("=" * 60)
    
    storage = CSVStorage(base_dir="data/test_csv", compress=True)
    
    try:
        # 테스트 데이터 생성 (15초 간격)
        records = []
        base_time = datetime.now()
        
        for i in range(240):  # 1시간 분량
            record = OrderbookRecord(
                timestamp=base_time + timedelta(seconds=i * 15),
                exchange="binance",
                symbol="BTC/USDT",
                best_bid=115000 + i * 10,
                best_ask=115010 + i * 10,
                bid_volume=10.5 + i * 0.01,
                ask_volume=10.3 + i * 0.01,
                spread=0.01 + i * 0.0001,
                mid_price=115005 + i * 10,
                liquidity_score=85.0 + (i % 10),
                imbalance=0.01 * (i % 21 - 10) / 10,  # -0.1 to 0.1
            )
            records.append(record)
        
        # 저장
        print(f"\nSaving {len(records)} orderbook records (15s intervals)...")
        start_time = time.time()
        storage.save_orderbook_data(records, "binance")
        save_time = time.time() - start_time
        
        print(f"  Save time: {save_time:.3f}s")
        print(f"  Records/sec: {len(records)/save_time:.0f}")
        
        # 로드
        print("\nLoading saved data...")
        start_time = time.time()
        df = storage.load_orderbook_data(
            base_time - timedelta(minutes=10),
            base_time + timedelta(hours=2),
            "binance"
        )
        load_time = time.time() - start_time
        
        print(f"  Load time: {load_time:.3f}s")
        print(f"  Records loaded: {len(df)}")
        
        if len(df) > 0:
            print(f"  Avg liquidity score: {df['liquidity_score'].mean():.1f}")
            print(f"  Avg spread: {df['spread'].mean():.4f}")
            print(f"  Imbalance range: [{df['imbalance'].min():.3f}, {df['imbalance'].max():.3f}]")
        
        # 검증
        if len(df) == len(records):
            print("  [OK] Orderbook data stored correctly")
            return True
        else:
            print(f"  [FAIL] Record count mismatch")
            return False
            
    except Exception as e:
        print(f"[ERROR] Orderbook test failed: {e}")
        return False


def test_premium_data_storage():
    """김치 프리미엄 데이터 저장 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Kimchi Premium Data Storage")
    print("=" * 60)
    
    storage = CSVStorage(base_dir="data/test_csv", compress=True)
    
    try:
        # 테스트 데이터 생성
        records = []
        base_time = datetime.now()
        
        for i in range(60):  # 1시간 분량 (1분 간격)
            premium_rate = 0.5 + i * 0.01  # 0.5% ~ 1.1%
            record = KimchiPremiumRecord(
                timestamp=base_time + timedelta(minutes=i),
                upbit_price=160000000 + i * 10000,
                binance_price=115000 + i * 10,
                exchange_rate=1386.14 + i * 0.1,
                premium_rate=premium_rate,
                premium_krw=premium_rate * 1000000,
                liquidity_score=80.0 + (i % 20),
                spread_upbit=0.05 + i * 0.001,
                spread_binance=0.02 + i * 0.0005,
                confidence=0.8 + (i % 10) * 0.02,
            )
            records.append(record)
        
        # 저장
        print(f"\nSaving {len(records)} premium records...")
        start_time = time.time()
        storage.save_premium_data(records)
        save_time = time.time() - start_time
        
        print(f"  Save time: {save_time:.3f}s")
        
        # 로드
        print("\nLoading saved data...")
        df = storage.load_premium_data(
            base_time - timedelta(minutes=10),
            base_time + timedelta(hours=2)
        )
        
        if len(df) > 0:
            print(f"  Records loaded: {len(df)}")
            print(f"  Premium range: {df['premium_rate'].min():.2f}% - {df['premium_rate'].max():.2f}%")
            print(f"  Avg confidence: {df['confidence'].mean():.2%}")
            print(f"  Avg liquidity: {df['liquidity_score'].mean():.1f}")
        
        # 검증
        if len(df) == len(records):
            print("  [OK] Premium data stored correctly")
            return True
        else:
            print(f"  [FAIL] Record count mismatch")
            return False
            
    except Exception as e:
        print(f"[ERROR] Premium test failed: {e}")
        return False


def test_data_partitioning():
    """일별 파티셔닝 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: Daily Partitioning")
    print("=" * 60)
    
    storage = CSVStorage(
        base_dir="data/test_csv",
        compress=True,
        partition_by_day=True
    )
    
    try:
        # 3일치 데이터 생성
        records = []
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for day in range(3):
            for hour in range(24):
                record = PriceRecord(
                    timestamp=base_time + timedelta(days=day, hours=hour),
                    exchange="test",
                    symbol="BTC/TEST",
                    open=100000 + day * 1000 + hour * 10,
                    high=100100 + day * 1000 + hour * 10,
                    low=99900 + day * 1000 + hour * 10,
                    close=100050 + day * 1000 + hour * 10,
                    volume=1.0,
                )
                records.append(record)
        
        # 저장
        print(f"\nSaving {len(records)} records across 3 days...")
        storage.save_price_data(records, "test")
        
        # 파티션 확인
        price_dir = Path("data/test_csv/price")
        partitions = sorted([d.name for d in price_dir.iterdir() if d.is_dir()])
        
        print(f"\nCreated partitions: {partitions}")
        
        # 각 파티션 파일 확인
        for partition in partitions:
            partition_dir = price_dir / partition
            files = list(partition_dir.glob("*.csv*"))
            if files:
                file_size = files[0].stat().st_size
                print(f"  {partition}: {files[0].name} ({file_size} bytes)")
        
        # 검증
        if len(partitions) == 3:
            print("  [OK] Daily partitioning working correctly")
            return True
        else:
            print(f"  [FAIL] Expected 3 partitions, got {len(partitions)}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Partitioning test failed: {e}")
        return False


def test_compression_efficiency():
    """압축 효율성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 5: Compression Efficiency")
    print("=" * 60)
    
    # 압축 없이
    storage_uncompressed = CSVStorage(
        base_dir="data/test_csv_uncompressed",
        compress=False
    )
    
    # 압축 사용
    storage_compressed = CSVStorage(
        base_dir="data/test_csv_compressed",
        compress=True
    )
    
    try:
        # 대량 데이터 생성
        records = []
        base_time = datetime.now()
        
        for i in range(1000):
            record = OrderbookRecord(
                timestamp=base_time + timedelta(seconds=i * 15),
                exchange="test",
                symbol="BTC/TEST",
                best_bid=100000.0,
                best_ask=100010.0,
                bid_volume=10.12345678,
                ask_volume=10.87654321,
                spread=0.01,
                mid_price=100005.0,
                liquidity_score=85.5,
                imbalance=0.05,
            )
            records.append(record)
        
        # 압축 없이 저장
        print(f"\nSaving {len(records)} records without compression...")
        storage_uncompressed.save_orderbook_data(records, "test")
        
        # 압축하여 저장
        print(f"Saving {len(records)} records with compression...")
        storage_compressed.save_orderbook_data(records, "test")
        
        # 파일 크기 비교
        uncompressed_file = list(Path("data/test_csv_uncompressed/orderbook").rglob("*.csv"))[0]
        compressed_file = list(Path("data/test_csv_compressed/orderbook").rglob("*.csv.gz"))[0]
        
        uncompressed_size = uncompressed_file.stat().st_size
        compressed_size = compressed_file.stat().st_size
        compression_ratio = (1 - compressed_size / uncompressed_size) * 100
        
        print(f"\nCompression results:")
        print(f"  Uncompressed: {uncompressed_size:,} bytes")
        print(f"  Compressed: {compressed_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print(f"  Space saved: {uncompressed_size - compressed_size:,} bytes")
        
        # 로드 속도 비교
        print("\nLoad speed comparison:")
        
        # 압축 없이 로드
        start_time = time.time()
        df1 = storage_uncompressed.load_orderbook_data(
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=5),
            "test"
        )
        uncompressed_load_time = time.time() - start_time
        
        # 압축 파일 로드
        start_time = time.time()
        df2 = storage_compressed.load_orderbook_data(
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=5),
            "test"
        )
        compressed_load_time = time.time() - start_time
        
        print(f"  Uncompressed load: {uncompressed_load_time:.3f}s")
        print(f"  Compressed load: {compressed_load_time:.3f}s")
        print(f"  Load time difference: {abs(compressed_load_time - uncompressed_load_time):.3f}s")
        
        # 검증
        if compression_ratio > 50 and len(df1) == len(df2) == len(records):
            print("  [OK] Compression working efficiently")
            return True
        else:
            print(f"  [FAIL] Compression ratio too low or data mismatch")
            return False
            
    except Exception as e:
        print(f"[ERROR] Compression test failed: {e}")
        return False


def test_buffer_and_batch():
    """버퍼링 및 배치 처리 테스트"""
    print("\n" + "=" * 60)
    print("TEST 6: Buffer and Batch Processing")
    print("=" * 60)
    
    storage = CSVStorage(base_dir="data/test_csv_buffer")
    storage.buffer_size = 100  # 100개씩 배치
    
    try:
        base_time = datetime.now()
        
        # 버퍼에 데이터 추가 (자동 플러시 테스트)
        print("\nAdding records to buffer...")
        for i in range(250):  # buffer_size(100) 초과
            record = PriceRecord(
                timestamp=base_time + timedelta(minutes=i),
                exchange="buffer_test",
                symbol="BTC/TEST",
                open=100000 + i,
                high=100100 + i,
                low=99900 + i,
                close=100050 + i,
                volume=1.0,
            )
            storage.add_to_buffer("price", record)
            
            if i == 99:
                print(f"  Buffer at {i+1} records (should auto-flush at 100)")
            elif i == 199:
                print(f"  Buffer at {i+1} records (should auto-flush at 200)")
        
        # 남은 버퍼 플러시
        print(f"\nFlushing remaining buffer...")
        storage.flush_buffer()
        
        # 저장된 데이터 확인
        df = storage.load_price_data(
            base_time - timedelta(hours=1),
            base_time + timedelta(hours=5)
        )
        
        print(f"\nTotal records saved: {len(df)}")
        
        # 검증
        if len(df) == 250:
            print("  [OK] Buffer and batch processing working correctly")
            return True
        else:
            print(f"  [FAIL] Expected 250 records, got {len(df)}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Buffer test failed: {e}")
        return False


def test_storage_stats():
    """저장소 통계 테스트"""
    print("\n" + "=" * 60)
    print("TEST 7: Storage Statistics")
    print("=" * 60)
    
    storage = CSVStorage(base_dir="data/test_csv")
    
    try:
        # 통계 수집
        stats = storage.get_storage_stats()
        
        print("\nStorage Statistics:")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        print(f"  Total files: {stats['file_count']}")
        
        for data_type, type_stats in stats["data_types"].items():
            if type_stats["file_count"] > 0:
                print(f"\n  {data_type.upper()}:")
                print(f"    Files: {type_stats['file_count']}")
                print(f"    Size: {type_stats['size_mb']:.2f} MB")
                if type_stats["oldest_date"]:
                    print(f"    Date range: {type_stats['oldest_date']} to {type_stats['newest_date']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Stats test failed: {e}")
        return False


def cleanup_test_data():
    """테스트 데이터 정리"""
    test_dirs = [
        "data/test_csv",
        "data/test_csv_uncompressed",
        "data/test_csv_compressed",
        "data/test_csv_buffer",
    ]
    
    for dir_path in test_dirs:
        path = Path(dir_path)
        if path.exists():
            import shutil
            shutil.rmtree(path)
            print(f"Cleaned up: {dir_path}")


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("CSV STORAGE SYSTEM TEST SUITE")
    print("=" * 60)
    print("\nTesting Task #12: CSV-based Data Storage System")
    
    tests = [
        ("Price Data Storage", test_price_data_storage),
        ("Orderbook Data Storage", test_orderbook_data_storage),
        ("Premium Data Storage", test_premium_data_storage),
        ("Daily Partitioning", test_data_partitioning),
        ("Compression Efficiency", test_compression_efficiency),
        ("Buffer & Batch Processing", test_buffer_and_batch),
        ("Storage Statistics", test_storage_stats),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
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
        print(f"{test_name:30} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    # 테스트 데이터 정리
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)
    cleanup_test_data()
    
    if passed == total:
        print("\n[SUCCESS] Task #12 COMPLETED! CSV storage system ready.")
        print("\nKey features implemented:")
        print("  1. Price, orderbook, and premium data storage")
        print("  2. Daily data partitioning")
        print("  3. GZIP compression (50%+ space savings)")
        print("  4. Batch processing with buffers")
        print("  5. Pydantic schema validation")
        print("  6. Storage statistics and cleanup")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)