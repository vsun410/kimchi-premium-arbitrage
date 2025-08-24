#!/usr/bin/env python3
"""
Historical Data Collection Script
실제 1년치 BTC 데이터 수집
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collectors.historical_collector import (
    HistoricalDataCollector,
    collect_all_exchanges,
)
from src.utils.logger import logger


def collect_test_data():
    """테스트용 소량 데이터 수집 (1주일)"""
    print("\n" + "=" * 60)
    print("  TEST DATA COLLECTION (1 Week)")
    print("=" * 60)
    
    results = {}
    
    # Binance 데이터 수집
    print("\n[1/2] Collecting Binance BTC/USDT...")
    try:
        binance_collector = HistoricalDataCollector('binance')
        binance_data = binance_collector.collect_historical_data(
            symbol='BTC/USDT',
            days=7,  # 1주일
            timeframe='15m',  # 15분 캔들 (빠른 수집)
            save_dir='data/historical/test'
        )
        results['binance'] = binance_data
        print(f"  [OK] Collected {len(binance_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Binance collection failed: {e}")
        results['binance'] = None
    
    # Upbit 데이터 수집
    print("\n[2/2] Collecting Upbit BTC/KRW...")
    try:
        upbit_collector = HistoricalDataCollector('upbit')
        upbit_data = upbit_collector.collect_historical_data(
            symbol='BTC/KRW',
            days=7,  # 1주일
            timeframe='15m',  # 15분 캔들
            save_dir='data/historical/test'
        )
        results['upbit'] = upbit_data
        print(f"  [OK] Collected {len(upbit_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Upbit collection failed: {e}")
        results['upbit'] = None
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    
    for exchange, data in results.items():
        if data is not None:
            print(f"\n{exchange.upper()}:")
            print(f"  Candles: {len(data)}")
            print(f"  Period: {data.index[0]} to {data.index[-1]}")
            print(f"  Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        else:
            print(f"\n{exchange.upper()}: Failed")
    
    return results


def collect_full_data():
    """실제 1년치 데이터 수집"""
    print("\n" + "=" * 60)
    print("  FULL DATA COLLECTION (1 Year)")
    print("=" * 60)
    print("\n[WARNING] This will take a long time (30-60 minutes)")
    print("API rate limits will slow down the collection")
    
    response = input("\nProceed with full collection? (y/n): ")
    if response.lower() != 'y':
        print("Collection cancelled")
        return None
    
    results = {}
    
    # Binance 데이터 수집
    print("\n[1/2] Collecting Binance BTC/USDT (365 days)...")
    try:
        binance_collector = HistoricalDataCollector('binance')
        binance_data = binance_collector.collect_historical_data(
            symbol='BTC/USDT',
            days=365,
            timeframe='1m',  # 1분 캔들 (PRD 요구사항)
            save_dir='data/historical/full'
        )
        results['binance'] = binance_data
        print(f"  [OK] Collected {len(binance_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Binance collection failed: {e}")
        results['binance'] = None
    
    # Upbit 데이터 수집
    print("\n[2/2] Collecting Upbit BTC/KRW (365 days)...")
    try:
        upbit_collector = HistoricalDataCollector('upbit')
        upbit_data = upbit_collector.collect_historical_data(
            symbol='BTC/KRW',
            days=365,
            timeframe='1m',  # 1분 캔들
            save_dir='data/historical/full'
        )
        results['upbit'] = upbit_data
        print(f"  [OK] Collected {len(upbit_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Upbit collection failed: {e}")
        results['upbit'] = None
    
    # 결과 저장
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    
    for exchange, data in results.items():
        if data is not None:
            print(f"\n{exchange.upper()}:")
            print(f"  Total candles: {len(data):,}")
            print(f"  Period: {data.index[0]} to {data.index[-1]}")
            print(f"  File size: ~{len(data) * 50 / 1024 / 1024:.1f} MB")
    
    return results


def collect_training_data():
    """ML 학습용 3개월 데이터 수집"""
    print("\n" + "=" * 60)
    print("  TRAINING DATA COLLECTION (3 Months)")
    print("=" * 60)
    
    results = {}
    
    # Binance 데이터 수집
    print("\n[1/2] Collecting Binance BTC/USDT (90 days)...")
    try:
        binance_collector = HistoricalDataCollector('binance')
        binance_data = binance_collector.collect_historical_data(
            symbol='BTC/USDT',
            days=90,  # 3개월
            timeframe='5m',  # 5분 캔들 (적당한 크기)
            save_dir='data/historical/training'
        )
        results['binance'] = binance_data
        print(f"  [OK] Collected {len(binance_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Binance collection failed: {e}")
        results['binance'] = None
    
    # Upbit 데이터 수집
    print("\n[2/2] Collecting Upbit BTC/KRW (90 days)...")
    try:
        upbit_collector = HistoricalDataCollector('upbit')
        upbit_data = upbit_collector.collect_historical_data(
            symbol='BTC/KRW',
            days=90,  # 3개월
            timeframe='5m',  # 5분 캔들
            save_dir='data/historical/training'
        )
        results['upbit'] = upbit_data
        print(f"  [OK] Collected {len(upbit_data)} candles")
    except Exception as e:
        print(f"  [ERROR] Upbit collection failed: {e}")
        results['upbit'] = None
    
    # 김치 프리미엄 계산
    if results.get('binance') is not None and results.get('upbit') is not None:
        print("\n[3/3] Calculating Kimchi Premium...")
        
        # 시간대 정렬
        binance_data = results['binance']
        upbit_data = results['upbit']
        
        # 공통 시간대 찾기
        common_index = binance_data.index.intersection(upbit_data.index)
        
        if len(common_index) > 0:
            # 김프 계산 (간단 버전, 환율 1200 가정)
            usd_krw = 1200
            binance_krw = binance_data.loc[common_index, 'close'] * usd_krw
            upbit_krw = upbit_data.loc[common_index, 'close']
            
            kimchi_premium = ((upbit_krw - binance_krw) / binance_krw * 100)
            
            print(f"  Average Kimchi Premium: {kimchi_premium.mean():.2f}%")
            print(f"  Min/Max: {kimchi_premium.min():.2f}% / {kimchi_premium.max():.2f}%")
            
            # 김프 데이터 저장
            import pandas as pd
            kimchi_df = pd.DataFrame({
                'binance_usd': binance_data.loc[common_index, 'close'],
                'upbit_krw': upbit_krw,
                'kimchi_premium': kimchi_premium
            })
            kimchi_df.to_csv('data/historical/training/kimchi_premium.csv')
            print(f"  [OK] Kimchi premium data saved")
    
    return results


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("  HISTORICAL DATA COLLECTOR")
    print("=" * 60)
    
    print("\nSelect collection mode:")
    print("1. Test (1 week, fast)")
    print("2. Training (3 months, for ML)")
    print("3. Full (1 year, very slow)")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == '1':
        results = collect_test_data()
    elif choice == '2':
        results = collect_training_data()
    elif choice == '3':
        results = collect_full_data()
    else:
        print("Invalid choice")
        return 1
    
    # 성공 여부 확인
    if results and any(v is not None for v in results.values()):
        print("\n[SUCCESS] Data collection completed!")
        print("\nNext steps:")
        print("1. Train ML models with collected data")
        print("2. Run backtesting")
        return 0
    else:
        print("\n[FAILED] Data collection failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)