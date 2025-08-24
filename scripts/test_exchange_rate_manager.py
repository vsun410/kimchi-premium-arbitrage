"""
Test Exchange Rate Manager
환율 관리자 테스트 및 검증
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exchange_rate_manager import (
    ExchangeRateManager, 
    get_current_exchange_rate,
    calculate_kimchi_premium
)


def test_rate_manager():
    """환율 관리자 테스트"""
    
    print("=" * 60)
    print("  EXCHANGE RATE MANAGER TEST")
    print("=" * 60)
    
    # 1. 환율 관리자 초기화
    print("\n[1] Initializing Exchange Rate Manager...")
    manager = ExchangeRateManager()
    
    # 2. 현재 환율
    print(f"\n[2] Current Exchange Rate")
    current_rate = manager.current_rate
    print(f"Current rate: {current_rate:.2f} KRW/USD")
    
    # 3. 통계
    print(f"\n[3] Exchange Rate Statistics")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 4. 특정 시간 환율 조회
    print(f"\n[4] Rate at Specific Times")
    test_times = [
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=6),
        datetime.now() - timedelta(hours=24),
        datetime.now()
    ]
    
    for t in test_times:
        try:
            rate = manager.get_rate_at_time(t)
            print(f"  {t.strftime('%Y-%m-%d %H:%M')}: {rate:.2f} KRW/USD")
        except Exception as e:
            print(f"  {t.strftime('%Y-%m-%d %H:%M')}: Error - {e}")
    
    # 5. 김치 프리미엄 계산 테스트
    print(f"\n[5] Kimchi Premium Calculation Test")
    
    # 테스트 가격
    test_cases = [
        (159400000, 114850),  # 현재 가격
        (160000000, 115000),  # 가상 가격 1
        (158000000, 114000),  # 가상 가격 2
    ]
    
    for upbit, binance in test_cases:
        premium = manager.calculate_kimchi_premium(upbit, binance)
        print(f"\n  Upbit: KRW {upbit:,}")
        print(f"  Binance: ${binance:,}")
        print(f"  Rate: {current_rate:.2f}")
        print(f"  Binance in KRW: {binance * current_rate:,.0f}")
        print(f"  Premium: {premium:.2f}%")
    
    # 6. 간편 함수 테스트
    print(f"\n[6] Convenience Functions Test")
    
    # get_current_exchange_rate() 테스트
    quick_rate = get_current_exchange_rate()
    print(f"Quick rate lookup: {quick_rate:.2f} KRW/USD")
    
    # calculate_kimchi_premium() 테스트
    quick_premium = calculate_kimchi_premium(159400000, 114850)
    print(f"Quick premium calc: {quick_premium:.2f}%")
    
    # 7. 설정 저장
    print(f"\n[7] Saving Configuration...")
    manager.save_config()
    print("Config saved to config/exchange_rate_config.json")
    
    # 8. 검증
    print(f"\n[8] Validation")
    
    # 하드코딩 체크
    if ExchangeRateManager.DEFAULT_RATE is not None:
        print("[ERROR] DEFAULT_RATE is hardcoded! This must be None!")
    else:
        print("[OK] No hardcoded exchange rates found")
    
    # 데이터 유효성
    if manager.current_rate > 1000 and manager.current_rate < 1500:
        print(f"[OK] Current rate {manager.current_rate:.2f} is in valid range")
    else:
        print(f"[WARNING] Current rate {manager.current_rate:.2f} seems unusual")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_rate_manager()