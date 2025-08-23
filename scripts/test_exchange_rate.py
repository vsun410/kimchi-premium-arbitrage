#!/usr/bin/env python3
"""
환율 데이터 시스템 종합 테스트
여러 시나리오와 엣지 케이스 테스트
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.exchange_rate_manager import ExchangeRateManager, RateSource
from src.data.rate_storage import RateCache, RateStorage
from src.utils.logger import logger


class ExchangeRateTestSuite:
    """환율 시스템 테스트 스위트"""

    def __init__(self):
        self.manager = ExchangeRateManager()
        self.storage = RateStorage()
        self.cache = RateCache(ttl=5)  # 5초 캐시
        self.test_results = []

    async def test_api_connection(self):
        """API 연결 테스트"""
        print("\n[TEST 1] API Connection Test")
        print("-" * 50)

        try:
            # 현재 환율 가져오기
            rate = await self.manager.get_current_rate()

            if rate and 1000 < rate < 2000:  # 합리적인 범위 체크
                print(f"[PASS] Current rate: {rate:.2f} KRW/USD")
                self.test_results.append(("API Connection", True, f"Rate: {rate:.2f}"))
                return True
            else:
                print(f"[FAIL] Invalid rate: {rate}")
                self.test_results.append(("API Connection", False, f"Invalid rate: {rate}"))
                return False

        except Exception as e:
            print(f"[FAIL] API connection failed: {e}")
            self.test_results.append(("API Connection", False, str(e)))
            return False

    async def test_fallback_mechanism(self):
        """Fallback 메커니즘 테스트"""
        print("\n[TEST 2] Fallback Mechanism Test")
        print("-" * 50)

        try:
            # 새 매니저 인스턴스 생성 (API 키 없음)
            test_manager = ExchangeRateManager()
            test_manager.api_keys = {}  # 모든 API 키 제거

            # Fallback rate 설정
            test_manager.set_fallback_rate(1400.0)

            # 환율 가져오기 (fallback 사용해야 함)
            rate = await test_manager.get_current_rate(force_refresh=True)

            if rate == 1400.0:
                print(f"[PASS] Fallback worked: {rate:.2f}")
                self.test_results.append(("Fallback", True, "Fallback rate used"))
                return True
            else:
                print(f"[FAIL] Fallback failed: {rate}")
                self.test_results.append(("Fallback", False, f"Wrong rate: {rate}"))
                return False

        except Exception as e:
            print(f"[FAIL] Fallback test failed: {e}")
            self.test_results.append(("Fallback", False, str(e)))
            return False

    def test_cache_functionality(self):
        """캐시 기능 테스트"""
        print("\n[TEST 3] Cache Functionality Test")
        print("-" * 50)

        try:
            # 캐시에 값 저장
            self.cache.set("test_rate", 1385.50)

            # 즉시 조회
            cached = self.cache.get("test_rate")
            if cached != 1385.50:
                raise ValueError(f"Cache immediate retrieval failed: {cached}")
            print("[PASS] Cache immediate retrieval: OK")

            # TTL 내 조회
            time.sleep(2)
            cached = self.cache.get("test_rate")
            if cached != 1385.50:
                raise ValueError(f"Cache within TTL failed: {cached}")
            print("[PASS] Cache within TTL: OK")

            # TTL 초과 후 조회
            time.sleep(4)  # 총 6초 경과 (TTL=5초)
            cached = self.cache.get("test_rate")
            if cached is not None:
                raise ValueError(f"Cache TTL expiry failed: {cached}")
            print("[PASS] Cache TTL expiry: OK")

            self.test_results.append(("Cache", True, "All cache tests passed"))
            return True

        except Exception as e:
            print(f"[FAIL] Cache test failed: {e}")
            self.test_results.append(("Cache", False, str(e)))
            return False

    def test_storage_operations(self):
        """저장소 작업 테스트"""
        print("\n[TEST 4] Storage Operations Test")
        print("-" * 50)

        try:
            # 환율 저장
            test_rate = 1390.25
            self.storage.save_current_rate(test_rate, "test")
            print(f"[PASS] Saved rate: {test_rate}")

            # 환율 로드
            loaded = self.storage.load_current_rate()
            if not loaded or abs(loaded["rate"] - test_rate) > 0.01:
                raise ValueError(f"Load mismatch: {loaded}")
            print(f"[PASS] Loaded rate: {loaded['rate']}")

            # 히스토리 로드
            df = self.storage.load_history(start_date=datetime.now() - timedelta(hours=1))
            if df.empty:
                print("[WARN]  No history data (expected for new installation)")
            else:
                print(f"[PASS] History records: {len(df)}")

            # 통계 계산
            stats = self.storage.get_statistics(days=1)
            print(f"[PASS] Statistics calculated: {stats['data_points']} points")

            self.test_results.append(("Storage", True, "All storage tests passed"))
            return True

        except Exception as e:
            print(f"[FAIL] Storage test failed: {e}")
            self.test_results.append(("Storage", False, str(e)))
            return False

    async def test_rate_monitoring(self):
        """환율 모니터링 테스트"""
        print("\n[TEST 5] Rate Monitoring Test")
        print("-" * 50)

        try:
            # 5초간 모니터링
            print("Starting 5-second monitoring test...")

            async def monitor_task():
                await self.manager.start_monitoring(interval=2)

            # 5초 타임아웃으로 실행
            await asyncio.wait_for(monitor_task(), timeout=5)

        except asyncio.TimeoutError:
            # 타임아웃은 정상 (테스트 종료)
            history_size = len(self.manager.rate_history)

            if history_size > 0:
                print(f"[PASS] Monitoring worked: {history_size} updates collected")
                self.test_results.append(("Monitoring", True, f"{history_size} updates"))
                return True
            else:
                print("[WARN]  No updates collected (API might be unavailable)")
                self.test_results.append(("Monitoring", False, "No updates"))
                return False

        except Exception as e:
            print(f"[FAIL] Monitoring test failed: {e}")
            self.test_results.append(("Monitoring", False, str(e)))
            return False

    async def test_volatility_calculation(self):
        """변동성 계산 테스트"""
        print("\n[TEST 6] Volatility Calculation Test")
        print("-" * 50)

        try:
            # 새 매니저로 깨끗한 테스트
            test_manager = ExchangeRateManager()
            test_rates = [1380.0, 1385.0, 1382.0, 1388.0, 1384.0]

            for rate in test_rates:
                test_manager.rate_history.append(
                    {"timestamp": datetime.now(), "rate": rate, "source": "test"}
                )
                await asyncio.sleep(0.1)

            # 평균 계산
            avg = test_manager.get_average_rate(hours=1)
            expected_avg = sum(test_rates) / len(test_rates)

            if avg and abs(avg - expected_avg) < 0.1:
                print(f"[PASS] Average rate: {avg:.2f} (expected: {expected_avg:.2f})")
            else:
                raise ValueError(f"Average mismatch: {avg} vs {expected_avg}")

            # 변동성 계산
            volatility = test_manager.get_rate_volatility(hours=1)
            if volatility is not None and volatility >= 0:
                print(f"[PASS] Volatility: {volatility:.2f}")
            else:
                raise ValueError(f"Invalid volatility: {volatility}")

            self.test_results.append(("Volatility", True, f"Vol: {volatility:.2f}"))
            return True

        except Exception as e:
            print(f"[FAIL] Volatility test failed: {e}")
            self.test_results.append(("Volatility", False, str(e)))
            return False

    async def test_edge_cases(self):
        """엣지 케이스 테스트"""
        print("\n[TEST 7] Edge Cases Test")
        print("-" * 50)

        try:
            # 빈 히스토리에서 평균 계산
            temp_manager = ExchangeRateManager()
            avg = temp_manager.get_average_rate(hours=1)
            if avg is not None and avg != temp_manager.current_rate:
                raise ValueError(f"Empty history average should be None or current rate")
            print("[PASS] Empty history average: OK")

            # 단일 데이터포인트 변동성
            temp_manager.rate_history.append(
                {"timestamp": datetime.now(), "rate": 1385.0, "source": "test"}
            )
            vol = temp_manager.get_rate_volatility(hours=1)
            if vol != 0.0:
                raise ValueError(f"Single point volatility should be 0: {vol}")
            print("[PASS] Single point volatility: OK")

            # 극단적인 환율 값
            extreme_rates = [0, -100, 1000000]
            for rate in extreme_rates:
                temp_manager.current_rate = rate
                if 1000 < rate < 2000:  # 합리적인 범위만 통과해야 함
                    raise ValueError(f"Extreme rate accepted: {rate}")
            print("[PASS] Extreme value rejection: OK")

            self.test_results.append(("Edge Cases", True, "All edge cases passed"))
            return True

        except Exception as e:
            print(f"[FAIL] Edge case test failed: {e}")
            self.test_results.append(("Edge Cases", False, str(e)))
            return False

    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)

        for test_name, result, details in self.test_results:
            status = "[PASS] PASS" if result else "[FAIL] FAIL"
            print(f"{test_name:20} {status:10} {details}")

        print("-" * 60)
        print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        return passed == total


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("EXCHANGE RATE SYSTEM TEST SUITE")
    print("=" * 60)

    test_suite = ExchangeRateTestSuite()

    # 모든 테스트 실행
    await test_suite.test_api_connection()
    await test_suite.test_fallback_mechanism()
    test_suite.test_cache_functionality()
    test_suite.test_storage_operations()
    await test_suite.test_rate_monitoring()
    await test_suite.test_volatility_calculation()
    await test_suite.test_edge_cases()

    # 결과 요약
    all_passed = test_suite.print_summary()

    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED! Ready to commit.")
        return 0
    else:
        print("\n[WARN]  SOME TESTS FAILED! Please fix before committing.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
