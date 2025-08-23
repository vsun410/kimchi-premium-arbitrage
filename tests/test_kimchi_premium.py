#!/usr/bin/env python3
"""
김치 프리미엄 계산기 테스트 스위트
"""

import asyncio
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kimchi_premium import KimchiPremiumCalculator, KimchiPremiumData, PremiumSignal


class TestKimchiPremiumCalculator(unittest.TestCase):
    """김치 프리미엄 계산기 테스트"""

    def setUp(self):
        """테스트 초기화"""
        self.calculator = KimchiPremiumCalculator()

    def tearDown(self):
        """테스트 정리"""
        self.calculator = None

    async def async_test_premium_calculation(self):
        """프리미엄 계산 테스트"""
        # 테스트 데이터
        upbit_price = 159_400_000  # 159.4M KRW
        binance_price = 115_000  # 115K USDT
        exchange_rate = 1386.14  # KRW/USD

        # 계산
        with patch("src.data.exchange_rate_manager.rate_manager.get_current_rate") as mock_rate:
            mock_rate.return_value = exchange_rate

            data = await self.calculator.calculate_premium(upbit_price, binance_price)

        # 검증
        self.assertIsNotNone(data)
        self.assertIsInstance(data, KimchiPremiumData)

        # 예상 김프: (159,400,000 / (115,000 * 1386.14) - 1) * 100 ≈ 0.0%
        expected_binance_krw = binance_price * exchange_rate
        expected_premium = ((upbit_price / expected_binance_krw) - 1) * 100

        self.assertAlmostEqual(data.premium_rate, expected_premium, places=2)
        self.assertEqual(data.upbit_price, upbit_price)
        self.assertEqual(data.binance_price, binance_price)
        self.assertEqual(data.exchange_rate, exchange_rate)

    def test_premium_calculation(self):
        """동기 래퍼"""
        asyncio.run(self.async_test_premium_calculation())

    def test_signal_determination(self):
        """시그널 결정 테스트"""
        test_cases = [
            (6.0, PremiumSignal.STRONG_BUY),  # > 5%
            (4.5, PremiumSignal.BUY),  # > 4%
            (3.0, PremiumSignal.NEUTRAL),  # 2-4%
            (1.5, PremiumSignal.SELL),  # < 2%
            (0.5, PremiumSignal.STRONG_SELL),  # < 1%
            (15.0, PremiumSignal.ANOMALY),  # > 10% (이상치)
            (-5.0, PremiumSignal.STRONG_SELL),  # 역프리미엄
        ]

        for premium_rate, expected_signal in test_cases:
            signal = self.calculator._determine_signal(premium_rate)
            self.assertEqual(
                signal, expected_signal, f"Premium {premium_rate}% should give {expected_signal}"
            )

    def test_confidence_calculation(self):
        """신뢰도 계산 테스트"""
        # 정상 케이스
        confidence = self.calculator._calculate_confidence(
            premium_rate=3.0, liquidity_score=80.0, spread_upbit=0.05, spread_binance=0.02
        )
        self.assertGreater(confidence, 0.8)
        self.assertLessEqual(confidence, 1.0)

        # 이상치 케이스
        confidence = self.calculator._calculate_confidence(
            premium_rate=15.0,  # 이상치
            liquidity_score=80.0,
            spread_upbit=0.05,
            spread_binance=0.02,
        )
        self.assertLess(confidence, 0.6)  # 신뢰도 낮아야 함

        # 높은 스프레드 케이스
        confidence = self.calculator._calculate_confidence(
            premium_rate=3.0,
            liquidity_score=80.0,
            spread_upbit=0.15,  # 높은 스프레드
            spread_binance=0.10,
        )
        self.assertLess(confidence, 0.8)  # 스프레드 때문에 낮아짐

    def test_history_update(self):
        """히스토리 업데이트 테스트"""
        # 빈 히스토리에서 시작
        self.assertEqual(len(self.calculator.premium_history), 0)

        # 데이터 추가
        test_data = KimchiPremiumData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400.0,
            premium_rate=2.0,
            premium_krw=2_000_000,
            signal=PremiumSignal.NEUTRAL,
            liquidity_score=75.0,
            spread_upbit=0.05,
            spread_binance=0.02,
            confidence=0.85,
        )

        self.calculator._update_history(test_data)

        # 검증
        self.assertEqual(len(self.calculator.premium_history), 1)
        self.assertEqual(len(self.calculator.ma_short), 1)
        self.assertEqual(len(self.calculator.ma_long), 1)

    def test_moving_average_cross(self):
        """이동평균 교차 테스트"""
        # 데이터 부족 시
        self.assertIsNone(self.calculator.get_ma_cross_signal())

        # 충분한 데이터 추가 (60개)
        for i in range(60):
            if i < 30:
                rate = 2.0 + i * 0.1  # 상승 추세
            else:
                rate = 5.0 - (i - 30) * 0.1  # 하락 추세

            self.calculator.ma_short.append(rate)
            self.calculator.ma_long.append(rate)

        # MA 교차 시그널 확인
        signal = self.calculator.get_ma_cross_signal()
        self.assertIn(signal, ["golden_cross", "death_cross", "neutral"])

    def test_volatility_calculation(self):
        """변동성 계산 테스트"""
        # 데이터 없을 때
        vol = self.calculator.get_volatility(24)
        self.assertIsNone(vol)

        # 데이터 추가
        now = datetime.now()
        test_rates = [2.0, 2.5, 3.0, 2.8, 3.2, 2.9]

        for i, rate in enumerate(test_rates):
            data = KimchiPremiumData(
                timestamp=now - timedelta(minutes=i),
                upbit_price=100_000_000,
                binance_price=70_000,
                exchange_rate=1400.0,
                premium_rate=rate,
                premium_krw=2_000_000,
                signal=PremiumSignal.NEUTRAL,
                liquidity_score=75.0,
                spread_upbit=0.05,
                spread_binance=0.02,
                confidence=0.85,
            )
            self.calculator.premium_history.append(data)

        # 변동성 계산
        vol = self.calculator.get_volatility(1)
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0)

    def test_anomaly_detection(self):
        """이상치 감지 테스트"""
        # 데이터 부족 시 (임계값 방식)
        self.assertTrue(self.calculator.detect_anomaly(15.0))  # > 10%
        self.assertFalse(self.calculator.detect_anomaly(5.0))  # < 10%

        # 충분한 데이터로 Z-score 테스트
        now = datetime.now()
        for i in range(100):
            rate = 3.0 + (i % 10) * 0.1  # 2.0 ~ 4.0 범위
            data = KimchiPremiumData(
                timestamp=now - timedelta(minutes=i),
                upbit_price=100_000_000,
                binance_price=70_000,
                exchange_rate=1400.0,
                premium_rate=rate,
                premium_krw=2_000_000,
                signal=PremiumSignal.NEUTRAL,
                liquidity_score=75.0,
                spread_upbit=0.05,
                spread_binance=0.02,
                confidence=0.85,
            )
            self.calculator.premium_history.append(data)

        # 정상 범위
        self.assertFalse(self.calculator.detect_anomaly(3.5))

        # 이상치 (3 표준편차 이상)
        self.assertTrue(self.calculator.detect_anomaly(10.0))

    def test_statistics(self):
        """통계 정보 테스트"""
        stats = self.calculator.get_statistics()

        # 기본 필드 확인
        self.assertIn("current_premium", stats)
        self.assertIn("daily_high", stats)
        self.assertIn("daily_low", stats)
        self.assertIn("daily_avg", stats)
        self.assertIn("volatility_24h", stats)
        self.assertIn("ma_signal", stats)
        self.assertIn("history_size", stats)
        self.assertIn("last_calculation", stats)

        # 초기 상태
        self.assertIsNone(stats["current_premium"])
        self.assertEqual(stats["history_size"], 0)


class TestPremiumIntegration(unittest.TestCase):
    """통합 테스트"""

    async def async_test_with_real_exchange_rate(self):
        """실제 환율 API 테스트"""
        calculator = KimchiPremiumCalculator()

        # 실제 환율로 테스트
        data = await calculator.calculate_premium(
            upbit_price=159_400_000, binance_price=115_000, exchange_rate=None  # 자동 조회
        )

        # 결과 확인
        self.assertIsNotNone(data)
        self.assertGreater(data.exchange_rate, 1000)
        self.assertLess(data.exchange_rate, 2000)

        print(f"\n[Integration Test Results]")
        print(f"Exchange Rate: {data.exchange_rate:.2f} KRW/USD")
        print(f"Kimchi Premium: {data.premium_rate:.2f}%")
        print(f"Signal: {data.signal.value}")
        print(f"Confidence: {data.confidence:.2%}")

    def test_with_real_exchange_rate(self):
        """동기 래퍼"""
        asyncio.run(self.async_test_with_real_exchange_rate())


def run_tests():
    """테스트 실행"""
    print("\n" + "=" * 60)
    print("KIMCHI PREMIUM CALCULATOR TEST SUITE")
    print("=" * 60)

    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestKimchiPremiumCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestPremiumIntegration))

    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "-" * 60)
    if result.wasSuccessful():
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"[FAIL] {len(result.failures)} tests failed")
        print(f"[ERROR] {len(result.errors)} tests had errors")
        return 1


if __name__ == "__main__":
    exit(run_tests())
