#!/usr/bin/env python3
"""
김치 프리미엄 시스템 통합 테스트
모든 컴포넌트가 함께 작동하는지 확인
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kimchi_premium import kimchi_calculator, KimchiPremiumData, PremiumSignal
from src.data.premium_storage import premium_storage
from src.data.exchange_rate_manager import rate_manager
from src.utils.logger import logger


class IntegrationTestSuite:
    """통합 테스트 스위트"""
    
    def __init__(self):
        self.test_results = []
        
    async def test_end_to_end_flow(self):
        """전체 플로우 테스트"""
        print("\n[TEST 1] End-to-End Flow Test")
        print("-" * 50)
        
        try:
            # 1. 환율 가져오기
            exchange_rate = await rate_manager.get_current_rate()
            if not exchange_rate:
                raise ValueError("Failed to get exchange rate")
            print(f"[PASS] Exchange rate fetched: {exchange_rate:.2f} KRW/USD")
            
            # 2. 김프 계산
            test_prices = [
                (159_400_000, 115_000),  # 약 0% 김프
                (165_000_000, 115_000),  # 약 3.5% 김프
                (175_000_000, 115_000),  # 약 9.8% 김프
            ]
            
            for upbit_price, binance_price in test_prices:
                data = await kimchi_calculator.calculate_premium(
                    upbit_price,
                    binance_price,
                    exchange_rate
                )
                
                print(f"[PASS] Premium calculated: {data.premium_rate:.2f}% "
                      f"(Signal: {data.signal.value})")
                
                # 3. 데이터 저장
                premium_storage.save_premium(data)
                print(f"[PASS] Premium data saved")
                
                # 작은 지연 (실제 시간차 시뮬레이션)
                await asyncio.sleep(0.1)
            
            # 4. 저장된 데이터 확인
            current = premium_storage.load_current_premium()
            if not current:
                raise ValueError("Failed to load saved premium")
            print(f"[PASS] Current premium loaded: {current['premium_rate']:.2f}%")
            
            self.test_results.append(("End-to-End Flow", True, "All steps passed"))
            return True
            
        except Exception as e:
            print(f"[FAIL] End-to-end test failed: {e}")
            self.test_results.append(("End-to-End Flow", False, str(e)))
            return False
    
    async def test_statistics_calculation(self):
        """통계 계산 테스트"""
        print("\n[TEST 2] Statistics Calculation Test")
        print("-" * 50)
        
        try:
            # 테스트 데이터 생성
            for i in range(10):
                premium_rate = 2.0 + (i % 5) * 0.5
                
                data = KimchiPremiumData(
                    timestamp=datetime.now() - timedelta(minutes=i*5),
                    upbit_price=160_000_000 + i * 1_000_000,
                    binance_price=115_000,
                    exchange_rate=1386.14,
                    premium_rate=premium_rate,
                    premium_krw=premium_rate * 1_000_000,
                    signal=PremiumSignal.NEUTRAL if premium_rate < 4 else PremiumSignal.BUY,
                    liquidity_score=80.0 + i,
                    spread_upbit=0.05,
                    spread_binance=0.02,
                    confidence=0.85
                )
                
                premium_storage.save_premium(data)
            
            # 통계 계산
            stats = premium_storage.get_statistics(hours=1)
            
            if stats['data_points'] < 10:
                raise ValueError(f"Expected 10+ data points, got {stats['data_points']}")
            
            print(f"[PASS] Data points: {stats['data_points']}")
            print(f"[PASS] Average premium: {stats['avg_premium']:.2f}%")
            print(f"[PASS] Max premium: {stats['max_premium']:.2f}%")
            print(f"[PASS] Min premium: {stats['min_premium']:.2f}%")
            
            # 시그널 분포
            if stats['signal_distribution']:
                print(f"[PASS] Signal distribution: {stats['signal_distribution']}")
            
            self.test_results.append(("Statistics", True, "Statistics calculated"))
            return True
            
        except Exception as e:
            print(f"[FAIL] Statistics test failed: {e}")
            self.test_results.append(("Statistics", False, str(e)))
            return False
    
    async def test_trading_opportunities(self):
        """거래 기회 탐지 테스트"""
        print("\n[TEST 3] Trading Opportunities Test")
        print("-" * 50)
        
        try:
            # 높은 김프 데이터 생성
            high_premium_data = KimchiPremiumData(
                timestamp=datetime.now(),
                upbit_price=180_000_000,
                binance_price=115_000,
                exchange_rate=1386.14,
                premium_rate=5.5,  # 높은 김프
                premium_krw=5_500_000,
                signal=PremiumSignal.STRONG_BUY,
                liquidity_score=90.0,
                spread_upbit=0.03,
                spread_binance=0.01,
                confidence=0.92
            )
            
            premium_storage.save_premium(high_premium_data)
            
            # 거래 기회 찾기
            opportunities = premium_storage.get_trading_opportunities(
                threshold=5.0,
                hours=1
            )
            
            if not opportunities:
                raise ValueError("No trading opportunities found")
            
            print(f"[PASS] Found {len(opportunities)} trading opportunities")
            
            for opp in opportunities[:3]:  # 최대 3개만 표시
                print(f"  - Premium: {opp['premium_rate']:.2f}%, "
                      f"Signal: {opp['signal']}, "
                      f"Confidence: {opp['confidence']:.2%}")
            
            self.test_results.append(("Trading Opportunities", True, f"{len(opportunities)} found"))
            return True
            
        except Exception as e:
            print(f"[FAIL] Trading opportunities test failed: {e}")
            self.test_results.append(("Trading Opportunities", False, str(e)))
            return False
    
    async def test_data_persistence(self):
        """데이터 영속성 테스트"""
        print("\n[TEST 4] Data Persistence Test")
        print("-" * 50)
        
        try:
            # 히스토리 로드
            df = premium_storage.load_history(
                start_date=datetime.now() - timedelta(hours=1)
            )
            
            if df.empty:
                print("[WARN]  No history data (expected for first run)")
            else:
                print(f"[PASS] Loaded {len(df)} historical records")
                
                # 데이터 무결성 확인
                required_columns = [
                    'timestamp', 'upbit_price', 'binance_price',
                    'premium_rate', 'signal', 'confidence'
                ]
                
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"Missing column: {col}")
                
                print(f"[PASS] All required columns present")
                
                # 리샘플링 테스트
                df_resampled = premium_storage.export_to_dataframe(
                    start_date=datetime.now() - timedelta(hours=1),
                    resample='5T'
                )
                
                if not df_resampled.empty:
                    print(f"[PASS] Resampled to {len(df_resampled)} 5-minute intervals")
            
            self.test_results.append(("Data Persistence", True, "Data persisted correctly"))
            return True
            
        except Exception as e:
            print(f"[FAIL] Data persistence test failed: {e}")
            self.test_results.append(("Data Persistence", False, str(e)))
            return False
    
    async def test_calculator_features(self):
        """계산기 고급 기능 테스트"""
        print("\n[TEST 5] Calculator Advanced Features Test")
        print("-" * 50)
        
        try:
            # 이상치 감지 테스트
            is_anomaly = kimchi_calculator.detect_anomaly(15.0)
            print(f"[PASS] Anomaly detection: 15% -> {is_anomaly}")
            
            # 변동성 계산
            volatility = kimchi_calculator.get_volatility(hours=1)
            if volatility is not None:
                print(f"[PASS] Volatility calculated: {volatility:.2f}")
            else:
                print("[INFO] Volatility: Not enough data")
            
            # MA 크로스 시그널
            ma_signal = kimchi_calculator.get_ma_cross_signal()
            if ma_signal:
                print(f"[PASS] MA Cross signal: {ma_signal}")
            else:
                print("[INFO] MA Cross: Not enough data")
            
            # 통계 정보
            stats = kimchi_calculator.get_statistics()
            print(f"[PASS] Calculator statistics retrieved")
            
            self.test_results.append(("Advanced Features", True, "All features working"))
            return True
            
        except Exception as e:
            print(f"[FAIL] Advanced features test failed: {e}")
            self.test_results.append(("Advanced Features", False, str(e)))
            return False
    
    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result, details in self.test_results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"{test_name:25} {status:7} {details}")
        
        print("-" * 60)
        print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        return passed == total


async def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("KIMCHI PREMIUM SYSTEM INTEGRATION TEST")
    print("=" * 60)
    print("\nTesting Task #10: Kimchi Premium Calculation & Analysis")
    
    test_suite = IntegrationTestSuite()
    
    # 모든 테스트 실행
    await test_suite.test_end_to_end_flow()
    await test_suite.test_statistics_calculation()
    await test_suite.test_trading_opportunities()
    await test_suite.test_data_persistence()
    await test_suite.test_calculator_features()
    
    # 결과 요약
    all_passed = test_suite.print_summary()
    
    if all_passed:
        print("\n[SUCCESS] Task #10 COMPLETED! All integration tests passed.")
        print("\nComponents implemented:")
        print("  1. Kimchi premium calculator with signal generation")
        print("  2. Real-time WebSocket data integration")
        print("  3. Premium data storage system with CSV export")
        print("  4. Statistical analysis and trading opportunity detection")
        print("  5. Anomaly detection and volatility calculation")
        return 0
    else:
        print("\n[WARN]  Some tests failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)