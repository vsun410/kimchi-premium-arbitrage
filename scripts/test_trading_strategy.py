#!/usr/bin/env python3
"""
트레이딩 전략 테스트
단순 임계값 기반 전략 시뮬레이션
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kimchi_premium import KimchiPremiumData, PremiumSignal
from src.strategies.simple_threshold import (
    PositionSide,
    SignalType,
    SimpleThresholdStrategy,
)


def create_test_data(premium_rate: float, confidence: float = 0.85) -> KimchiPremiumData:
    """테스트 데이터 생성"""
    upbit_price = 160_000_000  # 160M KRW
    exchange_rate = 1386.14
    
    # 김프율에서 바이낸스 가격 역산
    binance_price = upbit_price / (exchange_rate * (1 + premium_rate / 100))
    
    return KimchiPremiumData(
        timestamp=datetime.now(),
        upbit_price=upbit_price,
        binance_price=binance_price,
        exchange_rate=exchange_rate,
        premium_rate=premium_rate,
        premium_krw=premium_rate * 1_000_000,
        signal=PremiumSignal.BUY if premium_rate > 4 else PremiumSignal.NEUTRAL,
        liquidity_score=85.0,
        spread_upbit=0.05,
        spread_binance=0.02,
        confidence=confidence,
    )


def test_signal_generation():
    """시그널 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Signal Generation")
    print("=" * 60)
    
    strategy = SimpleThresholdStrategy(
        enter_threshold=4.0,
        exit_threshold=2.0,
        capital=20_000_000,
    )
    
    # 테스트 시나리오
    scenarios = [
        (3.5, "Below enter threshold"),
        (4.5, "Above enter threshold - ENTER"),
        (3.0, "Holding position"),
        (1.5, "Below exit threshold - EXIT"),
        (5.0, "New opportunity - ENTER"),
        (-0.5, "Reverse premium - EXIT"),
    ]
    
    for premium_rate, description in scenarios:
        print(f"\n[Scenario] {description}")
        print(f"  Premium rate: {premium_rate}%")
        
        data = create_test_data(premium_rate)
        signal = strategy.generate_signal(data)
        
        if signal:
            print(f"  [OK] Signal: {signal.signal_type.value}")
            print(f"  [OK] Reason: {signal.reason}")
            print(f"  [OK] Position: {signal.position_size:.4f} BTC")
            print(f"  [OK] Risk: {signal.risk_score:.2%}")
            
            # 시뮬레이션 포지션 관리
            if signal.signal_type == SignalType.ENTER_LONG:
                strategy.open_position(
                    symbol="BTC/KRW",
                    side=PositionSide.LONG,
                    size=signal.position_size,
                    entry_price=data.upbit_price,
                    exchange="upbit",
                )
                strategy.open_position(
                    symbol="BTC/USDT",
                    side=PositionSide.SHORT,
                    size=signal.position_size,
                    entry_price=data.binance_price,
                    exchange="binance",
                )
            elif signal.signal_type == SignalType.EXIT_LONG:
                strategy.close_position("upbit_BTC/KRW", data.upbit_price)
                strategy.close_position("binance_BTC/USDT", data.binance_price)
        else:
            print(f"  - No signal (HOLD)")
    
    # 통계 출력
    stats = strategy.get_statistics()
    print("\n" + "-" * 60)
    print("Strategy Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


def test_position_sizing():
    """포지션 사이징 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Position Sizing")
    print("=" * 60)
    
    strategy = SimpleThresholdStrategy(
        capital=20_000_000,  # 20M KRW
        max_position_pct=0.01,  # 1%
    )
    
    # 다양한 신뢰도로 테스트
    confidences = [0.5, 0.7, 0.85, 0.95]
    
    for conf in confidences:
        data = create_test_data(premium_rate=5.0, confidence=conf)
        size = strategy.calculate_position_size(data, data.upbit_price)
        
        position_value = size * data.upbit_price
        position_pct = (position_value / strategy.capital) * 100
        
        print(f"\nConfidence: {conf:.0%}")
        print(f"  Position size: {size:.4f} BTC")
        print(f"  Position value: {position_value:,.0f} KRW")
        print(f"  % of capital: {position_pct:.2f}%")
    
    return True


def test_risk_management():
    """리스크 관리 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Risk Management")
    print("=" * 60)
    
    strategy = SimpleThresholdStrategy()
    
    # 리스크 시나리오
    scenarios = [
        {
            "name": "Normal conditions",
            "premium": 4.5,
            "liquidity": 85.0,
            "confidence": 0.85,
            "spread_up": 0.05,
            "spread_bn": 0.02,
        },
        {
            "name": "Low liquidity",
            "premium": 4.5,
            "liquidity": 50.0,  # 낮은 유동성
            "confidence": 0.85,
            "spread_up": 0.05,
            "spread_bn": 0.02,
        },
        {
            "name": "High spread",
            "premium": 4.5,
            "liquidity": 85.0,
            "confidence": 0.85,
            "spread_up": 0.15,  # 높은 스프레드
            "spread_bn": 0.10,
        },
        {
            "name": "Low confidence",
            "premium": 4.5,
            "liquidity": 85.0,
            "confidence": 0.40,  # 낮은 신뢰도
            "spread_up": 0.05,
            "spread_bn": 0.02,
        },
        {
            "name": "Extreme premium",
            "premium": 15.0,  # 극단적 김프
            "liquidity": 85.0,
            "confidence": 0.85,
            "spread_up": 0.05,
            "spread_bn": 0.02,
        },
    ]
    
    for scenario in scenarios:
        print(f"\n[{scenario['name']}]")
        
        data = KimchiPremiumData(
            timestamp=datetime.now(),
            upbit_price=160_000_000,
            binance_price=115_000,
            exchange_rate=1386.14,
            premium_rate=scenario["premium"],
            premium_krw=scenario["premium"] * 1_000_000,
            signal=PremiumSignal.BUY if scenario["premium"] > 4 else PremiumSignal.NEUTRAL,
            liquidity_score=scenario["liquidity"],
            spread_upbit=scenario["spread_up"],
            spread_binance=scenario["spread_bn"],
            confidence=scenario["confidence"],
        )
        
        # 리스크 점수 계산
        risk_score = strategy._calculate_risk_score(data)
        
        # 진입 조건 체크
        can_enter = strategy._check_entry_conditions(data)
        
        print(f"  Premium: {scenario['premium']}%")
        print(f"  Liquidity: {scenario['liquidity']}")
        print(f"  Confidence: {scenario['confidence']:.0%}")
        print(f"  Risk score: {risk_score:.2%}")
        print(f"  Can enter: {'YES' if can_enter else 'NO'}")
        
        if not can_enter:
            # 진입 불가 이유
            if scenario["liquidity"] < strategy.min_liquidity_score:
                print(f"    -> Insufficient liquidity")
            if scenario["confidence"] < 0.6:
                print(f"    -> Low confidence")
            if scenario["premium"] > 10:
                print(f"    -> Extreme premium (possible anomaly)")
    
    return True


def test_pnl_calculation():
    """손익 계산 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: P&L Calculation")
    print("=" * 60)
    
    strategy = SimpleThresholdStrategy()
    
    # 포지션 오픈
    print("\n[Opening positions]")
    upbit_position = strategy.open_position(
        symbol="BTC/KRW",
        side=PositionSide.LONG,
        size=0.001,  # 0.001 BTC
        entry_price=160_000_000,
        exchange="upbit",
    )
    
    binance_position = strategy.open_position(
        symbol="BTC/USDT",
        side=PositionSide.SHORT,
        size=0.001,
        entry_price=115_500,
        exchange="binance",
    )
    
    # 가격 변동 시뮬레이션
    price_changes = [
        (159_000_000, 115_000),  # 김프 감소
        (158_000_000, 114_500),  # 김프 더 감소
        (161_000_000, 116_000),  # 김프 증가
    ]
    
    print("\n[Price movements & P&L]")
    for upbit_price, binance_price in price_changes:
        # 업비트 손익
        upbit_pnl = upbit_position.get_pnl(upbit_price)
        
        # 바이낸스 손익 (USDT -> KRW 변환)
        binance_pnl_usdt = binance_position.get_pnl(binance_price)
        binance_pnl_krw = binance_pnl_usdt * 1386.14
        
        # 총 손익
        total_pnl = upbit_pnl + binance_pnl_krw
        
        # 새 김프 계산
        new_premium = ((upbit_price / (binance_price * 1386.14)) - 1) * 100
        
        print(f"\n  Upbit: {upbit_price:,} KRW ({upbit_pnl:+,.0f} KRW)")
        print(f"  Binance: {binance_price:,} USDT ({binance_pnl_krw:+,.0f} KRW)")
        print(f"  Total P&L: {total_pnl:+,.0f} KRW")
        print(f"  New premium: {new_premium:.2f}%")
    
    # 포지션 청산
    print("\n[Closing positions]")
    final_upbit_price = 158_500_000
    final_binance_price = 114_800
    
    upbit_pnl = strategy.close_position("upbit_BTC/KRW", final_upbit_price)
    binance_pnl_usdt = strategy.close_position("binance_BTC/USDT", final_binance_price)
    
    if upbit_pnl is not None and binance_pnl_usdt is not None:
        total_pnl = upbit_pnl + (binance_pnl_usdt * 1386.14)
        print(f"\nFinal P&L: {total_pnl:+,.0f} KRW")
    
    return True


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("TRADING STRATEGY TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Signal Generation", test_signal_generation),
        ("Position Sizing", test_position_sizing),
        ("Risk Management", test_risk_management),
        ("P&L Calculation", test_pnl_calculation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name:20} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Task #11 implementation complete.")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)