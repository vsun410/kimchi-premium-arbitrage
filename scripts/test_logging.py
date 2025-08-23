#!/usr/bin/env python3
"""
로깅 및 모니터링 시스템 테스트
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger, trading_logger, log_execution_time
from src.utils.metrics import metrics_collector, MetricsTimer
from src.utils.alerts import alert_manager, AlertLevel, AlertChannel


def test_basic_logging():
    """기본 로깅 테스트"""
    print("\n[TEST] Basic Logging")
    print("-" * 50)
    
    # 다양한 레벨 로그
    logger.debug("Debug message - detailed information")
    logger.info("Info message - general information")
    logger.warning("Warning message - potential issue")
    
    # 추가 필드와 함께 로깅
    logger.info("Trade execution started", 
               exchange="upbit", 
               symbol="BTC/KRW",
               action="BUY")
    
    print("[OK] Basic logging test completed")


def test_trading_logger():
    """거래 전용 로거 테스트"""
    print("\n[TEST] Trading Logger")
    print("-" * 50)
    
    # 거래 신호 로그
    trading_logger.log_signal(
        signal_type="ENTER",
        confidence=0.85,
        kimchi_premium=4.5,
        reason="Premium threshold exceeded"
    )
    
    # 주문 로그
    trading_logger.log_order(
        order_id="ORDER-123456",
        exchange="upbit",
        symbol="BTC/KRW",
        side="buy",
        price=100000000,
        amount=0.01,
        order_type="limit"
    )
    
    # 포지션 로그
    trading_logger.log_position(
        position_id="POS-789",
        status="opened",
        pnl=0,
        entry_price=100000000,
        current_price=100500000
    )
    
    # 리스크 경고
    trading_logger.log_risk_alert(
        alert_type="HIGH_VOLATILITY",
        message="Market volatility exceeds safe threshold",
        volatility=0.05,
        threshold=0.03
    )
    
    print("[OK] Trading logger test completed")


def test_metrics_collection():
    """메트릭 수집 테스트"""
    print("\n[TEST] Metrics Collection")
    print("-" * 50)
    
    # 시스템 메트릭 수집
    system_metrics = metrics_collector.collect_system_metrics()
    if system_metrics:
        print(f"CPU Usage: {system_metrics.cpu_percent}%")
        print(f"Memory Usage: {system_metrics.memory_percent}%")
        print(f"Disk Usage: {system_metrics.disk_percent}%")
    
    # API 지연시간 기록
    metrics_collector.record_api_latency("upbit", "fetch_ticker", 25.5)
    metrics_collector.record_api_latency("binance", "fetch_ticker", 15.3)
    metrics_collector.record_api_latency("upbit", "fetch_balance", 30.2)
    
    # 평균 지연시간 조회
    avg_latency = metrics_collector.get_average_latency("upbit", "fetch_ticker")
    if avg_latency:
        print(f"Average Upbit ticker latency: {avg_latency:.2f}ms")
    
    # 거래 기록
    metrics_collector.record_trade(
        exchange="upbit",
        symbol="BTC/KRW",
        side="buy",
        amount=0.01,
        price=100000000,
        success=True
    )
    
    # 김프 업데이트
    metrics_collector.update_kimchi_premium(4.5)
    
    # 잔고 업데이트
    metrics_collector.update_balance("upbit", "KRW", 10000000)
    metrics_collector.update_balance("binance", "USDT", 7000)
    
    # 손익 기록
    metrics_collector.record_pnl(150000)
    
    # 에러 기록
    metrics_collector.record_error("API_TIMEOUT", "upbit", "Connection timeout after 30s")
    
    print("[OK] Metrics collection test completed")


def test_metrics_timer():
    """메트릭 타이머 테스트"""
    print("\n[TEST] Metrics Timer")
    print("-" * 50)
    
    # API 호출 시뮬레이션
    with MetricsTimer("upbit", "fetch_orderbook"):
        time.sleep(0.05)  # 50ms 대기
        print("Simulating API call...")
    
    with MetricsTimer("binance", "place_order"):
        time.sleep(0.03)  # 30ms 대기
        print("Simulating order placement...")
    
    print("[OK] Metrics timer test completed")


@log_execution_time
def sample_function_with_decorator():
    """실행 시간 측정 데코레이터 테스트"""
    time.sleep(0.1)
    return "Completed"


def test_execution_time_decorator():
    """실행 시간 데코레이터 테스트"""
    print("\n[TEST] Execution Time Decorator")
    print("-" * 50)
    
    result = sample_function_with_decorator()
    print(f"Function returned: {result}")
    
    print("[OK] Execution time decorator test completed")


async def test_alerts():
    """알림 시스템 테스트"""
    print("\n[TEST] Alert System")
    print("-" * 50)
    
    # 정보 알림
    await alert_manager.send_alert(
        message="System started successfully",
        level=AlertLevel.INFO,
        title="System Status"
    )
    
    # 경고 알림
    await alert_manager.send_alert(
        message="High memory usage detected",
        level=AlertLevel.WARNING,
        title="Resource Warning",
        details={
            "Memory Usage": "85%",
            "Available": "2GB"
        }
    )
    
    # 거래 알림
    await alert_manager.send_trade_alert(
        action="BUY",
        exchange="upbit",
        symbol="BTC/KRW",
        amount=0.01,
        price=100000000,
        profit=150000
    )
    
    # 리스크 알림
    await alert_manager.send_risk_alert(
        risk_type="Liquidity",
        message="Low liquidity detected in orderbook",
        bid_volume=0.5,
        ask_volume=0.3,
        threshold=1.0
    )
    
    print("[OK] Alert system test completed")


def test_metrics_export():
    """메트릭 내보내기 테스트"""
    print("\n[TEST] Metrics Export")
    print("-" * 50)
    
    # 메트릭 내보내기
    metrics_collector.export_metrics()
    
    # 내보낸 파일 확인
    metrics_dir = Path("logs/metrics")
    if metrics_dir.exists():
        files = list(metrics_dir.glob("*.json"))
        print(f"Exported {len(files)} metric files:")
        for file in files:
            print(f"  - {file.name}")
    
    print("[OK] Metrics export test completed")


def test_system_status():
    """시스템 상태 조회 테스트"""
    print("\n[TEST] System Status")
    print("-" * 50)
    
    # 몇 개의 시스템 메트릭 수집
    for _ in range(3):
        metrics_collector.collect_system_metrics()
        time.sleep(1)
    
    # 시스템 상태 조회
    status = metrics_collector.get_system_status()
    
    if status:
        print("Current System Status:")
        print(f"  CPU: {status['current']['cpu_percent']}%")
        print(f"  Memory: {status['current']['memory_percent']}%")
        print(f"  Disk: {status['current']['disk_percent']}%")
        
        if 'average_5min' in status:
            print("\n5-minute Average:")
            print(f"  CPU: {status['average_5min']['cpu_percent']:.1f}%")
            print(f"  Memory: {status['average_5min']['memory_percent']:.1f}%")
    
    print("[OK] System status test completed")


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("LOGGING AND MONITORING SYSTEM TEST")
    print("=" * 60)
    
    try:
        # 기본 로깅 테스트
        test_basic_logging()
        
        # 거래 로거 테스트
        test_trading_logger()
        
        # 메트릭 수집 테스트
        test_metrics_collection()
        
        # 메트릭 타이머 테스트
        test_metrics_timer()
        
        # 실행 시간 데코레이터 테스트
        test_execution_time_decorator()
        
        # 시스템 상태 테스트
        test_system_status()
        
        # 메트릭 내보내기 테스트
        test_metrics_export()
        
        # 알림 시스템 테스트 (환경변수 설정 필요)
        if os.getenv('SLACK_WEBHOOK_URL') or os.getenv('DISCORD_WEBHOOK_URL'):
            print("\n[INFO] Running alert system test...")
            asyncio.run(test_alerts())
        else:
            print("\n[INFO] Skipping alert test (no webhooks configured)")
        
        # 백그라운드 수집기 테스트
        print("\n[TEST] Background Collector")
        print("-" * 50)
        metrics_collector.start_collector()
        print("Background collector started")
        time.sleep(2)
        metrics_collector.stop_collector()
        print("Background collector stopped")
        print("[OK] Background collector test completed")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # 로그 파일 위치 안내
        print("\nLog files location:")
        print(f"  - Application logs: logs/")
        print(f"  - Metrics data: logs/metrics/")
        print("\nCloudWatch config: configs/cloudwatch-config.json")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        logger.error(f"Test failure: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())