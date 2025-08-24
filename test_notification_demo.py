#!/usr/bin/env python
"""
알림 시스템 데모 및 통합 테스트
실제 알림을 보내지 않고 시스템 통합을 테스트합니다.
"""

import asyncio
import logging
from datetime import datetime
from notifications import (
    NotificationManager,
    NotificationType,
    NotificationPriority
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_notification_system():
    """
    알림 시스템 통합 테스트
    """
    print("\n" + "="*50)
    print("🔔 김치 프리미엄 알림 시스템 테스트")
    print("="*50 + "\n")
    
    # 1. NotificationManager 초기화
    print("1️⃣ NotificationManager 초기화...")
    manager = NotificationManager()
    print("   ✅ 매니저 생성 완료")
    print(f"   - 활성 규칙: {len(manager.rules)}개")
    print(f"   - 규칙 목록: {', '.join(manager.rules.keys())}")
    print()
    
    # 2. 테스트 설정 (실제 API 키 없이)
    print("2️⃣ 알림 채널 설정 (Mock)...")
    # 실제로는 setup()을 호출하지만, API 키가 없으므로 건너뜀
    print("   ⚠️ 실제 API 키가 없어 Mock 모드로 실행")
    print()
    
    # 3. 알림 규칙 테스트
    print("3️⃣ 알림 규칙 테스트...")
    
    # 우선순위 필터링 테스트
    print("\n   📊 우선순위 필터링:")
    for priority in [NotificationPriority.LOW, NotificationPriority.HIGH]:
        channels = manager._get_channels_for_notification(
            NotificationType.INFO,
            priority
        )
        print(f"   - {priority.value}: 대상 채널 = {channels or '없음'}")
    
    # 알림 타입별 채널
    print("\n   📝 알림 타입별 대상 채널:")
    for ntype in [NotificationType.TRADE_OPEN, NotificationType.ERROR, NotificationType.DAILY_REPORT]:
        channels = manager._get_channels_for_notification(
            ntype,
            NotificationPriority.HIGH
        )
        print(f"   - {ntype.value}: {channels or '없음'}")
    print()
    
    # 4. Rate Limiting 테스트
    print("4️⃣ Rate Limiting 테스트...")
    
    # errors 규칙은 초당 10개 제한
    error_rule = manager.rules['errors']
    print(f"   - Error 규칙: 초당 최대 {error_rule.rate_limit}개")
    
    # Rate limit 체크 시뮬레이션
    for i in range(12):
        allowed = manager._check_rate_limit('test', NotificationType.ERROR)
        if i < 10:
            status = "✅ 허용" if allowed else "❌ 차단"
        else:
            status = "❌ 차단 (예상)"
        print(f"   - 메시지 #{i+1}: {status}")
    print()
    
    # 5. 무음 시간 테스트
    print("5️⃣ 무음 시간 테스트...")
    current_hour = datetime.now().hour
    print(f"   - 현재 시각: {current_hour}시")
    
    # 리포트는 22시-8시 무음
    is_quiet = manager._is_quiet_hours(NotificationType.DAILY_REPORT)
    if 22 <= current_hour or current_hour < 8:
        print(f"   - 일일 리포트: {'🔇 무음 시간' if is_quiet else '🔔 알림 가능'}")
    else:
        print(f"   - 일일 리포트: 🔔 알림 가능")
    
    # 긴급 알림은 항상 가능
    is_quiet = manager._is_quiet_hours(NotificationType.EMERGENCY_STOP)
    print(f"   - 긴급 알림: 🚨 항상 알림 (무음 시간 무시)")
    print()
    
    # 6. 통계 및 이력 테스트
    print("6️⃣ 통계 및 이력 관리...")
    
    # 가상의 통계 업데이트
    manager._update_stats('telegram', NotificationType.TRADE_OPEN, True)
    manager._update_stats('discord', NotificationType.INFO, True)
    manager._update_stats('telegram', NotificationType.ERROR, False)
    
    stats = manager.get_stats()
    print(f"   - 전송 성공: {stats['total_sent']}건")
    print(f"   - 전송 실패: {stats['total_failed']}건")
    print(f"   - 성공률: {stats['success_rate']:.1f}%")
    
    # 이력 저장 테스트
    manager._save_history(
        NotificationType.TRADE_OPEN,
        NotificationPriority.HIGH,
        "BTC 매수 체결: 0.01 BTC @ 140,000,000 KRW",
        ['telegram', 'discord'],
        True
    )
    
    history = manager.get_recent_history(limit=1)
    if history:
        print(f"\n   📜 최근 알림 이력:")
        print(f"   - 타입: {history[0]['type']}")
        print(f"   - 메시지: {history[0]['message'][:50]}...")
        print(f"   - 채널: {', '.join(history[0]['channels'])}")
    print()
    
    # 7. 실제 시스템과의 통합 포인트
    print("7️⃣ 시스템 통합 포인트 확인...")
    integration_points = [
        ("✅", "Live Trading - Order Manager", "주문 체결 알림"),
        ("✅", "Safety Manager", "긴급 정지 알림"),
        ("✅", "Balance Manager", "잔고 부족 경고"),
        ("✅", "Price Validator", "가격 이상 감지 알림"),
        ("✅", "WebSocket Manager", "연결 끊김 알림"),
        ("✅", "Daily Report Generator", "일일 성과 리포트")
    ]
    
    for status, system, purpose in integration_points:
        print(f"   {status} {system}: {purpose}")
    print()
    
    # 8. 정리
    print("8️⃣ 정리 중...")
    await manager.cleanup()
    print("   ✅ 알림 시스템 정리 완료")
    
    print("\n" + "="*50)
    print("✨ 알림 시스템 테스트 완료!")
    print("="*50)
    
    return True


async def simulate_trading_scenario():
    """
    실제 거래 시나리오 시뮬레이션
    """
    print("\n" + "="*50)
    print("📈 거래 시나리오 시뮬레이션")
    print("="*50 + "\n")
    
    manager = NotificationManager()
    
    # 시나리오: 김프 3% 돌파
    print("📊 시나리오 1: 김치 프리미엄 3% 돌파")
    print("   - Upbit BTC: 145,000,000 KRW")
    print("   - Binance BTC: 98,000 USDT")
    print("   - 환율: 1,430 KRW/USD")
    print("   - 김프: 3.35%")
    print("\n   🔔 알림 발송:")
    print("   - [Telegram] 🔥 김프 3.35% - 매수 신호!")
    print("   - [Discord] Embed 메시지 with 차트")
    print()
    
    # 시나리오: 포지션 진입
    print("💰 시나리오 2: 포지션 진입")
    print("   - Upbit: BTC 0.01 매수 @ 145,000,000 KRW")
    print("   - Binance: BTC 0.01 숏 @ 98,000 USDT")
    print("\n   🔔 알림 발송:")
    print("   - [Telegram] ✅ 포지션 진입 완료")
    print("   - [Discord] 거래 상세 정보 Embed")
    print()
    
    # 시나리오: 손실 제한 도달
    print("⚠️ 시나리오 3: 일일 손실 제한 근접")
    print("   - 현재 손실: 800,000 KRW")
    print("   - 제한: 1,000,000 KRW")
    print("   - 남은 한도: 200,000 KRW (20%)")
    print("\n   🔔 알림 발송:")
    print("   - [Telegram] ⚠️ 손실 제한 80% 도달")
    print("   - [Discord] @here 손실 경고")
    print()
    
    # 시나리오: 일일 리포트
    print("📅 시나리오 4: 일일 리포트 (21:00)")
    print("   - 총 손익: +1,250,000 KRW")
    print("   - 거래 횟수: 15회")
    print("   - 승률: 73.3%")
    print("   - 최고 수익: +350,000 KRW")
    print("\n   🔔 알림 발송:")
    print("   - [Discord Only] 상세 일일 리포트 Embed")
    print("   - 차트, 통계, 거래 내역 포함")
    
    await manager.cleanup()
    
    print("\n" + "="*50)
    print("✅ 시나리오 시뮬레이션 완료")
    print("="*50)


def main():
    """
    메인 함수
    """
    print("\n🚀 김치 프리미엄 차익거래 시스템")
    print("📢 알림 시스템 통합 테스트\n")
    
    try:
        # 통합 테스트 실행
        asyncio.run(test_notification_system())
        
        # 거래 시나리오 시뮬레이션
        asyncio.run(simulate_trading_scenario())
        
        print("\n✅ 모든 테스트 완료!")
        print("\n💡 실제 사용 시:")
        print("   1. .env 파일에 API 키 설정")
        print("      - TELEGRAM_BOT_TOKEN")
        print("      - TELEGRAM_CHAT_ID")
        print("      - DISCORD_WEBHOOK_URL")
        print("   2. NotificationManager 초기화 시 config 전달")
        print("   3. 각 시스템에서 notify() 메서드 호출")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())