"""
알림 시스템 베이스 클래스
모든 알림 채널이 구현해야 할 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """알림 우선순위"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(Enum):
    """알림 타입"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    DAILY_REPORT = "daily_report"
    EMERGENCY_STOP = "emergency_stop"


class BaseNotifier(ABC):
    """
    알림 시스템 베이스 클래스
    
    모든 알림 채널(Telegram, Discord 등)이 상속받아 구현
    """
    
    def __init__(self, name: str):
        """
        초기화
        
        Args:
            name: 알림 채널 이름
        """
        self.name = name
        self.enabled = False
        self.stats = {
            'sent': 0,
            'failed': 0,
            'last_sent': None
        }
        
        logger.info(f"{name} notifier initialized")
    
    @abstractmethod
    async def setup(self, config: Dict[str, Any]) -> bool:
        """
        알림 채널 설정
        
        Args:
            config: 설정 정보
            
        Returns:
            설정 성공 여부
        """
        pass
    
    @abstractmethod
    async def send_message(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        **kwargs
    ) -> bool:
        """
        메시지 전송
        
        Args:
            message: 메시지 내용
            notification_type: 알림 타입
            priority: 우선순위
            **kwargs: 추가 파라미터
            
        Returns:
            전송 성공 여부
        """
        pass
    
    @abstractmethod
    async def send_trade_alert(
        self,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        reason: str = "",
        **kwargs
    ) -> bool:
        """
        거래 알림 전송
        
        Args:
            action: 거래 액션 (BUY/SELL/CLOSE)
            symbol: 심볼
            amount: 수량
            price: 가격
            reason: 거래 사유
            **kwargs: 추가 정보
            
        Returns:
            전송 성공 여부
        """
        pass
    
    @abstractmethod
    async def send_daily_report(
        self,
        date: datetime,
        pnl: float,
        trades_count: int,
        win_rate: float,
        details: Dict[str, Any]
    ) -> bool:
        """
        일일 리포트 전송
        
        Args:
            date: 날짜
            pnl: 손익
            trades_count: 거래 횟수
            win_rate: 승률
            details: 상세 정보
            
        Returns:
            전송 성공 여부
        """
        pass
    
    def format_number(self, value: float, decimals: int = 2) -> str:
        """
        숫자 포맷팅
        
        Args:
            value: 값
            decimals: 소수점 자리수
            
        Returns:
            포맷된 문자열
        """
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:,.{decimals}f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:,.{decimals}f}K"
        else:
            return f"{value:,.{decimals}f}"
    
    def format_percentage(self, value: float) -> str:
        """
        퍼센트 포맷팅
        
        Args:
            value: 값 (0.1 = 10%)
            
        Returns:
            포맷된 문자열
        """
        return f"{value * 100:+.2f}%"
    
    def get_emoji(self, notification_type: NotificationType) -> str:
        """
        알림 타입별 이모지
        
        Args:
            notification_type: 알림 타입
            
        Returns:
            이모지
        """
        emoji_map = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.TRADE_OPEN: "📈",
            NotificationType.TRADE_CLOSE: "📊",
            NotificationType.DAILY_REPORT: "📅",
            NotificationType.EMERGENCY_STOP: "🚨"
        }
        return emoji_map.get(notification_type, "📢")
    
    def get_priority_prefix(self, priority: NotificationPriority) -> str:
        """
        우선순위별 접두사
        
        Args:
            priority: 우선순위
            
        Returns:
            접두사
        """
        prefix_map = {
            NotificationPriority.LOW: "",
            NotificationPriority.MEDIUM: "",
            NotificationPriority.HIGH: "⚡ ",
            NotificationPriority.CRITICAL: "🔴 URGENT: "
        }
        return prefix_map.get(priority, "")
    
    def update_stats(self, success: bool):
        """
        통계 업데이트
        
        Args:
            success: 성공 여부
        """
        if success:
            self.stats['sent'] += 1
            self.stats['last_sent'] = datetime.now()
        else:
            self.stats['failed'] += 1
    
    def get_stats(self) -> Dict:
        """
        통계 조회
        
        Returns:
            통계 정보
        """
        return {
            'channel': self.name,
            'enabled': self.enabled,
            'sent': self.stats['sent'],
            'failed': self.stats['failed'],
            'last_sent': self.stats['last_sent'].isoformat() if self.stats['last_sent'] else None,
            'success_rate': (
                self.stats['sent'] / (self.stats['sent'] + self.stats['failed']) * 100
                if (self.stats['sent'] + self.stats['failed']) > 0
                else 0
            )
        }