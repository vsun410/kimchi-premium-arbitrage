"""
통합 알림 관리자
모든 알림 채널을 관리하고 조정
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os

from .base_notifier import (
    BaseNotifier,
    NotificationType,
    NotificationPriority
)
from .telegram_notifier import TelegramNotifier
from .discord_notifier import DiscordNotifier

logger = logging.getLogger(__name__)


@dataclass
class NotificationRule:
    """알림 규칙"""
    name: str
    enabled: bool = True
    channels: Set[str] = field(default_factory=set)  # 사용할 채널
    types: Set[NotificationType] = field(default_factory=set)  # 허용할 타입
    min_priority: NotificationPriority = NotificationPriority.LOW  # 최소 우선순위
    rate_limit: int = 0  # 초당 최대 메시지 수 (0=무제한)
    quiet_hours: Optional[tuple] = None  # (start_hour, end_hour) 무음 시간


@dataclass
class NotificationHistory:
    """알림 이력"""
    timestamp: datetime
    notification_type: NotificationType
    priority: NotificationPriority
    message: str
    channels_sent: List[str]
    success: bool
    error: Optional[str] = None


class NotificationManager:
    """
    통합 알림 관리자
    
    핵심 기능:
    1. 여러 채널 통합 관리
    2. 알림 규칙 및 필터링
    3. Rate limiting
    4. 무음 시간 관리
    5. 알림 이력 및 통계
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 알림 채널들
        self.notifiers: Dict[str, BaseNotifier] = {}
        
        # 알림 규칙
        self.rules: Dict[str, NotificationRule] = {
            'default': NotificationRule(
                name='default',
                enabled=True,
                channels={'telegram', 'discord'},
                types=set(NotificationType),
                min_priority=NotificationPriority.LOW
            ),
            'trades': NotificationRule(
                name='trades',
                enabled=True,
                channels={'telegram', 'discord'},
                types={NotificationType.TRADE_OPEN, NotificationType.TRADE_CLOSE},
                min_priority=NotificationPriority.MEDIUM
            ),
            'errors': NotificationRule(
                name='errors',
                enabled=True,
                channels={'telegram', 'discord'},
                types={NotificationType.ERROR, NotificationType.EMERGENCY_STOP},
                min_priority=NotificationPriority.HIGH,
                rate_limit=10  # 초당 최대 10개
            ),
            'reports': NotificationRule(
                name='reports',
                enabled=True,
                channels={'discord'},  # 보고서는 Discord로만
                types={NotificationType.DAILY_REPORT},
                min_priority=NotificationPriority.LOW,
                quiet_hours=(22, 8)  # 22시-08시 무음
            )
        }
        
        # Rate limiting
        self.rate_limiter: Dict[str, List[datetime]] = {}
        
        # 알림 이력
        self.history: List[NotificationHistory] = []
        self.max_history = 1000
        
        # 통계
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_type': {},
            'by_channel': {},
            'last_error': None
        }
        
        # 설정 로드
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        logger.info("NotificationManager initialized")
    
    async def setup(self, config: Dict[str, Any]):
        """
        알림 채널 설정
        
        Args:
            config: {
                'telegram': {
                    'bot_token': '...',
                    'chat_ids': [...]
                },
                'discord': {
                    'webhook_urls': [...]
                },
                'rules': {...}  # optional
            }
        """
        # Telegram 설정
        if 'telegram' in config:
            telegram = TelegramNotifier()
            if await telegram.setup(config['telegram']):
                self.notifiers['telegram'] = telegram
                logger.info("Telegram notifier added")
            else:
                logger.error("Failed to setup Telegram notifier")
        
        # Discord 설정
        if 'discord' in config:
            discord = DiscordNotifier()
            if await discord.setup(config['discord']):
                self.notifiers['discord'] = discord
                logger.info("Discord notifier added")
            else:
                logger.error("Failed to setup Discord notifier")
        
        # 규칙 업데이트
        if 'rules' in config:
            self.update_rules(config['rules'])
        
        logger.info(f"NotificationManager setup complete: {len(self.notifiers)} channels active")
    
    def update_rules(self, rules_config: Dict[str, Any]):
        """
        알림 규칙 업데이트
        
        Args:
            rules_config: 규칙 설정
        """
        for rule_name, rule_data in rules_config.items():
            if rule_name in self.rules:
                rule = self.rules[rule_name]
                
                if 'enabled' in rule_data:
                    rule.enabled = rule_data['enabled']
                
                if 'channels' in rule_data:
                    rule.channels = set(rule_data['channels'])
                
                if 'min_priority' in rule_data:
                    rule.min_priority = NotificationPriority[rule_data['min_priority'].upper()]
                
                if 'rate_limit' in rule_data:
                    rule.rate_limit = rule_data['rate_limit']
                
                if 'quiet_hours' in rule_data:
                    rule.quiet_hours = tuple(rule_data['quiet_hours'])
                
                logger.info(f"Updated rule: {rule_name}")
    
    async def notify(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        알림 전송
        
        Args:
            message: 메시지
            notification_type: 알림 타입
            priority: 우선순위
            channels: 특정 채널 지정 (None=규칙에 따라)
            **kwargs: 추가 파라미터
            
        Returns:
            전송 성공 여부
        """
        # 사용할 채널 결정
        if channels:
            target_channels = set(channels) & set(self.notifiers.keys())
        else:
            target_channels = self._get_channels_for_notification(
                notification_type,
                priority
            )
        
        if not target_channels:
            logger.warning("No channels available for notification")
            return False
        
        # 무음 시간 체크
        if self._is_quiet_hours(notification_type):
            logger.info("Notification skipped due to quiet hours")
            return False
        
        # Rate limiting 체크
        for channel in list(target_channels):
            if not self._check_rate_limit(channel, notification_type):
                logger.warning(f"Rate limit exceeded for {channel}")
                target_channels.remove(channel)
        
        if not target_channels:
            logger.warning("All channels rate limited")
            return False
        
        # 채널별 전송
        success_channels = []
        error_message = None
        
        for channel_name in target_channels:
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                continue
            
            try:
                success = await notifier.send_message(
                    message,
                    notification_type,
                    priority,
                    **kwargs
                )
                
                if success:
                    success_channels.append(channel_name)
                    self._update_stats(channel_name, notification_type, True)
                else:
                    self._update_stats(channel_name, notification_type, False)
                    
            except Exception as e:
                logger.error(f"Error sending to {channel_name}: {e}")
                error_message = str(e)
                self._update_stats(channel_name, notification_type, False)
        
        # 이력 저장
        self._save_history(
            notification_type,
            priority,
            message,
            success_channels,
            len(success_channels) > 0,
            error_message
        )
        
        return len(success_channels) > 0
    
    async def notify_trade(
        self,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        reason: str = "",
        **kwargs
    ) -> bool:
        """
        거래 알림
        
        Args:
            action: 거래 액션
            symbol: 심볼
            amount: 수량
            price: 가격
            reason: 사유
            **kwargs: 추가 정보
            
        Returns:
            전송 성공 여부
        """
        notification_type = (
            NotificationType.TRADE_OPEN if action in ['BUY', 'SELL']
            else NotificationType.TRADE_CLOSE
        )
        
        channels = self._get_channels_for_notification(
            notification_type,
            NotificationPriority.HIGH
        )
        
        success_count = 0
        
        for channel_name in channels:
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                continue
            
            try:
                success = await notifier.send_trade_alert(
                    action,
                    symbol,
                    amount,
                    price,
                    reason,
                    **kwargs
                )
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error sending trade alert to {channel_name}: {e}")
        
        return success_count > 0
    
    async def notify_daily_report(
        self,
        date: datetime,
        pnl: float,
        trades_count: int,
        win_rate: float,
        details: Dict[str, Any]
    ) -> bool:
        """
        일일 리포트 알림
        
        Args:
            date: 날짜
            pnl: 손익
            trades_count: 거래 횟수
            win_rate: 승률
            details: 상세 정보
            
        Returns:
            전송 성공 여부
        """
        channels = self._get_channels_for_notification(
            NotificationType.DAILY_REPORT,
            NotificationPriority.MEDIUM
        )
        
        success_count = 0
        
        for channel_name in channels:
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                continue
            
            try:
                success = await notifier.send_daily_report(
                    date,
                    pnl,
                    trades_count,
                    win_rate,
                    details
                )
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Error sending daily report to {channel_name}: {e}")
        
        return success_count > 0
    
    async def notify_emergency(
        self,
        message: str,
        details: Dict[str, Any] = None
    ) -> bool:
        """
        긴급 알림 (모든 채널로 즉시 전송)
        
        Args:
            message: 긴급 메시지
            details: 상세 정보
            
        Returns:
            전송 성공 여부
        """
        # 모든 활성 채널로 전송
        channels = list(self.notifiers.keys())
        
        # 긴급 메시지 구성
        emergency_message = f"🚨 **EMERGENCY ALERT** 🚨\n\n{message}"
        
        if details:
            emergency_message += "\n\n**Details:**"
            for key, value in details.items():
                emergency_message += f"\n• {key}: {value}"
        
        emergency_message += f"\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 전송 (무음 시간, rate limit 무시)
        return await self.notify(
            emergency_message,
            NotificationType.EMERGENCY_STOP,
            NotificationPriority.CRITICAL,
            channels=channels
        )
    
    def _get_channels_for_notification(
        self,
        notification_type: NotificationType,
        priority: NotificationPriority
    ) -> Set[str]:
        """
        알림에 사용할 채널 결정
        
        Args:
            notification_type: 알림 타입
            priority: 우선순위
            
        Returns:
            채널 집합
        """
        channels = set()
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 타입 체크
            if notification_type not in rule.types:
                continue
            
            # 우선순위 체크
            priority_values = {
                NotificationPriority.LOW: 0,
                NotificationPriority.MEDIUM: 1,
                NotificationPriority.HIGH: 2,
                NotificationPriority.CRITICAL: 3
            }
            
            if priority_values[priority] < priority_values[rule.min_priority]:
                continue
            
            # 채널 추가
            channels.update(rule.channels)
        
        # 활성 채널만 필터링
        active_channels = set(self.notifiers.keys())
        return channels & active_channels
    
    def _is_quiet_hours(self, notification_type: NotificationType) -> bool:
        """
        무음 시간 체크
        
        Args:
            notification_type: 알림 타입
            
        Returns:
            무음 시간 여부
        """
        # 긴급 알림은 무음 시간 무시
        if notification_type == NotificationType.EMERGENCY_STOP:
            return False
        
        current_hour = datetime.now().hour
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if notification_type not in rule.types:
                continue
            
            if rule.quiet_hours:
                start, end = rule.quiet_hours
                
                if start < end:
                    # 예: 22-08 (같은 날)
                    if start <= current_hour < end:
                        return True
                else:
                    # 예: 22-08 (다음 날)
                    if current_hour >= start or current_hour < end:
                        return True
        
        return False
    
    def _check_rate_limit(
        self,
        channel: str,
        notification_type: NotificationType
    ) -> bool:
        """
        Rate limit 체크
        
        Args:
            channel: 채널 이름
            notification_type: 알림 타입
            
        Returns:
            허용 여부
        """
        # 긴급 알림은 rate limit 무시
        if notification_type == NotificationType.EMERGENCY_STOP:
            return True
        
        # 규칙별 rate limit 확인
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if notification_type not in rule.types:
                continue
            
            if rule.rate_limit <= 0:
                continue
            
            # Rate limit 계산
            key = f"{channel}_{rule.name}"
            now = datetime.now()
            
            if key not in self.rate_limiter:
                self.rate_limiter[key] = []
            
            # 1초 이내 메시지 필터링
            recent = [
                ts for ts in self.rate_limiter[key]
                if (now - ts).total_seconds() < 1
            ]
            
            if len(recent) >= rule.rate_limit:
                return False
            
            self.rate_limiter[key] = recent + [now]
        
        return True
    
    def _update_stats(
        self,
        channel: str,
        notification_type: NotificationType,
        success: bool
    ):
        """
        통계 업데이트
        
        Args:
            channel: 채널
            notification_type: 알림 타입
            success: 성공 여부
        """
        if success:
            self.stats['total_sent'] += 1
        else:
            self.stats['total_failed'] += 1
        
        # 타입별 통계
        type_key = notification_type.value
        if type_key not in self.stats['by_type']:
            self.stats['by_type'][type_key] = {'sent': 0, 'failed': 0}
        
        if success:
            self.stats['by_type'][type_key]['sent'] += 1
        else:
            self.stats['by_type'][type_key]['failed'] += 1
        
        # 채널별 통계
        if channel not in self.stats['by_channel']:
            self.stats['by_channel'][channel] = {'sent': 0, 'failed': 0}
        
        if success:
            self.stats['by_channel'][channel]['sent'] += 1
        else:
            self.stats['by_channel'][channel]['failed'] += 1
    
    def _save_history(
        self,
        notification_type: NotificationType,
        priority: NotificationPriority,
        message: str,
        channels_sent: List[str],
        success: bool,
        error: Optional[str] = None
    ):
        """
        알림 이력 저장
        
        Args:
            notification_type: 알림 타입
            priority: 우선순위
            message: 메시지
            channels_sent: 전송된 채널
            success: 성공 여부
            error: 에러 메시지
        """
        history_item = NotificationHistory(
            timestamp=datetime.now(),
            notification_type=notification_type,
            priority=priority,
            message=message[:200],  # 길이 제한
            channels_sent=channels_sent,
            success=success,
            error=error
        )
        
        self.history.append(history_item)
        
        # 최대 개수 유지
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        if error:
            self.stats['last_error'] = {
                'timestamp': datetime.now().isoformat(),
                'message': error
            }
    
    def get_stats(self) -> Dict:
        """
        통계 조회
        
        Returns:
            통계 정보
        """
        stats = self.stats.copy()
        
        # 채널별 상태
        stats['channels'] = {}
        for channel_name, notifier in self.notifiers.items():
            stats['channels'][channel_name] = notifier.get_stats()
        
        # 성공률 계산
        total = stats['total_sent'] + stats['total_failed']
        stats['success_rate'] = (
            stats['total_sent'] / total * 100 if total > 0 else 0
        )
        
        return stats
    
    def get_recent_history(self, limit: int = 50) -> List[Dict]:
        """
        최근 알림 이력 조회
        
        Args:
            limit: 조회 개수
            
        Returns:
            알림 이력
        """
        recent = self.history[-limit:] if len(self.history) > limit else self.history
        
        return [
            {
                'timestamp': item.timestamp.isoformat(),
                'type': item.notification_type.value,
                'priority': item.priority.value,
                'message': item.message,
                'channels': item.channels_sent,
                'success': item.success,
                'error': item.error
            }
            for item in reversed(recent)
        ]
    
    def save_config(self, path: Optional[str] = None):
        """
        설정 저장
        
        Args:
            path: 저장 경로
        """
        path = path or self.config_path
        if not path:
            logger.warning("No config path specified")
            return
        
        config = {
            'rules': {},
            'stats': self.stats
        }
        
        # 규칙 직렬화
        for name, rule in self.rules.items():
            config['rules'][name] = {
                'enabled': rule.enabled,
                'channels': list(rule.channels),
                'types': [t.value for t in rule.types],
                'min_priority': rule.min_priority.value,
                'rate_limit': rule.rate_limit,
                'quiet_hours': rule.quiet_hours
            }
        
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Config saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def load_config(self, path: str):
        """
        설정 로드
        
        Args:
            path: 설정 파일 경로
        """
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # 규칙 로드
            if 'rules' in config:
                for name, rule_data in config['rules'].items():
                    if name not in self.rules:
                        self.rules[name] = NotificationRule(name=name)
                    
                    rule = self.rules[name]
                    rule.enabled = rule_data.get('enabled', True)
                    rule.channels = set(rule_data.get('channels', []))
                    rule.types = {
                        NotificationType[t.upper()]
                        for t in rule_data.get('types', [])
                    }
                    rule.min_priority = NotificationPriority[
                        rule_data.get('min_priority', 'LOW').upper()
                    ]
                    rule.rate_limit = rule_data.get('rate_limit', 0)
                    rule.quiet_hours = (
                        tuple(rule_data['quiet_hours'])
                        if 'quiet_hours' in rule_data
                        else None
                    )
            
            # 통계 로드
            if 'stats' in config:
                self.stats.update(config['stats'])
            
            logger.info(f"Config loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    async def cleanup(self):
        """정리 작업"""
        # 모든 채널 정리
        for notifier in self.notifiers.values():
            await notifier.cleanup()
        
        # 설정 저장
        self.save_config()
        
        logger.info("NotificationManager cleaned up")