"""
Discord 알림 시스템
Discord Webhook을 통한 메시지 전송
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
import aiohttp
import json

from .base_notifier import (
    BaseNotifier,
    NotificationType,
    NotificationPriority
)

logger = logging.getLogger(__name__)


class DiscordNotifier(BaseNotifier):
    """
    Discord 알림 전송자
    
    Webhook을 통해 Discord 채널로 알림 전송
    """
    
    MAX_EMBED_FIELDS = 25  # Discord embed 최대 필드 수
    MAX_MESSAGE_LENGTH = 2000  # Discord 메시지 최대 길이
    
    def __init__(self):
        """초기화"""
        super().__init__("Discord")
        
        self.webhook_urls: List[str] = []  # 여러 Webhook 지원
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 메시지 설정
        self.username = "Kimchi Premium Bot"  # 봇 이름
        self.avatar_url = None  # 봇 아바타 URL
        
        # 색상 설정 (embed용)
        self.colors = {
            NotificationType.INFO: 0x3498db,      # 파란색
            NotificationType.SUCCESS: 0x2ecc71,   # 초록색
            NotificationType.WARNING: 0xf39c12,   # 주황색
            NotificationType.ERROR: 0xe74c3c,     # 빨간색
            NotificationType.TRADE_OPEN: 0x9b59b6,   # 보라색
            NotificationType.TRADE_CLOSE: 0x34495e,  # 회색
            NotificationType.DAILY_REPORT: 0x1abc9c,  # 청록색
            NotificationType.EMERGENCY_STOP: 0xc0392b  # 진한 빨간색
        }
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    async def setup(self, config: Dict[str, Any]) -> bool:
        """
        Discord 설정
        
        Args:
            config: {
                'webhook_urls': ['WEBHOOK_URL_1', 'WEBHOOK_URL_2'],
                'username': 'Bot Name',  # optional
                'avatar_url': 'https://...'  # optional
            }
            
        Returns:
            설정 성공 여부
        """
        try:
            # Webhook URL 설정
            webhook_urls = config.get('webhook_urls', [])
            if isinstance(webhook_urls, str):
                self.webhook_urls = [webhook_urls]
            else:
                self.webhook_urls = webhook_urls
            
            if not self.webhook_urls:
                logger.error("No webhook URLs provided")
                return False
            
            # 옵션 설정
            self.username = config.get('username', self.username)
            self.avatar_url = config.get('avatar_url', self.avatar_url)
            
            # HTTP 세션 생성
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Webhook 테스트
            if await self._test_webhook():
                self.enabled = True
                logger.info(f"Discord notifier setup successful for {len(self.webhook_urls)} webhooks")
                return True
            else:
                logger.error("Failed to verify webhook")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup Discord: {e}")
            return False
    
    async def _test_webhook(self) -> bool:
        """
        Webhook 연결 테스트
        
        Returns:
            테스트 성공 여부
        """
        try:
            test_payload = {
                'content': '🤖 Discord notifier connected successfully!',
                'username': self.username
            }
            
            if self.avatar_url:
                test_payload['avatar_url'] = self.avatar_url
            
            # 첫 번째 webhook로 테스트
            async with self.session.post(self.webhook_urls[0], json=test_payload) as response:
                if response.status in [200, 204]:
                    logger.info("Webhook test successful")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Webhook test failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Webhook test error: {e}")
            return False
    
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
            **kwargs: 추가 파라미터 (embed, files 등)
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            logger.warning("Discord notifier is not enabled")
            return False
        
        # 페이로드 생성
        payload = self._create_payload(message, notification_type, priority, **kwargs)
        
        # 모든 webhook에 전송
        success_count = 0
        for webhook_url in self.webhook_urls:
            if await self._send_to_webhook(webhook_url, payload):
                success_count += 1
        
        # 통계 업데이트
        success = success_count > 0
        self.update_stats(success)
        
        return success
    
    def _create_payload(
        self,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority,
        **kwargs
    ) -> Dict:
        """
        Discord 페이로드 생성
        
        Args:
            message: 메시지
            notification_type: 알림 타입
            priority: 우선순위
            **kwargs: 추가 파라미터
            
        Returns:
            Webhook 페이로드
        """
        payload = {
            'username': self.username
        }
        
        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url
        
        # 우선순위가 높으면 @everyone 멘션
        if priority == NotificationPriority.CRITICAL:
            message = f"@everyone\n{message}"
        elif priority == NotificationPriority.HIGH:
            message = f"@here\n{message}"
        
        # Embed 사용 여부
        use_embed = kwargs.get('use_embed', True)
        
        if use_embed:
            # Embed 메시지
            embed = self._create_embed(message, notification_type, priority)
            
            # 추가 필드
            if 'fields' in kwargs:
                embed['fields'] = kwargs['fields'][:self.MAX_EMBED_FIELDS]
            
            # 이미지
            if 'image_url' in kwargs:
                embed['image'] = {'url': kwargs['image_url']}
            
            # 썸네일
            if 'thumbnail_url' in kwargs:
                embed['thumbnail'] = {'url': kwargs['thumbnail_url']}
            
            payload['embeds'] = [embed]
        else:
            # 일반 텍스트 메시지
            emoji = self.get_emoji(notification_type)
            prefix = self.get_priority_prefix(priority)
            formatted_message = f"{emoji} {prefix}{message}"
            
            # 길이 체크
            if len(formatted_message) > self.MAX_MESSAGE_LENGTH:
                formatted_message = formatted_message[:self.MAX_MESSAGE_LENGTH-3] + "..."
            
            payload['content'] = formatted_message
        
        return payload
    
    def _create_embed(
        self,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority
    ) -> Dict:
        """
        Discord Embed 생성
        
        Args:
            message: 메시지
            notification_type: 알림 타입
            priority: 우선순위
            
        Returns:
            Embed 데이터
        """
        emoji = self.get_emoji(notification_type)
        color = self.colors.get(notification_type, 0x95a5a6)  # 기본 회색
        
        # 제목 설정
        title_map = {
            NotificationType.INFO: f"{emoji} Information",
            NotificationType.SUCCESS: f"{emoji} Success",
            NotificationType.WARNING: f"{emoji} Warning",
            NotificationType.ERROR: f"{emoji} Error",
            NotificationType.TRADE_OPEN: f"{emoji} Trade Opened",
            NotificationType.TRADE_CLOSE: f"{emoji} Trade Closed",
            NotificationType.DAILY_REPORT: f"{emoji} Daily Report",
            NotificationType.EMERGENCY_STOP: f"{emoji} EMERGENCY STOP"
        }
        
        embed = {
            'title': title_map.get(notification_type, emoji),
            'description': message,
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'footer': {
                'text': 'Kimchi Premium Arbitrage System'
            }
        }
        
        # 우선순위 표시
        if priority == NotificationPriority.CRITICAL:
            embed['title'] = f"🔴 {embed['title']}"
        elif priority == NotificationPriority.HIGH:
            embed['title'] = f"⚡ {embed['title']}"
        
        return embed
    
    async def _send_to_webhook(
        self,
        webhook_url: str,
        payload: Dict
    ) -> bool:
        """
        특정 Webhook에 메시지 전송
        
        Args:
            webhook_url: Webhook URL
            payload: 페이로드
            
        Returns:
            전송 성공 여부
        """
        # 재시도 로직
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(webhook_url, json=payload) as response:
                    if response.status in [200, 204]:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Discord API error: {error_text}")
                        
            except Exception as e:
                logger.error(f"Failed to send to webhook: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
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
        # Embed 필드 구성
        fields = [
            {'name': '💱 Symbol', 'value': symbol, 'inline': True},
            {'name': '🎯 Action', 'value': action, 'inline': True},
            {'name': '💰 Price', 'value': self.format_number(price, 2), 'inline': True},
            {'name': '📦 Amount', 'value': self.format_number(amount, 4), 'inline': True},
        ]
        
        if reason:
            fields.append({'name': '📝 Reason', 'value': reason, 'inline': False})
        
        # 추가 정보
        if 'exchange' in kwargs:
            fields.append({'name': '🏦 Exchange', 'value': kwargs['exchange'], 'inline': True})
        
        if 'expected_profit' in kwargs:
            profit = kwargs['expected_profit']
            fields.append({
                'name': '💸 Expected Profit',
                'value': f"{self.format_number(profit, 2)} ({self.format_percentage(profit/price)})",
                'inline': True
            })
        
        # 메시지 설명
        description = f"A new {action.lower()} order has been executed."
        
        # 타입 결정
        notification_type = (
            NotificationType.TRADE_OPEN if action in ['BUY', 'SELL']
            else NotificationType.TRADE_CLOSE
        )
        
        # 전송
        return await self.send_message(
            description,
            notification_type=notification_type,
            priority=NotificationPriority.HIGH,
            use_embed=True,
            fields=fields
        )
    
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
        # 기본 필드
        fields = [
            {
                'name': '📈 Total P&L',
                'value': f"{self.format_number(pnl, 2)} KRW\n({self.format_percentage(pnl/1000000)})",
                'inline': True
            },
            {
                'name': '🔄 Trades',
                'value': str(trades_count),
                'inline': True
            },
            {
                'name': '🎯 Win Rate',
                'value': self.format_percentage(win_rate),
                'inline': True
            }
        ]
        
        # 상세 정보 추가
        if details:
            # 최고/최저 거래
            if 'best_trade' in details:
                best = details['best_trade']
                fields.append({
                    'name': '✅ Best Trade',
                    'value': f"{self.format_number(best.get('pnl', 0), 2)} KRW",
                    'inline': True
                })
            
            if 'worst_trade' in details:
                worst = details['worst_trade']
                fields.append({
                    'name': '❌ Worst Trade',
                    'value': f"{self.format_number(worst.get('pnl', 0), 2)} KRW",
                    'inline': True
                })
            
            # 평균 거래 시간
            if 'avg_trade_duration' in details:
                fields.append({
                    'name': '⏱️ Avg Duration',
                    'value': details['avg_trade_duration'],
                    'inline': True
                })
            
            # 거래소별 분포
            if 'exchange_distribution' in details:
                dist = details['exchange_distribution']
                dist_text = "\n".join([f"{ex}: {count}" for ex, count in dist.items()])
                fields.append({
                    'name': '🏦 Exchange Distribution',
                    'value': dist_text,
                    'inline': False
                })
            
            # 전략별 성과
            if 'strategy_performance' in details:
                perf = details['strategy_performance']
                perf_text = "\n".join([f"{s}: {p:.2f}%" for s, p in perf.items()])
                fields.append({
                    'name': '📊 Strategy Performance',
                    'value': perf_text,
                    'inline': False
                })
        
        # 설명 메시지
        date_str = date.strftime('%Y-%m-%d')
        if pnl > 0:
            description = f"🎉 Great job! Today was a profitable day.\n**Date:** {date_str}"
        else:
            description = f"💪 Tomorrow will be better!\n**Date:** {date_str}"
        
        # 전송
        return await self.send_message(
            description,
            notification_type=NotificationType.DAILY_REPORT,
            priority=NotificationPriority.MEDIUM,
            use_embed=True,
            fields=fields
        )
    
    async def send_error_alert(
        self,
        error_message: str,
        error_type: str = "Unknown",
        traceback: str = None
    ) -> bool:
        """
        에러 알림 전송
        
        Args:
            error_message: 에러 메시지
            error_type: 에러 타입
            traceback: 트레이스백
            
        Returns:
            전송 성공 여부
        """
        fields = [
            {'name': '🔴 Error Type', 'value': error_type, 'inline': True},
            {'name': '🕑 Time', 'value': datetime.now().strftime('%H:%M:%S'), 'inline': True}
        ]
        
        if traceback:
            # 트레이스백이 너무 길면 자르기
            if len(traceback) > 1000:
                traceback = traceback[:997] + "..."
            fields.append({
                'name': '🔍 Traceback',
                'value': f"```python\n{traceback}\n```",
                'inline': False
            })
        
        return await self.send_message(
            error_message,
            notification_type=NotificationType.ERROR,
            priority=NotificationPriority.HIGH,
            use_embed=True,
            fields=fields
        )
    
    async def cleanup(self):
        """정리 작업"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.enabled = False
        logger.info("Discord notifier cleaned up")