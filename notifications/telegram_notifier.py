"""
Telegram 알림 시스템
Telegram Bot API를 통한 메시지 전송
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


class TelegramNotifier(BaseNotifier):
    """
    Telegram 알림 전송자
    
    Bot API를 통해 Telegram 채널/그룹으로 알림 전송
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}"
    MAX_MESSAGE_LENGTH = 4096  # Telegram 메시지 최대 길이
    
    def __init__(self):
        """초기화"""
        super().__init__("Telegram")
        
        self.bot_token: Optional[str] = None
        self.chat_ids: List[str] = []  # 여러 채팅방 지원
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 메시지 포맷 설정
        self.parse_mode = "HTML"  # HTML or Markdown
        self.disable_notification = False  # 무음 알림 여부
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    async def setup(self, config: Dict[str, Any]) -> bool:
        """
        Telegram 설정
        
        Args:
            config: {
                'bot_token': 'YOUR_BOT_TOKEN',
                'chat_ids': ['CHAT_ID_1', 'CHAT_ID_2'],
                'parse_mode': 'HTML',  # optional
                'disable_notification': False  # optional
            }
            
        Returns:
            설정 성공 여부
        """
        try:
            # 봇 토큰 설정
            self.bot_token = config.get('bot_token')
            if not self.bot_token:
                logger.error("Bot token not provided")
                return False
            
            # 채팅 ID 설정
            chat_ids = config.get('chat_ids', [])
            if isinstance(chat_ids, str):
                self.chat_ids = [chat_ids]
            else:
                self.chat_ids = chat_ids
            
            if not self.chat_ids:
                logger.error("No chat IDs provided")
                return False
            
            # 옵션 설정
            self.parse_mode = config.get('parse_mode', 'HTML')
            self.disable_notification = config.get('disable_notification', False)
            
            # HTTP 세션 생성
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # 봇 테스트
            if await self._test_bot():
                self.enabled = True
                logger.info(f"Telegram notifier setup successful for {len(self.chat_ids)} chats")
                return True
            else:
                logger.error("Failed to verify bot")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup Telegram: {e}")
            return False
    
    async def _test_bot(self) -> bool:
        """
        봇 연결 테스트
        
        Returns:
            테스트 성공 여부
        """
        try:
            url = f"{self.BASE_URL.format(token=self.bot_token)}/getMe"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        bot_info = data.get('result', {})
                        logger.info(f"Bot verified: @{bot_info.get('username')}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Bot test failed: {e}")
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
            **kwargs: 추가 파라미터 (reply_markup, photo 등)
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            logger.warning("Telegram notifier is not enabled")
            return False
        
        # 메시지 포맷팅
        formatted_message = self._format_message(message, notification_type, priority)
        
        # 길이 체크
        if len(formatted_message) > self.MAX_MESSAGE_LENGTH:
            formatted_message = formatted_message[:self.MAX_MESSAGE_LENGTH-3] + "..."
        
        # 모든 채팅방에 전송
        success_count = 0
        for chat_id in self.chat_ids:
            if await self._send_to_chat(chat_id, formatted_message, **kwargs):
                success_count += 1
        
        # 통계 업데이트
        success = success_count > 0
        self.update_stats(success)
        
        return success
    
    async def _send_to_chat(
        self,
        chat_id: str,
        message: str,
        **kwargs
    ) -> bool:
        """
        특정 채팅방에 메시지 전송
        
        Args:
            chat_id: 채팅 ID
            message: 메시지
            **kwargs: 추가 파라미터
            
        Returns:
            전송 성공 여부
        """
        url = f"{self.BASE_URL.format(token=self.bot_token)}/sendMessage"
        
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': self.parse_mode,
            'disable_notification': self.disable_notification
        }
        
        # 추가 파라미터 병합
        payload.update(kwargs)
        
        # 재시도 로직
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {error_text}")
                        
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
    def _format_message(
        self,
        message: str,
        notification_type: NotificationType,
        priority: NotificationPriority
    ) -> str:
        """
        메시지 포맷팅
        
        Args:
            message: 원본 메시지
            notification_type: 알림 타입
            priority: 우선순위
            
        Returns:
            포맷된 메시지
        """
        # 이모지와 접두사
        emoji = self.get_emoji(notification_type)
        prefix = self.get_priority_prefix(priority)
        
        # HTML 포맷팅
        if self.parse_mode == "HTML":
            # 타입별 강조
            if notification_type == NotificationType.ERROR:
                formatted = f"<b>{emoji} {prefix}ERROR</b>\n\n{message}"
            elif notification_type == NotificationType.WARNING:
                formatted = f"<b>{emoji} {prefix}WARNING</b>\n\n{message}"
            elif notification_type == NotificationType.SUCCESS:
                formatted = f"<b>{emoji} {prefix}SUCCESS</b>\n\n{message}"
            else:
                formatted = f"{emoji} {prefix}{message}"
        else:
            # Markdown 포맷팅
            if notification_type in [NotificationType.ERROR, NotificationType.WARNING]:
                formatted = f"**{emoji} {prefix}{notification_type.value.upper()}**\n\n{message}"
            else:
                formatted = f"{emoji} {prefix}{message}"
        
        return formatted
    
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
        # 메시지 구성
        if self.parse_mode == "HTML":
            message = f"""<b>🔔 거래 알림</b>

<b>액션:</b> {action}
<b>심볼:</b> {symbol}
<b>수량:</b> {self.format_number(amount, 4)}
<b>가격:</b> {self.format_number(price, 2)}"""
        else:
            message = f"""**🔔 거래 알림**

**액션:** {action}
**심볼:** {symbol}
**수량:** {self.format_number(amount, 4)}
**가격:** {self.format_number(price, 2)}"""
        
        if reason:
            message += f"\n<b>사유:</b> {reason}" if self.parse_mode == "HTML" else f"\n**사유:** {reason}"
        
        # 추가 정보
        if kwargs:
            if self.parse_mode == "HTML":
                message += "\n\n<b>상세 정보:</b>"
                for key, value in kwargs.items():
                    message += f"\n• {key}: {value}"
            else:
                message += "\n\n**상세 정보:**"
                for key, value in kwargs.items():
                    message += f"\n• {key}: {value}"
        
        # 타임스탬프
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message += f"\n\n⏰ {timestamp}"
        
        # 전송
        notification_type = (
            NotificationType.TRADE_OPEN if action in ['BUY', 'SELL']
            else NotificationType.TRADE_CLOSE
        )
        
        return await self.send_message(
            message,
            notification_type=notification_type,
            priority=NotificationPriority.HIGH
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
        # 메시지 구성
        date_str = date.strftime('%Y-%m-%d')
        pnl_emoji = "💰" if pnl >= 0 else "💸"
        
        if self.parse_mode == "HTML":
            message = f"""<b>📊 일일 거래 리포트</b>
<i>{date_str}</i>

<b>📈 총 손익:</b> {pnl_emoji} {self.format_number(pnl, 2)} KRW ({self.format_percentage(pnl/1000000)})
<b>🔄 거래 횟수:</b> {trades_count}
<b>🎯 승률:</b> {self.format_percentage(win_rate)}"""
        else:
            message = f"""**📊 일일 거래 리포트**
_{date_str}_

**📈 총 손익:** {pnl_emoji} {self.format_number(pnl, 2)} KRW ({self.format_percentage(pnl/1000000)})
**🔄 거래 횟수:** {trades_count}
**🎯 승률:** {self.format_percentage(win_rate)}"""
        
        # 상세 정보 추가
        if details:
            if self.parse_mode == "HTML":
                message += "\n\n<b>📋 상세 내역:</b>"
            else:
                message += "\n\n**📋 상세 내역:**"
            
            # 최고 수익 거래
            if 'best_trade' in details:
                best = details['best_trade']
                message += f"\n✅ 최고 수익: {self.format_number(best.get('pnl', 0), 2)} KRW"
            
            # 최대 손실 거래
            if 'worst_trade' in details:
                worst = details['worst_trade']
                message += f"\n❌ 최대 손실: {self.format_number(worst.get('pnl', 0), 2)} KRW"
            
            # 평균 거래 시간
            if 'avg_trade_duration' in details:
                duration = details['avg_trade_duration']
                message += f"\n⏱️ 평균 보유 시간: {duration}"
            
            # 거래소별 분포
            if 'exchange_distribution' in details:
                dist = details['exchange_distribution']
                message += "\n\n거래소별 거래:"
                for exchange, count in dist.items():
                    message += f"\n• {exchange}: {count}건"
        
        # 마무리 메시지
        if pnl > 0:
            message += "\n\n🎉 수고하셨습니다! 좋은 하루였네요."
        else:
            message += "\n\n💪 내일은 더 나은 결과가 있을 거예요."
        
        # 전송
        return await self.send_message(
            message,
            notification_type=NotificationType.DAILY_REPORT,
            priority=NotificationPriority.MEDIUM
        )
    
    async def send_photo(
        self,
        photo_path: str,
        caption: str = "",
        notification_type: NotificationType = NotificationType.INFO
    ) -> bool:
        """
        사진 전송
        
        Args:
            photo_path: 사진 파일 경로
            caption: 캡션
            notification_type: 알림 타입
            
        Returns:
            전송 성공 여부
        """
        if not self.enabled:
            return False
        
        url = f"{self.BASE_URL.format(token=self.bot_token)}/sendPhoto"
        
        # 캡션 포맷팅
        if caption:
            caption = self._format_message(caption, notification_type, NotificationPriority.MEDIUM)
        
        success_count = 0
        
        for chat_id in self.chat_ids:
            try:
                with open(photo_path, 'rb') as photo:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', chat_id)
                    data.add_field('photo', photo, filename='photo.png')
                    data.add_field('caption', caption)
                    data.add_field('parse_mode', self.parse_mode)
                    
                    async with self.session.post(url, data=data) as response:
                        if response.status == 200:
                            success_count += 1
                            
            except Exception as e:
                logger.error(f"Failed to send photo to {chat_id}: {e}")
        
        return success_count > 0
    
    async def cleanup(self):
        """정리 작업"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.enabled = False
        logger.info("Telegram notifier cleaned up")