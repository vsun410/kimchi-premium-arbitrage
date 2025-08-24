"""
알림 시스템 테스트
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from notifications import (
    NotificationManager,
    TelegramNotifier,
    DiscordNotifier,
    NotificationType,
    NotificationPriority
)


class TestNotificationManager:
    """NotificationManager 테스트"""
    
    @pytest_asyncio.fixture
    async def manager(self):
        """NotificationManager 픽스처"""
        manager = NotificationManager()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """초기화 테스트"""
        assert manager is not None
        assert len(manager.notifiers) == 0  # 채널 미설정
        assert 'default' in manager.rules
        assert 'trades' in manager.rules
        assert 'errors' in manager.rules
        assert 'reports' in manager.rules
    
    @pytest.mark.asyncio
    async def test_setup_channels(self, manager):
        """채널 설정 테스트"""
        # Mock 설정
        with patch.object(TelegramNotifier, 'setup', return_value=True):
            with patch.object(DiscordNotifier, 'setup', return_value=True):
                config = {
                    'telegram': {
                        'bot_token': 'test_token',
                        'chat_ids': ['123456']
                    },
                    'discord': {
                        'webhook_urls': ['https://discord.com/webhook']
                    }
                }
                
                await manager.setup(config)
                
                assert 'telegram' in manager.notifiers
                assert 'discord' in manager.notifiers
                assert len(manager.notifiers) == 2
    
    @pytest.mark.asyncio
    async def test_notify_basic(self, manager):
        """기본 알림 테스트"""
        # Mock notifier 추가
        mock_notifier = AsyncMock()
        mock_notifier.send_message.return_value = True
        mock_notifier.get_stats.return_value = {}
        manager.notifiers['test'] = mock_notifier
        
        # 규칙 업데이트
        manager.rules['default'].channels = {'test'}
        
        # 알림 전송
        result = await manager.notify(
            "Test message",
            NotificationType.INFO,
            NotificationPriority.MEDIUM
        )
        
        assert result is True
        mock_notifier.send_message.assert_called_once()
        assert manager.stats['total_sent'] == 1
    
    @pytest.mark.asyncio
    async def test_priority_filtering(self, manager):
        """우선순위 필터링 테스트"""
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.send_message.return_value = True
        manager.notifiers['test'] = mock_notifier
        
        # HIGH 우선순위 이상만 허용
        manager.rules['default'].channels = {'test'}
        manager.rules['default'].min_priority = NotificationPriority.HIGH
        
        # LOW 우선순위 - 차단되어야 함
        result = await manager.notify(
            "Low priority",
            NotificationType.INFO,
            NotificationPriority.LOW
        )
        assert result is False
        mock_notifier.send_message.assert_not_called()
        
        # HIGH 우선순위 - 허용되어야 함
        result = await manager.notify(
            "High priority",
            NotificationType.INFO,
            NotificationPriority.HIGH
        )
        assert result is True
        mock_notifier.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, manager):
        """Rate limiting 테스트"""
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.send_message.return_value = True
        mock_notifier.get_stats.return_value = {}
        manager.notifiers['test'] = mock_notifier
        
        # Rate limit 설정 (초당 2개)
        manager.rules['default'].channels = {'test'}
        manager.rules['default'].rate_limit = 2
        
        # 2개는 통과
        for i in range(2):
            result = await manager.notify(
                f"Message {i}",
                NotificationType.INFO,
                NotificationPriority.MEDIUM
            )
            assert result is True
        
        # 3번째는 차단
        result = await manager.notify(
            "Message 3",
            NotificationType.INFO,
            NotificationPriority.MEDIUM
        )
        assert result is False
        
        # 2번만 호출되어야 함
        assert mock_notifier.send_message.call_count == 2
    
    @pytest.mark.asyncio
    async def test_quiet_hours(self, manager):
        """무음 시간 테스트"""
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.send_message.return_value = True
        manager.notifiers['test'] = mock_notifier
        
        # 현재 시간 mock
        current_hour = 23  # 23시
        
        with patch('notifications.notification_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = current_hour
            mock_datetime.now.return_value.strftime = MagicMock(return_value="2025-08-24 23:00:00")
            mock_datetime.now.return_value.isoformat = MagicMock(return_value="2025-08-24T23:00:00")
            
            # 무음 시간 설정 (22시-8시)
            manager.rules['reports'].channels = {'test'}
            manager.rules['reports'].quiet_hours = (22, 8)
            manager.rules['reports'].types = {NotificationType.DAILY_REPORT}
            
            # 무음 시간에 리포트 - 차단되어야 함
            result = await manager.notify(
                "Daily report",
                NotificationType.DAILY_REPORT,
                NotificationPriority.MEDIUM
            )
            assert result is False
            mock_notifier.send_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_emergency_notification(self, manager):
        """긴급 알림 테스트"""
        # Mock notifiers
        mock_telegram = AsyncMock()
        mock_telegram.send_message.return_value = True
        mock_discord = AsyncMock()
        mock_discord.send_message.return_value = True
        
        manager.notifiers['telegram'] = mock_telegram
        manager.notifiers['discord'] = mock_discord
        
        # 긴급 알림
        result = await manager.notify_emergency(
            "System critical error!",
            {'error_code': 'E001', 'severity': 'CRITICAL'}
        )
        
        assert result is True
        # 모든 채널로 전송되어야 함
        mock_telegram.send_message.assert_called_once()
        mock_discord.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trade_notification(self, manager):
        """거래 알림 테스트"""
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.send_trade_alert.return_value = True
        manager.notifiers['test'] = mock_notifier
        
        # 거래 규칙
        manager.rules['trades'].channels = {'test'}
        
        # 거래 알림
        result = await manager.notify_trade(
            action='BUY',
            symbol='BTC/USDT',
            amount=0.01,
            price=100000,
            reason='김프 3% 포착'
        )
        
        assert result is True
        mock_notifier.send_trade_alert.assert_called_once_with(
            'BUY',
            'BTC/USDT',
            0.01,
            100000,
            '김프 3% 포착'
        )
    
    @pytest.mark.asyncio
    async def test_daily_report(self, manager):
        """일일 리포트 테스트"""
        # Mock notifier
        mock_notifier = AsyncMock()
        mock_notifier.send_daily_report.return_value = True
        manager.notifiers['test'] = mock_notifier
        
        # 리포트 규칙
        manager.rules['reports'].channels = {'test'}
        manager.rules['reports'].quiet_hours = None  # 무음 시간 해제
        
        # 리포트 전송
        result = await manager.notify_daily_report(
            date=datetime.now(),
            pnl=150000,
            trades_count=25,
            win_rate=0.72,
            details={
                'best_trade': {'pnl': 50000},
                'worst_trade': {'pnl': -10000}
            }
        )
        
        assert result is True
        mock_notifier.send_daily_report.assert_called_once()
    
    def test_stats_tracking(self, manager):
        """통계 추적 테스트"""
        # 통계 업데이트
        manager._update_stats('telegram', NotificationType.INFO, True)
        manager._update_stats('telegram', NotificationType.ERROR, False)
        manager._update_stats('discord', NotificationType.TRADE_OPEN, True)
        
        stats = manager.get_stats()
        
        assert stats['total_sent'] == 2
        assert stats['total_failed'] == 1
        assert stats['by_channel']['telegram']['sent'] == 1
        assert stats['by_channel']['telegram']['failed'] == 1
        assert stats['by_channel']['discord']['sent'] == 1
        assert stats['by_type']['info']['sent'] == 1
        assert stats['by_type']['error']['failed'] == 1
    
    def test_history_tracking(self, manager):
        """이력 추적 테스트"""
        # 이력 저장
        for i in range(5):
            manager._save_history(
                NotificationType.INFO,
                NotificationPriority.MEDIUM,
                f"Message {i}",
                ['telegram'],
                True,
                None
            )
        
        # 이력 조회
        history = manager.get_recent_history(limit=3)
        
        assert len(history) == 3
        assert history[0]['message'] == "Message 4"  # 최신 먼저
        assert history[2]['message'] == "Message 2"


class TestTelegramNotifier:
    """TelegramNotifier 테스트"""
    
    @pytest_asyncio.fixture
    async def notifier(self):
        """TelegramNotifier 픽스처"""
        notifier = TelegramNotifier()
        yield notifier
        await notifier.cleanup()
    
    @pytest.mark.asyncio
    async def test_setup(self, notifier):
        """설정 테스트"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'ok': True,
                'result': {'username': 'test_bot'}
            }
            
            # Mock session with async close method
            mock_session_instance = AsyncMock()
            mock_get = AsyncMock()
            mock_get.__aenter__.return_value = mock_response
            mock_get.__aexit__.return_value = None
            mock_session_instance.get.return_value = mock_get
            mock_session.return_value = mock_session_instance
            
            config = {
                'bot_token': 'test_token',
                'chat_ids': ['123456']
            }
            
            result = await notifier.setup(config)
            
            assert result is True
            assert notifier.enabled is True
            assert notifier.bot_token == 'test_token'
            assert '123456' in notifier.chat_ids
    
    @pytest.mark.asyncio
    async def test_message_formatting(self, notifier):
        """메시지 포맷팅 테스트"""
        # HTML 포맷
        notifier.parse_mode = "HTML"
        formatted = notifier._format_message(
            "Test message",
            NotificationType.ERROR,
            NotificationPriority.HIGH
        )
        
        assert "<b>" in formatted
        assert "ERROR" in formatted
        assert "⚡" in formatted  # HIGH priority prefix
        
        # Markdown 포맷
        notifier.parse_mode = "Markdown"
        formatted = notifier._format_message(
            "Test message",
            NotificationType.SUCCESS,
            NotificationPriority.LOW
        )
        
        assert "**" not in formatted  # LOW priority에는 강조 없음
        assert "✅" in formatted  # SUCCESS emoji


class TestDiscordNotifier:
    """DiscordNotifier 테스트"""
    
    @pytest_asyncio.fixture
    async def notifier(self):
        """DiscordNotifier 픽스처"""
        notifier = DiscordNotifier()
        yield notifier
        await notifier.cleanup()
    
    @pytest.mark.asyncio
    async def test_setup(self, notifier):
        """설정 테스트"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204
            
            # Mock session with async close method
            mock_session_instance = AsyncMock()
            mock_post = AsyncMock()
            mock_post.__aenter__.return_value = mock_response
            mock_post.__aexit__.return_value = None
            mock_session_instance.post.return_value = mock_post
            mock_session.return_value = mock_session_instance
            
            config = {
                'webhook_urls': ['https://discord.com/webhook'],
                'username': 'Test Bot'
            }
            
            result = await notifier.setup(config)
            
            assert result is True
            assert notifier.enabled is True
            assert 'https://discord.com/webhook' in notifier.webhook_urls
            assert notifier.username == 'Test Bot'
    
    def test_embed_creation(self, notifier):
        """Embed 생성 테스트"""
        embed = notifier._create_embed(
            "Test message",
            NotificationType.TRADE_OPEN,
            NotificationPriority.HIGH
        )
        
        assert embed['title'] == "⚡ 📈 Trade Opened"
        assert embed['description'] == "Test message"
        assert embed['color'] == 0x9b59b6  # TRADE_OPEN color
        assert 'timestamp' in embed
        assert 'footer' in embed
    
    def test_payload_creation(self, notifier):
        """페이로드 생성 테스트"""
        # Embed 페이로드
        payload = notifier._create_payload(
            "Test message",
            NotificationType.INFO,
            NotificationPriority.MEDIUM,
            use_embed=True,
            fields=[{'name': 'Field1', 'value': 'Value1'}]
        )
        
        assert 'embeds' in payload
        assert len(payload['embeds']) == 1
        assert payload['embeds'][0]['fields'][0]['name'] == 'Field1'
        
        # 텍스트 페이로드
        payload = notifier._create_payload(
            "Test message",
            NotificationType.INFO,
            NotificationPriority.CRITICAL,
            use_embed=False
        )
        
        assert 'content' in payload
        assert '@everyone' in payload['content']  # CRITICAL priority


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])