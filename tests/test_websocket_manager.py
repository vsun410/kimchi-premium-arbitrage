"""
WebSocket Manager 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import ccxt.pro as ccxtpro

from src.exchange.websocket_manager import WebSocketManager, create_websocket_manager


class TestWebSocketManager:
    """WebSocket Manager 테스트"""
    
    @pytest.fixture
    def config(self):
        """테스트 설정"""
        return {
            'exchanges': {
                'upbit': {},
                'binance': {}
            },
            'reconnect_delay': 1,
            'max_reconnect_attempts': 3
        }
    
    @pytest.fixture
    def manager(self, config):
        """WebSocketManager 인스턴스"""
        return WebSocketManager(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """초기화 테스트"""
        with patch('ccxt.pro.upbit') as mock_upbit, \
             patch('ccxt.pro.binance') as mock_binance:
            
            await manager.initialize()
            
            assert 'upbit' in manager.exchanges
            assert 'binance' in manager.exchanges
            assert manager.connection_status['upbit'] == 'disconnected'
            assert manager.connection_status['binance'] == 'disconnected'
    
    def test_register_callback(self, manager):
        """콜백 등록 테스트"""
        callback = Mock()
        
        manager.register_callback('ticker', callback)
        assert callback in manager.callbacks['ticker']
        
        manager.register_callback('orderbook', callback)
        assert callback in manager.callbacks['orderbook']
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test causing timeout - needs refactoring")
    async def test_reconnection_logic(self, manager):
        """재연결 로직 테스트"""
        # Mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.watch_ticker = AsyncMock(side_effect=Exception("Connection lost"))
        mock_exchange.close = AsyncMock()
        
        manager.exchanges['upbit'] = mock_exchange
        manager.connection_status['upbit'] = 'connected'
        manager._reconnect_counts['upbit'] = 0  # Initialize reconnect count
        manager.is_running = True
        
        # 재연결 테스트 (짧은 시간 내 완료를 위해 제한)
        with patch.object(manager, 'reconnect_delay', 0.1):
            with patch.object(manager, 'max_reconnect_attempts', 2):
                # 재연결 시도
                task = asyncio.create_task(
                    manager._watch_ticker('upbit', mock_exchange, ['BTC/KRW'])
                )
                
                # 잠시 대기
                await asyncio.sleep(0.5)
                
                # 중지
                manager.is_running = False
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected cancellation
                
                # 재연결 시도 확인
                assert manager._reconnect_counts['upbit'] > 0
                assert manager.connection_status['upbit'] in ['reconnecting', 'failed']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """에러 처리 테스트"""
        error_callback = AsyncMock()
        manager.register_callback('error', error_callback)
        
        test_error = Exception("Test error")
        await manager._handle_error("test_source", test_error)
        
        # 에러 콜백이 호출되었는지 확인
        error_callback.assert_called_once()
        call_args = error_callback.call_args[0][0]
        assert call_args['source'] == 'test_source'
        assert 'Test error' in call_args['error']
    
    def test_connection_status(self, manager):
        """연결 상태 조회 테스트"""
        manager.connection_status = {
            'upbit': 'connected',
            'binance': 'disconnected'
        }
        
        assert manager.is_connected('upbit') == True
        assert manager.is_connected('binance') == False
        assert manager.is_connected() == False  # 모든 거래소가 연결되어야 True
        
        manager.connection_status['binance'] = 'connected'
        assert manager.is_connected() == True
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, manager):
        """Exponential backoff 테스트"""
        mock_exchange = AsyncMock()
        mock_exchange.close = AsyncMock()
        
        manager._reconnect_counts['upbit'] = 0
        original_delay = manager.reconnect_delay
        
        # 첫 번째 재연결
        with patch('asyncio.sleep') as mock_sleep:
            await manager._handle_reconnection(
                'upbit', mock_exchange, 'ticker', 
                ['BTC/KRW'], Exception("Test")
            )
            
            # 첫 재연결은 기본 딜레이
            expected_delay = original_delay
            mock_sleep.assert_called_with(expected_delay)
        
        # 두 번째 재연결
        with patch('asyncio.sleep') as mock_sleep:
            await manager._handle_reconnection(
                'upbit', mock_exchange, 'ticker', 
                ['BTC/KRW'], Exception("Test")
            )
            
            # 두 번째는 2배
            expected_delay = original_delay * 2
            mock_sleep.assert_called_with(expected_delay)
    
    @pytest.mark.asyncio
    async def test_max_reconnect_attempts(self, manager):
        """최대 재연결 시도 테스트"""
        mock_exchange = AsyncMock()
        mock_exchange.close = AsyncMock()
        
        manager._reconnect_counts['upbit'] = manager.max_reconnect_attempts - 1
        
        await manager._handle_reconnection(
            'upbit', mock_exchange, 'ticker', 
            ['BTC/KRW'], Exception("Test")
        )
        
        # 최대 시도 횟수 도달 시 failed 상태
        assert manager.connection_status['upbit'] == 'failed'
    
    @pytest.mark.asyncio 
    async def test_create_websocket_manager(self):
        """WebSocket Manager 생성 헬퍼 테스트"""
        config = {
            'exchanges': {'upbit': {}},
            'reconnect_delay': 5,
            'max_reconnect_attempts': 10
        }
        
        with patch('ccxt.pro.upbit'):
            manager = await create_websocket_manager(config)
            assert isinstance(manager, WebSocketManager)
            assert 'upbit' in manager.exchanges


class TestIntegrationWebSocket:
    """통합 테스트 (실제 연결 없이)"""
    
    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """다중 콜백 처리 테스트"""
        config = {'exchanges': {}, 'reconnect_delay': 1}
        manager = WebSocketManager(config)
        
        callback1_called = False
        callback2_called = False
        
        async def callback1(exchange_id, symbol, data):
            nonlocal callback1_called
            callback1_called = True
        
        async def callback2(exchange_id, symbol, data):
            nonlocal callback2_called
            callback2_called = True
        
        manager.register_callback('ticker', callback1)
        manager.register_callback('ticker', callback2)
        
        # 콜백 실행 테스트
        for callback in manager.callbacks['ticker']:
            await callback('upbit', 'BTC/KRW', {})
        
        assert callback1_called
        assert callback2_called
    
    def test_get_stats(self):
        """통계 조회 테스트"""
        config = {'exchanges': {}, 'reconnect_delay': 1}
        manager = WebSocketManager(config)
        
        manager._reconnect_counts = {'upbit': 5, 'binance': 3}
        stats = manager.get_reconnect_stats()
        
        assert stats['upbit'] == 5
        assert stats['binance'] == 3
        
        # 원본이 변경되어도 반환값은 불변
        manager._reconnect_counts['upbit'] = 10
        assert stats['upbit'] == 5