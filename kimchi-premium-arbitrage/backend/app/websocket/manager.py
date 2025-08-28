"""
WebSocket connection manager
"""
import json
import asyncio
import logging
from typing import Dict, Set, Any
from datetime import datetime
import socketio
from app.config import settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        # Create Socket.IO server
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins='*',
            logger=False,
            engineio_logger=False
        )
        
        # Create ASGI app
        self.app = socketio.ASGIApp(
            self.sio,
            socketio_path='/ws/socket.io'
        )
        
        # Track connected clients and their subscriptions
        self.clients: Set[str] = set()
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Background tasks
        self.background_tasks = []
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"Client {sid} connected")
            self.clients.add(sid)
            await self.sio.emit('connected', {'status': 'connected'}, to=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Client {sid} disconnected")
            self.clients.discard(sid)
            # Clean up subscriptions
            for channel in list(self.subscriptions.keys()):
                self.subscriptions[channel].discard(sid)
                if not self.subscriptions[channel]:
                    del self.subscriptions[channel]
        
        @self.sio.event
        async def subscribe(sid, data):
            """Handle subscription to channels"""
            channels = data.get('channels', [])
            for channel in channels:
                if channel not in self.subscriptions:
                    self.subscriptions[channel] = set()
                self.subscriptions[channel].add(sid)
                logger.info(f"Client {sid} subscribed to {channel}")
            
            await self.sio.emit('subscribed', {
                'channels': channels,
                'status': 'success'
            }, to=sid)
        
        @self.sio.event
        async def unsubscribe(sid, data):
            """Handle unsubscription from channels"""
            channels = data.get('channels', [])
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(sid)
                    if not self.subscriptions[channel]:
                        del self.subscriptions[channel]
                logger.info(f"Client {sid} unsubscribed from {channel}")
            
            await self.sio.emit('unsubscribed', {
                'channels': channels,
                'status': 'success'
            }, to=sid)
        
        @self.sio.event
        async def ping(sid):
            """Handle ping for connection keep-alive"""
            await self.sio.emit('pong', {'timestamp': datetime.utcnow().isoformat()}, to=sid)
    
    async def startup(self):
        """Start background tasks"""
        # Start market data broadcaster
        task = asyncio.create_task(self.broadcast_market_data())
        self.background_tasks.append(task)
        
        # Start system status broadcaster
        task = asyncio.create_task(self.broadcast_system_status())
        self.background_tasks.append(task)
        
        logger.info("WebSocket manager started")
    
    async def shutdown(self):
        """Shutdown background tasks"""
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        logger.info("WebSocket manager shutdown")
    
    async def broadcast_market_data(self):
        """Broadcast market data to subscribed clients"""
        while True:
            try:
                # TODO: Get actual market data
                # For now, send dummy data
                if 'premium' in self.subscriptions:
                    data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'upbit_price': 89000000,
                        'binance_price': 65000,
                        'exchange_rate': 1370,
                        'premium_rate': 2.5
                    }
                    
                    for sid in self.subscriptions['premium']:
                        await self.sio.emit('premium.update', data, to=sid)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error broadcasting market data: {e}")
                await asyncio.sleep(1)
    
    async def broadcast_system_status(self):
        """Broadcast system status to all clients"""
        while True:
            try:
                # TODO: Get actual system status
                status = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'operational',
                    'connected_clients': len(self.clients),
                    'active_strategies': 2,
                    'open_positions': 1
                }
                
                await self.sio.emit('system.status', status)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting system status: {e}")
                await asyncio.sleep(5)
    
    async def emit_trade_executed(self, trade_data: Dict[str, Any]):
        """Emit trade execution event"""
        if 'trades' in self.subscriptions:
            for sid in self.subscriptions['trades']:
                await self.sio.emit('trade.executed', trade_data, to=sid)
    
    async def emit_position_update(self, position_data: Dict[str, Any]):
        """Emit position update event"""
        if 'positions' in self.subscriptions:
            for sid in self.subscriptions['positions']:
                await self.sio.emit('position.update', position_data, to=sid)
    
    async def emit_alert(self, alert_data: Dict[str, Any]):
        """Emit alert to all clients"""
        await self.sio.emit('alert.new', alert_data)
    
    async def emit_backtest_progress(self, progress_data: Dict[str, Any]):
        """Emit backtest progress update"""
        if 'backtests' in self.subscriptions:
            for sid in self.subscriptions['backtests']:
                await self.sio.emit('backtest.progress', progress_data, to=sid)
    
    async def emit_paper_trading_update(self, update_data: Dict[str, Any]):
        """Emit paper trading update"""
        if 'paper_trading' in self.subscriptions:
            for sid in self.subscriptions['paper_trading']:
                await self.sio.emit('paper_trading.update', update_data, to=sid)


# Create global instance
websocket_manager = WebSocketManager()