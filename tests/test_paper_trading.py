"""
Paper Trading 테스트
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.trading.paper_trading import (
    PaperTradingEngine, 
    PaperTradingManager, 
    PaperOrder,
    PaperPosition,
    OrderStatus,
    OrderType,
    OrderSide,
    create_paper_trading_manager
)


class TestPaperTradingEngine:
    """Paper Trading 엔진 테스트"""
    
    @pytest.fixture
    def engine(self):
        """테스트용 엔진"""
        initial_balance = {
            'upbit': {'KRW': 20000000},
            'binance': {'USDT': 14000}
        }
        return PaperTradingEngine(initial_balance)
    
    def test_initialization(self, engine):
        """초기화 테스트"""
        assert engine.balance['upbit']['KRW'] == 20000000
        assert engine.balance['binance']['USDT'] == 14000
        assert len(engine.orders) == 0
        assert len(engine.positions) == 0
    
    @pytest.mark.asyncio
    async def test_market_buy_order(self, engine):
        """시장가 매수 주문 테스트"""
        # 가격 설정
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        
        # 매수 주문
        order = await engine.place_order(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.BUY,
            amount=0.01,
            order_type=OrderType.MARKET
        )
        
        assert order.status == OrderStatus.FILLED
        assert order.executed_amount == 0.01
        assert order.executed_price == 100000000
        assert order.fee > 0
        
        # 잔고 확인
        assert engine.balance['upbit']['KRW'] < 20000000
        assert engine.balance['upbit'].get('BTC', 0) == 0.01
    
    @pytest.mark.asyncio
    async def test_market_sell_order(self, engine):
        """시장가 매도 주문 테스트"""
        # BTC 잔고 추가
        engine.balance['upbit']['BTC'] = 0.01
        
        # 가격 설정
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        
        # 매도 주문
        order = await engine.place_order(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.SELL,
            amount=0.01,
            order_type=OrderType.MARKET
        )
        
        assert order.status == OrderStatus.FILLED
        assert engine.balance['upbit']['BTC'] == 0
        assert engine.balance['upbit']['KRW'] > 20000000  # 초기값보다 증가
    
    @pytest.mark.asyncio
    async def test_insufficient_balance(self, engine):
        """잔고 부족 테스트"""
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        
        # 큰 금액 주문 (잔고 초과)
        order = await engine.place_order(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.BUY,
            amount=1,  # 1 BTC = 1억원 > 2천만원
            order_type=OrderType.MARKET
        )
        
        assert order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_futures_long_position(self, engine):
        """선물 롱 포지션 테스트"""
        engine.update_price('binance', 'BTC/USDT', 70000)
        
        # 롱 포지션 진입
        order = await engine.place_order(
            exchange='binance',
            symbol='BTC/USDT',
            side=OrderSide.LONG,
            amount=0.1,
            order_type=OrderType.MARKET
        )
        
        assert order.status == OrderStatus.FILLED
        assert 'binance:BTC/USDT' in engine.positions
        
        position = engine.positions['binance:BTC/USDT']
        assert position.side == 'long'
        assert position.amount == 0.1
        assert position.entry_price == 70000
    
    @pytest.mark.asyncio
    async def test_position_pnl(self, engine):
        """포지션 PnL 계산 테스트"""
        # 포지션 생성
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        await engine.place_order(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.BUY,
            amount=0.01,
            order_type=OrderType.MARKET
        )
        
        # 가격 상승
        engine.update_price('upbit', 'BTC/KRW', 105000000)
        
        position = engine.positions['upbit:BTC/KRW']
        pnl = position.calculate_pnl(105000000)
        
        assert pnl > 0  # 수익 발생
        assert pnl == (105000000 - 100000000) * 0.01
    
    @pytest.mark.asyncio
    async def test_close_position(self, engine):
        """포지션 청산 테스트"""
        # BTC 보유
        engine.balance['upbit']['BTC'] = 0.01
        
        # 포지션 생성
        position = PaperPosition(
            exchange='upbit',
            symbol='BTC/KRW',
            side='buy',
            amount=0.01,
            entry_price=100000000
        )
        engine.positions['upbit:BTC/KRW'] = position
        
        # 가격 설정 및 청산
        engine.update_price('upbit', 'BTC/KRW', 105000000)
        
        await engine.place_order(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.SELL,
            amount=0.01,
            order_type=OrderType.MARKET
        )
        
        # 포지션 확인
        assert 'upbit:BTC/KRW' not in engine.positions
        assert engine.stats['total_trades'] == 1
        assert engine.stats['total_pnl'] > 0
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, engine):
        """주문 취소 테스트"""
        # 지정가 주문
        order = PaperOrder(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0.01,
            price=90000000,
            status=OrderStatus.PENDING
        )
        engine.orders.append(order)
        
        # 취소
        success = await engine.cancel_order(order.order_id)
        
        assert success
        assert order not in engine.orders
        assert engine.stats['cancelled_orders'] == 1
    
    def test_portfolio_value(self, engine):
        """포트폴리오 가치 계산 테스트"""
        # 초기 포트폴리오 가치
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        engine.update_price('binance', 'BTC/USDT', 70000)
        
        initial_value = engine.get_portfolio_value()
        
        # BTC 보유 추가
        engine.balance['upbit']['BTC'] = 0.1
        engine.balance['upbit']['KRW'] = 10000000  # 잔고 감소
        
        # 포트폴리오 가치 재계산
        new_value = engine.get_portfolio_value()
        
        assert new_value == initial_value  # 가치 보존 (수수료 제외)
    
    def test_statistics(self, engine):
        """통계 테스트"""
        # 초기 통계
        stats = engine.get_statistics()
        
        assert stats['total_orders'] == 0
        assert stats['win_rate'] == 0
        assert stats['max_drawdown'] == 0
    
    def test_reset(self, engine):
        """리셋 테스트"""
        # 데이터 추가
        engine.stats['total_orders'] = 10
        engine.positions['test'] = Mock()
        
        # 리셋
        engine.reset()
        
        assert engine.stats['total_orders'] == 0
        assert len(engine.positions) == 0
        assert engine.balance['upbit']['KRW'] == 20000000


class TestPaperTradingManager:
    """Paper Trading 매니저 테스트"""
    
    @pytest.fixture
    def manager(self):
        """테스트용 매니저"""
        config = {
            'initial_balance': {
                'upbit': {'KRW': 10000000},
                'binance': {'USDT': 7000}
            }
        }
        return PaperTradingManager(config)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """시작/중지 테스트"""
        await manager.start()
        assert manager.is_running
        
        await manager.stop()
        assert not manager.is_running
    
    def test_price_update(self, manager):
        """가격 업데이트 테스트"""
        manager.on_price_update('upbit', 'BTC/KRW', 100000000)
        
        assert manager.engine.current_prices['upbit:BTC/KRW'] == 100000000
    
    def test_orderbook_update(self, manager):
        """오더북 업데이트 테스트"""
        orderbook = {
            'bids': [[99000000, 0.5]],
            'asks': [[101000000, 0.5]]
        }
        
        manager.on_orderbook_update('upbit', 'BTC/KRW', orderbook)
        
        assert manager.engine.orderbooks['upbit:BTC/KRW'] == orderbook
    
    @pytest.mark.asyncio
    async def test_execute_signal(self, manager):
        """신호 실행 테스트"""
        # 가격 설정
        manager.on_price_update('upbit', 'BTC/KRW', 100000000)
        
        # 신호 생성
        signal = {
            'exchange': 'upbit',
            'symbol': 'BTC/KRW',
            'side': 'buy',
            'amount': 0.001,
            'order_type': 'market'
        }
        
        # 실행
        order = await manager.execute_signal(signal)
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.executed_amount == 0.001
    
    @pytest.mark.asyncio
    async def test_pending_order_fill(self, manager):
        """지정가 주문 체결 테스트"""
        await manager.start()
        
        # 초기 가격
        manager.on_price_update('upbit', 'BTC/KRW', 100000000)
        
        # 지정가 매수 주문 (현재가보다 낮은 가격)
        signal = {
            'exchange': 'upbit',
            'symbol': 'BTC/KRW',
            'side': 'buy',
            'amount': 0.001,
            'order_type': 'limit',
            'price': 95000000
        }
        
        order = await manager.execute_signal(signal)
        assert order.status == OrderStatus.PENDING
        
        # 가격 하락
        manager.on_price_update('upbit', 'BTC/KRW', 94000000)
        
        # 체결 확인
        await manager._check_pending_orders()
        
        # 주문이 체결되었는지 확인
        assert order not in manager.engine.orders
        
        await manager.stop()
    
    def test_create_manager(self):
        """매니저 생성 테스트"""
        config = {'initial_balance': {'upbit': {'KRW': 5000000}}}
        
        manager = create_paper_trading_manager(config)
        
        assert isinstance(manager, PaperTradingManager)
        assert manager.engine.balance['upbit']['KRW'] == 5000000


class TestIntegrationPaperTrading:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self):
        """완전한 거래 사이클 테스트"""
        # 매니저 생성
        config = {
            'initial_balance': {
                'upbit': {'KRW': 20000000},
                'binance': {'USDT': 14000}
            }
        }
        
        manager = create_paper_trading_manager(config)
        await manager.start()
        
        # 가격 설정
        manager.on_price_update('upbit', 'BTC/KRW', 100000000)
        manager.on_price_update('binance', 'BTC/USDT', 70000)
        
        # 김프 진입: 업비트 매수 + 바이낸스 숏
        upbit_signal = {
            'exchange': 'upbit',
            'symbol': 'BTC/KRW',
            'side': 'buy',
            'amount': 0.01,
            'order_type': 'market'
        }
        
        binance_signal = {
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'side': 'short',
            'amount': 0.01,
            'order_type': 'market'
        }
        
        upbit_order = await manager.execute_signal(upbit_signal)
        binance_order = await manager.execute_signal(binance_signal)
        
        assert upbit_order.status == OrderStatus.FILLED
        assert binance_order.status == OrderStatus.FILLED
        
        # 포지션 확인
        positions = manager.engine.get_positions()
        assert len(positions) == 2
        
        # 가격 변동
        manager.on_price_update('upbit', 'BTC/KRW', 95000000)
        manager.on_price_update('binance', 'BTC/USDT', 68000)
        
        # 김프 청산: 업비트 매도 + 바이낸스 롱
        # 먼저 BTC 잔고 추가 (매도를 위해)
        manager.engine.balance['upbit']['BTC'] = 0.01
        
        upbit_close = {
            'exchange': 'upbit',
            'symbol': 'BTC/KRW',
            'side': 'sell',
            'amount': 0.01,
            'order_type': 'market'
        }
        
        await manager.execute_signal(upbit_close)
        
        # 통계 확인
        stats = manager.engine.get_statistics()
        assert stats['total_orders'] > 0
        assert stats['filled_orders'] > 0
        
        await manager.stop()
    
    @pytest.mark.asyncio  
    async def test_risk_management(self):
        """리스크 관리 테스트"""
        engine = PaperTradingEngine(
            initial_balance={'upbit': {'KRW': 10000000}}
        )
        
        # 최대 낙폭 추적
        engine.update_price('upbit', 'BTC/KRW', 100000000)
        
        initial_value = engine.get_portfolio_value()
        engine.stats['peak_balance'] = initial_value
        
        # BTC 매수
        engine.balance['upbit']['BTC'] = 0.1
        engine.balance['upbit']['KRW'] = 0
        
        # 가격 하락
        engine.update_price('upbit', 'BTC/KRW', 80000000)
        
        # 포트폴리오 가치 및 낙폭 계산
        current_value = engine.get_portfolio_value()
        
        assert current_value < initial_value
        assert engine.stats['max_drawdown'] > 0


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, '-v'])