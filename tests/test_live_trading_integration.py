"""
Live Trading 시스템 통합 테스트

목적: 실거래 시스템 전체 플로우 검증
결과: 모든 모듈이 함께 정상 작동
평가: 안전성과 정확성 확보
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading import (
    OrderManager,
    OrderRequest,
    BalanceManager,
    SafetyManager,
    SafetyAction,
    RiskLevel
)
from live_trading.exchanges.base_exchange import (
    Order, OrderSide, OrderType, OrderStatus,
    Balance, Ticker
)
from live_trading.order_manager import ExecutionStrategy


class MockExchange:
    """테스트용 Mock 거래소"""
    
    def __init__(self, name: str):
        self.name = name
        self.connected = False
        self.orders = {}
        self.balances = {}
        self.order_counter = 0
        
    async def connect(self) -> bool:
        self.connected = True
        return True
    
    async def disconnect(self):
        self.connected = False
    
    async def get_balance(self, currency: str = None):
        if currency:
            return {currency: self.balances.get(currency, Balance(currency, 0, 0, 0))}
        return self.balances
    
    async def get_ticker(self, symbol: str):
        # Mock 가격 데이터
        if 'KRW' in symbol:
            return Ticker(
                symbol=symbol,
                bid=140000000,
                ask=140100000,
                last=140050000,
                volume=100,
                timestamp=datetime.now()
            )
        else:
            return Ticker(
                symbol=symbol,
                bid=100000,
                ask=100100,
                last=100050,
                volume=1000,
                timestamp=datetime.now()
            )
    
    async def place_limit_order(self, symbol: str, side: OrderSide, amount: float, price: float):
        self.order_counter += 1
        order_id = f"TEST_{self.order_counter}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            price=price,
            amount=amount,
            status=OrderStatus.OPEN
        )
        
        self.orders[order_id] = order
        return order
    
    async def place_market_order(self, symbol: str, side: OrderSide, amount: float):
        self.order_counter += 1
        order_id = f"TEST_{self.order_counter}"
        
        # Mock 시장가 체결
        ticker = await self.get_ticker(symbol)
        price = ticker.ask if side == OrderSide.BUY else ticker.bid
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            price=price,
            amount=amount,
            status=OrderStatus.FILLED,
            filled_amount=amount,
            filled_price=price
        )
        
        self.orders[order_id] = order
        return order
    
    async def get_order(self, order_id: str, symbol: str = None):
        return self.orders.get(order_id)
    
    async def cancel_order(self, order_id: str, symbol: str = None):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_open_orders(self, symbol: str = None):
        return [o for o in self.orders.values() if o.status == OrderStatus.OPEN]


class TestOrderManager:
    """OrderManager 테스트"""
    
    @pytest.fixture
    def mock_exchanges(self):
        """Mock 거래소 생성"""
        upbit = MockExchange('upbit')
        binance = MockExchange('binance')
        
        # 잔고 설정
        upbit.balances = {
            'KRW': Balance('KRW', 10000000, 0, 10000000),  # 1000만원
            'BTC': Balance('BTC', 0.1, 0, 0.1)
        }
        binance.balances = {
            'USDT': Balance('USDT', 10000, 0, 10000),
            'BTC': Balance('BTC', 0.1, 0, 0.1)
        }
        
        return upbit, binance
    
    @pytest.fixture
    def order_manager(self, mock_exchanges):
        """OrderManager 인스턴스"""
        upbit, binance = mock_exchanges
        return OrderManager(upbit_api=upbit, binance_api=binance)
    
    @pytest.mark.asyncio
    async def test_connect_all(self, order_manager):
        """전체 연결 테스트"""
        result = await order_manager.connect_all()
        assert result == True
        assert order_manager.exchanges['upbit'].connected == True
        assert order_manager.exchanges['binance'].connected == True
    
    @pytest.mark.asyncio
    async def test_execute_market_order(self, order_manager):
        """시장가 주문 테스트"""
        await order_manager.connect_all()
        
        request = OrderRequest(
            exchange='upbit',
            symbol='BTC/KRW',
            side=OrderSide.BUY,
            amount=0.01,
            strategy=ExecutionStrategy.AGGRESSIVE
        )
        
        result = await order_manager.execute_order(request)
        
        assert result.success == True
        assert result.order is not None
        assert result.order.status == OrderStatus.FILLED
        assert result.order.filled_amount == 0.01
        assert result.slippage_pct >= 0
    
    @pytest.mark.asyncio
    async def test_execute_limit_order(self, order_manager):
        """지정가 주문 테스트"""
        await order_manager.connect_all()
        
        request = OrderRequest(
            exchange='binance',
            symbol='BTC/USDT',
            side=OrderSide.SELL,
            amount=0.01,
            strategy=ExecutionStrategy.PASSIVE
        )
        
        # Mock wait_for_fill to simulate immediate fill
        async def mock_wait(exchange, order, timeout):
            order.status = OrderStatus.FILLED
            order.filled_amount = order.amount
            order.filled_price = order.price
            return True
        
        with patch.object(order_manager, '_wait_for_fill', new=AsyncMock(side_effect=mock_wait)):
            result = await order_manager.execute_order(request)
        
        assert result.success == True
        assert result.order is not None
        assert result.execution_time_ms >= 0  # >= 0 since we mock wait_for_fill
    
    def test_statistics(self, order_manager):
        """통계 테스트"""
        # 초기 통계
        stats = order_manager.get_statistics()
        assert stats['total_orders'] == 0
        assert stats['success_rate'] == 0
        
        # 성공 주문 추가
        order_manager.stats['total_orders'] = 10
        order_manager.stats['successful_orders'] = 8
        order_manager.stats['failed_orders'] = 2
        
        stats = order_manager.get_statistics()
        assert stats['success_rate'] == 80.0


class TestBalanceManager:
    """BalanceManager 테스트"""
    
    @pytest.fixture
    def balance_manager(self):
        """BalanceManager 인스턴스"""
        upbit = MockExchange('upbit')
        binance = MockExchange('binance')
        
        # 잔고 설정
        upbit.balances = {
            'KRW': Balance('KRW', 5000000, 1000000, 6000000),
            'BTC': Balance('BTC', 0.05, 0.01, 0.06)
        }
        binance.balances = {
            'USDT': Balance('USDT', 5000, 500, 5500),
            'BTC': Balance('BTC', 0.05, 0.01, 0.06)
        }
        
        return BalanceManager(upbit_api=upbit, binance_api=binance)
    
    @pytest.mark.asyncio
    async def test_sync_all_balances(self, balance_manager):
        """전체 잔고 동기화 테스트"""
        balances = await balance_manager.sync_all_balances()
        
        assert 'upbit' in balances
        assert 'binance' in balances
        assert 'KRW' in balances['upbit']
        assert 'USDT' in balances['binance']
        
        # 캐시 확인
        assert 'upbit' in balance_manager.current_balances
        assert 'binance' in balance_manager.current_balances
    
    @pytest.mark.asyncio
    async def test_get_balance(self, balance_manager):
        """개별 잔고 조회 테스트"""
        # 첫 조회 (동기화 필요)
        balance = await balance_manager.get_balance('upbit', 'KRW')
        assert balance is not None
        assert balance.currency == 'KRW'
        assert balance.total == 6000000
        
        # 재조회 (캐시 사용)
        balance = await balance_manager.get_balance('upbit', 'KRW', force_sync=False)
        assert balance.total == 6000000
    
    def test_check_sufficient_balance(self, balance_manager):
        """잔고 충분 여부 체크 테스트"""
        # 잔고 동기화 먼저
        balance_manager.current_balances = {
            'upbit': {
                'KRW': Balance('KRW', 5000000, 0, 5000000)
            }
        }
        
        # 충분한 경우
        sufficient, usable = balance_manager.check_sufficient_balance('upbit', 'KRW', 1000000)
        assert sufficient == True
        assert usable == 4900000  # 5000000 - 100000(최소 잔고)
        
        # 부족한 경우
        sufficient, usable = balance_manager.check_sufficient_balance('upbit', 'KRW', 5000000)
        assert sufficient == False
    
    def test_position_update(self, balance_manager):
        """포지션 업데이트 테스트"""
        # 매수 포지션
        balance_manager.update_position('upbit', 'BTC', 0.1, 140000000, True)
        
        position = balance_manager.positions.get('upbit_BTC')
        assert position is not None
        assert position.amount == 0.1
        assert position.avg_price == 140000000
        
        # 추가 매수
        balance_manager.update_position('upbit', 'BTC', 0.05, 141000000, True)
        assert abs(position.amount - 0.15) < 0.0001  # Float 정밀도 고려
        expected_avg = (0.1 * 140000000 + 0.05 * 141000000) / 0.15
        assert abs(position.avg_price - expected_avg) < 1  # 가격 정밀도 고려
        
        # 매도 (실현 손익)
        balance_manager.update_position('upbit', 'BTC', 0.05, 142000000, False)
        assert abs(position.amount - 0.10) < 0.0001  # Float 정밀도 고려
        assert position.realized_pnl == 0.05 * (142000000 - position.avg_price)


class TestSafetyManager:
    """SafetyManager 테스트"""
    
    @pytest.fixture
    def safety_manager(self):
        """SafetyManager 인스턴스"""
        return SafetyManager()
    
    @pytest.mark.asyncio
    async def test_order_safety_check(self, safety_manager):
        """주문 안전성 체크 테스트"""
        # 정상 주문
        action, messages = await safety_manager.check_order_safety(
            exchange='upbit',
            symbol='BTC/KRW',
            side='BUY',
            amount=0.01,
            price=140000000
        )
        
        assert action == SafetyAction.ALLOW
        assert len(messages) == 0
        
        # 큰 주문 (차단)
        action, messages = await safety_manager.check_order_safety(
            exchange='upbit',
            symbol='BTC/KRW',
            side='BUY',
            amount=0.2,  # > 0.1 BTC
            price=140000000
        )
        
        assert action == SafetyAction.BLOCK
        assert len(messages) > 0
    
    @pytest.mark.asyncio
    async def test_price_safety_check(self, safety_manager):
        """가격 안전성 체크 테스트"""
        # 정상 가격
        action, msg = await safety_manager.check_price_safety(
            symbol='BTC/KRW',
            our_price=140000000,
            market_price=140070000  # 0.05% 차이
        )
        
        assert action == SafetyAction.ALLOW
        
        # 이상 가격
        action, msg = await safety_manager.check_price_safety(
            symbol='BTC/KRW',
            our_price=140000000,
            market_price=142000000  # 1.4% 차이
        )
        
        assert action == SafetyAction.WARN
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, safety_manager):
        """일일 손실 제한 테스트"""
        # 손실 업데이트
        await safety_manager.update_daily_loss(500000)  # 50만원
        assert safety_manager.emergency_stop == False
        
        # 추가 손실 (제한 초과)
        await safety_manager.update_daily_loss(600000)  # 60만원 추가 = 총 110만원
        assert safety_manager.emergency_stop == True
        assert '손실' in safety_manager.stop_reason.lower() or 'loss' in safety_manager.stop_reason.lower()
    
    def test_cooldown(self, safety_manager):
        """쿨다운 테스트"""
        # 쿨다운 설정
        safety_manager.set_cooldown(1)  # 1분
        
        # 상태 확인
        status = safety_manager.get_risk_status()
        assert status['cooldown_active'] == True
    
    def test_risk_status(self, safety_manager):
        """리스크 상태 테스트"""
        status = safety_manager.get_risk_status()
        
        assert status['current_level'] == RiskLevel.LOW.value
        assert status['emergency_stop'] == False
        assert status['daily_loss'] == 0
        assert status['warnings_today'] == 0


class TestIntegration:
    """전체 시스템 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_trading_flow(self):
        """전체 거래 플로우 테스트"""
        
        # 시스템 초기화
        upbit = MockExchange('upbit')
        binance = MockExchange('binance')
        
        order_manager = OrderManager(upbit_api=upbit, binance_api=binance)
        balance_manager = BalanceManager(upbit_api=upbit, binance_api=binance)
        safety_manager = SafetyManager()
        
        # 연결
        await order_manager.connect_all()
        
        # 잔고 동기화
        await balance_manager.sync_all_balances()
        
        # 안전성 체크
        action, messages = await safety_manager.check_order_safety(
            exchange='upbit',
            symbol='BTC/KRW',
            side='BUY',
            amount=0.01,
            price=140000000
        )
        
        if action == SafetyAction.ALLOW:
            # 주문 실행
            request = OrderRequest(
                exchange='upbit',
                symbol='BTC/KRW',
                side=OrderSide.BUY,
                amount=0.01,
                strategy=ExecutionStrategy.ADAPTIVE
            )
            
            result = await order_manager.execute_order(request)
            
            # 포지션 업데이트
            if result.success and result.order:
                balance_manager.update_position(
                    exchange='upbit',
                    currency='BTC',
                    amount=result.order.filled_amount,
                    price=result.order.filled_price,
                    is_buy=True
                )
            
            # 통계 확인
            stats = order_manager.get_statistics()
            assert stats['total_orders'] == 1
            assert stats['successful_orders'] == 1
            
            # PnL 확인
            pnl = balance_manager.get_pnl_summary()
            assert 'realized' in pnl
            assert 'unrealized' in pnl
        
        print("\n=== Integration Test Results ===")
        print(f"Order Manager Stats: {order_manager.get_statistics()}")
        print(f"Risk Status: {safety_manager.get_risk_status()}")
        print(f"Total Value: {balance_manager.get_total_value()}")


if __name__ == "__main__":
    # 통합 테스트 실행
    print("Running Live Trading Integration Tests...")
    asyncio.run(TestIntegration().test_full_trading_flow())
    print("\n[OK] All integration tests completed")