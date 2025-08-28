"""
자동 복구 시스템 테스트
SafetyManager의 자동 복구 기능 검증
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.safety_manager import (
    SafetyManager, 
    SafetyRule, 
    SafetyAction, 
    RiskLevel,
    SafetyAlert
)


class TestAutoRecovery:
    """자동 복구 시스템 테스트"""
    
    @pytest.fixture
    def safety_manager(self):
        """테스트용 SafetyManager 인스턴스"""
        manager = SafetyManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_emergency_stop_triggers_auto_recovery(self, safety_manager):
        """긴급 정지시 자동 복구 모니터 시작 확인"""
        # 자동 복구 활성화
        safety_manager.auto_recovery_enabled = True
        
        # 긴급 정지 발동
        await safety_manager.trigger_emergency_stop("Test emergency")
        
        # 확인
        assert safety_manager.emergency_stop == True
        assert safety_manager.stop_reason == "Test emergency"
        assert safety_manager.stop_time is not None
        assert safety_manager.recovery_task is not None
        
        # 정리
        safety_manager.release_emergency_stop()
    
    @pytest.mark.asyncio
    async def test_manual_release_cancels_auto_recovery(self, safety_manager):
        """수동 해제시 자동 복구 취소 확인"""
        safety_manager.auto_recovery_enabled = True
        
        # 긴급 정지 발동
        await safety_manager.trigger_emergency_stop("Test emergency")
        recovery_task = safety_manager.recovery_task
        
        # 수동 해제
        safety_manager.release_emergency_stop()
        
        # 확인
        assert safety_manager.emergency_stop == False
        assert safety_manager.recovery_task is None
        assert recovery_task.cancelled() or recovery_task.done()
    
    @pytest.mark.asyncio
    async def test_recovery_conditions_min_time(self, safety_manager):
        """최소 복구 시간 조건 테스트"""
        safety_manager.min_recovery_time = 10  # 10초로 설정
        safety_manager.stop_time = datetime.now()
        
        # 즉시 체크 - False
        result = await safety_manager._check_recovery_conditions(0)
        assert result == False
        
        # 5초 후 - 여전히 False
        result = await safety_manager._check_recovery_conditions(5)
        assert result == False
        
        # 15초 후 - True (모든 조건 충족시)
        with patch.object(safety_manager, '_check_market_stability', return_value=True), \
             patch.object(safety_manager, '_check_system_health', return_value=True), \
             patch.object(safety_manager, '_check_risk_levels', return_value=True), \
             patch.object(safety_manager, '_check_no_recent_errors', return_value=True):
            result = await safety_manager._check_recovery_conditions(15)
            assert result == True
    
    @pytest.mark.asyncio
    async def test_recovery_conditions_max_time(self, safety_manager):
        """최대 복구 시간 강제 복구 테스트"""
        safety_manager.max_recovery_time = 60  # 60초로 설정
        safety_manager.stop_time = datetime.now()
        
        # 최대 시간 도달시 조건 무시하고 복구
        with patch.object(safety_manager, '_check_market_stability', return_value=False), \
             patch.object(safety_manager, '_check_system_health', return_value=False):
            result = await safety_manager._check_recovery_conditions(61)
            assert result == True
    
    @pytest.mark.asyncio
    async def test_partial_recovery_conditions(self, safety_manager):
        """부분 복구 조건 테스트 (1시간 후 3/4 조건)"""
        safety_manager.min_recovery_time = 30
        safety_manager.stop_time = datetime.now()
        
        # 3/4 조건 충족 + 1시간 경과
        with patch.object(safety_manager, '_check_market_stability', return_value=True), \
             patch.object(safety_manager, '_check_system_health', return_value=True), \
             patch.object(safety_manager, '_check_risk_levels', return_value=True), \
             patch.object(safety_manager, '_check_no_recent_errors', return_value=False):
            # 30분 후 - False (1시간 미만)
            result = await safety_manager._check_recovery_conditions(1800)
            assert result == False
            
            # 1시간 후 - True (3/4 조건 + 1시간)
            result = await safety_manager._check_recovery_conditions(3601)
            assert result == True
    
    def test_risk_level_check(self, safety_manager):
        """리스크 레벨 체크 테스트"""
        safety_manager.daily_stats['total_loss'] = -400000  # 40만원 손실
        safety_manager.rules['max_daily_loss'].threshold = 1000000  # 100만원 한도
        
        # 50% 미만이므로 True
        result = safety_manager._check_risk_levels()
        assert result == True
        
        # 60만원 손실 - 50% 초과
        safety_manager.daily_stats['total_loss'] = -600000
        result = safety_manager._check_risk_levels()
        assert result == False
    
    def test_recent_errors_check(self, safety_manager):
        """최근 에러 체크 테스트"""
        # 에러 없음 - True
        assert safety_manager._check_no_recent_errors() == True
        
        # 오래된 Critical 에러 - True
        old_alert = SafetyAlert(
            timestamp=datetime.now() - timedelta(minutes=15),
            level=RiskLevel.CRITICAL,
            rule_name="test",
            message="old error",
            action_taken=SafetyAction.BLOCK
        )
        safety_manager.alerts.append(old_alert)
        assert safety_manager._check_no_recent_errors() == True
        
        # 최근 Critical 에러 - False
        recent_alert = SafetyAlert(
            timestamp=datetime.now() - timedelta(minutes=5),
            level=RiskLevel.CRITICAL,
            rule_name="test",
            message="recent error",
            action_taken=SafetyAction.EMERGENCY_STOP
        )
        safety_manager.alerts.append(recent_alert)
        assert safety_manager._check_no_recent_errors() == False
    
    @pytest.mark.asyncio
    async def test_gradual_recovery_execution(self, safety_manager):
        """점진적 복구 실행 테스트"""
        safety_manager.stop_time = datetime.now() - timedelta(minutes=35)
        safety_manager.stop_reason = "Test stop"
        safety_manager.emergency_stop = True
        
        # Mock 시스템 체크
        with patch.object(safety_manager, '_perform_system_check', new=AsyncMock()), \
             patch.object(safety_manager, '_create_recovery_alert', new=AsyncMock()), \
             patch.object(safety_manager, 'set_cooldown') as mock_cooldown:
            
            await safety_manager._execute_gradual_recovery()
            
            # 확인
            assert safety_manager.emergency_stop == False
            assert safety_manager.stop_reason is None
            assert safety_manager.stop_time is None
            assert safety_manager.recovery_count == 1
            mock_cooldown.assert_called_once_with(30)  # 30분 쿨다운
    
    def test_risk_parameter_adjustment(self, safety_manager):
        """리스크 파라미터 조정 테스트"""
        # 초기값 저장
        original_values = {}
        for rule_name in ['max_position_loss', 'max_position_size']:
            if rule_name in safety_manager.rules:
                original_values[rule_name] = safety_manager.rules[rule_name].threshold
        
        # 보수적으로 조정
        safety_manager._adjust_risk_parameters(conservative=True)
        
        # 확인 - 값이 감소해야 함
        for rule_name in ['max_position_loss', 'max_position_size']:
            if rule_name in safety_manager.rules:
                assert safety_manager.rules[rule_name].threshold < original_values[rule_name]
    
    @pytest.mark.asyncio
    async def test_recovery_alert_creation(self, safety_manager):
        """복구 알림 생성 테스트"""
        safety_manager.recovery_count = 2
        
        # 성공 알림
        await safety_manager._create_recovery_alert("success", 45.5)
        
        assert len(safety_manager.alerts) == 1
        alert = safety_manager.alerts[-1]
        assert alert.level == RiskLevel.MEDIUM
        assert "successful" in alert.message
        assert alert.details['recovery_count'] == 2
        assert alert.details['downtime_minutes'] == 45.5
        
        # 실패 알림
        await safety_manager._create_recovery_alert("failed", 30.0, "Connection error")
        
        assert len(safety_manager.alerts) == 2
        alert = safety_manager.alerts[-1]
        assert alert.level == RiskLevel.CRITICAL
        assert "failed" in alert.message
        assert alert.details['error'] == "Connection error"
    
    @pytest.mark.asyncio
    async def test_auto_recovery_disabled(self, safety_manager):
        """자동 복구 비활성화시 동작 테스트"""
        safety_manager.auto_recovery_enabled = False
        
        # 긴급 정지 발동
        await safety_manager.trigger_emergency_stop("Test emergency")
        
        # 자동 복구 태스크가 생성되지 않아야 함
        assert safety_manager.recovery_task is None
        assert safety_manager.emergency_stop == True
        
        # 정리
        safety_manager.release_emergency_stop()
    
    @pytest.mark.asyncio
    async def test_recovery_monitor_error_handling(self, safety_manager):
        """복구 모니터 에러 처리 테스트"""
        safety_manager.auto_recovery_enabled = True
        safety_manager.recovery_check_interval = 0.1  # 빠른 테스트
        safety_manager.min_recovery_time = 0.1
        
        # 에러 발생 시뮬레이션
        with patch.object(safety_manager, '_check_recovery_conditions', 
                         side_effect=Exception("Test error")):
            await safety_manager.trigger_emergency_stop("Test")
            
            # 잠시 대기 (에러 발생)
            await asyncio.sleep(0.2)
            
            # 에러가 발생해도 모니터가 계속 실행되어야 함
            assert safety_manager.recovery_task is not None
            assert not safety_manager.recovery_task.done()
            
            # 정리
            safety_manager.release_emergency_stop()


class TestIntegrationAutoRecovery:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_recovery_cycle(self):
        """전체 복구 사이클 테스트"""
        manager = SafetyManager()
        manager.auto_recovery_enabled = True
        manager.min_recovery_time = 0.5  # 빠른 테스트
        manager.recovery_check_interval = 0.1
        
        # 일일 손실 설정 (임계값 이하)
        manager.daily_stats['total_loss'] = -300000
        
        # Mock 설정
        with patch.object(manager, '_perform_system_check', new=AsyncMock()), \
             patch.object(manager, 'set_cooldown'), \
             patch('asyncio.sleep', new=AsyncMock()):
            
            # 긴급 정지 발동
            await manager.trigger_emergency_stop("Integration test")
            assert manager.emergency_stop == True
            
            # 복구 조건 충족 시뮬레이션
            manager.stop_time = datetime.now() - timedelta(seconds=1)
            
            # 수동으로 복구 실행
            await manager._execute_gradual_recovery()
            
            # 확인
            assert manager.emergency_stop == False
            assert manager.recovery_count == 1
            assert len(manager.alerts) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_recovery_attempts(self):
        """다중 복구 시도 테스트"""
        manager = SafetyManager()
        
        # 첫 번째 복구
        manager.stop_time = datetime.now() - timedelta(minutes=35)
        manager.emergency_stop = True
        
        with patch.object(manager, '_perform_system_check', new=AsyncMock()), \
             patch.object(manager, 'set_cooldown'), \
             patch('asyncio.sleep', new=AsyncMock()):
            
            await manager._execute_gradual_recovery()
            assert manager.recovery_count == 1
            
            # 두 번째 긴급 정지 및 복구
            await manager.trigger_emergency_stop("Second stop")
            manager.stop_time = datetime.now() - timedelta(minutes=35)
            await manager._execute_gradual_recovery()
            assert manager.recovery_count == 2


if __name__ == "__main__":
    pytest.main([__file__, '-v'])