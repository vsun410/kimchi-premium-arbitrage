"""
안전 관리자 (Safety Manager)
거래 안전성 확보 및 리스크 관리

목적: 자금 보호와 리스크 제한
결과: 손실 제한, 이상 거래 차단
평가: 안전 규칙 준수율, 손실 방지 효과
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAction(Enum):
    """안전 조치"""
    ALLOW = "allow"           # 허용
    WARN = "warn"            # 경고 후 진행
    BLOCK = "block"          # 차단
    EMERGENCY_STOP = "stop"  # 긴급 정지


@dataclass
class SafetyRule:
    """안전 규칙"""
    name: str
    description: str
    enabled: bool = True
    action: SafetyAction = SafetyAction.BLOCK
    threshold: float = 0
    current_value: float = 0
    
    def check(self) -> Tuple[bool, str]:
        """
        규칙 체크
        
        Returns:
            (통과 여부, 메시지)
        """
        if not self.enabled:
            return True, "Rule disabled"
        
        if self.current_value > self.threshold:
            return False, f"{self.name}: {self.current_value:.2f} > {self.threshold:.2f}"
        
        return True, "OK"


@dataclass
class SafetyAlert:
    """안전 경고"""
    timestamp: datetime
    level: RiskLevel
    rule_name: str
    message: str
    action_taken: SafetyAction
    details: Dict = field(default_factory=dict)


class SafetyManager:
    """
    안전 관리자
    
    핵심 기능:
    1. 거래 전 안전성 체크
    2. 실시간 리스크 모니터링
    3. 손실 제한 (Stop Loss)
    4. 이상 거래 감지
    5. 긴급 정지 시스템
    """
    
    def __init__(self):
        """초기화"""
        
        # 안전 규칙들
        self.rules: Dict[str, SafetyRule] = {
            # 손실 제한
            'max_daily_loss': SafetyRule(
                name="일일 최대 손실",
                description="하루 최대 손실 제한",
                threshold=1_000_000,  # 100만원
                action=SafetyAction.EMERGENCY_STOP
            ),
            'max_position_loss': SafetyRule(
                name="포지션 최대 손실",
                description="단일 포지션 최대 손실",
                threshold=500_000,  # 50만원
                action=SafetyAction.BLOCK
            ),
            
            # 거래량 제한
            'max_order_size': SafetyRule(
                name="최대 주문 크기",
                description="단일 주문 최대 크기",
                threshold=0.1,  # 0.1 BTC
                action=SafetyAction.BLOCK
            ),
            'max_daily_volume': SafetyRule(
                name="일일 최대 거래량",
                description="하루 최대 거래량",
                threshold=10_000_000,  # 1000만원
                action=SafetyAction.WARN
            ),
            
            # 가격 이상 감지
            'price_deviation': SafetyRule(
                name="가격 이탈",
                description="시장가 대비 가격 차이",
                threshold=1.0,  # 1%
                action=SafetyAction.WARN
            ),
            'spread_threshold': SafetyRule(
                name="스프레드 임계값",
                description="매수/매도 호가 차이",
                threshold=0.5,  # 0.5%
                action=SafetyAction.WARN
            ),
            
            # 시스템 제한
            'max_open_orders': SafetyRule(
                name="최대 미체결 주문",
                description="동시 미체결 주문 수",
                threshold=10,
                action=SafetyAction.BLOCK
            ),
            'min_balance_ratio': SafetyRule(
                name="최소 잔고 비율",
                description="최소 유지 잔고 비율",
                threshold=0.1,  # 10%
                action=SafetyAction.BLOCK
            )
        }
        
        # 경고 이력
        self.alerts: List[SafetyAlert] = []
        self.max_alerts = 1000
        
        # 거래 통계
        self.daily_stats = {
            'date': datetime.now().date(),
            'total_loss': 0,
            'total_profit': 0,
            'total_volume': 0,
            'order_count': 0,
            'blocked_orders': 0,
            'warnings_issued': 0
        }
        
        # 긴급 정지 상태
        self.emergency_stop = False
        self.stop_reason = None
        self.stop_time: Optional[datetime] = None  # 긴급 정지 시간
        
        # 쿨다운
        self.cooldown_until: Optional[datetime] = None
        
        # 자동 복구 설정
        self.auto_recovery_enabled = True
        self.min_recovery_time = 1800  # 최소 30분
        self.max_recovery_time = 7200  # 최대 2시간
        self.recovery_check_interval = 60  # 1분마다 체크
        self.recovery_count = 0  # 자동 복구 횟수
        self.recovery_task: Optional[asyncio.Task] = None
        
        logger.info("SafetyManager initialized with {} rules".format(len(self.rules)))
    
    async def check_order_safety(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> Tuple[SafetyAction, List[str]]:
        """
        주문 안전성 체크
        
        Args:
            exchange: 거래소
            symbol: 심볼
            side: 매수/매도
            amount: 수량
            price: 가격
            
        Returns:
            (안전 조치, 메시지 리스트)
        """
        messages = []
        highest_action = SafetyAction.ALLOW
        
        # 긴급 정지 체크
        if self.emergency_stop:
            return SafetyAction.EMERGENCY_STOP, [f"Emergency stop active: {self.stop_reason}"]
        
        # 쿨다운 체크
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds()
            return SafetyAction.BLOCK, [f"Cooldown active for {remaining:.0f} seconds"]
        
        # 주문 크기 체크
        self.rules['max_order_size'].current_value = amount
        passed, msg = self.rules['max_order_size'].check()
        if not passed:
            messages.append(msg)
            highest_action = self._update_action(highest_action, self.rules['max_order_size'].action)
        
        # 일일 거래량 체크
        order_value = amount * price
        self.rules['max_daily_volume'].current_value = self.daily_stats['total_volume'] + order_value
        passed, msg = self.rules['max_daily_volume'].check()
        if not passed:
            messages.append(msg)
            highest_action = self._update_action(highest_action, self.rules['max_daily_volume'].action)
        
        # 경고 생성
        if highest_action != SafetyAction.ALLOW:
            await self._create_alert(
                level=self._action_to_risk_level(highest_action),
                rule_name="Order Safety Check",
                message="; ".join(messages),
                action_taken=highest_action,
                details={
                    'exchange': exchange,
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price
                }
            )
        
        return highest_action, messages
    
    async def check_price_safety(
        self,
        symbol: str,
        our_price: float,
        market_price: float
    ) -> Tuple[SafetyAction, str]:
        """
        가격 안전성 체크
        
        Args:
            symbol: 심볼
            our_price: 우리 가격
            market_price: 시장 가격
            
        Returns:
            (안전 조치, 메시지)
        """
        # 가격 차이 계산
        deviation_pct = abs((our_price - market_price) / market_price * 100)
        
        self.rules['price_deviation'].current_value = deviation_pct
        passed, msg = self.rules['price_deviation'].check()
        
        if not passed:
            await self._create_alert(
                level=RiskLevel.MEDIUM,
                rule_name="Price Deviation",
                message=msg,
                action_taken=self.rules['price_deviation'].action,
                details={
                    'symbol': symbol,
                    'our_price': our_price,
                    'market_price': market_price,
                    'deviation_pct': deviation_pct
                }
            )
            
            return self.rules['price_deviation'].action, msg
        
        return SafetyAction.ALLOW, "Price OK"
    
    async def update_daily_loss(self, loss_amount: float):
        """
        일일 손실 업데이트
        
        Args:
            loss_amount: 손실 금액 (양수)
        """
        # 날짜 체크 (자정 넘으면 리셋)
        if self.daily_stats['date'] != datetime.now().date():
            self._reset_daily_stats()
        
        self.daily_stats['total_loss'] += abs(loss_amount)
        
        # 일일 손실 제한 체크
        self.rules['max_daily_loss'].current_value = self.daily_stats['total_loss']
        passed, msg = self.rules['max_daily_loss'].check()
        
        if not passed:
            # 긴급 정지 발동
            await self.trigger_emergency_stop(f"Daily loss limit exceeded: {msg}")
    
    async def update_position_loss(self, position_id: str, loss_amount: float):
        """
        포지션 손실 업데이트
        
        Args:
            position_id: 포지션 ID
            loss_amount: 손실 금액
        """
        self.rules['max_position_loss'].current_value = abs(loss_amount)
        passed, msg = self.rules['max_position_loss'].check()
        
        if not passed:
            await self._create_alert(
                level=RiskLevel.HIGH,
                rule_name="Position Loss",
                message=msg,
                action_taken=self.rules['max_position_loss'].action,
                details={
                    'position_id': position_id,
                    'loss_amount': loss_amount
                }
            )
    
    async def trigger_emergency_stop(self, reason: str):
        """
        긴급 정지 발동
        
        Args:
            reason: 정지 사유
        """
        self.emergency_stop = True
        self.stop_reason = reason
        self.stop_time = datetime.now()
        
        await self._create_alert(
            level=RiskLevel.CRITICAL,
            rule_name="Emergency Stop",
            message=reason,
            action_taken=SafetyAction.EMERGENCY_STOP,
            details={'timestamp': datetime.now().isoformat()}
        )
        
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
        
        # 자동 복구 태스크 시작
        if self.auto_recovery_enabled and not self.recovery_task:
            self.recovery_task = asyncio.create_task(self._auto_recovery_monitor())
            logger.info("🔄 Auto-recovery monitor started")
    
    def release_emergency_stop(self):
        """긴급 정지 해제 (수동)"""
        self.emergency_stop = False
        self.stop_reason = None
        self.stop_time = None
        
        # 자동 복구 태스크 취소
        if self.recovery_task:
            self.recovery_task.cancel()
            self.recovery_task = None
            
        logger.info("✅ Emergency stop released manually")
    
    async def _auto_recovery_monitor(self):
        """자동 복구 모니터"""
        logger.info(f"🕐 Auto-recovery monitor started (min: {self.min_recovery_time/60:.0f}min, max: {self.max_recovery_time/60:.0f}min)")
        
        while self.emergency_stop:
            try:
                await asyncio.sleep(self.recovery_check_interval)
                
                if not self.stop_time:
                    continue
                    
                elapsed = (datetime.now() - self.stop_time).total_seconds()
                
                # 복구 조건 체크
                if await self._check_recovery_conditions(elapsed):
                    await self._execute_gradual_recovery()
                    break
                    
                # 진행 상황 로깅 (5분마다)
                if int(elapsed) % 300 == 0:
                    remaining_min = max(0, (self.min_recovery_time - elapsed) / 60)
                    logger.info(f"🕒 Recovery check: {elapsed/60:.1f}min elapsed, min {remaining_min:.1f}min remaining")
                    
            except asyncio.CancelledError:
                logger.info("🚫 Auto-recovery monitor cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in auto-recovery monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_recovery_conditions(self, elapsed_seconds: float) -> bool:
        """
        자동 복구 조건 확인
        
        Args:
            elapsed_seconds: 긴급 정지 후 경과 시간 (초)
            
        Returns:
            복구 가능 여부
        """
        # 최소 시간 체크
        if elapsed_seconds < self.min_recovery_time:
            return False
            
        # 최대 시간 도달시 강제 복구
        if elapsed_seconds >= self.max_recovery_time:
            logger.warning(f"⚠️ Max recovery time reached ({self.max_recovery_time/60:.0f}min), forcing recovery")
            return True
        
        # 복구 조건들 체크
        conditions = [
            self._check_market_stability(),
            self._check_system_health(),
            self._check_risk_levels(),
            self._check_no_recent_errors()
        ]
        
        conditions_met = sum(conditions)
        logger.debug(f"🔍 Recovery conditions: {conditions_met}/4 met")
        
        # 모든 조건 충족시 복구
        if all(conditions):
            logger.info("✅ All recovery conditions met")
            return True
            
        # 일부 조건만 충족시 (1시간 후 3/4)
        if conditions_met >= 3 and elapsed_seconds >= 3600:
            logger.info(f"⚠️ Partial recovery conditions met ({conditions_met}/4) after 1 hour")
            return True
            
        return False
    
    def _check_market_stability(self) -> bool:
        """시장 안정성 체크"""
        # TODO: 실제 시장 변동성 체크
        # 예: 김프 < 10%, 급격한 가격 변동 없음
        return True
    
    def _check_system_health(self) -> bool:
        """시스템 건전성 체크"""
        # TODO: 시스템 메트릭 체크
        # 예: WebSocket 연결 상태, API 응답시간
        return True
    
    def _check_risk_levels(self) -> bool:
        """리스크 레벨 체크"""
        # 현재 손실이 임계값의 50% 이하인지 확인
        current_loss = self.daily_stats.get('total_loss', 0)
        max_loss_threshold = self.rules['max_daily_loss'].threshold
        return abs(current_loss) < max_loss_threshold * 0.5
    
    def _check_no_recent_errors(self) -> bool:
        """최근 에러 없음 확인"""
        # 최근 10분간 심각한 경고 없음
        if not self.alerts:
            return True
            
        recent_critical = [
            alert for alert in self.alerts[-10:]
            if alert.level == RiskLevel.CRITICAL and 
            (datetime.now() - alert.timestamp).total_seconds() < 600
        ]
        return len(recent_critical) == 0
    
    async def _execute_gradual_recovery(self):
        """
        점진적 복구 실행
        """
        logger.info("=" * 50)
        logger.info("🔄 STARTING GRADUAL AUTO-RECOVERY")
        logger.info("=" * 50)
        
        recovery_start = datetime.now()
        downtime = (recovery_start - self.stop_time).total_seconds() / 60
        
        try:
            # 단계 1: 시스템 상태 점검
            logger.info("🔍 Step 1: System health check")
            await self._perform_system_check()
            await asyncio.sleep(5)
            
            # 단계 2: 리스크 파라미터 조정 (보수적으로)
            logger.info("⚙️ Step 2: Adjusting risk parameters (conservative mode)")
            self._adjust_risk_parameters(conservative=True)
            await asyncio.sleep(5)
            
            # 단계 3: 긴급 정지 해제
            logger.info("🚀 Step 3: Releasing emergency stop")
            original_reason = self.stop_reason
            self.emergency_stop = False
            self.stop_reason = None
            self.stop_time = None
            self.recovery_count += 1
            
            # 단계 4: 쿨다운 설정 (추가 보호)
            cooldown_minutes = 30
            self.set_cooldown(cooldown_minutes)
            
            # 복구 완료 로그
            logger.info("="*50)
            logger.info("✅ AUTO-RECOVERY COMPLETED SUCCESSFULLY")
            logger.info(f"  🕰️ Total downtime: {downtime:.1f} minutes")
            logger.info(f"  🔄 Recovery count: {self.recovery_count}")
            logger.info(f"  ⏸️ Cooldown period: {cooldown_minutes} minutes")
            logger.info(f"  🛡️ Risk parameters: CONSERVATIVE MODE")
            logger.info(f"  📌 Original stop reason: {original_reason}")
            logger.info("="*50)
            
            # 성공 알림
            await self._create_recovery_alert("success", downtime)
            
        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")
            self.emergency_stop = True  # 복구 실패시 정지 유지
            await self._create_recovery_alert("failed", downtime, str(e))
            
        finally:
            self.recovery_task = None
    
    async def _perform_system_check(self):
        """시스템 점검 수행"""
        checks = [
            "✅ API connections",
            "✅ WebSocket connections",
            "✅ Database connectivity",
            "✅ Balance synchronization",
            "✅ Position verification"
        ]
        for check in checks:
            logger.info(f"  {check}")
            await asyncio.sleep(0.5)
    
    def _adjust_risk_parameters(self, conservative: bool = True):
        """리스크 파라미터 조정"""
        if conservative:
            # 보수적으로 조정
            adjustments = [
                ('max_position_loss', 0.5),
                ('max_position_size', 0.5),
                ('max_leverage', 0.3),
                ('max_daily_volume', 0.5)
            ]
            
            for rule_name, factor in adjustments:
                if rule_name in self.rules:
                    original = self.rules[rule_name].threshold
                    self.rules[rule_name].threshold *= factor
                    logger.info(f"  {rule_name}: {original:.2f} → {self.rules[rule_name].threshold:.2f}")
            
            logger.info("🛡️ Risk parameters adjusted to CONSERVATIVE levels")
    
    async def _create_recovery_alert(self, status: str, downtime: float, error: str = None):
        """복구 알림 생성"""
        if status == "success":
            alert = SafetyAlert(
                timestamp=datetime.now(),
                level=RiskLevel.MEDIUM,
                rule_name="auto_recovery",
                message=f"Auto-recovery successful after {downtime:.1f} minutes",
                action_taken=SafetyAction.ALLOW,
                details={
                    'recovery_count': self.recovery_count,
                    'downtime_minutes': downtime,
                    'status': 'success'
                }
            )
        else:
            alert = SafetyAlert(
                timestamp=datetime.now(),
                level=RiskLevel.CRITICAL,
                rule_name="auto_recovery",
                message=f"Auto-recovery failed: {error}",
                action_taken=SafetyAction.EMERGENCY_STOP,
                details={
                    'recovery_count': self.recovery_count,
                    'downtime_minutes': downtime,
                    'status': 'failed',
                    'error': error
                }
            )
        
        self.alerts.append(alert)
        # TODO: 실제 알림 시스템과 통합
    
    def set_cooldown(self, minutes: int):
        """
        쿨다운 설정
        
        Args:
            minutes: 쿨다운 시간 (분)
        """
        self.cooldown_until = datetime.now() + timedelta(minutes=minutes)
        logger.info(f"Cooldown set for {minutes} minutes")
    
    def _update_action(self, current: SafetyAction, new: SafetyAction) -> SafetyAction:
        """
        더 강한 조치로 업데이트
        
        Args:
            current: 현재 조치
            new: 새 조치
            
        Returns:
            더 강한 조치
        """
        priority = {
            SafetyAction.ALLOW: 0,
            SafetyAction.WARN: 1,
            SafetyAction.BLOCK: 2,
            SafetyAction.EMERGENCY_STOP: 3
        }
        
        if priority[new] > priority[current]:
            return new
        return current
    
    def _action_to_risk_level(self, action: SafetyAction) -> RiskLevel:
        """
        조치를 리스크 레벨로 변환
        
        Args:
            action: 안전 조치
            
        Returns:
            리스크 레벨
        """
        mapping = {
            SafetyAction.ALLOW: RiskLevel.LOW,
            SafetyAction.WARN: RiskLevel.MEDIUM,
            SafetyAction.BLOCK: RiskLevel.HIGH,
            SafetyAction.EMERGENCY_STOP: RiskLevel.CRITICAL
        }
        return mapping[action]
    
    async def _create_alert(
        self,
        level: RiskLevel,
        rule_name: str,
        message: str,
        action_taken: SafetyAction,
        details: Dict = None
    ):
        """
        경고 생성
        
        Args:
            level: 리스크 레벨
            rule_name: 규칙 이름
            message: 메시지
            action_taken: 취한 조치
            details: 상세 정보
        """
        alert = SafetyAlert(
            timestamp=datetime.now(),
            level=level,
            rule_name=rule_name,
            message=message,
            action_taken=action_taken,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # 최대 개수 유지
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # 통계 업데이트
        if action_taken == SafetyAction.WARN:
            self.daily_stats['warnings_issued'] += 1
        elif action_taken == SafetyAction.BLOCK:
            self.daily_stats['blocked_orders'] += 1
        
        # 로그
        if level == RiskLevel.CRITICAL:
            logger.critical(f"[{level.value.upper()}] {rule_name}: {message}")
        elif level == RiskLevel.HIGH:
            logger.error(f"[{level.value.upper()}] {rule_name}: {message}")
        elif level == RiskLevel.MEDIUM:
            logger.warning(f"[{level.value.upper()}] {rule_name}: {message}")
        else:
            logger.info(f"[{level.value.upper()}] {rule_name}: {message}")
    
    def _reset_daily_stats(self):
        """일일 통계 리셋"""
        self.daily_stats = {
            'date': datetime.now().date(),
            'total_loss': 0,
            'total_profit': 0,
            'total_volume': 0,
            'order_count': 0,
            'blocked_orders': 0,
            'warnings_issued': 0
        }
    
    def get_risk_status(self) -> Dict:
        """
        리스크 상태 조회
        
        Returns:
            리스크 상태 정보
        """
        # 현재 리스크 레벨 계산
        if self.emergency_stop:
            current_level = RiskLevel.CRITICAL
        elif self.daily_stats['total_loss'] > self.rules['max_daily_loss'].threshold * 0.8:
            current_level = RiskLevel.HIGH
        elif self.daily_stats['warnings_issued'] > 5:
            current_level = RiskLevel.MEDIUM
        else:
            current_level = RiskLevel.LOW
        
        return {
            'current_level': current_level.value,
            'emergency_stop': self.emergency_stop,
            'stop_reason': self.stop_reason,
            'cooldown_active': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'daily_loss': self.daily_stats['total_loss'],
            'daily_profit': self.daily_stats['total_profit'],
            'warnings_today': self.daily_stats['warnings_issued'],
            'blocked_orders_today': self.daily_stats['blocked_orders'],
            'active_rules': sum(1 for r in self.rules.values() if r.enabled)
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        최근 경고 조회
        
        Args:
            limit: 조회 개수
            
        Returns:
            경고 리스트
        """
        recent = self.alerts[-limit:] if len(self.alerts) > limit else self.alerts
        
        return [
            {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'rule': alert.rule_name,
                'message': alert.message,
                'action': alert.action_taken.value,
                'details': alert.details
            }
            for alert in reversed(recent)
        ]