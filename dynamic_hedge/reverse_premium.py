"""
Task 32: Reverse Premium Handler
역프리미엄 대응 시스템 - 손실을 수익으로 전환
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PremiumState(Enum):
    """프리미엄 상태"""
    NORMAL = "normal"  # 정상 김프 (양수)
    REVERSE = "reverse"  # 역프 (음수)
    NEUTRAL = "neutral"  # 중립 (±0.5%)


@dataclass
class ReversePremiumEvent:
    """역프리미엄 이벤트"""
    start_time: datetime
    end_time: Optional[datetime]
    min_premium: float  # 최저 프리미엄 (음수)
    max_loss: float  # 최대 손실
    recovery_profit: float  # 회복 시 수익
    status: str  # 'active', 'recovered', 'expired'
    trend_strength: float  # 추세 강도
    
    
@dataclass
class OptimalExitPoint:
    """최적 청산 포인트"""
    timestamp: datetime
    premium_level: float
    expected_profit: float
    confidence: float
    strategy: str  # 'immediate', 'wait_recovery', 'partial_close'
    reasoning: str


class ReversePremiumHandler:
    """
    역프리미엄 대응 시스템
    - 역프리미엄 감지 및 알림
    - 추세 강도 측정
    - 최적 청산 타이밍 계산
    - 손실 → 수익 전환 로직
    """
    
    def __init__(self, alert_threshold: float = -0.005,
                 recovery_target: float = 0.01):
        """
        Args:
            alert_threshold: 역프 알림 임계값 (-0.5%)
            recovery_target: 회복 목표 프리미엄 (1%)
        """
        self.alert_threshold = alert_threshold
        self.recovery_target = recovery_target
        self.premium_history: List[float] = []
        self.reverse_events: List[ReversePremiumEvent] = []
        self.active_event: Optional[ReversePremiumEvent] = None
        self.alert_sent = False
        
    def update(self, current_premium: float, market_data: Dict) -> Dict:
        """
        역프리미엄 상태 업데이트
        
        Args:
            current_premium: 현재 김치 프리미엄
            market_data: {
                'upbit_price': float,
                'binance_price': float,
                'volume': float,
                'trend': str
            }
            
        Returns:
            상태 및 권장 액션
        """
        self.premium_history.append(current_premium)
        
        # 상태 판단
        state = self._classify_premium_state(current_premium)
        
        result = {
            'state': state,
            'current_premium': current_premium,
            'is_reverse': state == PremiumState.REVERSE,
            'alert': False,
            'action': None,
            'analysis': {}
        }
        
        # 역프 진입 감지
        if state == PremiumState.REVERSE and not self.active_event:
            self._start_reverse_event(current_premium, market_data)
            result['alert'] = True
            result['action'] = self._analyze_initial_response(current_premium, market_data)
            
        # 역프 진행 중
        elif self.active_event:
            self._update_active_event(current_premium, market_data)
            result['analysis'] = self._analyze_reverse_premium(market_data)
            result['action'] = self._determine_optimal_action(current_premium, market_data)
            
            # 역프 종료 확인
            if state != PremiumState.REVERSE:
                self._end_reverse_event(current_premium)
                result['alert'] = True
                
        return result
    
    def _classify_premium_state(self, premium: float) -> PremiumState:
        """
        프리미엄 상태 분류
        
        Args:
            premium: 현재 프리미엄
            
        Returns:
            프리미엄 상태
        """
        if premium < self.alert_threshold:
            return PremiumState.REVERSE
        elif abs(premium) < 0.005:  # ±0.5%
            return PremiumState.NEUTRAL
        else:
            return PremiumState.NORMAL
    
    def _start_reverse_event(self, premium: float, market_data: Dict):
        """
        역프 이벤트 시작
        
        Args:
            premium: 현재 프리미엄
            market_data: 시장 데이터
        """
        self.active_event = ReversePremiumEvent(
            start_time=datetime.now(),
            end_time=None,
            min_premium=premium,
            max_loss=0,
            recovery_profit=0,
            status='active',
            trend_strength=self._measure_trend_strength(market_data)
        )
        
        logger.warning(f"Reverse premium detected: {premium:.2%}")
        self.alert_sent = True
    
    def _update_active_event(self, premium: float, market_data: Dict):
        """
        활성 역프 이벤트 업데이트
        
        Args:
            premium: 현재 프리미엄
            market_data: 시장 데이터
        """
        if not self.active_event:
            return
            
        # 최저 프리미엄 업데이트
        self.active_event.min_premium = min(self.active_event.min_premium, premium)
        
        # 추세 강도 업데이트
        self.active_event.trend_strength = self._measure_trend_strength(market_data)
        
        # 예상 손실/수익 계산
        position_value = market_data.get('position_value', 0)
        if position_value > 0:
            self.active_event.max_loss = abs(self.active_event.min_premium) * position_value
            self.active_event.recovery_profit = (self.recovery_target - premium) * position_value
    
    def _end_reverse_event(self, final_premium: float):
        """
        역프 이벤트 종료
        
        Args:
            final_premium: 종료 시 프리미엄
        """
        if not self.active_event:
            return
            
        self.active_event.end_time = datetime.now()
        self.active_event.status = 'recovered' if final_premium > 0 else 'expired'
        
        self.reverse_events.append(self.active_event)
        
        logger.info(f"Reverse premium ended: duration={(self.active_event.end_time - self.active_event.start_time).seconds}s, "
                   f"min={self.active_event.min_premium:.2%}, final={final_premium:.2%}")
        
        self.active_event = None
        self.alert_sent = False
    
    def _measure_trend_strength(self, market_data: Dict) -> float:
        """
        추세 강도 측정
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            추세 강도 (0-1)
        """
        if len(self.premium_history) < 10:
            return 0.5
            
        # 최근 프리미엄 추세
        recent_premiums = self.premium_history[-10:]
        
        # 선형 회귀로 추세 계산
        x = np.arange(len(recent_premiums))
        y = np.array(recent_premiums)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            # 추세 강도 (기울기 절대값)
            trend_strength = min(abs(slope) * 100, 1.0)
        else:
            trend_strength = 0.5
            
        # 거래량 가중치
        volume_ratio = market_data.get('volume', 1.0) / market_data.get('avg_volume', 1.0)
        volume_weight = min(volume_ratio, 2.0) / 2.0
        
        return trend_strength * 0.7 + volume_weight * 0.3
    
    def _analyze_initial_response(self, premium: float, 
                                market_data: Dict) -> Dict:
        """
        역프 진입 시 초기 대응 분석
        
        Args:
            premium: 현재 프리미엄
            market_data: 시장 데이터
            
        Returns:
            권장 액션
        """
        # 역프 깊이
        depth = abs(premium)
        
        if depth > 0.02:  # -2% 이상
            return {
                'type': 'immediate_action',
                'action': 'close_long',  # 롱 포지션 청산
                'reasoning': 'Deep reverse premium requires immediate risk reduction',
                'urgency': 'high'
            }
        elif depth > 0.01:  # -1% 이상
            return {
                'type': 'monitor',
                'action': 'prepare_exit',
                'reasoning': 'Moderate reverse premium, prepare for potential exit',
                'urgency': 'medium'
            }
        else:
            return {
                'type': 'wait',
                'action': 'monitor_trend',
                'reasoning': 'Shallow reverse premium, monitor for trend development',
                'urgency': 'low'
            }
    
    def _analyze_reverse_premium(self, market_data: Dict) -> Dict:
        """
        역프리미엄 상황 분석
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            분석 결과
        """
        if not self.active_event:
            return {}
            
        duration = (datetime.now() - self.active_event.start_time).seconds / 60  # 분
        
        analysis = {
            'duration_minutes': duration,
            'min_premium': self.active_event.min_premium,
            'trend_strength': self.active_event.trend_strength,
            'estimated_loss': self.active_event.max_loss,
            'recovery_potential': self.active_event.recovery_profit,
            'risk_level': self._calculate_risk_level()
        }
        
        # 회복 가능성 평가
        if duration < 30:
            analysis['recovery_probability'] = 0.7  # 단기 역프는 회복 가능성 높음
        elif duration < 120:
            analysis['recovery_probability'] = 0.5
        else:
            analysis['recovery_probability'] = 0.3  # 장기 역프는 회복 어려움
            
        return analysis
    
    def _determine_optimal_action(self, current_premium: float,
                                 market_data: Dict) -> Dict:
        """
        최적 액션 결정
        
        Args:
            current_premium: 현재 프리미엄
            market_data: 시장 데이터
            
        Returns:
            최적 액션
        """
        if not self.active_event:
            return {'type': 'none'}
            
        analysis = self._analyze_reverse_premium(market_data)
        
        # 리스크 레벨에 따른 액션
        risk_level = analysis['risk_level']
        recovery_prob = analysis.get('recovery_probability', 0.5)
        
        if risk_level == 'critical':
            return {
                'type': 'immediate_exit',
                'action': 'close_all',
                'positions': ['long', 'short'],
                'reasoning': 'Critical risk level requires immediate exit',
                'expected_loss': self.active_event.max_loss
            }
            
        elif risk_level == 'high':
            if recovery_prob > 0.6:
                return {
                    'type': 'partial_exit',
                    'action': 'close_long',
                    'positions': ['long'],
                    'reasoning': 'High risk but recovery likely, close long only',
                    'expected_outcome': 'Minimize loss while maintaining hedge'
                }
            else:
                return {
                    'type': 'full_exit',
                    'action': 'close_all',
                    'positions': ['long', 'short'],
                    'reasoning': 'High risk with low recovery probability',
                    'expected_loss': self.active_event.max_loss * 0.7
                }
                
        elif risk_level == 'medium':
            # 추세 활용 전략
            if self.active_event.trend_strength > 0.7:
                return {
                    'type': 'trend_exploitation',
                    'action': 'adjust_position',
                    'strategy': 'Reduce long, maintain short for trend profit',
                    'reasoning': 'Strong trend allows profit from directional move',
                    'expected_outcome': 'Convert loss to profit'
                }
            else:
                return {
                    'type': 'wait_recovery',
                    'action': 'monitor',
                    'trigger': f'Exit when premium > {self.recovery_target:.2%}',
                    'reasoning': 'Medium risk, wait for recovery',
                    'max_wait_time': 120  # 분
                }
                
        else:  # low risk
            return {
                'type': 'hold',
                'action': 'maintain_hedge',
                'reasoning': 'Low risk, maintain positions for recovery',
                'monitor_interval': 15  # 분
            }
    
    def _calculate_risk_level(self) -> str:
        """
        리스크 레벨 계산
        
        Returns:
            'low', 'medium', 'high', 'critical'
        """
        if not self.active_event:
            return 'low'
            
        depth = abs(self.active_event.min_premium)
        duration = (datetime.now() - self.active_event.start_time).seconds / 60
        
        # 리스크 점수 계산
        risk_score = 0
        
        # 깊이 기반 점수
        if depth > 0.03:
            risk_score += 3
        elif depth > 0.02:
            risk_score += 2
        elif depth > 0.01:
            risk_score += 1
            
        # 지속 시간 기반 점수
        if duration > 180:
            risk_score += 3
        elif duration > 60:
            risk_score += 2
        elif duration > 30:
            risk_score += 1
            
        # 추세 강도 기반 점수
        if self.active_event.trend_strength > 0.8:
            risk_score += 2
        elif self.active_event.trend_strength > 0.6:
            risk_score += 1
            
        # 리스크 레벨 결정
        if risk_score >= 7:
            return 'critical'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def calculate_optimal_exit_timing(self, market_data: Dict) -> OptimalExitPoint:
        """
        최적 청산 타이밍 계산
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            최적 청산 포인트
        """
        if not self.active_event:
            return OptimalExitPoint(
                timestamp=datetime.now(),
                premium_level=0,
                expected_profit=0,
                confidence=0,
                strategy='none',
                reasoning='No active reverse premium event'
            )
            
        # 회복 패턴 분석
        recovery_pattern = self._analyze_recovery_pattern()
        
        # 최적 타이밍 계산
        if recovery_pattern['type'] == 'v_shaped':
            # V자 회복 - 빠른 청산
            optimal_time = datetime.now() + timedelta(minutes=15)
            strategy = 'immediate'
            confidence = 0.8
            
        elif recovery_pattern['type'] == 'u_shaped':
            # U자 회복 - 바닥 확인 후 청산
            optimal_time = datetime.now() + timedelta(minutes=60)
            strategy = 'wait_recovery'
            confidence = 0.6
            
        else:  # 'l_shaped' or 'unknown'
            # L자 또는 불명확 - 손절
            optimal_time = datetime.now()
            strategy = 'immediate'
            confidence = 0.5
            
        # 예상 프리미엄 레벨
        if strategy == 'wait_recovery':
            expected_premium = self.recovery_target * 0.5  # 보수적 목표
        else:
            expected_premium = self.premium_history[-1] if self.premium_history else 0
            
        # 예상 수익 계산
        position_value = market_data.get('position_value', 0)
        if strategy == 'immediate':
            expected_profit = -abs(self.active_event.min_premium) * position_value * 0.5
        else:
            expected_profit = (expected_premium - self.active_event.min_premium) * position_value
            
        return OptimalExitPoint(
            timestamp=optimal_time,
            premium_level=expected_premium,
            expected_profit=expected_profit,
            confidence=confidence,
            strategy=strategy,
            reasoning=recovery_pattern['reasoning']
        )
    
    def _analyze_recovery_pattern(self) -> Dict:
        """
        회복 패턴 분석
        
        Returns:
            패턴 분석 결과
        """
        if len(self.premium_history) < 5:
            return {
                'type': 'unknown',
                'reasoning': 'Insufficient data for pattern analysis'
            }
            
        recent = self.premium_history[-5:]
        
        # 변화율 계산
        changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        # 패턴 판단
        if all(c > 0 for c in changes[-2:]):
            return {
                'type': 'v_shaped',
                'reasoning': 'Sharp recovery detected, V-shaped pattern'
            }
        elif sum(changes) > 0 and np.std(changes) < 0.001:
            return {
                'type': 'u_shaped',
                'reasoning': 'Gradual recovery detected, U-shaped pattern'
            }
        elif all(abs(c) < 0.0005 for c in changes[-3:]):
            return {
                'type': 'l_shaped',
                'reasoning': 'No recovery detected, L-shaped pattern'
            }
        else:
            return {
                'type': 'unknown',
                'reasoning': 'Pattern unclear, mixed signals'
            }
    
    def get_historical_performance(self) -> Dict:
        """
        과거 역프 대응 성과 조회
        
        Returns:
            성과 통계
        """
        if not self.reverse_events:
            return {
                'total_events': 0,
                'successful_recoveries': 0,
                'average_duration': 0,
                'average_loss': 0,
                'recovery_rate': 0
            }
            
        total = len(self.reverse_events)
        successful = sum(1 for e in self.reverse_events if e.status == 'recovered')
        
        durations = [
            (e.end_time - e.start_time).seconds / 60
            for e in self.reverse_events
            if e.end_time
        ]
        
        losses = [e.max_loss for e in self.reverse_events]
        
        return {
            'total_events': total,
            'successful_recoveries': successful,
            'average_duration': np.mean(durations) if durations else 0,
            'average_loss': np.mean(losses) if losses else 0,
            'recovery_rate': successful / total if total > 0 else 0,
            'max_loss': max(losses) if losses else 0,
            'min_premium_seen': min(e.min_premium for e in self.reverse_events) if self.reverse_events else 0
        }