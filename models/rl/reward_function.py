"""
샤프비율 기반 보상 함수 (Task #17.2)
리스크 조정 수익률을 최대화하는 강화학습 보상 설계
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RewardFunction:
    """
    PPO 에이전트를 위한 고급 보상 함수
    
    주요 컴포넌트:
    1. 수익률 기반 보상
    2. 샤프비율 컴포넌트
    3. 리스크 패널티
    4. 거래 비용 고려
    5. 목표 달성 보너스
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 연간 무위험 수익률
        trading_fee: float = 0.001,    # 거래 수수료 0.1%
        max_drawdown_threshold: float = 0.1,  # 최대 허용 손실 10%
        target_sharpe: float = 1.5,    # 목표 샤프비율
        target_return: float = 0.05,   # 목표 수익률 5%
    ):
        """
        보상 함수 초기화
        
        Args:
            risk_free_rate: 무위험 수익률
            trading_fee: 거래 수수료율
            max_drawdown_threshold: 최대 허용 드로우다운
            target_sharpe: 목표 샤프비율
            target_return: 목표 수익률
        """
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.max_drawdown_threshold = max_drawdown_threshold
        self.target_sharpe = target_sharpe
        self.target_return = target_return
        
        # 성과 추적
        self.portfolio_history = deque(maxlen=1000)
        self.return_history = deque(maxlen=100)
        self.consecutive_losses = 0
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        
    def calculate_reward(
        self,
        action: int,
        current_value: float,
        prev_value: float,
        position: float,
        premium_rate: float,
        trade_executed: bool,
        info: Dict
    ) -> Tuple[float, Dict]:
        """
        종합 보상 계산
        
        Args:
            action: 수행한 액션 (0: Hold, 1: Enter, 2: Exit)
            current_value: 현재 포트폴리오 가치
            prev_value: 이전 포트폴리오 가치
            position: 현재 포지션 크기
            premium_rate: 현재 김치프리미엄율
            trade_executed: 거래 실행 여부
            info: 추가 정보
            
        Returns:
            (총 보상, 보상 컴포넌트 딕셔너리)
        """
        # 1. 기본 수익률 계산
        returns = (current_value / prev_value - 1) if prev_value > 0 else 0
        self.return_history.append(returns)
        self.portfolio_history.append(current_value)
        
        # 2. 드로우다운 업데이트
        self._update_drawdown(current_value)
        
        # 보상 컴포넌트들
        reward_components = {}
        
        # 3. 수익률 보상
        profit_reward = self._calculate_profit_reward(returns)
        reward_components['profit'] = profit_reward
        
        # 4. 샤프비율 보상
        sharpe_reward = self._calculate_sharpe_reward()
        reward_components['sharpe'] = sharpe_reward
        
        # 5. 리스크 패널티
        risk_penalty = self._calculate_risk_penalty()
        reward_components['risk'] = risk_penalty
        
        # 6. 거래 비용 패널티
        if trade_executed:
            fee_penalty = -self.trading_fee * 10  # 거래 비용의 10배 패널티
            reward_components['fee'] = fee_penalty
        else:
            reward_components['fee'] = 0
        
        # 7. 포지션 관련 보상
        position_reward = self._calculate_position_reward(
            action, position, premium_rate
        )
        reward_components['position'] = position_reward
        
        # 8. 목표 달성 보너스
        achievement_bonus = self._calculate_achievement_bonus()
        reward_components['achievement'] = achievement_bonus
        
        # 9. 연속 손실 패널티
        if returns < 0:
            self.consecutive_losses += 1
            loss_penalty = -0.01 * (1.5 ** self.consecutive_losses)
            reward_components['consecutive_loss'] = loss_penalty
        else:
            self.consecutive_losses = 0
            reward_components['consecutive_loss'] = 0
        
        # 총 보상 계산 (가중 합계)
        total_reward = (
            reward_components['profit'] * 1.0 +
            reward_components['sharpe'] * 0.5 +
            reward_components['risk'] * 1.5 +
            reward_components['fee'] * 0.3 +
            reward_components['position'] * 0.7 +
            reward_components['achievement'] * 2.0 +
            reward_components['consecutive_loss'] * 1.0
        )
        
        # 보상 클리핑 (안정성)
        total_reward = np.clip(total_reward, -10, 10)
        
        # 정보 업데이트
        info.update({
            'reward_components': reward_components,
            'total_reward': total_reward,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'cumulative_return': (current_value / self.portfolio_history[0] - 1) if len(self.portfolio_history) > 0 else 0
        })
        
        return total_reward, info
    
    def _calculate_profit_reward(self, returns: float) -> float:
        """수익률 기반 보상"""
        # 수익률에 비례하되, 손실은 더 크게 패널티
        if returns > 0:
            return returns * 100  # 수익은 100배 스케일
        else:
            return returns * 150  # 손실은 150배 스케일 (더 큰 패널티)
    
    def _calculate_sharpe_reward(self) -> float:
        """샤프비율 기반 보상"""
        if len(self.return_history) < 30:
            return 0
        
        sharpe = self._calculate_sharpe_ratio()
        
        # 목표 샤프비율 대비 보상/패널티
        if sharpe > self.target_sharpe:
            return (sharpe - self.target_sharpe) * 0.5  # 초과 달성 보상
        elif sharpe > 0:
            return sharpe * 0.1  # 양의 샤프비율 작은 보상
        else:
            return sharpe * 0.2  # 음의 샤프비율 패널티
    
    def _calculate_risk_penalty(self) -> float:
        """리스크 관련 패널티"""
        penalty = 0
        
        # 드로우다운 패널티
        if self.current_drawdown > 0:
            # 드로우다운이 클수록 지수적으로 증가하는 패널티
            dd_ratio = self.current_drawdown / self.max_drawdown_threshold
            penalty -= dd_ratio ** 2
        
        # 최대 드로우다운 초과 시 큰 패널티
        if self.current_drawdown > self.max_drawdown_threshold:
            penalty -= (self.current_drawdown - self.max_drawdown_threshold) * 10
        
        # 변동성 패널티
        if len(self.return_history) > 10:
            volatility = np.std(list(self.return_history)[-10:])
            if volatility > 0.02:  # 2% 이상 변동성
                penalty -= (volatility - 0.02) * 5
        
        return penalty
    
    def _calculate_position_reward(
        self, action: int, position: float, premium_rate: float
    ) -> float:
        """포지션 관련 보상"""
        reward = 0
        
        # 진입 액션 (action == 1)
        if action == 1:
            if premium_rate > 3.0:  # 김프 3% 이상일 때 진입
                reward += 0.1
            elif premium_rate > 2.0:  # 김프 2-3%
                reward += 0.05
            else:  # 김프 2% 미만일 때 진입
                reward -= 0.1
        
        # 청산 액션 (action == 2)
        elif action == 2:
            if premium_rate < 1.5:  # 김프 1.5% 미만일 때 청산
                reward += 0.1
            elif premium_rate < 0:  # 역프리미엄 시 청산
                reward += 0.2
            elif premium_rate > 3.0:  # 높은 김프에서 청산
                reward -= 0.2
        
        # 홀드 액션 (action == 0)
        else:
            if position > 0:
                # 포지션 보유 중 김프 변화에 따른 보상
                if 2.0 <= premium_rate <= 3.5:
                    reward += 0.02  # 적정 김프 구간 유지
                elif premium_rate < 1.0:
                    reward -= 0.05  # 낮은 김프에서 홀드 패널티
                elif premium_rate > 4.0:
                    reward -= 0.03  # 너무 높은 김프 리스크
        
        return reward
    
    def _calculate_achievement_bonus(self) -> float:
        """목표 달성 보너스"""
        bonus = 0
        
        if len(self.portfolio_history) < 2:
            return 0
        
        # 누적 수익률 계산
        total_return = (self.portfolio_history[-1] / self.portfolio_history[0] - 1)
        
        # 목표 수익률 달성
        if total_return > self.target_return:
            bonus += (total_return - self.target_return) * 2
        
        # 샤프비율 목표 달성
        sharpe = self._calculate_sharpe_ratio()
        if sharpe > self.target_sharpe:
            bonus += (sharpe - self.target_sharpe) * 0.3
        
        # 낮은 드로우다운 유지
        if self.max_drawdown < self.max_drawdown_threshold / 2:
            bonus += 0.1
        
        return bonus
    
    def _update_drawdown(self, current_value: float):
        """드로우다운 업데이트"""
        # Peak 업데이트
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0
        else:
            # 현재 드로우다운 계산
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
            # 최대 드로우다운 업데이트
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """샤프비율 계산"""
        if len(self.return_history) < 2:
            return 0
        
        returns = np.array(list(self.return_history))
        
        if np.std(returns) == 0:
            return 0
        
        # 분당 수익률을 연간화
        annual_return = np.mean(returns) * 525600  # 1년 = 525600분
        annual_std = np.std(returns) * np.sqrt(525600)
        
        # 분당 무위험 수익률
        risk_free_per_minute = self.risk_free_rate / 525600
        
        sharpe = (annual_return - risk_free_per_minute * 525600) / (annual_std + 1e-8)
        
        return sharpe
    
    def reset(self):
        """보상 함수 상태 초기화"""
        self.portfolio_history.clear()
        self.return_history.clear()
        self.consecutive_losses = 0
        self.peak_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0


class AdaptiveRewardFunction(RewardFunction):
    """
    적응형 보상 함수 - 학습 진행에 따라 보상 가중치 조정
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_count = 0
        self.best_sharpe = 0
        self.learning_phase = 'exploration'  # exploration, exploitation, fine-tuning
        
    def update_learning_phase(self, episode: int, current_sharpe: float):
        """학습 단계 업데이트"""
        self.episode_count = episode
        
        if current_sharpe > self.best_sharpe:
            self.best_sharpe = current_sharpe
        
        # 학습 단계 결정
        if episode < 100:
            self.learning_phase = 'exploration'
        elif episode < 500:
            self.learning_phase = 'exploitation'
        else:
            self.learning_phase = 'fine-tuning'
    
    def calculate_reward(self, *args, **kwargs) -> Tuple[float, Dict]:
        """학습 단계에 따라 조정된 보상 계산"""
        reward, info = super().calculate_reward(*args, **kwargs)
        
        # 학습 단계별 보상 조정
        if self.learning_phase == 'exploration':
            # 탐색 단계: 다양한 행동 장려
            reward *= 0.8
            reward += np.random.normal(0, 0.1)  # 노이즈 추가
        
        elif self.learning_phase == 'exploitation':
            # 활용 단계: 성과 중심
            if info.get('sharpe_ratio', 0) > self.target_sharpe:
                reward *= 1.2
        
        elif self.learning_phase == 'fine-tuning':
            # 미세조정 단계: 안정성 중시
            if info.get('max_drawdown', 0) < self.max_drawdown_threshold:
                reward *= 1.1
        
        info['learning_phase'] = self.learning_phase
        info['episode'] = self.episode_count
        
        return reward, info