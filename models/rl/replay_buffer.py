"""
경험 재생 버퍼 및 데이터 전처리 (Task #17.4)
우선순위 샘플링과 N-step returns를 지원하는 고급 버퍼
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, namedtuple
import random
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings('ignore')


# 경험 튜플 정의
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'info']
)


@dataclass
class Transition:
    """향상된 전환 데이터 구조"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict
    td_error: float = 0.0  # TD 오차 (우선순위용)
    n_step_return: float = 0.0  # N-step 리턴
    advantage: float = 0.0  # Advantage 추정치


class PrioritizedReplayBuffer:
    """
    우선순위 경험 재생 버퍼
    
    TD-error 기반으로 중요한 경험을 더 자주 샘플링
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # 우선순위 지수
        beta: float = 0.4,   # 중요도 샘플링 지수
        beta_increment: float = 0.001,  # 베타 증가율
        epsilon: float = 1e-6  # 우선순위 최소값
    ):
        """
        버퍼 초기화
        
        Args:
            capacity: 버퍼 최대 크기
            alpha: 우선순위 지수 (0: uniform, 1: full priority)
            beta: 중요도 샘플링 보정 (0: no correction, 1: full correction)
            beta_increment: 에피소드마다 베타 증가
            epsilon: 우선순위 계산 시 최소값
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 순환 버퍼
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        # 통계
        self.total_samples = 0
        self.max_priority = 1.0
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None
    ):
        """
        경험 추가
        
        새 경험은 최대 우선순위로 추가 (탐색 장려)
        """
        if info is None:
            info = {}
            
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
            td_error=self.max_priority  # 초기에는 최대 우선순위
        )
        
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
        self.total_samples += 1
        
    def sample(
        self,
        batch_size: int,
        n_step: int = 1,
        gamma: float = 0.99
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        우선순위 샘플링
        
        Args:
            batch_size: 배치 크기
            n_step: N-step returns 계산
            gamma: 할인 계수
            
        Returns:
            (transitions, weights, indices)
        """
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # 우선순위를 확률로 변환
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / (priorities.sum() + self.epsilon)
        
        # 샘플링
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities,
            replace=False
        )
        
        # 중요도 샘플링 가중치 계산
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 정규화
        
        # 샘플 추출
        transitions = [self.buffer[idx] for idx in indices]
        
        # N-step returns 계산
        if n_step > 1:
            transitions = self._calculate_n_step_returns(
                transitions, indices, n_step, gamma
            )
        
        # 베타 증가 (선형적으로 1에 수렴)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, weights, indices
    
    def _calculate_n_step_returns(
        self,
        transitions: List[Transition],
        indices: np.ndarray,
        n_step: int,
        gamma: float
    ) -> List[Transition]:
        """N-step returns 계산"""
        updated_transitions = []
        
        for i, idx in enumerate(indices):
            transition = transitions[i]
            n_step_return = transition.reward
            
            # N-step 앞선 리워드 누적
            for step in range(1, min(n_step, len(self.buffer) - idx)):
                if idx + step >= len(self.buffer):
                    break
                    
                next_trans = self.buffer[idx + step]
                n_step_return += (gamma ** step) * next_trans.reward
                
                if next_trans.done:
                    break
            
            # N-step return으로 업데이트
            transition.n_step_return = n_step_return
            updated_transitions.append(transition)
        
        return updated_transitions
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        TD 오차 기반 우선순위 업데이트
        
        Args:
            indices: 업데이트할 인덱스
            td_errors: TD 오차값
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
            self.buffer[idx].td_error = td_error
            
            # 최대 우선순위 추적
            if priority > self.max_priority:
                self.max_priority = priority
    
    def get_statistics(self) -> Dict:
        """버퍼 통계 반환"""
        if len(self.buffer) == 0:
            return {}
            
        rewards = [t.reward for t in self.buffer]
        td_errors = [t.td_error for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_samples': self.total_samples,
            'avg_reward': np.mean(rewards),
            'avg_td_error': np.mean(td_errors),
            'max_priority': self.max_priority,
            'beta': self.beta
        }
    
    def save(self, filepath: str):
        """버퍼 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'total_samples': self.total_samples,
                'max_priority': self.max_priority,
                'beta': self.beta
            }, f)
    
    def load(self, filepath: str):
        """버퍼 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            self.priorities = deque(data['priorities'], maxlen=self.capacity)
            self.total_samples = data['total_samples']
            self.max_priority = data['max_priority']
            self.beta = data['beta']
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()
        self.priorities.clear()
        self.total_samples = 0
        self.max_priority = 1.0


class DataAugmentationBuffer(PrioritizedReplayBuffer):
    """
    데이터 증강 기능이 추가된 재생 버퍼
    
    노이즈 추가, 시간 왜곡 등으로 데이터 다양성 증가
    """
    
    def __init__(
        self,
        *args,
        augment_prob: float = 0.3,
        noise_scale: float = 0.01,
        **kwargs
    ):
        """
        증강 버퍼 초기화
        
        Args:
            augment_prob: 증강 확률
            noise_scale: 노이즈 스케일
        """
        super().__init__(*args, **kwargs)
        self.augment_prob = augment_prob
        self.noise_scale = noise_scale
        
    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 증강
        
        Args:
            state: 원본 상태
            
        Returns:
            증강된 상태
        """
        augmented = state.copy()
        
        if random.random() < self.augment_prob:
            # 가우시안 노이즈 추가
            noise = np.random.normal(0, self.noise_scale, state.shape)
            augmented += noise
            
            # 클리핑
            augmented = np.clip(augmented, -3, 3)
        
        return augmented
    
    def _augment_reward(self, reward: float) -> float:
        """
        리워드 증강 (스케일링)
        
        Args:
            reward: 원본 리워드
            
        Returns:
            증강된 리워드
        """
        if random.random() < self.augment_prob:
            # 작은 랜덤 스케일링
            scale = np.random.uniform(0.95, 1.05)
            return reward * scale
        return reward
    
    def sample(
        self,
        batch_size: int,
        n_step: int = 1,
        gamma: float = 0.99,
        augment: bool = True
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        증강된 샘플링
        
        Args:
            batch_size: 배치 크기
            n_step: N-step returns
            gamma: 할인 계수
            augment: 증강 적용 여부
            
        Returns:
            (transitions, weights, indices)
        """
        transitions, weights, indices = super().sample(
            batch_size, n_step, gamma
        )
        
        if augment and len(transitions) > 0:
            # 데이터 증강 적용
            augmented_transitions = []
            
            for trans in transitions:
                aug_state = self._augment_state(trans.state)
                aug_next_state = self._augment_state(trans.next_state)
                aug_reward = self._augment_reward(trans.reward)
                
                aug_trans = Transition(
                    state=aug_state,
                    action=trans.action,
                    reward=aug_reward,
                    next_state=aug_next_state,
                    done=trans.done,
                    info=trans.info,
                    td_error=trans.td_error,
                    n_step_return=trans.n_step_return,
                    advantage=trans.advantage
                )
                
                augmented_transitions.append(aug_trans)
            
            return augmented_transitions, weights, indices
        
        return transitions, weights, indices


class EpisodeBuffer:
    """
    에피소드 단위 버퍼 - 전체 에피소드를 저장하고 처리
    """
    
    def __init__(self, max_episodes: int = 1000):
        """
        에피소드 버퍼 초기화
        
        Args:
            max_episodes: 최대 에피소드 수
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []
        
    def add_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None
    ):
        """에피소드에 스텝 추가"""
        self.current_episode.append(
            Experience(state, action, reward, next_state, done, info or {})
        )
        
        if done:
            self.finish_episode()
    
    def finish_episode(self):
        """현재 에피소드 종료 및 저장"""
        if self.current_episode:
            # GAE 계산 등 후처리 가능
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def get_recent_episodes(self, n: int = 10) -> List[List[Experience]]:
        """최근 n개 에피소드 반환"""
        return list(self.episodes)[-n:]
    
    def compute_returns(
        self,
        episode: List[Experience],
        gamma: float = 0.99
    ) -> np.ndarray:
        """
        에피소드의 discounted returns 계산
        
        Args:
            episode: 에피소드 경험 리스트
            gamma: 할인 계수
            
        Returns:
            각 스텝의 return
        """
        returns = np.zeros(len(episode))
        running_return = 0
        
        for t in reversed(range(len(episode))):
            if episode[t].done:
                running_return = 0
            running_return = episode[t].reward + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def compute_gae(
        self,
        episode: List[Experience],
        values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> np.ndarray:
        """
        Generalized Advantage Estimation (GAE) 계산
        
        Args:
            episode: 에피소드 경험 리스트
            values: 가치 함수 추정값
            gamma: 할인 계수
            lam: GAE 람다
            
        Returns:
            각 스텝의 advantage
        """
        advantages = np.zeros(len(episode))
        last_advantage = 0
        
        for t in reversed(range(len(episode))):
            if t == len(episode) - 1:
                next_value = 0 if episode[t].done else values[t + 1]
            else:
                next_value = values[t + 1]
            
            delta = episode[t].reward + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * last_advantage
            
            if episode[t].done:
                last_advantage = 0
            else:
                last_advantage = advantages[t]
        
        return advantages