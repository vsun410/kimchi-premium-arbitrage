"""
PPO 에이전트 및 신경망 아키텍처 (Task #17.3)
Stable-baselines3를 활용한 PPO 강화학습 구현
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Optional, Tuple, Type, Union
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')


class KimchiPremiumFeatureExtractor(BaseFeaturesExtractor):
    """
    김치프리미엄 거래를 위한 커스텀 특징 추출기
    
    더 복잡한 특징을 학습하기 위한 깊은 신경망 구조
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        """
        특징 추출기 초기화
        
        Args:
            observation_space: 관찰 공간
            features_dim: 출력 특징 차원
            dropout_rate: 드롭아웃 비율
        """
        super().__init__(observation_space, features_dim)
        
        n_input_features = observation_space.shape[0]
        
        # 깊은 신경망 구조
        self.network = nn.Sequential(
            # 첫 번째 블록
            nn.Linear(n_input_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 두 번째 블록
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 세 번째 블록
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 출력층
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()  # -1 ~ 1 범위로 정규화
        )
        
        # Attention 메커니즘 (선택적)
        self.attention = nn.MultiheadAttention(
            embed_dim=features_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        특징 추출 forward pass
        
        Args:
            observations: 입력 관찰값
            
        Returns:
            추출된 특징
        """
        # 기본 특징 추출
        features = self.network(observations)
        
        # Attention 적용 (선택적)
        # 단일 시퀀스로 변환하여 self-attention 적용
        if len(features.shape) == 2:
            features_seq = features.unsqueeze(1)  # [batch, 1, features]
            attended_features, _ = self.attention(
                features_seq, features_seq, features_seq
            )
            features = features + attended_features.squeeze(1)  # Residual connection
        
        return features


class PPOAgent:
    """
    김치프리미엄 차익거래를 위한 PPO 에이전트
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_custom_features: bool = True,
        tensorboard_log: str = "./ppo_kimchi_tensorboard/",
        verbose: int = 1
    ):
        """
        PPO 에이전트 초기화
        
        Args:
            env: 거래 환경
            learning_rate: 학습률
            n_steps: 롤아웃 버퍼 크기
            batch_size: 미니배치 크기
            n_epochs: 업데이트 에포크 수
            gamma: 할인 계수
            gae_lambda: GAE 람다
            clip_range: PPO 클리핑 범위
            ent_coef: 엔트로피 계수
            vf_coef: 가치 함수 계수
            max_grad_norm: 그래디언트 클리핑
            use_custom_features: 커스텀 특징 추출기 사용 여부
            tensorboard_log: 텐서보드 로그 경로
            verbose: 출력 상세도
        """
        self.env = env
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        
        # 정책 설정
        policy_kwargs = {}
        
        if use_custom_features:
            policy_kwargs = dict(
                features_extractor_class=KimchiPremiumFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=128,
                    dropout_rate=0.2
                ),
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Actor-Critic 구조
                activation_fn=nn.ReLU
            )
        else:
            # 기본 MLP 정책
            policy_kwargs = dict(
                net_arch=[dict(pi=[128, 64], vf=[128, 64])],
                activation_fn=nn.Tanh
            )
        
        # PPO 모델 생성
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self._linear_schedule(learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=None,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose
        )
        
        # 최적 모델 추적
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        
    def _linear_schedule(self, initial_value: float):
        """
        선형 학습률 스케줄링
        
        Args:
            initial_value: 초기 학습률
            
        Returns:
            스케줄 함수
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0 (end)
            """
            return progress_remaining * initial_value
        
        return func
    
    def train(
        self,
        total_timesteps: int = 1000000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_freq: int = 50000,
        save_path: str = "./ppo_models/",
        log_path: str = "./ppo_logs/"
    ):
        """
        PPO 에이전트 학습
        
        Args:
            total_timesteps: 총 학습 스텝
            eval_env: 평가 환경
            eval_freq: 평가 빈도
            n_eval_episodes: 평가 에피소드 수
            save_freq: 모델 저장 빈도
            save_path: 모델 저장 경로
            log_path: 로그 저장 경로
        """
        callbacks = []
        
        # 체크포인트 콜백
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="ppo_kimchi_model",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
        
        # 평가 콜백
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # 커스텀 콜백
        custom_callback = TradingMetricsCallback()
        callbacks.append(custom_callback)
        
        # 학습 시작
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        return self.model
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        행동 예측
        
        Args:
            observation: 현재 관찰값
            deterministic: 결정적 예측 여부
            
        Returns:
            (행동, 행동 확률)
        """
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        # 행동 확률 계산
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            features = self.model.policy.features_extractor(obs_tensor)
            latent_pi, latent_vf = self.model.policy.mlp_extractor(features)
            distribution = self.model.policy.action_dist.proba_distribution(
                self.model.policy.action_net(latent_pi)
            )
            action_probs = distribution.distribution.probs.squeeze().numpy()
        
        return action, action_probs
    
    def save(self, path: str):
        """모델 저장"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """모델 로드"""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
        return self.model


class TradingMetricsCallback(BaseCallback):
    """
    거래 성과 메트릭을 추적하는 콜백
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.sharpe_ratios = []
        self.win_rates = []
        self.max_drawdowns = []
        
    def _on_step(self) -> bool:
        """각 스텝에서 호출"""
        # 에피소드 종료 시 메트릭 기록
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            
            # 메트릭 추출
            episode_reward = info.get("episode", {}).get("r", 0)
            episode_length = info.get("episode", {}).get("l", 0)
            sharpe_ratio = info.get("sharpe_ratio", 0)
            total_trades = info.get("total_trades", 0)
            max_drawdown = info.get("max_drawdown", 0)
            
            # 저장
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.sharpe_ratios.append(sharpe_ratio)
            self.max_drawdowns.append(max_drawdown)
            
            # Win rate 계산 (수익 에피소드 비율)
            if len(self.episode_rewards) > 0:
                win_rate = sum(1 for r in self.episode_rewards if r > 0) / len(self.episode_rewards)
                self.win_rates.append(win_rate)
            
            # 텐서보드에 기록
            self.logger.record("trading/episode_reward", episode_reward)
            self.logger.record("trading/sharpe_ratio", sharpe_ratio)
            self.logger.record("trading/max_drawdown", max_drawdown)
            self.logger.record("trading/total_trades", total_trades)
            
            # 100 에피소드마다 요약 출력
            if len(self.episode_rewards) % 100 == 0 and self.verbose > 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_sharpes = self.sharpe_ratios[-100:]
                
                print(f"\n=== Episode {len(self.episode_rewards)} Summary ===")
                print(f"Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"Avg Sharpe: {np.mean(recent_sharpes):.3f}")
                print(f"Win Rate: {self.win_rates[-1]:.2%}")
                print(f"Best Sharpe: {max(self.sharpe_ratios):.3f}")
                print("=" * 40)
        
        return True  # 학습 계속
    
    def _on_training_end(self) -> None:
        """학습 종료 시 호출"""
        print("\n=== Training Complete ===")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        
        if self.episode_rewards:
            print(f"Final Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        else:
            print("Final Avg Reward: N/A (no episodes)")
            
        if self.sharpe_ratios:
            print(f"Final Avg Sharpe: {np.mean(self.sharpe_ratios[-100:]):.3f}")
            print(f"Best Ever Sharpe: {max(self.sharpe_ratios):.3f}")
        else:
            print("Final Avg Sharpe: N/A (no episodes)")
            print("Best Ever Sharpe: N/A (no episodes)")
            
        if self.win_rates:
            print(f"Final Win Rate: {self.win_rates[-1]:.2%}")
        else:
            print("Final Win Rate: N/A (no episodes)")


class AdaptivePPOAgent(PPOAgent):
    """
    적응형 PPO 에이전트 - 학습 진행에 따라 하이퍼파라미터 조정
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_clip_range = kwargs.get('clip_range', 0.2)
        self.initial_ent_coef = kwargs.get('ent_coef', 0.01)
        
    def adjust_hyperparameters(self, progress: float):
        """
        학습 진행도에 따라 하이퍼파라미터 조정
        
        Args:
            progress: 학습 진행도 (0 ~ 1)
        """
        # 클리핑 범위 점진적 감소
        self.model.clip_range = self.initial_clip_range * (1 - progress * 0.5)
        
        # 엔트로피 계수 점진적 감소 (탐색 → 활용)
        self.model.ent_coef = self.initial_ent_coef * (1 - progress * 0.8)
        
        # 학습률은 이미 스케줄링됨
        
        if progress > 0.8:
            # 후반부: 미세 조정
            self.model.n_epochs = 5  # 에포크 감소
            self.model.batch_size = 32  # 배치 크기 감소