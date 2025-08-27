"""
PPO 에이전트 통합 테스트
Task #17 검증 및 성능 벤치마크
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.rl.trading_environment import KimchiPremiumTradingEnv
from models.rl.reward_function import AdaptiveRewardFunction
from models.rl.ppo_agent import PPOAgent
from models.rl.replay_buffer import PrioritizedReplayBuffer
from models.rl.train_ppo import PPOTrainer


class TestTradingEnvironment:
    """거래 환경 테스트"""
    
    def test_environment_creation(self):
        """환경 생성 테스트"""
        env = KimchiPremiumTradingEnv()
        assert env is not None
        assert env.observation_space.shape == (20,)
        assert env.action_space.n == 3
        
    def test_environment_reset(self):
        """환경 리셋 테스트"""
        env = KimchiPremiumTradingEnv()
        obs, info = env.reset()
        
        assert obs.shape == (20,)
        assert -3 <= obs.min() <= obs.max() <= 3
        assert isinstance(info, dict)
        
    def test_environment_step(self):
        """환경 스텝 테스트"""
        env = KimchiPremiumTradingEnv()
        obs, _ = env.reset()
        
        # 각 액션 테스트
        for action in [0, 1, 2]:
            next_obs, reward, done, truncated, info = env.step(action)
            
            assert next_obs.shape == (20,)
            assert isinstance(reward, (float, np.float32, np.float64))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            if done:
                break


class TestRewardFunction:
    """보상 함수 테스트"""
    
    def test_reward_calculation(self):
        """보상 계산 테스트"""
        reward_fn = AdaptiveRewardFunction()
        
        # 테스트 케이스
        test_cases = [
            # (action, current_value, prev_value, position, premium_rate, trade_executed)
            (1, 10100000, 10000000, 0, 3.5, True),  # 진입 with 좋은 김프
            (2, 10200000, 10100000, 0.1, 1.5, True),  # 청산 with 낮은 김프
            (0, 10150000, 10100000, 0.1, 2.5, False),  # 홀드
        ]
        
        for action, curr_val, prev_val, pos, premium, traded in test_cases:
            reward, info = reward_fn.calculate_reward(
                action, curr_val, prev_val, pos, premium, traded, {}
            )
            
            assert isinstance(reward, (float, np.float32, np.float64))
            assert -10 <= reward <= 10  # 클리핑 확인
            assert 'reward_components' in info
            assert 'sharpe_ratio' in info
    
    def test_sharpe_ratio_calculation(self):
        """샤프비율 계산 테스트"""
        reward_fn = AdaptiveRewardFunction()
        
        # 샘플 리턴 추가
        for _ in range(100):
            returns = np.random.normal(0.001, 0.01)
            reward_fn.return_history.append(returns)
        
        sharpe = reward_fn._calculate_sharpe_ratio()
        assert isinstance(sharpe, (float, np.float32, np.float64))
    
    def test_drawdown_tracking(self):
        """드로우다운 추적 테스트"""
        reward_fn = AdaptiveRewardFunction()
        
        # 포트폴리오 가치 시뮬레이션
        values = [10000000, 10500000, 10200000, 9800000, 10100000]
        
        for val in values:
            reward_fn._update_drawdown(val)
        
        assert reward_fn.max_drawdown > 0
        assert reward_fn.peak_value == 10500000


class TestPPOAgent:
    """PPO 에이전트 테스트"""
    
    @pytest.fixture
    def env_and_agent(self):
        """테스트용 환경과 에이전트"""
        env = KimchiPremiumTradingEnv()
        agent = PPOAgent(
            env=env,
            learning_rate=1e-3,
            n_steps=64,
            batch_size=16,
            n_epochs=2
        )
        return env, agent
    
    def test_agent_creation(self, env_and_agent):
        """에이전트 생성 테스트"""
        env, agent = env_and_agent
        assert agent is not None
        assert agent.model is not None
        
    def test_agent_prediction(self, env_and_agent):
        """에이전트 예측 테스트"""
        env, agent = env_and_agent
        
        obs, _ = env.reset()
        action, probs = agent.predict(obs)
        
        assert 0 <= action <= 2
        assert len(probs) == 3
        assert np.isclose(probs.sum(), 1.0)
    
    def test_short_training(self, env_and_agent):
        """짧은 학습 테스트"""
        env, agent = env_and_agent
        
        # 매우 짧은 학습 (테스트용)
        model = agent.train(
            total_timesteps=128,
            eval_env=None,
            eval_freq=None
        )
        
        assert model is not None


class TestReplayBuffer:
    """재생 버퍼 테스트"""
    
    def test_buffer_creation(self):
        """버퍼 생성 테스트"""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        assert buffer is not None
        assert buffer.capacity == 1000
        
    def test_buffer_add_sample(self):
        """버퍼 추가 및 샘플링 테스트"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # 샘플 추가
        for i in range(50):
            state = np.random.randn(20)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(20)
            done = i % 10 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        # 샘플링
        transitions, weights, indices = buffer.sample(batch_size=10)
        
        assert len(transitions) == 10
        assert len(weights) == 10
        assert len(indices) == 10
    
    def test_priority_update(self):
        """우선순위 업데이트 테스트"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # 샘플 추가
        for i in range(20):
            buffer.add(
                np.random.randn(20),
                np.random.randint(0, 3),
                np.random.randn(),
                np.random.randn(20),
                False
            )
        
        # 우선순위 업데이트
        indices = np.array([0, 5, 10])
        td_errors = np.array([0.5, 1.0, 0.1])
        
        buffer.update_priorities(indices, td_errors)
        
        assert buffer.priorities[0] > 0
        assert buffer.priorities[5] > buffer.priorities[10]


class TestIntegration:
    """통합 테스트"""
    
    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        # 트레이너 생성
        trainer = PPOTrainer(
            data_path=None,  # 더미 데이터 사용
            save_dir="./test_results"
        )
        
        # 설정 업데이트 (빠른 테스트용)
        trainer.config['training']['total_timesteps'] = 256
        trainer.config['training']['eval_freq'] = 128
        trainer.config['training']['n_eval_episodes'] = 2
        trainer.config['ppo']['n_steps'] = 64
        trainer.config['ppo']['batch_size'] = 16
        
        # 학습 실행
        results = trainer.train(use_adaptive=False)
        
        assert results is not None
        assert 'mean_reward' in results
        assert 'mean_sharpe' in results
        assert 'win_rate' in results
    
    def test_backtest(self):
        """백테스트 테스트"""
        trainer = PPOTrainer(save_dir="./test_results")
        
        # 간단한 학습
        trainer.config['training']['total_timesteps'] = 128
        trainer.config['ppo']['n_steps'] = 64
        trainer._create_environments()
        
        # 에이전트 생성
        trainer.agent = PPOAgent(
            env=trainer.env,
            n_steps=64,
            batch_size=16
        )
        
        # 백테스트
        results = trainer.backtest()
        
        assert results is not None
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results


def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print("\n" + "="*60)
    print("Running Performance Benchmark")
    print("="*60)
    
    env = KimchiPremiumTradingEnv()
    
    # 랜덤 에이전트 벤치마크
    random_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated
        
        random_rewards.append(episode_reward)
    
    print(f"Random Agent - Mean Reward: {np.mean(random_rewards):.2f}")
    
    # Buy-and-Hold 벤치마크
    obs, _ = env.reset()
    env.step(1)  # Enter position
    done = False
    hold_reward = 0
    
    while not done:
        obs, reward, done, truncated, info = env.step(0)  # Hold
        hold_reward += reward
        done = done or truncated
    
    print(f"Buy-and-Hold - Total Reward: {hold_reward:.2f}")
    
    # PPO 에이전트 (짧은 학습)
    agent = PPOAgent(env=env, n_steps=128, batch_size=32)
    agent.train(total_timesteps=512, eval_env=None)
    
    obs, _ = env.reset()
    done = False
    ppo_reward = 0
    
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ppo_reward += reward
        done = done or truncated
    
    print(f"PPO Agent - Total Reward: {ppo_reward:.2f}")
    print(f"Final Sharpe Ratio: {info.get('sharpe_ratio', 0):.3f}")
    
    # 성과 비교
    print("\n" + "="*60)
    if ppo_reward > hold_reward and ppo_reward > np.mean(random_rewards):
        print("✅ PPO outperforms baselines!")
    else:
        print("⚠️ PPO needs more training")
    print("="*60)


if __name__ == "__main__":
    # 기본 테스트 실행
    test_performance_benchmark()
    
    # pytest 실행 (설치된 경우)
    try:
        pytest.main([__file__, '-v'])
    except:
        print("\nRun 'pytest test_ppo_agent.py -v' for full test suite")