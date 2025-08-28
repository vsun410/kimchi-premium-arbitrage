"""
PPO 학습 루프 및 백테스팅 통합 (Task #17.5)
전체 학습 파이프라인과 성과 검증
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.rl.trading_environment import KimchiPremiumTradingEnv
from models.rl.ppo_agent import PPOAgent, AdaptivePPOAgent
from models.rl.reward_function import AdaptiveRewardFunction
from models.rl.replay_buffer import DataAugmentationBuffer

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import warnings
warnings.filterwarnings('ignore')


class PPOTrainer:
    """
    PPO 학습 관리자
    
    전체 학습 파이프라인을 관리하고 백테스팅과 통합
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        save_dir: str = "./ppo_results",
        config: Optional[Dict] = None
    ):
        """
        트레이너 초기화
        
        Args:
            data_path: 학습 데이터 경로
            save_dir: 결과 저장 디렉토리
            config: 설정 딕셔너리
        """
        self.data_path = data_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.config = config or self._get_default_config()
        
        # 데이터 로드
        self.df = self._load_data()
        
        # 환경 생성
        self.env = None
        self.eval_env = None
        self._create_environments()
        
        # 에이전트 생성
        self.agent = None
        
        # 학습 기록
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'sharpe_ratios': [],
            'win_rates': [],
            'drawdowns': []
        }
        
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            # 환경 설정
            'env': {
                'initial_balance': 10000000,  # 1천만원
                'trading_fee': 0.001,
                'max_position_size': 0.3,
                'episode_length': 1440,  # 24시간
                'lookback_period': 60
            },
            # PPO 설정
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            # 학습 설정
            'training': {
                'total_timesteps': 1000000,
                'eval_freq': 10000,
                'n_eval_episodes': 10,
                'save_freq': 50000,
                'target_sharpe': 1.5,
                'early_stopping_patience': 50
            }
        }
    
    def _load_data(self) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        if self.data_path and Path(self.data_path).exists():
            # 실제 데이터 로드
            df = pd.read_csv(self.data_path)
            print(f"Loaded data from {self.data_path}: {len(df)} samples")
            return df
        else:
            # 더미 데이터 사용 (테스트용)
            print("No data file found, using dummy data for testing")
            return None
    
    def _create_environments(self):
        """학습 및 평가 환경 생성"""
        # 학습 환경
        def make_env():
            env = KimchiPremiumTradingEnv(
                df=self.df,
                **self.config['env']
            )
            # 보상 함수 통합
            env.reward_function = AdaptiveRewardFunction(
                target_sharpe=self.config['training']['target_sharpe']
            )
            return Monitor(env)
        
        # 벡터화 환경 (병렬 처리)
        n_envs = 4  # 병렬 환경 수
        self.env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # 평가 환경 (단일)
        self.eval_env = make_env()
    
    def train(
        self,
        resume_from: Optional[str] = None,
        use_adaptive: bool = True
    ) -> Dict:
        """
        PPO 학습 실행
        
        Args:
            resume_from: 재개할 모델 경로
            use_adaptive: 적응형 에이전트 사용 여부
            
        Returns:
            학습 결과 딕셔너리
        """
        print("\n" + "="*60)
        print("Starting PPO Training for Kimchi Premium Arbitrage")
        print("="*60)
        
        # 에이전트 생성
        if use_adaptive:
            self.agent = AdaptivePPOAgent(
                env=self.env,
                **self.config['ppo'],
                tensorboard_log=str(self.save_dir / "tensorboard")
            )
        else:
            self.agent = PPOAgent(
                env=self.env,
                **self.config['ppo'],
                tensorboard_log=str(self.save_dir / "tensorboard")
            )
        
        # 모델 재개
        if resume_from and Path(resume_from).exists():
            self.agent.load(resume_from)
            print(f"Resumed training from {resume_from}")
        
        # 콜백 설정
        callbacks = self._setup_callbacks()
        
        # 학습 실행
        try:
            self.agent.train(
                total_timesteps=self.config['training']['total_timesteps'],
                eval_env=self.eval_env,
                eval_freq=self.config['training']['eval_freq'],
                n_eval_episodes=self.config['training']['n_eval_episodes'],
                save_freq=self.config['training']['save_freq'],
                save_path=str(self.save_dir / "models"),
                log_path=str(self.save_dir / "logs")
            )
            
            print("\n✅ Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        
        # 최종 평가
        final_results = self.evaluate()
        
        # 결과 저장
        self.save_results(final_results)
        
        return final_results
    
    def _setup_callbacks(self) -> CallbackList:
        """학습 콜백 설정"""
        callbacks = []
        
        # 체크포인트 콜백
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['save_freq'],
            save_path=str(self.save_dir / "checkpoints"),
            name_prefix="ppo_kimchi"
        )
        callbacks.append(checkpoint_callback)
        
        # 커스텀 콜백 (성과 추적)
        performance_callback = PerformanceTrackingCallback(
            trainer=self,
            target_sharpe=self.config['training']['target_sharpe'],
            patience=self.config['training']['early_stopping_patience']
        )
        callbacks.append(performance_callback)
        
        return CallbackList(callbacks)
    
    def evaluate(self, n_episodes: int = 100) -> Dict:
        """
        모델 평가
        
        Args:
            n_episodes: 평가 에피소드 수
            
        Returns:
            평가 결과
        """
        print(f"\nEvaluating model for {n_episodes} episodes...")
        
        # 평가 실행
        mean_reward, std_reward = evaluate_policy(
            self.agent.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=False
        )
        
        # 상세 평가
        episode_rewards = []
        episode_sharpes = []
        episode_drawdowns = []
        episode_trades = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()[0]
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.agent.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = done or truncated
            
            # 메트릭 수집
            episode_rewards.append(episode_reward)
            episode_sharpes.append(info.get('sharpe_ratio', 0))
            episode_drawdowns.append(info.get('max_drawdown', 0))
            episode_trades.append(info.get('total_trades', 0))
        
        # 결과 계산
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_sharpe': np.mean(episode_sharpes),
            'best_sharpe': np.max(episode_sharpes),
            'mean_drawdown': np.mean(episode_drawdowns),
            'worst_drawdown': np.max(episode_drawdowns),
            'win_rate': sum(1 for r in episode_rewards if r > 0) / len(episode_rewards),
            'avg_trades': np.mean(episode_trades)
        }
        
        # 결과 출력
        print("\n" + "="*50)
        print("Evaluation Results:")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Sharpe: {results['mean_sharpe']:.3f}")
        print(f"Best Sharpe: {results['best_sharpe']:.3f}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Mean Drawdown: {results['mean_drawdown']:.2%}")
        print(f"Worst Drawdown: {results['worst_drawdown']:.2%}")
        print("="*50)
        
        return results
    
    def backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        백테스팅 실행
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            백테스트 결과
        """
        print("\nRunning backtest...")
        
        # 백테스트 환경 생성
        backtest_env = KimchiPremiumTradingEnv(
            df=self.df,
            **self.config['env']
        )
        
        # 백테스트 실행
        obs, _ = backtest_env.reset()
        done = False
        
        trades = []
        portfolio_values = []
        
        while not done:
            action, _ = self.agent.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = backtest_env.step(action)
            done = done or truncated
            
            portfolio_values.append(info['portfolio_value'])
            
            # 거래 기록
            if len(backtest_env.trades) > len(trades):
                trades.append(backtest_env.trades[-1])
        
        # 성과 계산
        initial_value = self.config['env']['initial_balance']
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # 최대 드로우다운
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # 샤프비율
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 1440)
        
        results = {
            'total_return': total_return,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'portfolio_values': portfolio_values,
            'trades': trades
        }
        
        print(f"\nBacktest Results:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Final Value: ₩{final_value:,.0f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Total Trades: {len(trades)}")
        
        return results
    
    def save_results(self, results: Dict):
        """결과 저장"""
        # JSON 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 파일
        results_file = self.save_dir / f"results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # NumPy 배열을 리스트로 변환
            json_results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
        
        # 모델 저장
        model_file = self.save_dir / f"final_model_{timestamp}"
        self.agent.save(str(model_file))
        
        print(f"\n✅ Results saved to {self.save_dir}")
    
    def plot_results(self, results: Dict):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 포트폴리오 가치
        if 'portfolio_values' in results:
            axes[0, 0].plot(results['portfolio_values'])
            axes[0, 0].set_title('Portfolio Value')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value (KRW)')
        
        # 학습 곡선
        if self.training_history['episodes']:
            axes[0, 1].plot(self.training_history['rewards'])
            axes[0, 1].set_title('Training Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
        
        # 샤프비율
        if self.training_history['sharpe_ratios']:
            axes[1, 0].plot(self.training_history['sharpe_ratios'])
            axes[1, 0].axhline(y=self.config['training']['target_sharpe'], 
                              color='r', linestyle='--', label='Target')
            axes[1, 0].set_title('Sharpe Ratio')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Sharpe')
            axes[1, 0].legend()
        
        # Win Rate
        if self.training_history['win_rates']:
            axes[1, 1].plot(self.training_history['win_rates'])
            axes[1, 1].axhline(y=0.6, color='r', linestyle='--', label='Target 60%')
            axes[1, 1].set_title('Win Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_file = self.save_dir / f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        plt.show()
        
        print(f"Plot saved to {plot_file}")


class PerformanceTrackingCallback:
    """성과 추적 콜백"""
    
    def __init__(
        self,
        trainer: PPOTrainer,
        target_sharpe: float = 1.5,
        patience: int = 50
    ):
        self.trainer = trainer
        self.target_sharpe = target_sharpe
        self.patience = patience
        self.best_sharpe = -np.inf
        self.patience_counter = 0
        
    def on_rollout_end(self):
        """롤아웃 종료 시 호출"""
        # 성과 기록
        if hasattr(self, 'locals'):
            info = self.locals.get('infos', [{}])[0]
            
            self.trainer.training_history['episodes'].append(
                len(self.trainer.training_history['episodes'])
            )
            self.trainer.training_history['rewards'].append(
                info.get('episode', {}).get('r', 0)
            )
            self.trainer.training_history['sharpe_ratios'].append(
                info.get('sharpe_ratio', 0)
            )
            
            # 조기 종료 체크
            current_sharpe = info.get('sharpe_ratio', 0)
            if current_sharpe > self.best_sharpe:
                self.best_sharpe = current_sharpe
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if current_sharpe >= self.target_sharpe:
                print(f"\n🎯 Target Sharpe ratio achieved: {current_sharpe:.3f}")
            
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping: No improvement for {self.patience} episodes")
                return False
        
        return True


if __name__ == "__main__":
    # 학습 실행
    trainer = PPOTrainer(
        data_path="../../data/historical/training/kimchi_premium.csv",
        save_dir="./ppo_results"
    )
    
    # 학습
    results = trainer.train(use_adaptive=True)
    
    # 백테스트
    backtest_results = trainer.backtest()
    
    # 시각화
    trainer.plot_results(results)
    
    print("\n✅ PPO Training Complete!")