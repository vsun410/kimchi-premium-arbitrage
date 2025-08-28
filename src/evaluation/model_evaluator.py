"""
통합 모델 평가 시스템 (Task #19)
LSTM, XGBoost, PPO 모델의 성능을 종합적으로 평가
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 imports
from src.evaluation.performance_metrics import PerformanceMetrics


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    model_name: str
    model_type: str  # LSTM, XGBoost, PPO
    evaluation_date: datetime
    
    # 수익성 메트릭
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # 리스크 메트릭
    volatility: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    
    # 리스크 조정 메트릭
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # 거래 효율 메트릭
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # ML 메트릭
    mae: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    
    # 추가 정보
    total_trades: int = 0
    trading_days: int = 0
    best_day_return: float = 0
    worst_day_return: float = 0
    metadata: Dict = field(default_factory=dict)


class BaseModelEvaluator(ABC):
    """모델 평가 베이스 클래스"""
    
    @abstractmethod
    def evaluate(self, data: Any) -> EvaluationResult:
        """모델 평가"""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """예측 수행"""
        pass


class LSTMEvaluator(BaseModelEvaluator):
    """LSTM 모델 평가기"""
    
    def __init__(self, model_path: str):
        """
        LSTM 평가기 초기화
        
        Args:
            model_path: 모델 파일 경로
        """
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        import torch
        
        if Path(self.model_path).exists():
            return torch.load(self.model_path)
        return None
        
    def evaluate(self, test_data: pd.DataFrame) -> EvaluationResult:
        """LSTM 모델 평가"""
        predictions = self.predict(test_data)
        actual = test_data['premium_rate'].values
        
        # ML 메트릭 계산
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        
        # 방향성 정확도
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actual))
        accuracy = np.mean(pred_direction == actual_direction)
        
        # 거래 시뮬레이션
        portfolio_values = self._simulate_trading(test_data, predictions)
        metrics = PerformanceMetrics(portfolio_values)
        
        return EvaluationResult(
            model_name="LSTM",
            model_type="LSTM",
            evaluation_date=datetime.now(),
            total_return=metrics.total_return(),
            annual_return=metrics.annual_return(),
            monthly_return=metrics.monthly_return(),
            daily_return=metrics.daily_return(),
            volatility=metrics.volatility(),
            max_drawdown=metrics.max_drawdown(),
            var_95=metrics.value_at_risk(0.95),
            cvar_95=metrics.conditional_var(0.95),
            sharpe_ratio=metrics.sharpe_ratio(),
            sortino_ratio=metrics.sortino_ratio(),
            calmar_ratio=metrics.calmar_ratio(),
            information_ratio=metrics.information_ratio(),
            win_rate=metrics.win_rate(),
            profit_factor=metrics.profit_factor(),
            avg_win=metrics.avg_win(),
            avg_loss=metrics.avg_loss(),
            max_consecutive_wins=metrics.max_consecutive_wins(),
            max_consecutive_losses=metrics.max_consecutive_losses(),
            mae=mae,
            rmse=rmse,
            accuracy=accuracy,
            total_trades=metrics.total_trades(),
            trading_days=len(test_data)
        )
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """LSTM 예측"""
        if self.model is None:
            # 더미 예측 (테스트용)
            return np.random.randn(len(data)) * 2 + 3
        
        # 실제 모델 예측
        # features = self._prepare_features(data)
        # predictions = self.model.predict(features)
        return np.random.randn(len(data)) * 2 + 3  # 임시
    
    def _simulate_trading(self, data: pd.DataFrame, predictions: np.ndarray) -> List[float]:
        """거래 시뮬레이션"""
        portfolio = [10000000.0]  # 1천만원 시작
        position = 0
        
        for i in range(1, len(data)):
            # 예측 기반 거래 신호
            if predictions[i] > 3.0 and position == 0:
                # 진입
                position = portfolio[-1] * 0.3  # 30% 포지션
            elif predictions[i] < 2.0 and position > 0:
                # 청산
                returns = (data.iloc[i]['premium_rate'] - 3.0) / 100
                pnl = position * returns
                portfolio.append(portfolio[-1] + pnl)
                position = 0
            else:
                portfolio.append(portfolio[-1])
        
        return portfolio


class XGBoostEvaluator(BaseModelEvaluator):
    """XGBoost 모델 평가기"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        if Path(self.model_path).exists():
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def evaluate(self, test_data: pd.DataFrame) -> EvaluationResult:
        """XGBoost 모델 평가"""
        predictions = self.predict(test_data)
        
        # 분류 정확도
        actual_labels = (test_data['premium_rate'] > 3.0).astype(int)
        pred_labels = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_labels == actual_labels)
        
        # F1 Score
        from sklearn.metrics import f1_score
        f1 = f1_score(actual_labels, pred_labels)
        
        # 거래 시뮬레이션
        portfolio_values = self._simulate_trading(test_data, pred_labels)
        metrics = PerformanceMetrics(portfolio_values)
        
        return EvaluationResult(
            model_name="XGBoost",
            model_type="XGBoost",
            evaluation_date=datetime.now(),
            total_return=metrics.total_return(),
            annual_return=metrics.annual_return(),
            monthly_return=metrics.monthly_return(),
            daily_return=metrics.daily_return(),
            volatility=metrics.volatility(),
            max_drawdown=metrics.max_drawdown(),
            var_95=metrics.value_at_risk(0.95),
            cvar_95=metrics.conditional_var(0.95),
            sharpe_ratio=metrics.sharpe_ratio(),
            sortino_ratio=metrics.sortino_ratio(),
            calmar_ratio=metrics.calmar_ratio(),
            information_ratio=metrics.information_ratio(),
            win_rate=metrics.win_rate(),
            profit_factor=metrics.profit_factor(),
            avg_win=metrics.avg_win(),
            avg_loss=metrics.avg_loss(),
            max_consecutive_wins=metrics.max_consecutive_wins(),
            max_consecutive_losses=metrics.max_consecutive_losses(),
            accuracy=accuracy,
            f1_score=f1,
            total_trades=metrics.total_trades(),
            trading_days=len(test_data)
        )
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """XGBoost 예측"""
        if self.model is None:
            # 더미 예측
            return np.random.random(len(data))
        
        # 실제 예측
        # features = self._prepare_features(data)
        # predictions = self.model.predict_proba(features)[:, 1]
        return np.random.random(len(data))  # 임시
    
    def _simulate_trading(self, data: pd.DataFrame, signals: np.ndarray) -> List[float]:
        """거래 시뮬레이션"""
        portfolio = [10000000.0]
        position = 0
        
        for i in range(1, len(data)):
            if signals[i] == 1 and position == 0:
                position = portfolio[-1] * 0.3
            elif signals[i] == 0 and position > 0:
                returns = (data.iloc[i]['premium_rate'] - data.iloc[i-1]['premium_rate']) / 100
                pnl = position * returns
                portfolio.append(portfolio[-1] + pnl)
                position = 0
            else:
                portfolio.append(portfolio[-1])
        
        return portfolio


class PPOEvaluator(BaseModelEvaluator):
    """PPO 모델 평가기"""
    
    def __init__(self, model_path: str, env):
        self.model_path = model_path
        self.env = env
        self.model = self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        from stable_baselines3 import PPO
        
        if Path(self.model_path).exists():
            return PPO.load(self.model_path)
        return None
    
    def evaluate(self, test_episodes: int = 100) -> EvaluationResult:
        """PPO 모델 평가"""
        episode_rewards = []
        episode_lengths = []
        portfolio_history = []
        
        for episode in range(test_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                done = done or truncated
                
                if 'portfolio_value' in info:
                    portfolio_history.append(info['portfolio_value'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # 메트릭 계산
        metrics = PerformanceMetrics(portfolio_history)
        
        return EvaluationResult(
            model_name="PPO",
            model_type="PPO",
            evaluation_date=datetime.now(),
            total_return=metrics.total_return(),
            annual_return=metrics.annual_return(),
            monthly_return=metrics.monthly_return(),
            daily_return=metrics.daily_return(),
            volatility=metrics.volatility(),
            max_drawdown=metrics.max_drawdown(),
            var_95=metrics.value_at_risk(0.95),
            cvar_95=metrics.conditional_var(0.95),
            sharpe_ratio=metrics.sharpe_ratio(),
            sortino_ratio=metrics.sortino_ratio(),
            calmar_ratio=metrics.calmar_ratio(),
            information_ratio=0.0,  # PPO doesn't have benchmark
            win_rate=metrics.win_rate(),
            profit_factor=metrics.profit_factor(),
            avg_win=metrics.avg_win(),
            avg_loss=metrics.avg_loss(),
            max_consecutive_wins=metrics.max_consecutive_wins(),
            max_consecutive_losses=metrics.max_consecutive_losses(),
            total_trades=metrics.total_trades(),
            trading_days=np.mean(episode_lengths),
            metadata={
                'avg_episode_reward': np.mean(episode_rewards),
                'std_episode_reward': np.std(episode_rewards),
                'max_episode_reward': np.max(episode_rewards),
                'min_episode_reward': np.min(episode_rewards)
            }
        )
    
    def predict(self, obs: np.ndarray) -> Tuple[int, None]:
        """PPO 예측"""
        if self.model is None:
            return np.random.randint(0, 3), None
        
        action, _ = self.model.predict(obs, deterministic=True)
        return action, None


class ModelEvaluator:
    """통합 모델 평가 시스템"""
    
    def __init__(self, save_dir: str = "./evaluation_results"):
        """
        통합 평가기 초기화
        
        Args:
            save_dir: 평가 결과 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.evaluators = {}
        self.results = {}
        
    def add_evaluator(self, name: str, evaluator: BaseModelEvaluator):
        """평가기 추가"""
        self.evaluators[name] = evaluator
        
    def evaluate_all(self, **kwargs) -> Dict[str, EvaluationResult]:
        """모든 모델 평가"""
        for name, evaluator in self.evaluators.items():
            print(f"Evaluating {name}...")
            
            if isinstance(evaluator, PPOEvaluator):
                result = evaluator.evaluate(kwargs.get('test_episodes', 100))
            else:
                result = evaluator.evaluate(kwargs.get('test_data'))
            
            self.results[name] = result
            
        return self.results
    
    def compare_models(self) -> pd.DataFrame:
        """모델 비교 테이블 생성"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Type': result.model_type,
                'Total Return (%)': f"{result.total_return:.2f}",
                'Annual Return (%)': f"{result.annual_return:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
                'Calmar Ratio': f"{result.calmar_ratio:.3f}",
                'Max Drawdown (%)': f"{result.max_drawdown:.2f}",
                'Win Rate (%)': f"{result.win_rate:.1f}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Total Trades': result.total_trades
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Sharpe Ratio', ascending=False)
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Tuple[str, EvaluationResult]:
        """최고 성능 모델 반환"""
        if not self.results:
            return None, None
        
        best_name = None
        best_value = -float('inf')
        
        for name, result in self.results.items():
            value = getattr(result, metric)
            if value > best_value:
                best_value = value
                best_name = name
        
        return best_name, self.results[best_name]
    
    def save_results(self):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON으로 저장
        results_dict = {}
        for name, result in self.results.items():
            results_dict[name] = {
                k: v for k, v in result.__dict__.items()
                if not k.startswith('_')
            }
            # datetime 변환
            results_dict[name]['evaluation_date'] = result.evaluation_date.isoformat()
        
        json_file = self.save_dir / f"evaluation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # 비교 테이블 CSV 저장
        comparison_df = self.compare_models()
        csv_file = self.save_dir / f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {self.save_dir}")
        
    def print_summary(self):
        """평가 요약 출력"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        comparison_df = self.compare_models()
        print(comparison_df.to_string(index=False))
        
        # 최고 모델
        best_name, best_result = self.get_best_model('sharpe_ratio')
        if best_name:
            print(f"\n🏆 Best Model (by Sharpe Ratio): {best_name}")
            print(f"   - Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
            print(f"   - Total Return: {best_result.total_return:.2f}%")
            print(f"   - Max Drawdown: {best_result.max_drawdown:.2f}%")
        
        print("="*80)