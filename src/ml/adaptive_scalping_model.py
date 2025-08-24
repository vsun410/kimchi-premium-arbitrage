"""
Adaptive Scalping Model with Online Learning
점진적 학습과 과적합 방지를 위한 적응형 모델
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradingDecision:
    """거래 결정"""
    action: str  # 'ENTER_LONG', 'ENTER_SHORT', 'EXIT', 'HOLD'
    confidence: float
    expected_return: float
    risk_score: float
    reason: str


class AdaptiveScalpingModel:
    """
    적응형 스캘핑 모델
    - 온라인 학습
    - 과적합 방지
    - 점진적 목표 상향
    """
    
    def __init__(
        self,
        initial_target: float = 0.15,  # 초기 목표 0.15% (월 1.5%)
        max_target: float = 0.25,      # 최종 목표 0.25% (월 2.5%)
        learning_rate: float = 0.01,   # 학습률
        memory_size: int = 1000,       # 경험 메모리 크기
        min_confidence: float = 0.6    # 최소 신뢰도
    ):
        self.current_target = initial_target
        self.max_target = max_target
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        
        # 경험 메모리 (과적합 방지)
        self.memory = deque(maxlen=memory_size)
        
        # 모델들 (앙상블)
        self.models = {
            'trend': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,  # 얕은 트리로 과적합 방지
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'reversal': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=43
            ),
            'volatility': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=44
            )
        }
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 성과 추적
        self.performance_history = []
        self.win_rate_history = deque(maxlen=100)
        self.is_trained = False
        
        # 적응형 파라미터
        self.adaptive_params = {
            'entry_threshold': 0.2,   # 진입 임계값
            'exit_threshold': 0.1,    # 청산 임계값
            'stop_loss': 0.1,         # 손절
            'max_holding': 30,        # 최대 보유 시간(분)
            'position_size': 0.02     # 포지션 크기
        }
        
    def calculate_features(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """
        특징 계산 (과적합 방지를 위해 단순하게)
        
        Args:
            data: OHLCV 데이터
            idx: 현재 인덱스
            
        Returns:
            특징 벡터
        """
        if idx < 30:
            return None
        
        features = []
        
        # 현재 김프
        current_premium = data['kimchi_premium'].iloc[idx]
        features.append(current_premium)
        
        # 이동평균 (5, 15, 30분)
        for window in [5, 15, 30]:
            ma = data['kimchi_premium'].iloc[idx-window:idx].mean()
            features.append(ma)
            features.append(current_premium - ma)  # 편차
        
        # 변동성 (5, 15분)
        for window in [5, 15]:
            std = data['kimchi_premium'].iloc[idx-window:idx].std()
            features.append(std)
        
        # 모멘텀 (1, 5, 15분)
        for lag in [1, 5, 15]:
            momentum = current_premium - data['kimchi_premium'].iloc[idx-lag]
            features.append(momentum)
        
        # 볼륨 비율 (업비트/바이낸스)
        if 'upbit_volume' in data.columns and 'binance_volume' in data.columns:
            vol_ratio = data['upbit_volume'].iloc[idx] / (data['binance_volume'].iloc[idx] + 1e-8)
            features.append(np.log1p(vol_ratio))
        else:
            features.append(0)
        
        # 시간 특징 (과적합 방지를 위해 순환 인코딩)
        hour = data.index[idx].hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # RSI (14분)
        if idx >= 14:
            gains = []
            losses = []
            for i in range(idx-14, idx):
                change = data['kimchi_premium'].iloc[i] - data['kimchi_premium'].iloc[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            features.append(rsi)
        else:
            features.append(50)
        
        return np.array(features)
    
    def predict(self, features: np.ndarray) -> TradingDecision:
        """
        거래 결정 예측
        
        Args:
            features: 특징 벡터
            
        Returns:
            거래 결정
        """
        if not self.is_trained:
            # 학습 전에는 규칙 기반
            return self._rule_based_decision(features)
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 앙상블 예측
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0].max()
            predictions[name] = pred
            confidences[name] = prob
        
        # 가중 투표
        action_scores = {
            'ENTER_LONG': 0,
            'ENTER_SHORT': 0,
            'EXIT': 0,
            'HOLD': 0
        }
        
        for name, pred in predictions.items():
            weight = confidences[name]
            if name == 'trend':
                if pred == 1:
                    action_scores['ENTER_LONG'] += weight * 0.4
                elif pred == -1:
                    action_scores['ENTER_SHORT'] += weight * 0.4
            elif name == 'reversal':
                if pred == 1:
                    action_scores['ENTER_SHORT'] += weight * 0.3
                elif pred == -1:
                    action_scores['ENTER_LONG'] += weight * 0.3
            elif name == 'volatility':
                if pred == 1:
                    action_scores['ENTER_LONG'] += weight * 0.3
                    action_scores['ENTER_SHORT'] += weight * 0.3
                else:
                    action_scores['HOLD'] += weight
        
        # 최종 결정
        best_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[best_action]
        
        # 신뢰도 임계값
        if confidence < self.min_confidence:
            best_action = 'HOLD'
        
        # 기대 수익 계산
        expected_return = self._calculate_expected_return(features, best_action)
        
        # 리스크 점수
        risk_score = self._calculate_risk_score(features)
        
        return TradingDecision(
            action=best_action,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=risk_score,
            reason=f"Ensemble vote: {best_action} ({confidence:.2f})"
        )
    
    def _rule_based_decision(self, features: np.ndarray) -> TradingDecision:
        """규칙 기반 결정 (학습 전)"""
        current_premium = features[0]
        ma15 = features[4]
        std5 = features[9]
        
        # 단순 규칙
        if abs(current_premium - ma15) > self.adaptive_params['entry_threshold']:
            if current_premium > ma15:
                action = 'ENTER_SHORT'  # 평균 회귀 전략
            else:
                action = 'ENTER_LONG'
            
            confidence = min(abs(current_premium - ma15) / std5, 1.0) if std5 > 0 else 0.5
        else:
            action = 'HOLD'
            confidence = 0.3
        
        return TradingDecision(
            action=action,
            confidence=confidence,
            expected_return=self.current_target,
            risk_score=0.5,
            reason="Rule-based decision"
        )
    
    def _calculate_expected_return(self, features: np.ndarray, action: str) -> float:
        """기대 수익 계산"""
        if action == 'HOLD':
            return 0.0
        
        # 변동성 기반 예측
        std5 = features[9] if len(features) > 9 else 0.1
        
        # 보수적 추정
        if action in ['ENTER_LONG', 'ENTER_SHORT']:
            return min(std5 * 0.5, self.current_target)
        else:
            return 0.0
    
    def _calculate_risk_score(self, features: np.ndarray) -> float:
        """리스크 점수 계산 (0-1)"""
        std15 = features[10] if len(features) > 10 else 0.1
        
        # 변동성이 높을수록 리스크 높음
        risk = min(std15 / 2.0, 1.0)
        
        return risk
    
    def update_memory(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray):
        """
        경험 메모리 업데이트
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
        """
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': datetime.now()
        })
        
        # 성과 추적
        if action != 'HOLD':
            self.win_rate_history.append(1 if reward > 0 else 0)
    
    def train_online(self, min_samples: int = 100):
        """
        온라인 학습 (과적합 방지)
        
        Args:
            min_samples: 최소 학습 샘플 수
        """
        if len(self.memory) < min_samples:
            return
        
        # 최근 데이터와 과거 데이터 균형있게 샘플링
        recent_samples = list(self.memory)[-min_samples//2:]
        random_samples = np.random.choice(
            list(self.memory)[:-min_samples//2],
            size=min(min_samples//2, len(list(self.memory)[:-min_samples//2])),
            replace=False
        ).tolist() if len(self.memory) > min_samples//2 else []
        
        samples = recent_samples + random_samples
        
        # 특징과 레이블 준비
        X = np.array([s['state'] for s in samples])
        
        # 각 모델별 레이블 생성
        y_trend = []
        y_reversal = []
        y_volatility = []
        
        for s in samples:
            reward = s['reward']
            
            # Trend 모델: 모멘텀 추종
            if s['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                y_trend.append(1 if reward > 0 else -1)
            else:
                y_trend.append(0)
            
            # Reversal 모델: 평균 회귀
            if s['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                y_reversal.append(-1 if reward > 0 else 1)
            else:
                y_reversal.append(0)
            
            # Volatility 모델: 변동성 활용
            y_volatility.append(1 if abs(reward) > self.current_target * 0.5 else 0)
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        # 각 모델 학습 (점진적)
        for name, y in [('trend', y_trend), ('reversal', y_reversal), ('volatility', y_volatility)]:
            # 부분 학습 (과적합 방지)
            if self.is_trained:
                # 기존 모델 가중치 유지하며 업데이트
                self.models[name].n_estimators += 10
                self.models[name].fit(X_scaled, y)
            else:
                self.models[name].fit(X_scaled, y)
        
        self.is_trained = True
        
        # 목표 점진적 상향
        self._adjust_target()
    
    def _adjust_target(self):
        """목표 수익률 조정"""
        if len(self.win_rate_history) < 20:
            return
        
        recent_win_rate = np.mean(list(self.win_rate_history)[-20:])
        
        # 승률이 70% 이상이면 목표 상향
        if recent_win_rate > 0.7 and self.current_target < self.max_target:
            self.current_target = min(
                self.current_target * (1 + self.learning_rate),
                self.max_target
            )
            print(f"Target increased to {self.current_target:.3f}%")
        
        # 승률이 40% 이하면 목표 하향
        elif recent_win_rate < 0.4 and self.current_target > 0.1:
            self.current_target *= (1 - self.learning_rate)
            print(f"Target decreased to {self.current_target:.3f}%")
        
        # 파라미터도 조정
        self._adjust_parameters(recent_win_rate)
    
    def _adjust_parameters(self, win_rate: float):
        """적응형 파라미터 조정"""
        if win_rate > 0.6:
            # 성과 좋으면 더 공격적으로
            self.adaptive_params['entry_threshold'] *= 0.95
            self.adaptive_params['position_size'] = min(
                self.adaptive_params['position_size'] * 1.05,
                0.05
            )
        elif win_rate < 0.4:
            # 성과 나쁘면 보수적으로
            self.adaptive_params['entry_threshold'] *= 1.05
            self.adaptive_params['position_size'] = max(
                self.adaptive_params['position_size'] * 0.95,
                0.01
            )
    
    def get_performance_metrics(self) -> Dict:
        """성과 지표 반환"""
        if len(self.win_rate_history) == 0:
            return {
                'win_rate': 0,
                'current_target': self.current_target,
                'memory_size': len(self.memory),
                'is_trained': self.is_trained
            }
        
        return {
            'win_rate': np.mean(list(self.win_rate_history)),
            'recent_win_rate': np.mean(list(self.win_rate_history)[-20:]) if len(self.win_rate_history) >= 20 else 0,
            'current_target': self.current_target,
            'memory_size': len(self.memory),
            'is_trained': self.is_trained,
            'entry_threshold': self.adaptive_params['entry_threshold'],
            'position_size': self.adaptive_params['position_size']
        }
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'current_target': self.current_target,
            'adaptive_params': self.adaptive_params,
            'is_trained': self.is_trained,
            'memory': list(self.memory)[-100:]  # 최근 100개만 저장
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.current_target = model_data['current_target']
        self.adaptive_params = model_data['adaptive_params']
        self.is_trained = model_data['is_trained']
        
        # 메모리 복원
        for exp in model_data.get('memory', []):
            self.memory.append(exp)
        
        print(f"Model loaded from {filepath}")