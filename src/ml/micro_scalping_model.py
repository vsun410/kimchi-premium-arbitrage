"""
Micro Scalping Model for Small Movements
작은 김프 변동을 포착하는 마이크로 스쾘핑 모델
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MicroDecision:
    """마이크로 거래 결정"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    expected_move: float  # 예상 변동 (%)
    risk_score: float
    reason: str
    position_size: float  # BTC


class MicroScalpingModel:
    """
    마이크로 스쾘핑 모델
    - 0.05% 이상 움직임 포착
    - 빠른 진입/청산
    - 높은 빈도 거래
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.05,     # 0.05% 변동에서 진입
        target_profit: float = 0.03,       # 0.03% 목표 수익 (수수료 후)
        stop_loss: float = 0.02,           # 0.02% 손절
        max_holding_minutes: int = 15,     # 최대 15분 보유
        min_confidence: float = 0.55,      # 최소 신뢰도 (55%)
        position_size: float = 0.1         # 0.1 BTC per trade
    ):
        self.entry_threshold = entry_threshold
        self.target_profit = target_profit
        self.stop_loss = stop_loss
        self.max_holding_minutes = max_holding_minutes
        self.min_confidence = min_confidence
        self.base_position_size = position_size
        
        # 학습 메모리
        self.memory = deque(maxlen=2000)  # 최근 2000개 거래
        
        # 앙상블 모델
        self.models = {
            'momentum': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,  # 얕은 트리
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'mean_reversion': RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=43
            ),
            'volatility': RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=44
            )
        }
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 성과 추적
        self.recent_trades = deque(maxlen=100)
        self.daily_trades = {}
        self.is_trained = False
        
        # 현재 상태
        self.position_open = False
        self.entry_time = None
        self.entry_premium = None
        
        # 적응형 파라미터
        self.adaptive_threshold = entry_threshold
        self.adaptive_confidence = min_confidence
        
    def calculate_micro_features(self, data: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
        """
        마이크로 특징 계산 (5분 단위 최적화)
        
        Args:
            data: 김프 데이터
            idx: 현재 인덱스
            
        Returns:
            특징 벡터
        """
        if idx < 10:  # 최소 10개 데이터 필요
            return None
        
        features = []
        
        # 현재 김프
        current_premium = data['kimchi_premium'].iloc[idx]
        features.append(current_premium)
        
        # 최근 5분 변화
        if idx >= 1:
            change_1 = current_premium - data['kimchi_premium'].iloc[idx-1]
            features.append(change_1)
        else:
            features.append(0)
        
        # 5분, 10분 평균
        for window in [5, 10]:
            if idx >= window:
                ma = data['kimchi_premium'].iloc[idx-window:idx].mean()
                features.append(ma)
                features.append(current_premium - ma)
            else:
                features.append(current_premium)
                features.append(0)
        
        # 변동성 (5분)
        if idx >= 5:
            std5 = data['kimchi_premium'].iloc[idx-5:idx].std()
            features.append(std5)
            # 변동성 대비 현재 편차
            if std5 > 0:
                z_score = (current_premium - data['kimchi_premium'].iloc[idx-5:idx].mean()) / std5
                features.append(z_score)
            else:
                features.append(0)
        else:
            features.append(0.05)  # 기본 변동성
            features.append(0)
        
        # 최근 최고/최저 (10분)
        if idx >= 10:
            recent_high = data['kimchi_premium'].iloc[idx-10:idx].max()
            recent_low = data['kimchi_premium'].iloc[idx-10:idx].min()
            features.append(recent_high)
            features.append(recent_low)
            features.append((current_premium - recent_low) / (recent_high - recent_low + 1e-8))
        else:
            features.append(current_premium)
            features.append(current_premium)
            features.append(0.5)
        
        # 모멘텀 (1, 3, 5분)
        for lag in [1, 3, 5]:
            if idx >= lag:
                momentum = current_premium - data['kimchi_premium'].iloc[idx-lag]
                features.append(momentum)
            else:
                features.append(0)
        
        # 볼륨 비율 (선택적)
        if 'upbit_volume' in data.columns and 'binance_volume' in data.columns:
            vol_ratio = data['upbit_volume'].iloc[idx] / (data['binance_volume'].iloc[idx] + 1e-8)
            features.append(np.log1p(vol_ratio))
        else:
            features.append(0)
        
        # 시간 특징 (순환 인코딩)
        if hasattr(data.index[idx], 'hour'):
            hour = data.index[idx].hour
            minute = data.index[idx].minute
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(np.sin(2 * np.pi * minute / 60))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def predict_micro_movement(self, features: np.ndarray) -> MicroDecision:
        """
        마이크로 움직임 예측
        
        Args:
            features: 특징 벡터
            
        Returns:
            거래 결정
        """
        if not self.is_trained:
            # 학습 전에는 규칙 기반
            return self._rule_based_micro_decision(features)
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 앙상블 예측
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(features_scaled)[0]
                # 클래스: 0=HOLD, 1=BUY, 2=SELL
                predictions[name] = np.argmax(pred_proba)
                confidences[name] = pred_proba.max()
            except:
                predictions[name] = 0
                confidences[name] = 0.33
        
        # 가중 투표
        action_scores = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        for name, pred in predictions.items():
            weight = confidences[name]
            if name == 'momentum':
                # 모멘텀 모델: 추세 추종
                if pred == 1:
                    action_scores['BUY'] += weight * 0.4
                elif pred == 2:
                    action_scores['SELL'] += weight * 0.4
                else:
                    action_scores['HOLD'] += weight * 0.4
            elif name == 'mean_reversion':
                # 평균 회귀 모델: 반대 방향
                if pred == 1:
                    action_scores['SELL'] += weight * 0.3  # 오른 후 하락 예상
                elif pred == 2:
                    action_scores['BUY'] += weight * 0.3   # 내린 후 상승 예상
                else:
                    action_scores['HOLD'] += weight * 0.3
            elif name == 'volatility':
                # 변동성 모델: 높은 변동성에서 거래
                if pred != 0:
                    action_scores['BUY'] += weight * 0.15
                    action_scores['SELL'] += weight * 0.15
                else:
                    action_scores['HOLD'] += weight * 0.3
        
        # 최종 결정
        best_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[best_action] / sum(action_scores.values())
        
        # 신뢰도 임계값
        if confidence < self.adaptive_confidence:
            best_action = 'HOLD'
        
        # 기대 움직임 계산
        current_premium = features[0]
        change_1min = features[1] if len(features) > 1 else 0
        std5 = features[6] if len(features) > 6 else 0.05
        
        if std5 > 0:
            expected_move = min(std5 * 0.5, self.target_profit)
        else:
            expected_move = self.target_profit
        
        # 리스크 점수
        risk_score = min(std5 / 0.1, 1.0)  # 0.1% 변동성 기준
        
        # 포지션 크기 결정
        position_size = self.base_position_size
        if confidence > 0.7:
            position_size *= 1.2
        elif confidence < 0.6:
            position_size *= 0.8
        
        return MicroDecision(
            action=best_action,
            confidence=confidence,
            expected_move=expected_move,
            risk_score=risk_score,
            reason=f"Ensemble: {best_action} (conf={confidence:.2f})",
            position_size=position_size
        )
    
    def _rule_based_micro_decision(self, features: np.ndarray) -> MicroDecision:
        """규칙 기반 마이크로 결정"""
        current_premium = features[0]
        change_1min = features[1] if len(features) > 1 else 0
        ma5 = features[2] if len(features) > 2 else current_premium
        std5 = features[6] if len(features) > 6 else 0.05
        
        action = 'HOLD'
        confidence = 0.5
        
        # 급격한 변화 감지
        if abs(change_1min) > self.adaptive_threshold:
            if current_premium > ma5 + std5:
                action = 'SELL'  # 높은 김프는 하락 예상
                confidence = min(abs(change_1min) / self.adaptive_threshold, 0.8)
            elif current_premium < ma5 - std5:
                action = 'BUY'   # 낮은 김프는 상승 예상
                confidence = min(abs(change_1min) / self.adaptive_threshold, 0.8)
        
        return MicroDecision(
            action=action,
            confidence=confidence,
            expected_move=self.target_profit,
            risk_score=0.5,
            reason="Rule-based micro decision",
            position_size=self.base_position_size
        )
    
    def should_close_position(self, entry_premium: float, current_premium: float, 
                            holding_minutes: int) -> Tuple[bool, str]:
        """
        포지션 청산 여부 결정
        
        Args:
            entry_premium: 진입 시 김프
            current_premium: 현재 김프
            holding_minutes: 보유 시간(분)
            
        Returns:
            (uccad산 여부, 이유)
        """
        premium_change = abs(current_premium - entry_premium)
        
        # 1. 목표 달성
        if premium_change >= self.target_profit:
            return True, f"Target reached: {premium_change:.4f}%"
        
        # 2. 손절
        if premium_change >= self.stop_loss and \
           np.sign(current_premium - entry_premium) != np.sign(entry_premium):
            return True, f"Stop loss: {premium_change:.4f}%"
        
        # 3. 시간 초과
        if holding_minutes >= self.max_holding_minutes:
            return True, f"Time limit: {holding_minutes} minutes"
        
        # 4. 평균 회귀 완료 (김프가 0에 가까워지면)
        if abs(current_premium) < abs(entry_premium) * 0.3:
            return True, "Mean reversion complete"
        
        return False, "Hold position"
    
    def update_learning(self, state: np.ndarray, action: str, reward: float, 
                       next_state: np.ndarray):
        """
        학습 데이터 업데이트
        
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
        
        # 최근 거래 기록
        if action != 'HOLD':
            self.recent_trades.append({
                'action': action,
                'reward': reward,
                'timestamp': datetime.now()
            })
        
        # 적응형 파라미터 조정
        if len(self.recent_trades) >= 20:
            recent_rewards = [t['reward'] for t in list(self.recent_trades)[-20:]]
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            
            if win_rate > 0.65:
                # 성과 좋으면 더 공격적으로
                self.adaptive_threshold *= 0.95
                self.adaptive_confidence *= 0.98
            elif win_rate < 0.45:
                # 성과 나쁘면 보수적으로
                self.adaptive_threshold *= 1.05
                self.adaptive_confidence *= 1.02
            
            # 범위 제한
            self.adaptive_threshold = np.clip(self.adaptive_threshold, 0.03, 0.10)
            self.adaptive_confidence = np.clip(self.adaptive_confidence, 0.50, 0.70)
    
    def train_models(self, min_samples: int = 200):
        """
        모델 학습
        
        Args:
            min_samples: 최소 학습 샘플 수
        """
        if len(self.memory) < min_samples:
            return
        
        # 데이터 준비
        samples = list(self.memory)
        
        # 최근 데이터와 과거 데이터 균형
        if len(samples) > min_samples * 2:
            recent = samples[-min_samples:]
            historical = np.random.choice(samples[:-min_samples], 
                                        size=min_samples, 
                                        replace=False).tolist()
            samples = recent + historical
        
        X = np.array([s['state'] for s in samples])
        
        # 레이블 생성 (3클래스: HOLD=0, BUY=1, SELL=2)
        y_momentum = []
        y_reversion = []
        y_volatility = []
        
        for s in samples:
            reward = s['reward']
            action = s['action']
            
            # Momentum 레이블
            if action == 'BUY' and reward > 0:
                y_momentum.append(1)
            elif action == 'SELL' and reward > 0:
                y_momentum.append(2)
            else:
                y_momentum.append(0)
            
            # Mean Reversion 레이블 (반대)
            if action == 'BUY' and reward > 0:
                y_reversion.append(2)  # 매수 성공 = 매도 신호
            elif action == 'SELL' and reward > 0:
                y_reversion.append(1)  # 매도 성공 = 매수 신호
            else:
                y_reversion.append(0)
            
            # Volatility 레이블
            if abs(reward) > self.target_profit * 0.5:
                y_volatility.append(1 if action == 'BUY' else 2)
            else:
                y_volatility.append(0)
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 학습
        try:
            self.models['momentum'].fit(X_scaled, y_momentum)
            self.models['mean_reversion'].fit(X_scaled, y_reversion)
            self.models['volatility'].fit(X_scaled, y_volatility)
            self.is_trained = True
            print(f"Models trained with {len(samples)} samples")
        except Exception as e:
            print(f"Training failed: {e}")
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        if len(self.recent_trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_reward': 0,
                'current_threshold': self.adaptive_threshold,
                'current_confidence': self.adaptive_confidence
            }
        
        recent = list(self.recent_trades)
        rewards = [t['reward'] for t in recent]
        
        return {
            'total_trades': len(recent),
            'win_rate': sum(1 for r in rewards if r > 0) / len(rewards) * 100,
            'avg_reward': np.mean(rewards),
            'current_threshold': self.adaptive_threshold,
            'current_confidence': self.adaptive_confidence,
            'is_trained': self.is_trained
        }