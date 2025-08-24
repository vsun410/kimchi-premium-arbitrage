"""
Trading Strategy for Backtesting
Phase 3: 김치 프리미엄 거래 전략
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import joblib

from src.utils.logger import logger


class KimchiArbitrageStrategy:
    """
    김치 프리미엄 차익거래 전략
    
    Features:
    - ML 모델 기반 신호 생성
    - 리스크 관리
    - 동적 임계값
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        entry_threshold: float = 3.0,  # 진입 김프 (%)
        exit_threshold: float = 1.0,   # 청산 김프 (%)
        stop_loss: float = -2.0,       # 손절 (%)
        take_profit: float = 5.0,      # 익절 (%)
        use_ml_signal: bool = True
    ):
        """
        초기화
        
        Args:
            model_path: ML 모델 경로
            entry_threshold: 진입 임계값
            exit_threshold: 청산 임계값
            stop_loss: 손절선
            take_profit: 익절선
            use_ml_signal: ML 신호 사용 여부
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.use_ml_signal = use_ml_signal
        
        # ML 모델 로드
        self.model = None
        if model_path and use_ml_signal:
            try:
                self.model = joblib.load(model_path)
                logger.info(f"ML model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.use_ml_signal = False
        
        # 상태 추적
        self.position_open = False
        self.entry_premium = 0
        self.entry_price = 0
        self.position_pnl = 0
        
        # 기술적 지표 버퍼
        self.premium_history = []
        self.volatility_window = 20
        
    def calculate_features(self, data_row: pd.Series) -> np.ndarray:
        """
        ML 모델용 특징 계산
        
        Args:
            data_row: 현재 데이터 행
            
        Returns:
            특징 벡터
        """
        features = []
        
        # 기본 특징
        if 'binance_close' in data_row:
            features.append(data_row['binance_close'])
        if 'upbit_close' in data_row:
            features.append(data_row['upbit_close'])
        if 'binance_volume' in data_row:
            features.append(data_row['binance_volume'])
        if 'upbit_volume' in data_row:
            features.append(data_row['upbit_volume'])
        
        # 김프 이동평균
        if len(self.premium_history) > 0:
            features.append(np.mean(self.premium_history[-5:]))  # 5기간 MA
            features.append(np.mean(self.premium_history[-20:]) if len(self.premium_history) >= 20 else 0)  # 20기간 MA
        else:
            features.extend([0, 0])
        
        # 변동성
        if len(self.premium_history) >= self.volatility_window:
            volatility = np.std(self.premium_history[-self.volatility_window:])
            features.append(volatility)
        else:
            features.append(0)
        
        # 더미 특징 (147개 맞추기)
        while len(features) < 147:
            features.append(0)
        
        return np.array(features[:147]).reshape(1, -1)
    
    def get_ml_prediction(self, features: np.ndarray) -> float:
        """
        ML 모델 예측
        
        Args:
            features: 특징 벡터
            
        Returns:
            예측된 김프 변화
        """
        if self.model is None or not self.use_ml_signal:
            return 0.0
        
        try:
            prediction = self.model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 0.0
    
    def calculate_confidence(
        self,
        kimchi_premium: float,
        ml_prediction: float,
        volatility: float
    ) -> float:
        """
        신호 신뢰도 계산
        
        Args:
            kimchi_premium: 현재 김프
            ml_prediction: ML 예측값
            volatility: 김프 변동성
            
        Returns:
            신뢰도 (0-1)
        """
        confidence = 0.5
        
        # 김프 강도
        if abs(kimchi_premium) > self.entry_threshold:
            confidence += 0.2
        
        # ML 예측 일치
        if self.use_ml_signal:
            if (kimchi_premium > 0 and ml_prediction > 0) or \
               (kimchi_premium < 0 and ml_prediction < 0):
                confidence += 0.2
        
        # 낮은 변동성
        if volatility > 0 and volatility < 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_signal(
        self,
        timestamp: datetime,
        kimchi_premium: float,
        row: pd.Series
    ) -> Dict[str, Any]:
        """
        거래 신호 생성
        
        Args:
            timestamp: 현재 시간
            kimchi_premium: 현재 김프
            row: 현재 데이터 행
            
        Returns:
            거래 신호
        """
        # 김프 히스토리 업데이트
        self.premium_history.append(kimchi_premium)
        if len(self.premium_history) > 100:
            self.premium_history.pop(0)
        
        # 기본 신호
        signal = {
            'timestamp': timestamp,
            'action': 'HOLD',
            'confidence': 0.5,
            'kimchi_premium': kimchi_premium,
            'reason': ''
        }
        
        # ML 예측
        ml_prediction = 0
        if self.use_ml_signal:
            features = self.calculate_features(row)
            ml_prediction = self.get_ml_prediction(features)
        
        # 변동성 계산
        volatility = np.std(self.premium_history[-20:]) if len(self.premium_history) >= 20 else 0
        
        # 신뢰도 계산
        confidence = self.calculate_confidence(kimchi_premium, ml_prediction, volatility)
        
        # 포지션이 없을 때
        if not self.position_open:
            # 진입 조건
            if kimchi_premium > self.entry_threshold:
                # 롱 진입 (업비트 매수, 바이낸스 숏)
                signal['action'] = 'ENTER'
                signal['confidence'] = confidence
                signal['reason'] = f'Premium {kimchi_premium:.2f}% > threshold {self.entry_threshold}%'
                
                if self.use_ml_signal:
                    signal['ml_prediction'] = ml_prediction
                
                self.position_open = True
                self.entry_premium = kimchi_premium
                
            elif kimchi_premium < -self.entry_threshold:
                # 역프리미엄 진입 (업비트 매도, 바이낸스 롱)
                signal['action'] = 'ENTER'
                signal['confidence'] = confidence
                signal['reason'] = f'Reverse premium {kimchi_premium:.2f}% < -{self.entry_threshold}%'
                
                self.position_open = True
                self.entry_premium = kimchi_premium
        
        # 포지션이 있을 때
        else:
            # 청산 조건
            premium_change = kimchi_premium - self.entry_premium
            
            # 1. 목표 도달
            if abs(kimchi_premium) < self.exit_threshold:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Target reached: premium {kimchi_premium:.2f}% < {self.exit_threshold}%'
            
            # 2. 손절
            elif self.entry_premium > 0 and premium_change < self.stop_loss:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Stop loss: premium change {premium_change:.2f}% < {self.stop_loss}%'
            
            # 3. 익절
            elif abs(premium_change) > self.take_profit:
                signal['action'] = 'EXIT'
                signal['reason'] = f'Take profit: premium change {abs(premium_change):.2f}% > {self.take_profit}%'
            
            # 4. ML 신호 반전
            elif self.use_ml_signal and ml_prediction * self.entry_premium < 0:
                signal['action'] = 'EXIT'
                signal['reason'] = f'ML signal reversal: prediction {ml_prediction:.2f}'
            
            if signal['action'] == 'EXIT':
                self.position_open = False
                self.entry_premium = 0
        
        return signal


class SimpleThresholdStrategy:
    """
    단순 임계값 전략 (비교용)
    """
    
    def __init__(
        self,
        entry_threshold: float = 3.0,
        exit_threshold: float = 1.0
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_open = False
        
    def generate_signal(
        self,
        timestamp: datetime,
        kimchi_premium: float,
        row: pd.Series
    ) -> Dict[str, Any]:
        """
        단순 임계값 기반 신호
        """
        signal = {
            'timestamp': timestamp,
            'action': 'HOLD',
            'confidence': 0.5,
            'kimchi_premium': kimchi_premium
        }
        
        if not self.position_open:
            if abs(kimchi_premium) > self.entry_threshold:
                signal['action'] = 'ENTER'
                signal['confidence'] = min(abs(kimchi_premium) / 10, 1.0)
                self.position_open = True
        else:
            if abs(kimchi_premium) < self.exit_threshold:
                signal['action'] = 'EXIT'
                self.position_open = False
        
        return signal