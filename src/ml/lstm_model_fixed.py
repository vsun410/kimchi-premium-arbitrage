"""
Fixed LSTM Model with Attention for Kimchi Premium Prediction
차원 오류를 수정한 LSTM 모델
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler


@dataclass
class LSTMPrediction:
    """LSTM 예측 결과"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_premium: float
    attention_weights: np.ndarray
    reason: str


class AttentionLayer(nn.Module):
    """Attention 메커니즘"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = self.softmax(attention_weights)  # (batch, seq_len, 1)
        
        # Weighted sum
        weighted_output = lstm_output * attention_weights  # (batch, seq_len, hidden_dim)
        context = torch.sum(weighted_output, dim=1)  # (batch, hidden_dim)
        
        return context, attention_weights.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM with Attention for Kimchi Premium"""
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 3  # 3 classes: BUY, SELL, HOLD
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim*2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        # context: (batch, hidden_dim*2)
        
        # Classification
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out, attention_weights


class LSTMPredictor:
    """LSTM 기반 예측기"""
    
    def __init__(
        self,
        sequence_length: int = 48,  # 48시간 (2일) 히스토리
        feature_dim: int = 20,
        entry_threshold: float = 0.05,
        confidence_threshold: float = 0.6
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.entry_threshold = entry_threshold
        self.confidence_threshold = confidence_threshold
        
        # 모델 초기화
        self.model = LSTMModel(
            input_dim=feature_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2
        )
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 학습 상태
        self.is_trained = False
        self.training_history = []
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        LSTM을 위한 특징 준비
        
        Args:
            data: 가격 데이터 (최소 sequence_length 필요)
            
        Returns:
            특징 배열 (sequence_length, feature_dim)
        """
        if len(data) < self.sequence_length:
            return None
        
        features = []
        
        for i in range(len(data)):
            row_features = []
            
            # 기본 가격 정보
            if 'kimchi_premium' in data.columns:
                row_features.append(data['kimchi_premium'].iloc[i])
            
            if 'upbit_close' in data.columns:
                row_features.append(np.log(data['upbit_close'].iloc[i]))
            
            if 'binance_close' in data.columns:
                row_features.append(np.log(data['binance_close'].iloc[i]))
            
            # 기술적 지표
            if i >= 5:
                # 5기간 이동평균
                ma5 = data['kimchi_premium'].iloc[i-5:i].mean()
                row_features.append(ma5)
                row_features.append(data['kimchi_premium'].iloc[i] - ma5)
            else:
                row_features.extend([0, 0])
            
            if i >= 20:
                # 20기간 이동평균
                ma20 = data['kimchi_premium'].iloc[i-20:i].mean()
                row_features.append(ma20)
                row_features.append(data['kimchi_premium'].iloc[i] - ma20)
            else:
                row_features.extend([0, 0])
            
            # 변동성
            if i >= 10:
                std10 = data['kimchi_premium'].iloc[i-10:i].std()
                row_features.append(std10)
            else:
                row_features.append(0)
            
            # 모멘텀
            for lag in [1, 5, 10]:
                if i >= lag:
                    momentum = data['kimchi_premium'].iloc[i] - data['kimchi_premium'].iloc[i-lag]
                    row_features.append(momentum)
                else:
                    row_features.append(0)
            
            # RSI
            if i >= 14:
                gains = []
                losses = []
                for j in range(i-14, i):
                    change = data['kimchi_premium'].iloc[j] - data['kimchi_premium'].iloc[j-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                row_features.append(rsi / 100)  # Normalize
            else:
                row_features.append(0.5)
            
            # 볼륨 정보
            if 'upbit_volume' in data.columns and 'binance_volume' in data.columns:
                vol_ratio = data['upbit_volume'].iloc[i] / (data['binance_volume'].iloc[i] + 1e-8)
                row_features.append(np.log1p(vol_ratio))
            else:
                row_features.append(0)
            
            # 시간 특징
            if hasattr(data.index[i], 'hour'):
                hour = data.index[i].hour
                row_features.append(np.sin(2 * np.pi * hour / 24))
                row_features.append(np.cos(2 * np.pi * hour / 24))
            else:
                row_features.extend([0, 0])
            
            # Padding to reach feature_dim
            while len(row_features) < self.feature_dim:
                row_features.append(0)
            
            # Truncate if too many
            row_features = row_features[:self.feature_dim]
            
            features.append(row_features)
        
        return np.array(features)
    
    def predict(self, data: pd.DataFrame) -> LSTMPrediction:
        """
        LSTM으로 예측
        
        Args:
            data: 최근 sequence_length 시간의 데이터
            
        Returns:
            예측 결과
        """
        # 특징 준비
        features = self.prepare_features(data)
        
        if features is None or len(features) < self.sequence_length:
            return LSTMPrediction(
                action='HOLD',
                confidence=0.0,
                predicted_premium=0.0,
                attention_weights=np.array([]),
                reason='Insufficient data'
            )
        
        # 최근 sequence_length만큼 추출
        sequence = features[-self.sequence_length:]
        
        if not self.is_trained:
            # 학습 전에는 규칙 기반
            current_premium = data['kimchi_premium'].iloc[-1]
            ma20 = data['kimchi_premium'].iloc[-20:].mean()
            
            if abs(current_premium - ma20) > self.entry_threshold:
                action = 'SELL' if current_premium > ma20 else 'BUY'
                confidence = min(abs(current_premium - ma20) / self.entry_threshold, 1.0)
            else:
                action = 'HOLD'
                confidence = 0.5
            
            return LSTMPrediction(
                action=action,
                confidence=confidence,
                predicted_premium=ma20,
                attention_weights=np.ones(self.sequence_length) / self.sequence_length,
                reason='Rule-based (not trained)'
            )
        
        # 정규화
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, self.feature_dim))
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        # 텐서 변환
        x = torch.FloatTensor(sequence_scaled)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            output, attention_weights = self.model(x)
        
        # 결과 해석
        probs = output.numpy()[0]
        attention = attention_weights.numpy()[0]
        
        # 클래스: 0=HOLD, 1=BUY, 2=SELL
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        if predicted_class == 1:
            action = 'BUY'
        elif predicted_class == 2:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # 신뢰도 임계값
        if confidence < self.confidence_threshold:
            action = 'HOLD'
        
        # 예상 김프 (간단한 추정)
        current_premium = data['kimchi_premium'].iloc[-1]
        if action == 'BUY':
            predicted_premium = current_premium - 0.1  # 하락 예상
        elif action == 'SELL':
            predicted_premium = current_premium + 0.1  # 상승 예상
        else:
            predicted_premium = current_premium
        
        # 중요 시점 찾기 (attention 기반)
        important_times = np.argsort(attention)[-5:]  # Top 5
        reason = f"LSTM prediction based on attention to indices {important_times.tolist()}"
        
        return LSTMPrediction(
            action=action,
            confidence=float(confidence),
            predicted_premium=predicted_premium,
            attention_weights=attention,
            reason=reason
        )
    
    def train(self, train_data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """
        LSTM 모델 학습
        
        Args:
            train_data: 학습 데이터
            epochs: 에폭 수
            batch_size: 배치 크기
        """
        print(f"Training LSTM model with {len(train_data)} samples...")
        
        # 특징 준비
        features = self.prepare_features(train_data)
        if features is None or len(features) < self.sequence_length * 2:
            print("Insufficient data for training")
            return
        
        # 시퀀스 생성
        X = []
        y = []
        
        for i in range(self.sequence_length, len(features) - 1):
            # 입력: 과거 sequence_length 시간
            X.append(features[i-self.sequence_length:i])
            
            # 레이블: 다음 시간의 변화 방향
            current_premium = train_data['kimchi_premium'].iloc[i]
            next_premium = train_data['kimchi_premium'].iloc[i+1]
            change = next_premium - current_premium
            
            if abs(change) < 0.02:  # 작은 변화
                label = 0  # HOLD
            elif change > 0:  # 상승
                label = 2  # SELL (김프 상승 = 매도 기회)
            else:  # 하락
                label = 1  # BUY (김프 하락 = 매수 기회)
            
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # 정규화
        X_reshaped = X.reshape(-1, self.feature_dim)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(-1, self.sequence_length, self.feature_dim)
        
        # 텐서 변환
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # 데이터셋
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 옵티마이저와 손실 함수
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 학습
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                # Forward
                output, _ = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 통계
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # 에폭 결과
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total * 100
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
        
        self.is_trained = True
        print("LSTM training completed!")
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {filepath}")