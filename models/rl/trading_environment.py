"""
김치프리미엄 거래 환경 클래스 (OpenAI Gym 기반)
Task #17.1: Custom trading environment for PPO agent
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class KimchiPremiumTradingEnv(gym.Env):
    """
    김치프리미엄 차익거래를 위한 커스텀 Gym 환경
    
    State Space (20 features):
    - 김치프리미엄율 (현재, 평균, 표준편차)
    - 가격 데이터 (BTC 업비트, 바이낸스)
    - 기술 지표 (RSI, MACD, 볼린저 밴드)
    - 오더북 불균형
    - 포지션 상태
    - LSTM 예측값
    
    Action Space (3 actions):
    - 0: Hold (대기)
    - 1: Enter Position (진입)
    - 2: Exit Position (청산)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame = None,
        initial_balance: float = 10000000,  # 1천만원
        trading_fee: float = 0.001,  # 0.1% 수수료
        max_position_size: float = 0.3,  # 최대 30% 포지션
        episode_length: int = 1440,  # 24시간 (분 단위)
        lookback_period: int = 60,  # 60분 lookback
    ):
        """
        환경 초기화
        
        Args:
            df: 과거 가격 데이터 DataFrame
            initial_balance: 초기 자본금
            trading_fee: 거래 수수료
            max_position_size: 최대 포지션 크기
            episode_length: 에피소드 길이
            lookback_period: 과거 데이터 참조 기간
        """
        super().__init__()
        
        # 환경 파라미터
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.max_position_size = max_position_size
        self.episode_length = episode_length
        self.lookback_period = lookback_period
        
        # 데이터
        self.df = df if df is not None else self._generate_dummy_data()
        self.current_step = 0
        self.start_idx = 0
        
        # 계좌 상태
        self.balance = initial_balance
        self.position = 0  # 현재 포지션 (0: 없음, 양수: 롱)
        self.entry_price = 0  # 진입 가격
        self.trades = []  # 거래 기록
        
        # 성과 추적
        self.episode_returns = []
        self.price_history = deque(maxlen=lookback_period)
        self.premium_history = deque(maxlen=lookback_period)
        
        # Action & Observation Space
        self.action_space = spaces.Discrete(3)
        
        # State: 20차원 정규화된 특징
        self.observation_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(20,),
            dtype=np.float32
        )
        
        # 보상 관련
        self.last_portfolio_value = initial_balance
        
    def _generate_dummy_data(self) -> pd.DataFrame:
        """테스트용 더미 데이터 생성"""
        n_samples = 10000
        np.random.seed(42)
        
        # 기본 가격 생성
        time = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        btc_upbit = 50000000 + np.cumsum(np.random.randn(n_samples) * 10000)
        btc_binance = 35000 + np.cumsum(np.random.randn(n_samples) * 10)
        
        # 김치프리미엄 계산
        premium_rate = ((btc_upbit / 1400) / btc_binance - 1) * 100
        
        df = pd.DataFrame({
            'timestamp': time,
            'upbit_btc_krw': btc_upbit,
            'binance_btc_usdt': btc_binance,
            'premium_rate': premium_rate,
            'volume_upbit': np.random.uniform(10, 100, n_samples),
            'volume_binance': np.random.uniform(100, 1000, n_samples)
        })
        
        # 기술 지표 추가
        df['rsi'] = self._calculate_rsi(df['premium_rate'])
        df['macd'] = self._calculate_macd(df['premium_rate'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger(df['premium_rate'])
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, series: pd.Series) -> pd.Series:
        """MACD 계산"""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd.fillna(0)
    
    def _calculate_bollinger(self, series: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.fillna(series), lower.fillna(series)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 랜덤 시작 지점 선택
        max_start = len(self.df) - self.episode_length - self.lookback_period
        self.start_idx = np.random.randint(self.lookback_period, max_start)
        self.current_step = 0
        
        # 계좌 초기화
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        
        # 성과 추적 초기화
        self.episode_returns = []
        self.price_history.clear()
        self.premium_history.clear()
        self.last_portfolio_value = self.initial_balance
        
        # 초기 lookback 데이터 채우기
        for i in range(self.lookback_period):
            idx = self.start_idx - self.lookback_period + i
            self.price_history.append(self.df.iloc[idx]['upbit_btc_krw'])
            self.premium_history.append(self.df.iloc[idx]['premium_rate'])
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """현재 상태 벡터 생성"""
        idx = self.start_idx + self.current_step
        current = self.df.iloc[idx]
        
        # 1. 김치프리미엄 관련 (3)
        premium_current = current['premium_rate']
        premium_mean = np.mean(self.premium_history)
        premium_std = np.std(self.premium_history)
        
        # 2. 가격 관련 (4)
        price_return = (current['upbit_btc_krw'] / self.price_history[-2] - 1) if len(self.price_history) > 1 else 0
        price_volatility = np.std([self.price_history[i] / self.price_history[i-1] - 1 
                                   for i in range(1, len(self.price_history))]) if len(self.price_history) > 1 else 0
        upbit_price_norm = (current['upbit_btc_krw'] - np.mean(self.price_history)) / (np.std(self.price_history) + 1e-8)
        binance_price_norm = (current['binance_btc_usdt'] - 35000) / 1000  # 간단 정규화
        
        # 3. 기술 지표 (4)
        rsi_norm = (current['rsi'] - 50) / 50
        macd_norm = current['macd'] / 2  # 대략적 정규화
        bb_position = (premium_current - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'] + 1e-8)
        momentum = (premium_current - self.premium_history[-10]) / 10 if len(self.premium_history) >= 10 else 0
        
        # 4. 볼륨/유동성 (3)
        volume_ratio = current['volume_upbit'] / (current['volume_binance'] + 1e-8)
        volume_change = (current['volume_upbit'] / self.df.iloc[idx-1]['volume_upbit'] - 1) if idx > 0 else 0
        liquidity_score = np.log1p(current['volume_upbit'] * current['volume_binance']) / 20
        
        # 5. 포지션 상태 (3)
        has_position = float(self.position > 0)
        position_pnl = ((current['upbit_btc_krw'] / self.entry_price - 1) * 100) if self.position > 0 else 0
        position_duration = len([t for t in self.trades if t['exit_time'] is None]) if self.position > 0 else 0
        
        # 6. 시장 상태 (3)
        trend_strength = np.polyfit(range(len(self.premium_history)), list(self.premium_history), 1)[0] if len(self.premium_history) > 1 else 0
        market_hour = (idx % 1440) / 1440  # 시간대 (0-1)
        volatility_regime = premium_std / (premium_mean + 1e-8)
        
        # 상태 벡터 구성 (20차원)
        state = np.array([
            premium_current / 10,  # 정규화된 현재 프리미엄
            premium_mean / 10,
            premium_std / 5,
            price_return * 100,
            price_volatility * 100,
            upbit_price_norm,
            binance_price_norm,
            rsi_norm,
            macd_norm,
            bb_position,
            momentum,
            volume_ratio,
            volume_change,
            liquidity_score,
            has_position,
            position_pnl / 10,
            position_duration / 100,
            trend_strength * 10,
            market_hour,
            volatility_regime
        ], dtype=np.float32)
        
        # Clipping for stability
        return np.clip(state, -3.0, 3.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        idx = self.start_idx + self.current_step
        current = self.df.iloc[idx]
        
        # 이전 포트폴리오 가치
        prev_value = self._get_portfolio_value()
        
        # 액션 실행
        reward = 0
        if action == 1:  # Enter Position
            reward = self._enter_position(current)
        elif action == 2:  # Exit Position
            reward = self._exit_position(current)
        else:  # Hold
            reward = self._hold_position(current)
        
        # 스텝 업데이트
        self.current_step += 1
        self.price_history.append(current['upbit_btc_krw'])
        self.premium_history.append(current['premium_rate'])
        
        # 새로운 포트폴리오 가치
        new_value = self._get_portfolio_value()
        
        # 보상 계산 (Sharpe ratio 기반)
        returns = (new_value / prev_value - 1)
        self.episode_returns.append(returns)
        
        # Sharpe ratio 컴포넌트
        if len(self.episode_returns) > 30:
            recent_returns = self.episode_returns[-30:]
            sharpe_component = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
            reward += sharpe_component * 0.1
        
        # 에피소드 종료 체크
        done = self.current_step >= self.episode_length
        truncated = False
        
        # 정보 딕셔너리
        info = {
            'portfolio_value': new_value,
            'position': self.position,
            'premium_rate': current['premium_rate'],
            'total_trades': len(self.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        self.last_portfolio_value = new_value
        
        return self._get_observation(), reward, done, truncated, info
    
    def _enter_position(self, current: pd.Series) -> float:
        """포지션 진입"""
        if self.position > 0:
            # 이미 포지션이 있으면 패널티
            return -0.01
        
        # 포지션 크기 계산
        position_value = self.balance * self.max_position_size
        position_size = position_value / current['upbit_btc_krw']
        
        # 수수료 차감
        fee = position_value * self.trading_fee
        self.balance -= (position_value + fee)
        
        # 포지션 설정
        self.position = position_size
        self.entry_price = current['upbit_btc_krw']
        
        # 거래 기록
        self.trades.append({
            'type': 'enter',
            'price': current['upbit_btc_krw'],
            'size': position_size,
            'time': current['timestamp'],
            'premium': current['premium_rate'],
            'exit_time': None,
            'exit_price': None,
            'pnl': None
        })
        
        # 진입 보상 (프리미엄이 높을 때 진입하면 보상)
        premium_reward = current['premium_rate'] / 100 if current['premium_rate'] > 2 else -0.02
        
        return premium_reward
    
    def _exit_position(self, current: pd.Series) -> float:
        """포지션 청산"""
        if self.position <= 0:
            # 포지션이 없으면 패널티
            return -0.01
        
        # 수익 계산
        exit_value = self.position * current['upbit_btc_krw']
        fee = exit_value * self.trading_fee
        net_value = exit_value - fee
        
        # PnL 계산
        entry_value = self.position * self.entry_price
        pnl = (net_value - entry_value) / entry_value
        
        # 잔고 업데이트
        self.balance += net_value
        
        # 거래 기록 업데이트
        if self.trades and self.trades[-1]['exit_time'] is None:
            self.trades[-1]['exit_time'] = current['timestamp']
            self.trades[-1]['exit_price'] = current['upbit_btc_krw']
            self.trades[-1]['pnl'] = pnl
        
        # 포지션 초기화
        self.position = 0
        self.entry_price = 0
        
        # 청산 보상 (수익률 기반)
        return pnl * 10  # 수익률을 보상으로 스케일링
    
    def _hold_position(self, current: pd.Series) -> float:
        """포지션 유지"""
        if self.position > 0:
            # 포지션 보유 중 - 미실현 손익 기반 보상
            unrealized_pnl = (current['upbit_btc_krw'] / self.entry_price - 1)
            
            # 손실이 커지면 패널티
            if unrealized_pnl < -0.02:  # 2% 이상 손실
                return -0.05
            
            # 이익이 나면 작은 보상
            if unrealized_pnl > 0:
                return unrealized_pnl * 0.1
        
        return 0
    
    def _get_portfolio_value(self) -> float:
        """포트폴리오 총 가치 계산"""
        idx = self.start_idx + self.current_step
        if idx >= len(self.df):
            idx = len(self.df) - 1
        
        current_price = self.df.iloc[idx]['upbit_btc_krw']
        position_value = self.position * current_price if self.position > 0 else 0
        
        return self.balance + position_value
    
    def _calculate_sharpe_ratio(self) -> float:
        """Sharpe Ratio 계산"""
        if len(self.episode_returns) < 2:
            return 0
        
        returns = np.array(self.episode_returns)
        if np.std(returns) == 0:
            return 0
        
        # 연간화 (분 단위 데이터 기준)
        annual_return = np.mean(returns) * 525600  # 분당 수익률 * 1년 분
        annual_std = np.std(returns) * np.sqrt(525600)
        
        # Risk-free rate (연 2%)
        risk_free = 0.02 / 525600
        
        sharpe = (annual_return - risk_free) / (annual_std + 1e-8)
        
        return sharpe
    
    def render(self, mode='human'):
        """환경 렌더링 (선택사항)"""
        if mode == 'human':
            idx = self.start_idx + self.current_step
            current = self.df.iloc[idx]
            
            print(f"\n=== Step {self.current_step} ===")
            print(f"Premium Rate: {current['premium_rate']:.2f}%")
            print(f"Portfolio Value: {self._get_portfolio_value():,.0f} KRW")
            print(f"Position: {self.position:.4f} BTC" if self.position > 0 else "Position: None")
            print(f"Balance: {self.balance:,.0f} KRW")
            print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.3f}")
            print(f"Total Trades: {len(self.trades)}")
    
    def close(self):
        """환경 종료"""
        pass


if __name__ == "__main__":
    # 테스트
    env = KimchiPremiumTradingEnv()
    
    # 환경 리셋
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation sample: {obs[:5]}")
    
    # 랜덤 액션 테스트
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.4f}, Done: {done}")
        
        if done:
            break
    
    env.close()
    print("\nEnvironment test completed!")