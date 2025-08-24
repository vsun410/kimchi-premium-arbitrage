"""
Feature Engineering Pipeline
Task #14: 기술적 지표 및 시장 미시구조 특징 추출
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logger import logger

# PerformanceWarning 처리
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """특징 공학 파이프라인"""
    
    def __init__(
        self,
        scaler_type: str = "robust",
        feature_groups: Optional[List[str]] = None,
    ):
        """
        초기화
        
        Args:
            scaler_type: 스케일러 타입 ('standard', 'robust', 'none')
            feature_groups: 사용할 특징 그룹 리스트
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
        
        # 기본 특징 그룹
        self.feature_groups = feature_groups or [
            "price",
            "volume",
            "technical",
            "microstructure",
            "time",
        ]
        
        # 특징 이름 저장
        self.feature_names = []
        self.fitted = False
        
        logger.info(
            f"Feature engineer initialized: scaler={scaler_type}, "
            f"groups={self.feature_groups}"
        )
    
    def _create_scaler(self):
        """스케일러 생성"""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            return None
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        가격 관련 특징 생성
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            가격 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        # 기본 가격 특징
        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # 가격 비율
        features["high_low_ratio"] = df["high"] / df["low"]
        features["close_open_ratio"] = df["close"] / df["open"]
        
        # 이동 평균
        for period in [5, 10, 20, 50]:
            features[f"sma_{period}"] = df["close"].rolling(period).mean()
            features[f"price_to_sma_{period}"] = df["close"] / features[f"sma_{period}"]
        
        # 지수 이동 평균
        for period in [12, 26]:
            features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # 변동성 (returns를 features에서 사용)
        for period in [5, 10, 20]:
            features[f"volatility_{period}"] = features["returns"].rolling(period).std()
        
        # 가격 모멘텀
        for period in [1, 3, 5, 10]:
            features[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
            features[f"momentum_pct_{period}"] = df["close"].pct_change(period)
            features[f"roc_{period}"] = ((df["close"] - df["close"].shift(period)) / 
                                         df["close"].shift(period)) * 100
        
        # 추가 가격 특징
        # 가격 범위
        features["daily_range"] = df["high"] - df["low"]
        features["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        
        # 가격 위치
        features["price_position"] = ((df["close"] - df["low"]) / 
                                      (df["high"] - df["low"])).fillna(0.5)
        
        # 가격 채널
        features["highest_20"] = df["high"].rolling(20).max()
        features["lowest_20"] = df["low"].rolling(20).min()
        features["price_channel_position"] = ((df["close"] - features["lowest_20"]) / 
                                              (features["highest_20"] - features["lowest_20"])).fillna(0.5)
        
        # 가격 갭
        features["gap_open"] = df["open"] - df["close"].shift(1)
        features["gap_ratio"] = (features["gap_open"] / df["close"].shift(1)) * 100
        
        # 최대/최소 비율
        for period in [5, 10, 20]:
            features[f"high_low_ratio_{period}"] = (df["high"].rolling(period).max() / 
                                                     df["low"].rolling(period).min())
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 관련 특징 생성
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            거래량 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        # 기본 거래량 특징
        features["volume_ratio"] = df["volume"] / df["volume"].shift(1)
        features["volume_sma_5"] = df["volume"].rolling(5).mean()
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        
        # 거래량 가중 평균 가격 (VWAP)
        features["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        features["price_to_vwap"] = df["close"] / features["vwap"]
        
        # On-Balance Volume (OBV)
        obv = np.where(df["close"] > df["close"].shift(1), df["volume"], 
                      np.where(df["close"] < df["close"].shift(1), -df["volume"], 0))
        features["obv"] = np.cumsum(obv)
        features["obv_sma_5"] = features["obv"].rolling(5).mean()
        
        # Volume Rate of Change
        features["vroc_5"] = ((df["volume"] - df["volume"].shift(5)) / 
                             df["volume"].shift(5)) * 100
        
        # Money Flow
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        features["money_flow_5"] = money_flow.rolling(5).sum()
        
        return features
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 생성
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            기술적 지표 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI (Relative Strength Index)
        features["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        features["rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        
        # MACD
        macd = ta.trend.MACD(df["close"])
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_diff"] = macd.macd_diff()
        
        # Bollinger Bands
        bb_20 = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        features["bb_upper"] = bb_20.bollinger_hband()
        features["bb_middle"] = bb_20.bollinger_mavg()
        features["bb_lower"] = bb_20.bollinger_lband()
        features["bb_width"] = bb_20.bollinger_wband()
        features["bb_pband"] = bb_20.bollinger_pband()
        
        # ATR (Average True Range)
        features["atr_14"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        features["stoch_k"] = stoch.stoch()
        features["stoch_d"] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        features["adx"] = adx.adx()
        features["adx_pos"] = adx.adx_pos()
        features["adx_neg"] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        features["cci_20"] = ta.trend.CCIIndicator(
            df["high"], df["low"], df["close"], window=20
        ).cci()
        
        # Williams %R
        features["williams_r"] = ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"]
        ).williams_r()
        
        return features
    
    def create_microstructure_features(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        시장 미시구조 특징 생성
        
        Args:
            df: OHLCV 데이터프레임
            orderbook_data: 오더북 데이터 (선택)
            
        Returns:
            미시구조 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        # 스프레드 관련
        if "bid" in df.columns and "ask" in df.columns:
            features["bid_ask_spread"] = df["ask"] - df["bid"]
            features["bid_ask_spread_pct"] = (
                (df["ask"] - df["bid"]) / df["bid"] * 100
            )
            features["mid_price"] = (df["bid"] + df["ask"]) / 2
            features["price_to_mid"] = df["close"] / features["mid_price"]
        
        # 가격 효율성
        features["price_efficiency"] = abs(df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-10
        )
        
        # 롤링 상관관계 (자기상관)
        for lag in [1, 5, 10]:
            features[f"autocorr_{lag}"] = df["returns"].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Order Flow Imbalance (오더북 데이터가 있는 경우)
        if orderbook_data is not None and not orderbook_data.empty:
            if "bid_volume" in orderbook_data.columns and "ask_volume" in orderbook_data.columns:
                total_volume = orderbook_data["bid_volume"] + orderbook_data["ask_volume"]
                features["order_imbalance"] = (
                    (orderbook_data["bid_volume"] - orderbook_data["ask_volume"]) / 
                    (total_volume + 1e-10)
                )
                
                # 유동성 점수
                if "liquidity_score" in orderbook_data.columns:
                    features["liquidity_score"] = orderbook_data["liquidity_score"]
        
        # Kyle's Lambda (가격 영향력)
        if "volume" in df.columns:
            price_change = df["close"].diff()
            features["kyles_lambda"] = (
                price_change.rolling(20).std() / 
                (df["volume"].rolling(20).sum() + 1e-10)
            )
        
        # Amihud Illiquidity
        features["amihud_illiquidity"] = (
            abs(df["returns"]) / (df["volume"] * df["close"] + 1e-10)
        ).rolling(20).mean()
        
        return features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 관련 특징 생성
        
        Args:
            df: 시간 인덱스를 가진 데이터프레임
            
        Returns:
            시간 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        # 시간 특징 (datetime index 필요)
        if isinstance(df.index, pd.DatetimeIndex):
            # 시간대별 특징
            features["hour"] = df.index.hour
            features["day_of_week"] = df.index.dayofweek
            features["day_of_month"] = df.index.day
            features["month"] = df.index.month
            
            # 주기적 특징 (sin/cos 변환)
            features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            features["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # 거래 세션
            features["is_asian_session"] = (
                (df.index.hour >= 0) & (df.index.hour < 8)
            ).astype(int)
            features["is_european_session"] = (
                (df.index.hour >= 8) & (df.index.hour < 16)
            ).astype(int)
            features["is_american_session"] = (
                (df.index.hour >= 16) & (df.index.hour < 24)
            ).astype(int)
            
            # 주말 여부
            features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        
        return features
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        지연(Lag) 특징 생성
        
        Args:
            df: 데이터프레임
            columns: lag를 생성할 컬럼들
            lags: lag 기간들
            
        Returns:
            lag 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    features[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        return features
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        롤링 통계 특징 생성
        
        Args:
            df: 데이터프레임
            columns: 롤링 통계를 생성할 컬럼들
            windows: 윈도우 크기들
            
        Returns:
            롤링 특징 데이터프레임
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # 평균
                    features[f"{col}_mean_{window}"] = df[col].rolling(window).mean()
                    # 표준편차
                    features[f"{col}_std_{window}"] = df[col].rolling(window).std()
                    # 최대값
                    features[f"{col}_max_{window}"] = df[col].rolling(window).max()
                    # 최소값
                    features[f"{col}_min_{window}"] = df[col].rolling(window).min()
                    # 범위
                    features[f"{col}_range_{window}"] = (
                        features[f"{col}_max_{window}"] - features[f"{col}_min_{window}"]
                    )
                    # 왜도
                    features[f"{col}_skew_{window}"] = df[col].rolling(window).skew()
                    # 첨도
                    features[f"{col}_kurt_{window}"] = df[col].rolling(window).kurt()
        
        return features
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        include_lags: bool = True,
        include_rolling: bool = True
    ) -> pd.DataFrame:
        """
        전체 특징 공학 파이프라인
        
        Args:
            df: OHLCV 데이터프레임
            orderbook_data: 오더북 데이터 (선택)
            include_lags: lag 특징 포함 여부
            include_rolling: 롤링 특징 포함 여부
            
        Returns:
            모든 특징이 포함된 데이터프레임
        """
        all_features = []
        
        # 기본 returns 계산 (다른 특징에서 사용)
        df["returns"] = df["close"].pct_change()
        
        # 각 특징 그룹 생성
        if "price" in self.feature_groups:
            price_features = self.create_price_features(df)
            all_features.append(price_features)
        
        if "volume" in self.feature_groups:
            volume_features = self.create_volume_features(df)
            all_features.append(volume_features)
        
        if "technical" in self.feature_groups:
            technical_features = self.create_technical_indicators(df)
            all_features.append(technical_features)
        
        if "microstructure" in self.feature_groups:
            micro_features = self.create_microstructure_features(df, orderbook_data)
            all_features.append(micro_features)
        
        if "time" in self.feature_groups:
            time_features = self.create_time_features(df)
            all_features.append(time_features)
        
        # 모든 특징 결합
        features_df = pd.concat(all_features, axis=1)
        
        # Lag 특징 추가
        if include_lags:
            important_cols = ["returns", "volume_ratio", "rsi_14", "macd"]
            existing_cols = [col for col in important_cols if col in features_df.columns]
            if existing_cols:
                lag_features = self.create_lag_features(features_df, existing_cols)
                features_df = pd.concat([features_df, lag_features], axis=1)
        
        # 롤링 특징 추가
        if include_rolling:
            rolling_cols = ["returns", "volume_ratio"]
            existing_cols = [col for col in rolling_cols if col in features_df.columns]
            if existing_cols:
                rolling_features = self.create_rolling_features(features_df, existing_cols)
                features_df = pd.concat([features_df, rolling_features], axis=1)
        
        # 특징 이름 저장
        self.feature_names = features_df.columns.tolist()
        
        # 무한값 및 NaN 처리
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(
            f"Feature engineering complete: {len(features_df.columns)} features, "
            f"{len(features_df)} samples"
        )
        
        return features_df
    
    def fit_transform(
        self,
        features: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        특징 스케일링 (fit & transform)
        
        Args:
            features: 특징 데이터프레임
            exclude_cols: 스케일링에서 제외할 컬럼
            
        Returns:
            스케일된 특징 데이터프레임
        """
        if self.scaler is None:
            return features
        
        exclude_cols = exclude_cols or []
        scale_cols = [col for col in features.columns if col not in exclude_cols]
        
        if not scale_cols:
            return features
        
        # NaN이 있는 행 제거 (스케일러 학습용)
        train_data = features[scale_cols].dropna()
        
        if len(train_data) == 0:
            logger.warning("No valid data for scaling")
            return features
        
        # 스케일러 학습
        self.scaler.fit(train_data)
        self.fitted = True
        
        # 변환
        scaled_data = features.copy()
        mask = ~features[scale_cols].isna().any(axis=1)
        scaled_data.loc[mask, scale_cols] = self.scaler.transform(
            features.loc[mask, scale_cols]
        )
        
        logger.info(f"Features scaled using {self.scaler_type} scaler")
        
        return scaled_data
    
    def transform(
        self,
        features: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        특징 스케일링 (transform only)
        
        Args:
            features: 특징 데이터프레임
            exclude_cols: 스케일링에서 제외할 컬럼
            
        Returns:
            스케일된 특징 데이터프레임
        """
        if self.scaler is None or not self.fitted:
            logger.warning("Scaler not fitted, returning original features")
            return features
        
        exclude_cols = exclude_cols or []
        scale_cols = [col for col in features.columns if col not in exclude_cols]
        
        if not scale_cols:
            return features
        
        # 변환
        scaled_data = features.copy()
        mask = ~features[scale_cols].isna().any(axis=1)
        scaled_data.loc[mask, scale_cols] = self.scaler.transform(
            features.loc[mask, scale_cols]
        )
        
        return scaled_data
    
    def get_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "correlation"
    ) -> pd.Series:
        """
        특징 중요도 계산
        
        Args:
            features: 특징 데이터프레임
            target: 타겟 변수
            method: 중요도 계산 방법 ('correlation', 'mutual_info')
            
        Returns:
            특징 중요도 시리즈
        """
        if method == "correlation":
            # 상관계수 기반
            importance = features.corrwith(target).abs()
        
        elif method == "mutual_info":
            # 상호 정보량 기반
            from sklearn.feature_selection import mutual_info_regression
            
            # NaN 제거
            mask = ~(features.isna().any(axis=1) | target.isna())
            clean_features = features[mask]
            clean_target = target[mask]
            
            if len(clean_features) > 0:
                mi_scores = mutual_info_regression(clean_features, clean_target)
                importance = pd.Series(mi_scores, index=features.columns)
            else:
                importance = pd.Series(0, index=features.columns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 정렬
        importance = importance.sort_values(ascending=False)
        
        return importance
    
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        n_features: int = 50,
        method: str = "correlation"
    ) -> List[str]:
        """
        상위 N개 특징 선택
        
        Args:
            features: 특징 데이터프레임
            target: 타겟 변수
            n_features: 선택할 특징 수
            method: 선택 방법
            
        Returns:
            선택된 특징 이름 리스트
        """
        importance = self.get_feature_importance(features, target, method)
        selected = importance.head(n_features).index.tolist()
        
        logger.info(f"Selected top {n_features} features using {method}")
        
        return selected


# 싱글톤 인스턴스
feature_engineer = FeatureEngineer()


if __name__ == "__main__":
    # 테스트
    import numpy as np
    
    # 테스트 데이터 생성
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    test_data = pd.DataFrame({
        "open": 100 + np.random.randn(1000).cumsum(),
        "high": 102 + np.random.randn(1000).cumsum(),
        "low": 98 + np.random.randn(1000).cumsum(),
        "close": 100 + np.random.randn(1000).cumsum(),
        "volume": np.random.uniform(1000, 10000, 1000),
    }, index=dates)
    
    # 특징 생성
    engineer = FeatureEngineer()
    features = engineer.engineer_features(test_data)
    
    print(f"Generated {len(features.columns)} features")
    print(f"Feature groups: {engineer.feature_groups}")
    print(f"\nSample features:")
    print(features.head())
    
    # 스케일링
    scaled_features = engineer.fit_transform(features)
    print(f"\nScaled features shape: {scaled_features.shape}")
    
    # 특징 중요도
    target = pd.Series(np.random.randn(len(features)), index=features.index)
    importance = engineer.get_feature_importance(features, target)
    print(f"\nTop 10 important features:")
    print(importance.head(10))