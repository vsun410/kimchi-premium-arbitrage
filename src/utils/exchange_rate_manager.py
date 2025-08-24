"""
Exchange Rate Manager
환율 데이터를 중앙에서 관리하여 하드코딩 방지
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import glob
import json

from src.utils.logger import logger


class ExchangeRateManager:
    """
    환율 관리자
    - 실제 환율 데이터 로드
    - 시간대별 환율 조회
    - 환율 보간(interpolation)
    """
    
    # 절대 하드코딩하지 않음!
    DEFAULT_RATE = None  # 의도적으로 None으로 설정
    
    def __init__(self, data_dir: str = None):
        """
        초기화
        
        Args:
            data_dir: 환율 데이터 디렉토리
        """
        if data_dir is None:
            # 프로젝트 루트에서 상대 경로로 설정
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(project_root, "data", "exchange_rates")
        
        self.data_dir = data_dir
        self.rates_cache = {}
        self.current_rate = None
        self.rates_df = None
        
        # 초기화 시 데이터 로드
        self.load_rates()
        
    def load_rates(self) -> pd.DataFrame:
        """
        환율 데이터 로드
        
        Returns:
            환율 DataFrame
        """
        history_dir = os.path.join(self.data_dir, "history")
        
        if not os.path.exists(history_dir):
            raise FileNotFoundError(f"Exchange rate directory not found: {history_dir}")
        
        # 모든 환율 파일 찾기
        rate_files = glob.glob(os.path.join(history_dir, "rates_*.csv"))
        
        if not rate_files:
            raise FileNotFoundError(f"No exchange rate files found in {history_dir}")
        
        # 모든 파일 로드 및 병합
        all_rates = []
        
        for file in rate_files:
            try:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns and 'rate' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    all_rates.append(df[['timestamp', 'rate']])
                    logger.info(f"Loaded {len(df)} rates from {os.path.basename(file)}")
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
        
        if not all_rates:
            raise ValueError("No valid exchange rate data found")
        
        # 모든 데이터 병합
        self.rates_df = pd.concat(all_rates, ignore_index=True)
        self.rates_df = self.rates_df.sort_values('timestamp')
        self.rates_df = self.rates_df.drop_duplicates(subset=['timestamp'])
        self.rates_df.set_index('timestamp', inplace=True)
        
        # 현재 환율 설정 (가장 최근 값)
        self.current_rate = self.rates_df['rate'].iloc[-1]
        
        logger.info(f"Loaded total {len(self.rates_df)} exchange rates")
        logger.info(f"Period: {self.rates_df.index[0]} to {self.rates_df.index[-1]}")
        logger.info(f"Current rate: {self.current_rate:.2f} KRW/USD")
        
        # 통계
        stats = self.rates_df['rate'].describe()
        logger.info(f"Rate statistics - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
                   f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        
        return self.rates_df
    
    def get_rate_at_time(self, timestamp: datetime) -> float:
        """
        특정 시간의 환율 조회
        
        Args:
            timestamp: 조회할 시간
            
        Returns:
            환율 (KRW/USD)
        """
        if self.rates_df is None or len(self.rates_df) == 0:
            raise ValueError("No exchange rate data available")
        
        # 캐시 확인
        cache_key = timestamp.strftime('%Y%m%d%H')
        if cache_key in self.rates_cache:
            return self.rates_cache[cache_key]
        
        # 정확한 매칭 시도
        if timestamp in self.rates_df.index:
            rate = self.rates_df.loc[timestamp, 'rate']
            self.rates_cache[cache_key] = rate
            return rate
        
        # 가장 가까운 시간 찾기 (전후 1시간 이내)
        time_diff = abs(self.rates_df.index - timestamp)
        min_diff_idx = time_diff.argmin()
        min_diff = time_diff[min_diff_idx].total_seconds()
        
        # 1시간 이상 차이나면 보간
        if min_diff > 3600:  # 1시간
            # 선형 보간
            try:
                # 전후 데이터 포인트 찾기
                before = self.rates_df[self.rates_df.index < timestamp]
                after = self.rates_df[self.rates_df.index > timestamp]
                
                if len(before) > 0 and len(after) > 0:
                    # 선형 보간
                    t1 = before.index[-1]
                    t2 = after.index[0]
                    r1 = before['rate'].iloc[-1]
                    r2 = after['rate'].iloc[0]
                    
                    # 보간 계산
                    weight = (timestamp - t1).total_seconds() / (t2 - t1).total_seconds()
                    rate = r1 + (r2 - r1) * weight
                    
                    logger.debug(f"Interpolated rate for {timestamp}: {rate:.2f}")
                else:
                    # 가장 가까운 값 사용
                    rate = self.rates_df['rate'].iloc[min_diff_idx]
                    logger.debug(f"Using nearest rate for {timestamp}: {rate:.2f}")
            except:
                rate = self.rates_df['rate'].iloc[min_diff_idx]
        else:
            rate = self.rates_df['rate'].iloc[min_diff_idx]
        
        self.rates_cache[cache_key] = rate
        return rate
    
    def get_rates_for_period(self, start: datetime, end: datetime) -> pd.Series:
        """
        기간 동안의 환율 조회
        
        Args:
            start: 시작 시간
            end: 종료 시간
            
        Returns:
            환율 Series
        """
        mask = (self.rates_df.index >= start) & (self.rates_df.index <= end)
        return self.rates_df.loc[mask, 'rate']
    
    def calculate_kimchi_premium(
        self, 
        upbit_price: float, 
        binance_price: float,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        정확한 김치 프리미엄 계산
        
        Args:
            upbit_price: 업비트 가격 (KRW)
            binance_price: 바이낸스 가격 (USD)
            timestamp: 시간 (None이면 현재 환율)
            
        Returns:
            김치 프리미엄 (%)
        """
        if timestamp:
            rate = self.get_rate_at_time(timestamp)
        else:
            rate = self.current_rate
            
        if rate is None:
            raise ValueError("Exchange rate not available")
        
        binance_krw = binance_price * rate
        premium = ((upbit_price - binance_krw) / binance_krw) * 100
        
        return premium
    
    def get_statistics(self) -> Dict:
        """
        환율 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        if self.rates_df is None or len(self.rates_df) == 0:
            return {}
        
        stats = self.rates_df['rate'].describe()
        
        return {
            'current': self.current_rate,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'median': stats['50%'],
            'data_points': len(self.rates_df),
            'start_date': self.rates_df.index[0].isoformat(),
            'end_date': self.rates_df.index[-1].isoformat()
        }
    
    def save_config(self, filepath: str = "config/exchange_rate_config.json"):
        """
        환율 설정 저장 (다른 스크립트에서 참조용)
        
        Args:
            filepath: 저장 경로
        """
        config = {
            'current_rate': self.current_rate,
            'statistics': self.get_statistics(),
            'last_updated': datetime.now().isoformat(),
            'data_source': self.data_dir,
            'warning': "DO NOT HARDCODE EXCHANGE RATES! Always use ExchangeRateManager"
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Exchange rate config saved to {filepath}")
    
    @classmethod
    def get_current_rate(cls) -> float:
        """
        현재 환율 빠르게 조회 (클래스 메서드)
        
        Returns:
            현재 환율
        """
        manager = cls()
        return manager.current_rate
    
    def __repr__(self) -> str:
        return f"ExchangeRateManager(current_rate={self.current_rate:.2f}, data_points={len(self.rates_df) if self.rates_df is not None else 0})"


# 싱글톤 인스턴스 (전역 사용)
_rate_manager_instance = None


def get_exchange_rate_manager() -> ExchangeRateManager:
    """
    싱글톤 환율 관리자 반환
    
    Returns:
        ExchangeRateManager 인스턴스
    """
    global _rate_manager_instance
    if _rate_manager_instance is None:
        _rate_manager_instance = ExchangeRateManager()
    return _rate_manager_instance


def get_current_exchange_rate() -> float:
    """
    현재 환율 조회 (간편 함수)
    
    Returns:
        현재 환율 (KRW/USD)
    """
    manager = get_exchange_rate_manager()
    return manager.current_rate


def calculate_kimchi_premium(upbit_price: float, binance_price: float, timestamp: Optional[datetime] = None) -> float:
    """
    김치 프리미엄 계산 (간편 함수)
    
    Args:
        upbit_price: 업비트 가격 (KRW)
        binance_price: 바이낸스 가격 (USD)
        timestamp: 시간 (옵션)
        
    Returns:
        김치 프리미엄 (%)
    """
    manager = get_exchange_rate_manager()
    return manager.calculate_kimchi_premium(upbit_price, binance_price, timestamp)