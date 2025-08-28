"""
Data Loader for Backtesting
히스토리컬 데이터 로딩 및 전처리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataLoader:
    """
    백테스팅용 데이터 로더
    - CSV 파일에서 히스토리컬 데이터 로딩
    - 시계열 정렬 및 동기화
    - 리샘플링 지원
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        """
        Args:
            data_dir: 히스토리컬 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.upbit_data: Optional[pd.DataFrame] = None
        self.binance_data: Optional[pd.DataFrame] = None
        self.premium_data: Optional[pd.DataFrame] = None
        self.exchange_rate: float = 1350.0  # 기본 환율
        
    def load_all_data(self, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        모든 데이터 로드
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            데이터 딕셔너리
        """
        # Upbit 데이터 로드
        self.upbit_data = self._load_upbit_data()
        
        # Binance 데이터 로드
        self.binance_data = self._load_binance_data()
        
        # 날짜 필터링
        if start_date or end_date:
            self._filter_by_date(start_date, end_date)
        
        # 김치 프리미엄 계산 (필터링 후)
        self.premium_data = self._calculate_premium()
        
        # 데이터 동기화 (이미 align된 데이터이므로 필요없을 수 있음)
        # self._synchronize_data()
        
        return {
            'upbit': self.upbit_data,
            'binance': self.binance_data,
            'premium': self.premium_data
        }
    
    def _load_upbit_data(self) -> pd.DataFrame:
        """업비트 데이터 로드"""
        # Try sample data first
        sample_file = self.data_dir / "upbit_btc_krw_sample.csv"
        if sample_file.exists():
            df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} Upbit sample records")
            return df
            
        files = list(self.data_dir.glob("**/upbit_BTC_KRW*.csv"))
        
        if not files:
            raise FileNotFoundError("No Upbit data files found")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        # 모든 파일 합치기
        combined = pd.concat(dfs, ignore_index=True)
        
        # timestamp를 인덱스로 설정
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined.set_index('timestamp', inplace=True)
        combined.sort_index(inplace=True)
        
        # 중복 제거
        combined = combined[~combined.index.duplicated(keep='last')]
        
        logger.info(f"Loaded {len(combined)} Upbit records")
        return combined
    
    def _load_binance_data(self) -> pd.DataFrame:
        """바이낸스 데이터 로드"""
        # Try sample data first
        sample_file = self.data_dir / "binance_btc_usdt_sample.csv"
        if sample_file.exists():
            df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} Binance sample records")
            return df
            
        files = list(self.data_dir.glob("**/binance_BTC_USDT*.csv"))
        
        if not files:
            raise FileNotFoundError("No Binance data files found")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        # 모든 파일 합치기
        combined = pd.concat(dfs, ignore_index=True)
        
        # timestamp를 인덱스로 설정
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined.set_index('timestamp', inplace=True)
        combined.sort_index(inplace=True)
        
        # 중복 제거
        combined = combined[~combined.index.duplicated(keep='last')]
        
        logger.info(f"Loaded {len(combined)} Binance records")
        return combined
    
    def _load_premium_data(self) -> pd.DataFrame:
        """김치 프리미엄 데이터 로드"""
        premium_file = self.data_dir / "training" / "kimchi_premium.csv"
        
        if premium_file.exists():
            df = pd.read_csv(premium_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"Loaded {len(df)} premium records")
            return df
        else:
            # 프리미엄 데이터가 없으면 계산
            return self._calculate_premium()
    
    def _calculate_premium(self) -> pd.DataFrame:
        """김치 프리미엄 계산"""
        if self.upbit_data is None or self.binance_data is None:
            raise ValueError("Need both Upbit and Binance data to calculate premium")
        
        # 데이터 정렬
        aligned_upbit, aligned_binance = self.upbit_data.align(
            self.binance_data, 
            join='inner'
        )
        
        # 프리미엄 계산
        premium_df = pd.DataFrame(index=aligned_upbit.index)
        premium_df['upbit_krw'] = aligned_upbit['close']
        premium_df['binance_usd'] = aligned_binance['close']
        premium_df['binance_krw'] = premium_df['binance_usd'] * self.exchange_rate
        premium_df['kimchi_premium'] = (
            (premium_df['upbit_krw'] - premium_df['binance_krw']) 
            / premium_df['binance_krw'] * 100
        )
        
        logger.info(f"Calculated premium for {len(premium_df)} records")
        return premium_df
    
    def _filter_by_date(self, start_date: Optional[str], end_date: Optional[str]):
        """날짜 필터링"""
        if start_date:
            start = pd.to_datetime(start_date)
            if self.upbit_data is not None:
                self.upbit_data = self.upbit_data[self.upbit_data.index >= start]
            if self.binance_data is not None:
                self.binance_data = self.binance_data[self.binance_data.index >= start]
            if self.premium_data is not None:
                self.premium_data = self.premium_data[self.premium_data.index >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            if self.upbit_data is not None:
                self.upbit_data = self.upbit_data[self.upbit_data.index <= end]
            if self.binance_data is not None:
                self.binance_data = self.binance_data[self.binance_data.index <= end]
            if self.premium_data is not None:
                self.premium_data = self.premium_data[self.premium_data.index <= end]
    
    def _synchronize_data(self):
        """데이터 동기화 (같은 타임스탬프만 남기기)"""
        # 모든 데이터가 있는 경우에만 동기화
        if all([self.upbit_data is not None, 
                self.binance_data is not None]):
            
            # 공통 인덱스 찾기
            common_index = self.upbit_data.index.intersection(self.binance_data.index)
            
            if self.premium_data is not None:
                common_index = common_index.intersection(self.premium_data.index)
            
            # 공통 인덱스로 필터링
            self.upbit_data = self.upbit_data.loc[common_index]
            self.binance_data = self.binance_data.loc[common_index]
            
            if self.premium_data is not None:
                self.premium_data = self.premium_data.loc[common_index]
            
            logger.info(f"Synchronized data to {len(common_index)} common timestamps")
    
    def resample(self, freq: str = '5T') -> Dict[str, pd.DataFrame]:
        """
        데이터 리샘플링
        
        Args:
            freq: 리샘플링 주기 ('1T', '5T', '15T', '1H' 등)
            
        Returns:
            리샘플링된 데이터
        """
        resampled = {}
        
        if self.upbit_data is not None:
            resampled['upbit'] = self.upbit_data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        if self.binance_data is not None:
            resampled['binance'] = self.binance_data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        if self.premium_data is not None:
            if 'kimchi_premium' in self.premium_data.columns:
                resampled['premium'] = self.premium_data.resample(freq).agg({
                    'kimchi_premium': 'mean'
                }).dropna()
            else:
                resampled['premium'] = self.premium_data.resample(freq).agg({
                    'upbit_krw': 'last',
                    'binance_usd': 'last',
                    'binance_krw': 'last'
                }).dropna()
        
        return resampled
    
    def get_data_info(self) -> Dict:
        """데이터 정보 반환"""
        info = {}
        
        if self.upbit_data is not None:
            info['upbit'] = {
                'records': len(self.upbit_data),
                'start': str(self.upbit_data.index.min()),
                'end': str(self.upbit_data.index.max()),
                'columns': list(self.upbit_data.columns)
            }
        
        if self.binance_data is not None:
            info['binance'] = {
                'records': len(self.binance_data),
                'start': str(self.binance_data.index.min()),
                'end': str(self.binance_data.index.max()),
                'columns': list(self.binance_data.columns)
            }
        
        if self.premium_data is not None:
            info['premium'] = {
                'records': len(self.premium_data),
                'start': str(self.premium_data.index.min()),
                'end': str(self.premium_data.index.max()),
                'mean_premium': self.premium_data.get('kimchi_premium', pd.Series()).mean(),
                'max_premium': self.premium_data.get('kimchi_premium', pd.Series()).max(),
                'min_premium': self.premium_data.get('kimchi_premium', pd.Series()).min()
            }
        
        return info