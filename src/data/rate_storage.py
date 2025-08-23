"""
환율 데이터 저장 및 관리
CSV 파일로 환율 히스토리 저장
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from threading import Lock

from src.utils.logger import logger
from src.models.schemas import ExchangeRate


class RateStorage:
    """환율 데이터 저장소"""
    
    def __init__(self, data_dir: str = "data/exchange_rates"):
        """
        초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로
        self.current_file = self.data_dir / "current_rate.json"
        self.history_dir = self.data_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
        # 쓰기 잠금
        self.write_lock = Lock()
        
        logger.info(f"Rate storage initialized at {self.data_dir}")
    
    def save_current_rate(self, rate: float, source: str):
        """
        현재 환율 저장
        
        Args:
            rate: USD/KRW 환율
            source: 데이터 소스
        """
        with self.write_lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'rate': rate,
                'source': source
            }
            
            # JSON 파일로 저장
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # CSV 히스토리에도 추가
            self._append_to_history(data)
    
    def _append_to_history(self, data: Dict[str, Any]):
        """히스토리 CSV에 추가"""
        today = datetime.now().strftime('%Y%m%d')
        csv_file = self.history_dir / f"rates_{today}.csv"
        
        # 헤더 확인
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'rate', 'source'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)
    
    def load_current_rate(self) -> Optional[Dict[str, Any]]:
        """현재 환율 로드"""
        if not self.current_file.exists():
            return None
        
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load current rate: {e}")
            return None
    
    def load_history(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        히스토리 데이터 로드
        
        Args:
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            환율 데이터프레임
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        all_data = []
        
        # 날짜 범위의 파일들 읽기
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y%m%d')
            csv_file = self.history_dir / f"rates_{date_str}.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
                except Exception as e:
                    logger.error(f"Failed to load {csv_file}: {e}")
            
            current += timedelta(days=1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')
        
        return pd.DataFrame(columns=['timestamp', 'rate', 'source'])
    
    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        환율 통계 계산
        
        Args:
            days: 분석 기간 (일)
            
        Returns:
            통계 정보
        """
        start_date = datetime.now() - timedelta(days=days)
        df = self.load_history(start_date=start_date)
        
        if df.empty:
            return {
                'period_days': days,
                'data_points': 0,
                'average': None,
                'min': None,
                'max': None,
                'std': None,
                'change': None
            }
        
        stats = {
            'period_days': days,
            'data_points': len(df),
            'average': df['rate'].mean(),
            'min': df['rate'].min(),
            'max': df['rate'].max(),
            'std': df['rate'].std(),
            'change': None
        }
        
        # 변화율 계산
        if len(df) > 1:
            first_rate = df.iloc[0]['rate']
            last_rate = df.iloc[-1]['rate']
            stats['change'] = ((last_rate - first_rate) / first_rate) * 100
        
        return stats
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """
        오래된 파일 정리
        
        Args:
            days_to_keep: 보관 기간 (일)
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        
        for csv_file in self.history_dir.glob("rates_*.csv"):
            try:
                # 파일명에서 날짜 추출
                date_str = csv_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff:
                    csv_file.unlink()
                    logger.info(f"Deleted old rate file: {csv_file.name}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up {csv_file}: {e}")
    
    def export_to_dataframe(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           resample: Optional[str] = None) -> pd.DataFrame:
        """
        데이터프레임으로 내보내기
        
        Args:
            start_date: 시작일
            end_date: 종료일
            resample: 리샘플링 주기 ('1H', '1D' 등)
            
        Returns:
            환율 데이터프레임
        """
        df = self.load_history(start_date, end_date)
        
        if df.empty:
            return df
        
        # 인덱스를 timestamp로 설정
        df.set_index('timestamp', inplace=True)
        
        # 리샘플링
        if resample:
            df = df.resample(resample).agg({
                'rate': 'mean',
                'source': 'first'
            })
        
        return df
    
    def import_from_csv(self, csv_path: str):
        """
        외부 CSV 파일 임포트
        
        Args:
            csv_path: CSV 파일 경로
        """
        try:
            df = pd.read_csv(csv_path)
            
            # 필수 컬럼 확인
            required_cols = ['timestamp', 'rate']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must have columns: {required_cols}")
            
            # source 컬럼이 없으면 추가
            if 'source' not in df.columns:
                df['source'] = 'imported'
            
            # 각 행을 히스토리에 추가
            for _, row in df.iterrows():
                self._append_to_history(row.to_dict())
            
            logger.info(f"Imported {len(df)} rate records from {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to import CSV: {e}")
            raise


class RateCache:
    """환율 캐시"""
    
    def __init__(self, ttl: int = 60):
        """
        초기화
        
        Args:
            ttl: 캐시 유효 시간 (초)
        """
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[float]:
        """캐시에서 값 조회"""
        if key not in self.cache:
            return None
        
        # TTL 체크
        if datetime.now().timestamp() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: float):
        """캐시에 값 저장"""
        self.cache[key] = value
        self.timestamps[key] = datetime.now().timestamp()
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.timestamps.clear()


# 전역 저장소
rate_storage = RateStorage()
rate_cache = RateCache()


if __name__ == "__main__":
    # 저장소 테스트
    print("Rate Storage Test")
    print("-" * 40)
    
    # 현재 환율 저장
    rate_storage.save_current_rate(1385.50, "test_source")
    print("Saved current rate: 1385.50")
    
    # 현재 환율 로드
    current = rate_storage.load_current_rate()
    if current:
        print(f"Loaded current rate: {current['rate']}")
    
    # 통계 조회
    stats = rate_storage.get_statistics(days=1)
    print(f"\nStatistics (1 day):")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 히스토리 로드
    df = rate_storage.load_history()
    print(f"\nHistory records: {len(df)}")
    
    if not df.empty:
        print("\nLast 5 records:")
        print(df.tail())
    
    print("\nTest completed!")