"""
김치 프리미엄 데이터 저장 및 관리
CSV 파일로 김프 히스토리 저장 및 분석
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from threading import Lock

from src.utils.logger import logger
from src.analysis.kimchi_premium import KimchiPremiumData, PremiumSignal


class PremiumStorage:
    """김프 데이터 저장소"""
    
    def __init__(self, data_dir: str = "data/kimchi_premium"):
        """
        초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로
        self.current_file = self.data_dir / "current_premium.json"
        self.history_dir = self.data_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
        # 쓰기 잠금
        self.write_lock = Lock()
        
        logger.info(f"Premium storage initialized at {self.data_dir}")
    
    def save_premium(self, data: KimchiPremiumData):
        """
        김프 데이터 저장
        
        Args:
            data: 김프 데이터
        """
        with self.write_lock:
            # JSON 형식으로 변환
            json_data = {
                'timestamp': data.timestamp.isoformat(),
                'upbit_price': data.upbit_price,
                'binance_price': data.binance_price,
                'exchange_rate': data.exchange_rate,
                'premium_rate': data.premium_rate,
                'premium_krw': data.premium_krw,
                'signal': data.signal.value,
                'liquidity_score': data.liquidity_score,
                'spread_upbit': data.spread_upbit,
                'spread_binance': data.spread_binance,
                'confidence': data.confidence
            }
            
            # 현재 김프 저장
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            # CSV 히스토리에 추가
            self._append_to_history(json_data)
    
    def _append_to_history(self, data: Dict[str, Any]):
        """히스토리 CSV에 추가"""
        today = datetime.now().strftime('%Y%m%d')
        csv_file = self.history_dir / f"premium_{today}.csv"
        
        # 헤더 확인
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'upbit_price', 'binance_price', 'exchange_rate',
                'premium_rate', 'premium_krw', 'signal', 'liquidity_score',
                'spread_upbit', 'spread_binance', 'confidence'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)
    
    def load_current_premium(self) -> Optional[Dict[str, Any]]:
        """현재 김프 로드"""
        if not self.current_file.exists():
            return None
        
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load current premium: {e}")
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
            김프 데이터프레임
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        all_data = []
        
        # 날짜 범위의 파일들 읽기
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y%m%d')
            csv_file = self.history_dir / f"premium_{date_str}.csv"
            
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
        
        return pd.DataFrame()
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        김프 통계 계산
        
        Args:
            hours: 분석 기간 (시간)
            
        Returns:
            통계 정보
        """
        start_date = datetime.now() - timedelta(hours=hours)
        df = self.load_history(start_date=start_date)
        
        if df.empty:
            return {
                'period_hours': hours,
                'data_points': 0,
                'avg_premium': None,
                'max_premium': None,
                'min_premium': None,
                'std_premium': None,
                'avg_confidence': None,
                'signal_distribution': {}
            }
        
        # 기본 통계
        stats = {
            'period_hours': hours,
            'data_points': len(df),
            'avg_premium': df['premium_rate'].mean(),
            'max_premium': df['premium_rate'].max(),
            'min_premium': df['premium_rate'].min(),
            'std_premium': df['premium_rate'].std(),
            'avg_confidence': df['confidence'].mean()
        }
        
        # 시그널 분포
        signal_counts = df['signal'].value_counts().to_dict()
        stats['signal_distribution'] = signal_counts
        
        # 시간대별 평균
        df_hourly = df.set_index('timestamp').resample('1H')['premium_rate'].mean()
        if not df_hourly.empty:
            stats['hourly_avg'] = df_hourly.to_list()
        
        return stats
    
    def get_trading_opportunities(self, 
                                 threshold: float = 4.0,
                                 hours: int = 24) -> List[Dict[str, Any]]:
        """
        거래 기회 찾기
        
        Args:
            threshold: 김프 임계값 (%)
            hours: 검색 기간
            
        Returns:
            거래 기회 리스트
        """
        start_date = datetime.now() - timedelta(hours=hours)
        df = self.load_history(start_date=start_date)
        
        if df.empty:
            return []
        
        # 임계값 이상인 경우 필터링
        opportunities = df[
            (df['premium_rate'] > threshold) | 
            (df['premium_rate'] < -threshold)
        ]
        
        # 결과 변환
        results = []
        for _, row in opportunities.iterrows():
            results.append({
                'timestamp': row['timestamp'].isoformat(),
                'premium_rate': row['premium_rate'],
                'signal': row['signal'],
                'confidence': row['confidence'],
                'upbit_price': row['upbit_price'],
                'binance_price': row['binance_price']
            })
        
        return results
    
    def analyze_signal_performance(self, hours: int = 168) -> Dict[str, Any]:
        """
        시그널 성과 분석
        
        Args:
            hours: 분석 기간 (기본 1주일)
            
        Returns:
            시그널별 성과
        """
        start_date = datetime.now() - timedelta(hours=hours)
        df = self.load_history(start_date=start_date)
        
        if df.empty:
            return {}
        
        performance = {}
        
        # 각 시그널별 분석
        for signal in df['signal'].unique():
            signal_df = df[df['signal'] == signal]
            
            if len(signal_df) > 0:
                # 다음 김프와의 변화 계산
                signal_df = signal_df.sort_values('timestamp')
                signal_df['next_premium'] = signal_df['premium_rate'].shift(-1)
                signal_df['premium_change'] = signal_df['next_premium'] - signal_df['premium_rate']
                
                performance[signal] = {
                    'count': len(signal_df),
                    'avg_premium': signal_df['premium_rate'].mean(),
                    'avg_confidence': signal_df['confidence'].mean(),
                    'avg_change': signal_df['premium_change'].mean(),
                    'win_rate': (signal_df['premium_change'] > 0).mean() if len(signal_df) > 1 else None
                }
        
        return performance
    
    def export_to_dataframe(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           resample: Optional[str] = None) -> pd.DataFrame:
        """
        데이터프레임으로 내보내기
        
        Args:
            start_date: 시작일
            end_date: 종료일
            resample: 리샘플링 주기 ('5T', '1H', '1D' 등)
            
        Returns:
            김프 데이터프레임
        """
        df = self.load_history(start_date, end_date)
        
        if df.empty:
            return df
        
        # 인덱스를 timestamp로 설정
        df.set_index('timestamp', inplace=True)
        
        # 리샘플링
        if resample:
            numeric_cols = ['upbit_price', 'binance_price', 'exchange_rate',
                          'premium_rate', 'premium_krw', 'liquidity_score',
                          'spread_upbit', 'spread_binance', 'confidence']
            
            df_resampled = df[numeric_cols].resample(resample).agg({
                'upbit_price': 'mean',
                'binance_price': 'mean',
                'exchange_rate': 'mean',
                'premium_rate': 'mean',
                'premium_krw': 'mean',
                'liquidity_score': 'mean',
                'spread_upbit': 'mean',
                'spread_binance': 'mean',
                'confidence': 'mean'
            })
            
            return df_resampled
        
        return df
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """
        오래된 파일 정리
        
        Args:
            days_to_keep: 보관 기간 (일)
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        
        for csv_file in self.history_dir.glob("premium_*.csv"):
            try:
                # 파일명에서 날짜 추출
                date_str = csv_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff:
                    csv_file.unlink()
                    logger.info(f"Deleted old premium file: {csv_file.name}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up {csv_file}: {e}")


# 전역 저장소
premium_storage = PremiumStorage()


if __name__ == "__main__":
    # 저장소 테스트
    print("Premium Storage Test")
    print("-" * 40)
    
    from src.analysis.kimchi_premium import KimchiPremiumData, PremiumSignal
    
    # 테스트 데이터 생성
    test_data = KimchiPremiumData(
        timestamp=datetime.now(),
        upbit_price=159_400_000,
        binance_price=115_000,
        exchange_rate=1386.14,
        premium_rate=0.01,
        premium_krw=10_000,
        signal=PremiumSignal.STRONG_SELL,
        liquidity_score=85.0,
        spread_upbit=0.05,
        spread_binance=0.02,
        confidence=0.85
    )
    
    # 저장
    premium_storage.save_premium(test_data)
    print("Saved test premium data")
    
    # 로드
    current = premium_storage.load_current_premium()
    if current:
        print(f"\nLoaded current premium: {current['premium_rate']:.2f}%")
    
    # 통계
    stats = premium_storage.get_statistics(hours=24)
    print(f"\nStatistics (24h):")
    for key, value in stats.items():
        if key != 'hourly_avg':
            print(f"  {key}: {value}")
    
    # 히스토리
    df = premium_storage.load_history()
    print(f"\nHistory records: {len(df)}")
    
    if not df.empty:
        print("\nLast 5 records:")
        print(df.tail())
    
    print("\nTest completed!")