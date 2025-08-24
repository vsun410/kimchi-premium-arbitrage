#!/usr/bin/env python3
"""
Historical Data Collector for BTC
Task #7: BTC 1년치 히스토리컬 데이터 수집
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from pydantic import BaseModel, Field, validator

from src.utils.logger import logger


class CandleData(BaseModel):
    """캔들 데이터 스키마"""
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    
    @validator('high')
    def high_check(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= Low')
        return v


class HistoricalDataCollector:
    """
    거래소에서 히스토리컬 데이터 수집
    
    Features:
    - 업비트/바이낸스 1년치 데이터 수집
    - 1분 캔들 데이터
    - API 레이트 리밋 관리
    - 데이터 무결성 검증
    """
    
    def __init__(self, exchange_id: str = 'binance'):
        """
        초기화
        
        Args:
            exchange_id: 거래소 ID ('binance' or 'upbit')
        """
        self.exchange_id = exchange_id
        
        # CCXT 거래소 초기화
        if exchange_id == 'binance':
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,  # milliseconds
            })
        elif exchange_id == 'upbit':
            self.exchange = ccxt.upbit({
                'enableRateLimit': True,
                'rateLimit': 100,  # Upbit은 더 엄격함
            })
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        
        logger.info(f"Historical collector initialized for {exchange_id}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 가져오기
        
        Args:
            symbol: 심볼 (예: 'BTC/USDT', 'BTC/KRW')
            timeframe: 시간 프레임 ('1m', '5m', '1h', '1d')
            since: 시작 시간
            limit: 가져올 캔들 수
            
        Returns:
            OHLCV 데이터프레임
        """
        try:
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)
            
            # CCXT로 데이터 가져오기
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
            
            # DataFrame 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            raise
    
    def collect_historical_data(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = '1m',
        save_dir: str = 'data/historical'
    ) -> pd.DataFrame:
        """
        히스토리컬 데이터 수집 (긴 기간)
        
        Args:
            symbol: 심볼
            days: 수집할 일수
            timeframe: 시간 프레임
            save_dir: 저장 디렉토리
            
        Returns:
            전체 데이터프레임
        """
        logger.info(f"Collecting {days} days of {symbol} data from {self.exchange_id}")
        
        # 시간 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 시간프레임별 캔들 수 계산
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        minutes_per_request = timeframe_minutes.get(timeframe, 1)
        candles_per_request = min(1000, 1440 // minutes_per_request * 7)  # 최대 1주일씩
        
        # 데이터 수집
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            try:
                # 데이터 가져오기
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=candles_per_request
                )
                
                if df.empty:
                    logger.warning(f"No data for {current_date}")
                    current_date += timedelta(days=1)
                    continue
                
                all_data.append(df)
                
                # 진행 상황 로그
                last_timestamp = df.index[-1]
                progress = (last_timestamp - start_date).days / days * 100
                logger.info(f"Progress: {progress:.1f}% - Last: {last_timestamp}")
                
                # 다음 구간으로
                current_date = last_timestamp + timedelta(minutes=minutes_per_request)
                
                # API 레이트 리밋 대응
                time.sleep(0.5)  # 추가 대기
                
            except Exception as e:
                logger.error(f"Error fetching data for {current_date}: {e}")
                time.sleep(5)  # 에러 시 더 긴 대기
                continue
        
        # 데이터 병합
        if not all_data:
            raise ValueError("No data collected")
        
        full_data = pd.concat(all_data)
        full_data = full_data[~full_data.index.duplicated(keep='first')]
        full_data.sort_index(inplace=True)
        
        # 데이터 검증
        self._validate_data(full_data)
        
        # 저장
        self._save_data(full_data, symbol, save_dir)
        
        logger.info(f"Collected {len(full_data)} candles for {symbol}")
        
        return full_data
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        데이터 무결성 검증
        
        Args:
            df: 검증할 데이터프레임
        """
        # 1. Null 값 체크
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values")
            df.fillna(method='ffill', inplace=True)
        
        # 2. 중복 체크
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate timestamps")
        
        # 3. 시간 갭 체크
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=1)  # 1분 캔들 기준
        
        gaps = time_diff[time_diff > expected_diff * 2]
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} time gaps")
            for idx, gap in gaps.items():
                logger.debug(f"  Gap at {idx}: {gap}")
        
        # 4. 가격 이상치 체크
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            # 전일 대비 50% 이상 변동 체크
            pct_change = df[col].pct_change().abs()
            anomalies = pct_change[pct_change > 0.5]
            if len(anomalies) > 0:
                logger.warning(f"Found {len(anomalies)} price anomalies in {col}")
        
        # 5. OHLC 논리 체크
        invalid_high = (df['high'] < df['low']).sum()
        invalid_low = (df['low'] > df['high']).sum()
        
        if invalid_high > 0 or invalid_low > 0:
            logger.warning(f"Invalid OHLC relationships: high<low={invalid_high}, low>high={invalid_low}")
        
        logger.info("Data validation completed")
    
    def _save_data(self, df: pd.DataFrame, symbol: str, save_dir: str) -> None:
        """
        데이터 저장
        
        Args:
            df: 저장할 데이터프레임
            symbol: 심볼
            save_dir: 저장 디렉토리
        """
        # 디렉토리 생성
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        safe_symbol = symbol.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.exchange_id}_{safe_symbol}_{timestamp}.csv"
        filepath = Path(save_dir) / filename
        
        # CSV 저장
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        
        # 압축 버전도 저장
        compressed_file = filepath.with_suffix('.csv.gz')
        df.to_csv(compressed_file, compression='gzip')
        logger.info(f"Compressed data saved to {compressed_file}")
    
    def load_historical_data(
        self,
        symbol: str,
        save_dir: str = 'data/historical'
    ) -> Optional[pd.DataFrame]:
        """
        저장된 히스토리컬 데이터 로드
        
        Args:
            symbol: 심볼
            save_dir: 저장 디렉토리
            
        Returns:
            데이터프레임 또는 None
        """
        safe_symbol = symbol.replace('/', '_')
        pattern = f"{self.exchange_id}_{safe_symbol}_*.csv*"
        
        # 파일 찾기
        save_path = Path(save_dir)
        files = list(save_path.glob(pattern))
        
        if not files:
            logger.warning(f"No historical data found for {symbol}")
            return None
        
        # 가장 최근 파일 선택
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading historical data from {latest_file}")
        
        # 로드
        if latest_file.suffix == '.gz':
            df = pd.read_csv(latest_file, compression='gzip', index_col=0, parse_dates=True)
        else:
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        
        return df
    
    def update_historical_data(
        self,
        symbol: str,
        save_dir: str = 'data/historical'
    ) -> pd.DataFrame:
        """
        히스토리컬 데이터 업데이트 (최신 데이터 추가)
        
        Args:
            symbol: 심볼
            save_dir: 저장 디렉토리
            
        Returns:
            업데이트된 데이터프레임
        """
        # 기존 데이터 로드
        existing_data = self.load_historical_data(symbol, save_dir)
        
        if existing_data is None:
            # 전체 수집
            return self.collect_historical_data(symbol, days=365, save_dir=save_dir)
        
        # 마지막 타임스탬프부터 현재까지 수집
        last_timestamp = existing_data.index[-1]
        days_to_update = (datetime.now() - last_timestamp).days + 1
        
        if days_to_update <= 0:
            logger.info("Data is already up to date")
            return existing_data
        
        logger.info(f"Updating {days_to_update} days of data")
        
        # 새 데이터 수집
        new_data = self.collect_historical_data(
            symbol=symbol,
            days=days_to_update,
            save_dir=save_dir
        )
        
        # 병합
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data.sort_index(inplace=True)
        
        # 저장
        self._save_data(combined_data, symbol, save_dir)
        
        return combined_data


def collect_all_exchanges(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    모든 거래소에서 데이터 수집
    
    Args:
        days: 수집할 일수
        
    Returns:
        거래소별 데이터프레임 딕셔너리
    """
    data = {}
    
    # Binance BTC/USDT
    try:
        binance_collector = HistoricalDataCollector('binance')
        data['binance'] = binance_collector.collect_historical_data(
            'BTC/USDT',
            days=days
        )
    except Exception as e:
        logger.error(f"Failed to collect Binance data: {e}")
    
    # Upbit BTC/KRW
    try:
        upbit_collector = HistoricalDataCollector('upbit')
        data['upbit'] = upbit_collector.collect_historical_data(
            'BTC/KRW',
            days=days
        )
    except Exception as e:
        logger.error(f"Failed to collect Upbit data: {e}")
    
    return data


if __name__ == "__main__":
    # 테스트: 최근 7일 데이터 수집
    collector = HistoricalDataCollector('binance')
    df = collector.collect_historical_data('BTC/USDT', days=7)
    print(f"Collected {len(df)} candles")
    print(df.head())
    print(df.tail())