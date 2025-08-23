"""
CSV 데이터 저장 시스템
일별 파티셔닝, 압축, 스키마 검증 포함
"""

import csv
import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, validator

from src.utils.logger import logger


class DataRecord(BaseModel):
    """기본 데이터 레코드 모델"""
    
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PriceRecord(DataRecord):
    """가격 데이터 레코드"""
    
    exchange: str
    symbol: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    

class OrderbookRecord(DataRecord):
    """오더북 데이터 레코드"""
    
    exchange: str
    symbol: str
    best_bid: float = Field(gt=0)
    best_ask: float = Field(gt=0)
    bid_volume: float = Field(ge=0)
    ask_volume: float = Field(ge=0)
    spread: float = Field(ge=0)
    mid_price: float = Field(gt=0)
    liquidity_score: float = Field(ge=0, le=100)
    imbalance: float = Field(ge=-1, le=1)


class KimchiPremiumRecord(DataRecord):
    """김치 프리미엄 데이터 레코드"""
    
    upbit_price: float = Field(gt=0)
    binance_price: float = Field(gt=0)
    exchange_rate: float = Field(gt=0)
    premium_rate: float
    premium_krw: float
    liquidity_score: float = Field(ge=0, le=100)
    spread_upbit: float = Field(ge=0)
    spread_binance: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class CSVStorage:
    """CSV 데이터 저장 관리자"""
    
    def __init__(
        self,
        base_dir: str = "data/csv",
        compress: bool = True,
        partition_by_day: bool = True,
        max_file_size_mb: int = 100,
    ):
        """
        초기화
        
        Args:
            base_dir: 기본 저장 디렉토리
            compress: 압축 여부
            partition_by_day: 일별 파티셔닝 여부
            max_file_size_mb: 최대 파일 크기 (MB)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.compress = compress
        self.partition_by_day = partition_by_day
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # 데이터 타입별 디렉토리
        self.dirs = {
            "price": self.base_dir / "price",
            "orderbook": self.base_dir / "orderbook",
            "premium": self.base_dir / "premium",
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # 버퍼 (배치 쓰기용)
        self.buffers: Dict[str, List[Dict]] = {}
        self.buffer_size = 1000  # 1000개씩 배치 저장
        
        logger.info(
            f"CSV storage initialized: base_dir={base_dir}, "
            f"compress={compress}, partition={partition_by_day}"
        )
    
    def _get_file_path(
        self,
        data_type: str,
        date: datetime,
        suffix: str = "",
    ) -> Path:
        """
        파일 경로 생성
        
        Args:
            data_type: 데이터 타입 (price, orderbook, premium)
            date: 날짜
            suffix: 파일명 접미사
            
        Returns:
            파일 경로
        """
        base_dir = self.dirs.get(data_type, self.base_dir)
        
        if self.partition_by_day:
            # 일별 디렉토리
            date_str = date.strftime("%Y%m%d")
            dir_path = base_dir / date_str
            dir_path.mkdir(exist_ok=True)
            
            # 파일명
            if suffix:
                filename = f"{data_type}_{suffix}_{date_str}"
            else:
                filename = f"{data_type}_{date_str}"
        else:
            # 월별 파일
            date_str = date.strftime("%Y%m")
            dir_path = base_dir
            
            if suffix:
                filename = f"{data_type}_{suffix}_{date_str}"
            else:
                filename = f"{data_type}_{date_str}"
        
        # 확장자
        ext = ".csv.gz" if self.compress else ".csv"
        
        return dir_path / f"{filename}{ext}"
    
    def _check_file_size(self, file_path: Path) -> bool:
        """
        파일 크기 체크
        
        Args:
            file_path: 파일 경로
            
        Returns:
            크기 초과 여부
        """
        if not file_path.exists():
            return False
        
        return file_path.stat().st_size >= self.max_file_size_bytes
    
    def _rotate_file(self, file_path: Path) -> Path:
        """
        파일 로테이션 (크기 초과 시)
        
        Args:
            file_path: 원본 파일 경로
            
        Returns:
            새 파일 경로
        """
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%H%M%S")
        stem = file_path.stem.replace(".csv", "")
        suffix = file_path.suffix
        
        new_path = file_path.parent / f"{stem}_{timestamp}{suffix}"
        
        if file_path.exists():
            file_path.rename(new_path)
            logger.info(f"Rotated file: {file_path} -> {new_path}")
        
        return file_path
    
    def save_price_data(
        self,
        records: List[PriceRecord],
        exchange: Optional[str] = None,
    ):
        """
        가격 데이터 저장
        
        Args:
            records: 가격 레코드 리스트
            exchange: 거래소 (파일명 구분용)
        """
        if not records:
            return
        
        # 날짜별 그룹화
        grouped = {}
        for record in records:
            date_key = record.timestamp.date()
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(record.dict())
        
        # 각 날짜별로 저장
        for date, data_list in grouped.items():
            suffix = exchange or ""
            file_path = self._get_file_path("price", datetime.combine(date, datetime.min.time()), suffix)
            
            # 파일 크기 체크
            if self._check_file_size(file_path):
                file_path = self._rotate_file(file_path)
            
            # CSV 저장
            self._write_csv(file_path, data_list, append=True)
        
        logger.debug(f"Saved {len(records)} price records")
    
    def save_orderbook_data(
        self,
        records: List[OrderbookRecord],
        exchange: Optional[str] = None,
    ):
        """
        오더북 데이터 저장
        
        Args:
            records: 오더북 레코드 리스트
            exchange: 거래소
        """
        if not records:
            return
        
        # 날짜별 그룹화
        grouped = {}
        for record in records:
            date_key = record.timestamp.date()
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(record.dict())
        
        # 각 날짜별로 저장
        for date, data_list in grouped.items():
            suffix = exchange or ""
            file_path = self._get_file_path("orderbook", datetime.combine(date, datetime.min.time()), suffix)
            
            # 파일 크기 체크
            if self._check_file_size(file_path):
                file_path = self._rotate_file(file_path)
            
            # CSV 저장
            self._write_csv(file_path, data_list, append=True)
        
        logger.debug(f"Saved {len(records)} orderbook records")
    
    def save_premium_data(
        self,
        records: List[KimchiPremiumRecord],
    ):
        """
        김치 프리미엄 데이터 저장
        
        Args:
            records: 김프 레코드 리스트
        """
        if not records:
            return
        
        # 날짜별 그룹화
        grouped = {}
        for record in records:
            date_key = record.timestamp.date()
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(record.dict())
        
        # 각 날짜별로 저장
        for date, data_list in grouped.items():
            file_path = self._get_file_path("premium", datetime.combine(date, datetime.min.time()))
            
            # 파일 크기 체크
            if self._check_file_size(file_path):
                file_path = self._rotate_file(file_path)
            
            # CSV 저장
            self._write_csv(file_path, data_list, append=True)
        
        logger.debug(f"Saved {len(records)} premium records")
    
    def _write_csv(
        self,
        file_path: Path,
        data: List[Dict],
        append: bool = True,
    ):
        """
        CSV 파일 쓰기
        
        Args:
            file_path: 파일 경로
            data: 데이터 리스트
            append: 추가 모드
        """
        if not data:
            return
        
        # 파일 존재 여부 및 헤더 필요 여부
        file_exists = file_path.exists()
        write_header = not file_exists or not append
        
        # 열기 모드
        mode = "ab" if append and file_exists else "wb"
        
        # 압축 여부에 따른 파일 열기
        if self.compress:
            file_obj = gzip.open(file_path, mode)
        else:
            file_obj = open(file_path, mode)
        
        try:
            # DataFrame 변환
            df = pd.DataFrame(data)
            
            # timestamp 정렬
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            # CSV 쓰기
            csv_data = df.to_csv(
                None,
                index=False,
                header=write_header,
                encoding="utf-8",
            )
            
            # 바이트로 변환 (압축 모드)
            if self.compress:
                file_obj.write(csv_data.encode("utf-8"))
            else:
                file_obj.write(csv_data.encode("utf-8"))
            
        finally:
            file_obj.close()
    
    def add_to_buffer(
        self,
        data_type: str,
        record: BaseModel,
    ):
        """
        버퍼에 데이터 추가 (배치 처리용)
        
        Args:
            data_type: 데이터 타입
            record: 레코드
        """
        buffer_key = f"{data_type}_{record.timestamp.date()}"
        
        if buffer_key not in self.buffers:
            self.buffers[buffer_key] = []
        
        self.buffers[buffer_key].append(record.dict())
        
        # 버퍼 크기 초과 시 플러시
        if len(self.buffers[buffer_key]) >= self.buffer_size:
            self.flush_buffer(buffer_key)
    
    def flush_buffer(self, buffer_key: Optional[str] = None):
        """
        버퍼 플러시
        
        Args:
            buffer_key: 특정 버퍼만 플러시 (None이면 전체)
        """
        if buffer_key:
            # 특정 버퍼만
            if buffer_key in self.buffers and self.buffers[buffer_key]:
                # buffer_key 파싱 수정 - 날짜 포맷 고려
                parts = buffer_key.split("_")
                
                # 마지막 부분이 날짜인지 확인
                date_str = parts[-1]
                data_type = "_".join(parts[:-1])
                
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    # 날짜 형식이 아니면 오늘 날짜 사용
                    date = datetime.now().date()
                    data_type = buffer_key
                
                file_path = self._get_file_path(
                    data_type,
                    datetime.combine(date, datetime.min.time())
                )
                
                # 레코드 수 저장
                record_count = len(self.buffers[buffer_key])
                
                self._write_csv(file_path, self.buffers[buffer_key], append=True)
                del self.buffers[buffer_key]
                
                logger.debug(f"Flushed buffer: {buffer_key} ({record_count} records)")
        else:
            # 전체 버퍼
            for key in list(self.buffers.keys()):
                self.flush_buffer(key)
    
    def load_price_data(
        self,
        start_date: datetime,
        end_date: datetime,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        가격 데이터 로드
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            exchange: 거래소
            
        Returns:
            DataFrame
        """
        return self._load_data("price", start_date, end_date, exchange)
    
    def load_orderbook_data(
        self,
        start_date: datetime,
        end_date: datetime,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        오더북 데이터 로드
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            exchange: 거래소
            
        Returns:
            DataFrame
        """
        return self._load_data("orderbook", start_date, end_date, exchange)
    
    def load_premium_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        김프 데이터 로드
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            DataFrame
        """
        return self._load_data("premium", start_date, end_date)
    
    def _load_data(
        self,
        data_type: str,
        start_date: datetime,
        end_date: datetime,
        suffix: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        데이터 로드
        
        Args:
            data_type: 데이터 타입
            start_date: 시작 날짜
            end_date: 종료 날짜
            suffix: 파일명 접미사
            
        Returns:
            DataFrame
        """
        dfs = []
        
        # 날짜 범위 순회
        current_date = start_date.date()
        while current_date <= end_date.date():
            file_path = self._get_file_path(
                data_type,
                datetime.combine(current_date, datetime.min.time()),
                suffix or ""
            )
            
            if file_path.exists():
                try:
                    # 압축 여부에 따른 읽기
                    if self.compress:
                        df = pd.read_csv(file_path, compression="gzip")
                    else:
                        df = pd.read_csv(file_path)
                    
                    # timestamp 파싱
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    dfs.append(df)
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
            
            # 다음 날짜
            if self.partition_by_day:
                current_date += timedelta(days=1)
            else:
                # 월별인 경우 다음 달로
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)
        
        # 병합
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            
            # 시간 범위 필터
            if "timestamp" in df.columns:
                df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            
            return df.sort_values("timestamp").reset_index(drop=True)
        
        return pd.DataFrame()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        저장소 통계
        
        Returns:
            통계 정보
        """
        stats = {
            "total_size_mb": 0,
            "file_count": 0,
            "data_types": {},
        }
        
        for data_type, dir_path in self.dirs.items():
            type_stats = {
                "size_mb": 0,
                "file_count": 0,
                "oldest_date": None,
                "newest_date": None,
            }
            
            # 파일 순회
            for file_path in dir_path.rglob("*.csv*"):
                type_stats["file_count"] += 1
                type_stats["size_mb"] += file_path.stat().st_size / (1024 * 1024)
                
                # 날짜 추출
                try:
                    date_str = file_path.stem.split("_")[-1][:8]
                    file_date = datetime.strptime(date_str, "%Y%m%d").date()
                    
                    if type_stats["oldest_date"] is None or file_date < type_stats["oldest_date"]:
                        type_stats["oldest_date"] = file_date
                    
                    if type_stats["newest_date"] is None or file_date > type_stats["newest_date"]:
                        type_stats["newest_date"] = file_date
                        
                except:
                    pass
            
            stats["data_types"][data_type] = type_stats
            stats["total_size_mb"] += type_stats["size_mb"]
            stats["file_count"] += type_stats["file_count"]
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        오래된 데이터 정리
        
        Args:
            days_to_keep: 보관 일수
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for dir_path in self.dirs.values():
            for file_path in dir_path.rglob("*.csv*"):
                try:
                    # 파일 날짜 추출
                    date_str = file_path.stem.split("_")[-1][:8]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Deleted old file: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to cleanup {file_path}: {e}")


# 싱글톤 인스턴스
csv_storage = CSVStorage()


if __name__ == "__main__":
    # 테스트
    storage = CSVStorage()
    
    # 테스트 데이터
    price_records = [
        PriceRecord(
            timestamp=datetime.now(),
            exchange="upbit",
            symbol="BTC/KRW",
            open=160000000,
            high=161000000,
            low=159000000,
            close=160500000,
            volume=123.45,
        )
    ]
    
    orderbook_records = [
        OrderbookRecord(
            timestamp=datetime.now(),
            exchange="binance",
            symbol="BTC/USDT",
            best_bid=115000,
            best_ask=115100,
            bid_volume=10.5,
            ask_volume=9.8,
            spread=0.087,
            mid_price=115050,
            liquidity_score=85.5,
            imbalance=0.035,
        )
    ]
    
    premium_records = [
        KimchiPremiumRecord(
            timestamp=datetime.now(),
            upbit_price=160000000,
            binance_price=115000,
            exchange_rate=1386.14,
            premium_rate=0.42,
            premium_krw=420000,
            liquidity_score=85.0,
            spread_upbit=0.05,
            spread_binance=0.02,
            confidence=0.85,
        )
    ]
    
    # 저장 테스트
    storage.save_price_data(price_records, "upbit")
    storage.save_orderbook_data(orderbook_records, "binance")
    storage.save_premium_data(premium_records)
    
    # 통계
    stats = storage.get_storage_stats()
    print(f"Storage stats: {json.dumps(stats, indent=2, default=str)}")