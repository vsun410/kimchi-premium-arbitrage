"""
히스토리컬 데이터 수집 모듈
업비트와 바이낸스에서 과거 캔들 데이터 수집
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
import pandas as pd
from pydantic import BaseModel, Field

from src.utils.logger import logger


class CandleData(BaseModel):
    """캔들 데이터 모델"""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class HistoricalFetcher:
    """히스토리컬 데이터 수집기"""

    def __init__(self, data_dir: str = "data/historical"):
        """
        초기화

        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 거래소 인스턴스 (나중에 초기화)
        self.upbit = None
        self.binance = None

        # Rate limit 설정 (요청/초)
        self.rate_limits = {"upbit": 10, "binance": 20}  # API 제한

        logger.info("Historical fetcher initialized")

    async def initialize_exchanges(self):
        """거래소 초기화"""
        try:
            # Upbit 초기화
            self.upbit = ccxt.upbit(
                {
                    "enableRateLimit": True,
                    "rateLimit": 1000 / self.rate_limits["upbit"],  # ms per request
                }
            )

            # Binance 초기화
            self.binance = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "rateLimit": 1000 / self.rate_limits["binance"],
                }
            )

            logger.info("Exchanges initialized for historical data fetch")

        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            raise

    async def close_exchanges(self):
        """거래소 연결 종료"""
        if self.upbit:
            await self.upbit.close()
        if self.binance:
            await self.binance.close()

    async def fetch_candles(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List]:
        """
        캔들 데이터 가져오기

        Args:
            exchange: CCXT 거래소 객체
            symbol: 심볼 (예: 'BTC/KRW', 'BTC/USDT')
            timeframe: 시간 프레임 ('1m', '5m', '1h' 등)
            since: 시작 타임스탬프 (ms)
            limit: 가져올 캔들 수

        Returns:
            OHLCV 데이터 리스트
        """
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since, limit=limit
            )
            return ohlcv

        except Exception as e:
            logger.error(f"Failed to fetch candles from {exchange.name}: {e}")
            return []

    async def fetch_historical_data(
        self,
        exchange_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        히스토리컬 데이터 수집

        Args:
            exchange_name: 거래소 이름 ('upbit' 또는 'binance')
            symbol: 심볼
            start_date: 시작일
            end_date: 종료일
            timeframe: 시간 프레임

        Returns:
            캔들 데이터프레임
        """
        exchange = self.upbit if exchange_name == "upbit" else self.binance

        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not initialized")

        all_candles = []
        current_since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        # 시간 프레임별 밀리초
        timeframe_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }

        ms_per_candle = timeframe_ms.get(timeframe, 60 * 1000)
        batch_size = 500  # 한 번에 가져올 캔들 수

        logger.info(
            f"Fetching {exchange_name} {symbol} from {start_date} to {end_date}"
        )

        while current_since < end_timestamp:
            try:
                # 캔들 가져오기
                candles = await self.fetch_candles(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=batch_size,
                )

                if not candles:
                    logger.warning(f"No more data available at {current_since}")
                    break

                all_candles.extend(candles)

                # 다음 배치 시작점
                last_timestamp = candles[-1][0]
                current_since = last_timestamp + ms_per_candle

                # 진행 상황 로그
                fetched_date = datetime.fromtimestamp(last_timestamp / 1000)
                progress = (last_timestamp - start_date.timestamp() * 1000) / (
                    end_timestamp - start_date.timestamp() * 1000
                )
                logger.info(
                    f"Progress: {progress:.1%} - Fetched up to {fetched_date}"
                )

                # Rate limit 준수
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                await asyncio.sleep(1)  # 에러 시 잠시 대기
                continue

        # DataFrame 변환
        if all_candles:
            df = pd.DataFrame(
                all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.sort_values("timestamp")

            # 시간 범위 필터링
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            logger.info(
                f"Fetched {len(df)} candles for {exchange_name} {symbol}"
            )
            return df

        return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        데이터 유효성 검증

        Args:
            df: 캔들 데이터프레임

        Returns:
            (유효 여부, 문제점 리스트)
        """
        issues = []

        # 1. 누락값 체크
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")

        # 2. 중복 체크
        duplicates = df.duplicated(subset=["timestamp"]).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")

        # 3. 가격 일관성 체크 (high >= low)
        invalid_prices = df[df["high"] < df["low"]]
        if not invalid_prices.empty:
            issues.append(f"Found {len(invalid_prices)} candles with high < low")

        # 4. 시간 간격 체크
        if len(df) > 1:
            time_diffs = df["timestamp"].diff().dropna()
            expected_diff = pd.Timedelta(minutes=1)  # 1분 캔들 기준

            irregular_gaps = time_diffs[time_diffs != expected_diff]
            if not irregular_gaps.empty:
                issues.append(
                    f"Found {len(irregular_gaps)} irregular time gaps"
                )

        # 5. 음수 값 체크
        negative_values = (df[["open", "high", "low", "close", "volume"]] < 0).any()
        if negative_values.any():
            issues.append(f"Found negative values in: {negative_values[negative_values].index.tolist()}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def save_to_csv(
        self, df: pd.DataFrame, exchange: str, symbol: str, timeframe: str = "1m"
    ):
        """
        CSV 파일로 저장

        Args:
            df: 데이터프레임
            exchange: 거래소 이름
            symbol: 심볼
            timeframe: 시간 프레임
        """
        # 파일명 생성
        symbol_clean = symbol.replace("/", "_")
        filename = f"{exchange}_{symbol_clean}_{timeframe}.csv"
        filepath = self.data_dir / filename

        # 저장
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} candles to {filepath}")

    async def fetch_one_year_btc(self) -> Dict[str, pd.DataFrame]:
        """
        BTC 1년치 데이터 수집 (메인 함수)

        Returns:
            {'upbit': df, 'binance': df}
        """
        # 거래소 초기화
        await self.initialize_exchanges()

        # 날짜 설정 (1년 전 ~ 현재)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        results = {}

        try:
            # 1. Upbit BTC/KRW 수집
            logger.info("Fetching Upbit BTC/KRW data...")
            upbit_df = await self.fetch_historical_data(
                exchange_name="upbit",
                symbol="BTC/KRW",
                start_date=start_date,
                end_date=end_date,
                timeframe="1m",
            )

            if not upbit_df.empty:
                # 검증
                is_valid, issues = self.validate_data(upbit_df)
                if not is_valid:
                    logger.warning(f"Upbit data validation issues: {issues}")

                # 저장
                self.save_to_csv(upbit_df, "upbit", "BTC/KRW", "1m")
                results["upbit"] = upbit_df

            # 2. Binance BTC/USDT 수집
            logger.info("Fetching Binance BTC/USDT data...")
            binance_df = await self.fetch_historical_data(
                exchange_name="binance",
                symbol="BTC/USDT",
                start_date=start_date,
                end_date=end_date,
                timeframe="1m",
            )

            if not binance_df.empty:
                # 검증
                is_valid, issues = self.validate_data(binance_df)
                if not is_valid:
                    logger.warning(f"Binance data validation issues: {issues}")

                # 저장
                self.save_to_csv(binance_df, "binance", "BTC/USDT", "1m")
                results["binance"] = binance_df

            # 통계 출력
            self._print_statistics(results)

            return results

        finally:
            # 거래소 연결 종료
            await self.close_exchanges()

    def _print_statistics(self, results: Dict[str, pd.DataFrame]):
        """통계 정보 출력"""
        for exchange, df in results.items():
            if df.empty:
                continue

            logger.info(f"\n{exchange.upper()} Statistics:")
            logger.info(f"  - Total candles: {len(df):,}")
            logger.info(f"  - Date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            logger.info(f"  - Price range: {df['close'].min():,.0f} ~ {df['close'].max():,.0f}")
            logger.info(f"  - Avg daily volume: {df['volume'].mean():,.2f}")
            logger.info(f"  - Missing data: {df.isnull().sum().sum()}")


# 싱글톤 인스턴스
historical_fetcher = HistoricalFetcher()


if __name__ == "__main__":
    # 테스트
    async def test():
        results = await historical_fetcher.fetch_one_year_btc()
        print(f"\nFetched data: {list(results.keys())}")

    asyncio.run(test())