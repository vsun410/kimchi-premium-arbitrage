"""
오더북 데이터 수집 파이프라인
15초 간격으로 오더북 depth 20 수집 및 분석
"""

import asyncio
import gzip
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.utils.logger import logger


class OrderbookSnapshot(BaseModel):
    """오더북 스냅샷 모델"""

    timestamp: datetime
    exchange: str
    symbol: str
    bids: List[List[float]] = Field(description="[[price, amount], ...]")
    asks: List[List[float]] = Field(description="[[price, amount], ...]")
    bid_ask_spread: float = Field(ge=0)
    mid_price: float = Field(gt=0)
    imbalance: float = Field(ge=-1, le=1, description="Order imbalance (-1 to 1)")
    liquidity_score: float = Field(ge=0, le=100)
    depth_10_pct: Dict[str, float] = Field(description="10% depth liquidity")


class OrderbookCollector:
    """오더북 수집기"""

    def __init__(
        self,
        data_dir: str = "data/orderbook",
        depth: int = 20,
        interval_seconds: int = 15,
    ):
        """
        초기화

        Args:
            data_dir: 데이터 저장 디렉토리
            depth: 오더북 깊이
            interval_seconds: 수집 간격 (초)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.depth = depth
        self.interval_seconds = interval_seconds
        
        # 거래소 인스턴스
        self.exchanges = {}
        
        # 수집 상태
        self.is_running = False
        self.collection_task = None
        self.last_snapshots = {}
        
        logger.info(
            f"Orderbook collector initialized: "
            f"depth={depth}, interval={interval_seconds}s"
        )

    async def initialize_exchanges(self):
        """거래소 초기화"""
        try:
            # Upbit
            self.exchanges["upbit"] = ccxt.upbit({
                "enableRateLimit": True,
                "options": {
                    "fetchOrderBook": {"limit": self.depth}
                }
            })
            
            # Binance
            self.exchanges["binance"] = ccxt.binance({
                "enableRateLimit": True,
                "options": {
                    "fetchOrderBook": {"limit": self.depth}
                }
            })
            
            logger.info("Exchanges initialized for orderbook collection")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            raise

    async def close_exchanges(self):
        """거래소 연결 종료"""
        for name, exchange in self.exchanges.items():
            if exchange:
                await exchange.close()
                logger.info(f"Closed {name} exchange connection")

    async def fetch_orderbook(
        self, exchange_name: str, symbol: str
    ) -> Optional[Dict]:
        """
        오더북 가져오기

        Args:
            exchange_name: 거래소 이름
            symbol: 심볼

        Returns:
            오더북 데이터
        """
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not initialized")
            return None
        
        try:
            orderbook = await exchange.fetch_order_book(symbol, self.depth)
            return orderbook
            
        except Exception as e:
            logger.error(f"Failed to fetch orderbook from {exchange_name}: {e}")
            return None

    def calculate_liquidity_metrics(
        self, bids: List[List[float]], asks: List[List[float]]
    ) -> Dict:
        """
        유동성 메트릭 계산

        Args:
            bids: 매수 호가
            asks: 매도 호가

        Returns:
            메트릭 딕셔너리
        """
        if not bids or not asks:
            return {
                "bid_ask_spread": 0,
                "mid_price": 0,
                "imbalance": 0,
                "liquidity_score": 0,
                "depth_10_pct": {"bid": 0, "ask": 0},
            }
        
        # 스프레드 계산
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = (best_ask - best_bid) / best_bid * 100  # 퍼센트
        mid_price = (best_bid + best_ask) / 2
        
        # 주문 불균형 계산
        total_bid_volume = sum(bid[1] for bid in bids)
        total_ask_volume = sum(ask[1] for ask in asks)
        
        if total_bid_volume + total_ask_volume > 0:
            imbalance = (total_bid_volume - total_ask_volume) / (
                total_bid_volume + total_ask_volume
            )
        else:
            imbalance = 0
        
        # 10% 깊이 유동성
        depth_10_pct = self._calculate_depth_liquidity(
            bids, asks, mid_price, percentage=0.1
        )
        
        # 유동성 점수 (0-100)
        liquidity_score = self._calculate_liquidity_score(
            spread, total_bid_volume, total_ask_volume, depth_10_pct
        )
        
        return {
            "bid_ask_spread": spread,
            "mid_price": mid_price,
            "imbalance": imbalance,
            "liquidity_score": liquidity_score,
            "depth_10_pct": depth_10_pct,
        }

    def _calculate_depth_liquidity(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float,
        percentage: float = 0.1,
    ) -> Dict[str, float]:
        """
        특정 % 깊이의 유동성 계산

        Args:
            bids: 매수 호가
            asks: 매도 호가
            mid_price: 중간 가격
            percentage: 깊이 비율 (0.1 = 10%)

        Returns:
            매수/매도 깊이 유동성
        """
        bid_threshold = mid_price * (1 - percentage)
        ask_threshold = mid_price * (1 + percentage)
        
        bid_liquidity = sum(
            bid[1] for bid in bids if bid[0] >= bid_threshold
        )
        ask_liquidity = sum(
            ask[1] for ask in asks if ask[0] <= ask_threshold
        )
        
        return {"bid": bid_liquidity, "ask": ask_liquidity}

    def _calculate_liquidity_score(
        self,
        spread: float,
        bid_volume: float,
        ask_volume: float,
        depth_10_pct: Dict[str, float],
    ) -> float:
        """
        유동성 점수 계산 (0-100)

        Args:
            spread: 스프레드 (%)
            bid_volume: 총 매수 물량
            ask_volume: 총 매도 물량
            depth_10_pct: 10% 깊이 유동성

        Returns:
            유동성 점수
        """
        score = 100.0
        
        # 1. 스프레드 페널티 (높을수록 나쁨)
        if spread > 0.5:  # 0.5% 이상
            score -= min(spread * 10, 30)
        
        # 2. 볼륨 불균형 페널티
        if bid_volume + ask_volume > 0:
            imbalance_ratio = abs(bid_volume - ask_volume) / (bid_volume + ask_volume)
            score -= imbalance_ratio * 20
        
        # 3. 깊이 유동성 보너스
        depth_total = depth_10_pct["bid"] + depth_10_pct["ask"]
        if depth_total > 10:  # 10 BTC 이상
            score += min(depth_total / 10, 20)
        
        return max(0, min(100, score))

    async def collect_snapshot(
        self, exchange_name: str, symbol: str
    ) -> Optional[OrderbookSnapshot]:
        """
        오더북 스냅샷 수집

        Args:
            exchange_name: 거래소 이름
            symbol: 심볼

        Returns:
            오더북 스냅샷
        """
        orderbook = await self.fetch_orderbook(exchange_name, symbol)
        if not orderbook:
            return None
        
        # 메트릭 계산
        metrics = self.calculate_liquidity_metrics(
            orderbook["bids"], orderbook["asks"]
        )
        
        # 스냅샷 생성
        # timestamp가 없거나 None인 경우 현재 시간 사용
        if orderbook.get("timestamp"):
            timestamp = datetime.fromtimestamp(orderbook["timestamp"] / 1000)
        else:
            timestamp = datetime.now()
            
        snapshot = OrderbookSnapshot(
            timestamp=timestamp,
            exchange=exchange_name,
            symbol=symbol,
            bids=orderbook["bids"][:self.depth],  # depth 제한
            asks=orderbook["asks"][:self.depth],
            bid_ask_spread=metrics["bid_ask_spread"],
            mid_price=metrics["mid_price"],
            imbalance=metrics["imbalance"],
            liquidity_score=metrics["liquidity_score"],
            depth_10_pct=metrics["depth_10_pct"],
        )
        
        return snapshot

    def save_snapshot(self, snapshot: OrderbookSnapshot):
        """
        스냅샷 저장 (압축)

        Args:
            snapshot: 오더북 스냅샷
        """
        # 날짜별 디렉토리
        date_str = snapshot.timestamp.strftime("%Y%m%d")
        date_dir = self.data_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        # 시간별 파일
        hour_str = snapshot.timestamp.strftime("%H")
        filename = f"{snapshot.exchange}_{snapshot.symbol.replace('/', '_')}_{hour_str}.jsonl.gz"
        filepath = date_dir / filename
        
        # JSONL 형식으로 압축 저장
        data_line = snapshot.json() + "\n"
        
        mode = "ab" if filepath.exists() else "wb"
        with gzip.open(filepath, mode) as f:
            f.write(data_line.encode("utf-8"))
        
        logger.debug(f"Saved orderbook snapshot to {filepath}")

    async def collection_loop(self):
        """수집 루프"""
        logger.info("Starting orderbook collection loop")
        
        while self.is_running:
            start_time = time.time()
            
            try:
                # 모든 거래소에서 동시 수집
                tasks = [
                    self.collect_snapshot("upbit", "BTC/KRW"),
                    self.collect_snapshot("binance", "BTC/USDT"),
                ]
                
                snapshots = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 저장 및 캐싱
                for snapshot in snapshots:
                    if isinstance(snapshot, OrderbookSnapshot):
                        self.save_snapshot(snapshot)
                        self.last_snapshots[f"{snapshot.exchange}_{snapshot.symbol}"] = snapshot
                        
                        logger.info(
                            f"Collected {snapshot.exchange} {snapshot.symbol}: "
                            f"spread={snapshot.bid_ask_spread:.3f}%, "
                            f"liquidity={snapshot.liquidity_score:.1f}"
                        )
                    elif isinstance(snapshot, Exception):
                        logger.error(f"Collection error: {snapshot}")
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
            
            # 다음 수집까지 대기
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_seconds - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def start_collection(self):
        """수집 시작"""
        if self.is_running:
            logger.warning("Collection already running")
            return
        
        await self.initialize_exchanges()
        
        self.is_running = True
        self.collection_task = asyncio.create_task(self.collection_loop())
        
        logger.info("Orderbook collection started")

    async def stop_collection(self):
        """수집 중지"""
        if not self.is_running:
            logger.warning("Collection not running")
            return
        
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        await self.close_exchanges()
        
        logger.info("Orderbook collection stopped")

    def get_latest_liquidity(self) -> Dict[str, float]:
        """최신 유동성 정보"""
        result = {}
        
        for key, snapshot in self.last_snapshots.items():
            result[key] = {
                "liquidity_score": snapshot.liquidity_score,
                "spread": snapshot.bid_ask_spread,
                "imbalance": snapshot.imbalance,
                "timestamp": snapshot.timestamp.isoformat(),
            }
        
        return result

    def load_snapshots(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[OrderbookSnapshot]:
        """
        저장된 스냅샷 로드

        Args:
            exchange: 거래소
            symbol: 심볼
            start_time: 시작 시간
            end_time: 종료 시간

        Returns:
            스냅샷 리스트
        """
        snapshots = []
        symbol_clean = symbol.replace("/", "_")
        
        # 날짜 범위 순회
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            date_dir = self.data_dir / date_str
            
            if date_dir.exists():
                # 시간별 파일 검색
                for filepath in date_dir.glob(f"{exchange}_{symbol_clean}_*.jsonl.gz"):
                    with gzip.open(filepath, "rt", encoding="utf-8") as f:
                        for line in f:
                            snapshot = OrderbookSnapshot.parse_raw(line)
                            
                            # 시간 범위 필터
                            if start_time <= snapshot.timestamp <= end_time:
                                snapshots.append(snapshot)
            
            current_date += timedelta(days=1)
        
        return sorted(snapshots, key=lambda x: x.timestamp)


# 싱글톤 인스턴스
orderbook_collector = OrderbookCollector()


if __name__ == "__main__":
    # 테스트
    async def test():
        # 10초 간격으로 테스트
        collector = OrderbookCollector(interval_seconds=10)
        
        try:
            await collector.start_collection()
            
            # 30초 동안 수집
            await asyncio.sleep(30)
            
            # 최신 유동성 정보
            liquidity = collector.get_latest_liquidity()
            print("\nLatest Liquidity:")
            for key, info in liquidity.items():
                print(f"  {key}: {info}")
            
        finally:
            await collector.stop_collection()
    
    asyncio.run(test())