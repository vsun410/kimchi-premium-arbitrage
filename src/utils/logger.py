"""
로깅 시스템 설정
구조화된 JSON 로깅 및 파일 로테이션
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

# loguru import (더 나은 로깅)
try:
    from loguru import logger as loguru_logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """JSON 형식 포맷터"""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON으로 포맷"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # 추가 필드
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "exchange"):
            log_data["exchange"] = record.exchange

        if hasattr(record, "symbol"):
            log_data["symbol"] = record.symbol

        if hasattr(record, "action"):
            log_data["action"] = record.action

        # 예외 정보
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False)


class LoggerManager:
    """로거 관리자"""

    def __init__(self, name: str = "kimchi_arbitrage"):
        """
        초기화

        Args:
            name: 로거 이름
        """
        self.name = name
        self.logger = self._setup_logger()

        # Loguru 설정 (가능한 경우)
        if LOGURU_AVAILABLE:
            self._setup_loguru()

    def _setup_logger(self) -> logging.Logger:
        """표준 로거 설정"""
        logger = logging.getLogger(self.name)

        # 이미 설정된 경우 반환
        if logger.handlers:
            return logger

        # 로그 레벨 설정
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, log_level))

        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 파일 핸들러 (일반 로그)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        # 에러 로그 파일
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{self.name}_error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        logger.addHandler(error_handler)

        # 일별 로그 파일
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            LOG_DIR / f"{self.name}_daily.log",
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        daily_handler.setLevel(logging.INFO)
        daily_handler.setFormatter(JSONFormatter())
        logger.addHandler(daily_handler)

        return logger

    def _setup_loguru(self):
        """Loguru 설정"""
        # 기본 핸들러 제거
        loguru_logger.remove()

        # 콘솔 출력
        loguru_logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level="INFO",
            colorize=True,
        )

        # 파일 출력 (JSON)
        loguru_logger.add(
            LOG_DIR / "loguru_{time:YYYY-MM-DD}.log",
            format="{message}",
            level="DEBUG",
            rotation="00:00",
            retention="30 days",
            serialize=True,  # JSON 직렬화
            encoding="utf-8",
        )

        # 에러 로그
        loguru_logger.add(
            LOG_DIR / "loguru_error.log",
            format="{message}",
            level="ERROR",
            rotation="10 MB",
            retention="7 days",
            serialize=True,
            encoding="utf-8",
        )

    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        self.logger.debug(message, extra=kwargs)
        if LOGURU_AVAILABLE:
            loguru_logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """정보 로그"""
        self.logger.info(message, extra=kwargs)
        if LOGURU_AVAILABLE:
            loguru_logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """경고 로그"""
        self.logger.warning(message, extra=kwargs)
        if LOGURU_AVAILABLE:
            loguru_logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """에러 로그"""
        self.logger.error(message, exc_info=True, extra=kwargs)
        if LOGURU_AVAILABLE:
            loguru_logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """치명적 에러 로그"""
        self.logger.critical(message, exc_info=True, extra=kwargs)
        if LOGURU_AVAILABLE:
            loguru_logger.critical(message, **kwargs)

    def log_trade(
        self, action: str, exchange: str, symbol: str, price: float, amount: float, **kwargs
    ):
        """거래 로그"""
        trade_data = {
            "action": action,
            "exchange": exchange,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }

        self.info(
            f"Trade executed: {action} {amount} {symbol} @ {price} on {exchange}", **trade_data
        )

    def log_performance(self, metric: str, value: float, **kwargs):
        """성능 메트릭 로그"""
        perf_data = {
            "metric": metric,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }

        self.info(f"Performance metric: {metric}={value}", **perf_data)


# 전역 로거 인스턴스
logger = LoggerManager()


# 데코레이터
def log_execution_time(func):
    """실행 시간 로깅 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


def log_api_call(exchange: str):
    """API 호출 로깅 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"API call to {exchange}: {func.__name__}", exchange=exchange)
            try:
                result = func(*args, **kwargs)
                logger.debug(f"API call successful: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"API call failed: {func.__name__}: {e}", exchange=exchange)
                raise

        return wrapper

    return decorator


class TradingLogger:
    """거래 전용 로거"""

    def __init__(self):
        self.logger = LoggerManager("trading")

    def log_signal(self, signal_type: str, confidence: float, kimchi_premium: float, **kwargs):
        """거래 신호 로그"""
        self.logger.info(
            f"Signal generated: {signal_type} (confidence: {confidence:.2%})",
            signal_type=signal_type,
            confidence=confidence,
            kimchi_premium=kimchi_premium,
            **kwargs,
        )

    def log_order(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        **kwargs,
    ):
        """주문 로그"""
        self.logger.info(
            f"Order placed: {order_id} - {side} {amount} {symbol} @ {price} on {exchange}",
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            price=price,
            amount=amount,
            **kwargs,
        )

    def log_position(self, position_id: str, status: str, pnl: float, **kwargs):
        """포지션 로그"""
        self.logger.info(
            f"Position {position_id}: {status} (PnL: {pnl:+.2f})",
            position_id=position_id,
            status=status,
            pnl=pnl,
            **kwargs,
        )

    def log_risk_alert(self, alert_type: str, message: str, **kwargs):
        """리스크 경고 로그"""
        self.logger.warning(
            f"Risk Alert [{alert_type}]: {message}", alert_type=alert_type, **kwargs
        )


# 거래 로거 인스턴스
trading_logger = TradingLogger()


if __name__ == "__main__":
    # 로거 테스트
    print("로깅 시스템 테스트")
    print("-" * 40)

    # 기본 로그 테스트
    logger.debug("디버그 메시지")
    logger.info("정보 메시지")
    logger.warning("경고 메시지")

    # 거래 로그 테스트
    logger.log_trade(action="BUY", exchange="upbit", symbol="BTC/KRW", price=100000000, amount=0.01)

    # 성능 메트릭 테스트
    logger.log_performance("latency", 0.025)

    # 거래 전용 로거 테스트
    trading_logger.log_signal(signal_type="ENTER", confidence=0.85, kimchi_premium=4.5)

    print(f"\n로그 파일 확인: {LOG_DIR}")
    print("테스트 완료!")
