#!/usr/bin/env python3
"""
환경 설정 확인 스크립트
모든 의존성과 설정이 올바르게 구성되었는지 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Python 버전 확인"""
    print("\n[Python 버전 확인]")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAIL] Python 3.9+ 필요 (현재: {version.major}.{version.minor})")
        return False


def check_required_packages():
    """필수 패키지 확인"""
    print("\n[필수 패키지 확인]")
    required = {
        "ccxt": "CCXT (거래소 API)",
        "pandas": "Pandas (데이터 처리)",
        "numpy": "NumPy (수치 연산)",
        "pydantic": "Pydantic (데이터 검증)",
        "torch": "PyTorch (LSTM)",
        "transformers": "Transformers (HuggingFace)",
        "xgboost": "XGBoost",
        "stable_baselines3": "Stable-Baselines3 (RL)",
        "optuna": "Optuna (최적화)",
        "dotenv": "python-dotenv",
        "cryptography": "Cryptography",
        "loguru": "Loguru (로깅)",
    }

    all_installed = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} - 설치 필요")
            all_installed = False

    return all_installed


def check_environment_variables():
    """환경변수 확인"""
    print("\n[환경변수 확인]")

    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print(f"  [FAIL] .env 파일 없음 - .env.example을 복사하여 생성하세요")
        return False

    from dotenv import load_dotenv

    load_dotenv(env_file)

    required_vars = [
        "UPBIT_ACCESS_KEY",
        "UPBIT_SECRET_KEY",
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "EXCHANGE_RATE_API_KEY",
    ]

    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value and not value.startswith("your_"):
            print(f"  [OK] {var} 설정됨")
        else:
            print(f"  [WARN] {var} 미설정 또는 기본값")
            all_set = False

    return all_set


def check_directory_structure():
    """디렉토리 구조 확인"""
    print("\n[디렉토리 구조 확인]")

    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/cache",
        "models/lstm",
        "models/xgboost",
        "models/rl",
        "src/data_collectors",
        "src/strategies",
        "src/utils",
        "src/config",
        "tests",
        "logs",
        "configs",
        "scripts",
        "docs",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [FAIL] {dir_path} 없음")
            all_exist = False

    return all_exist


def check_configuration():
    """설정 파일 로드 테스트"""
    print("\n[설정 파일 확인]")

    try:
        from src.config.settings import settings

        print(f"  [OK] 설정 로드 성공")
        print(f"     - 환경: {settings.environment}")
        print(f"     - 디버그: {settings.debug_mode}")
        print(f"     - 자본금: {settings.trading.capital_per_exchange:,}원")
        return True
    except Exception as e:
        print(f"  [FAIL] 설정 로드 실패: {e}")
        return False


def check_data_models():
    """데이터 모델 확인"""
    print("\n[데이터 모델 확인]")

    try:
        from src.models.schemas import (
            ExchangeRate,
            OrderBookData,
            Position,
            PriceData,
            Signal,
            TradingMetrics,
        )

        print(f"  [OK] 데이터 모델 로드 성공")

        # 샘플 데이터로 검증 테스트
        from datetime import datetime

        sample_price = PriceData(
            timestamp=datetime.now(),
            exchange="upbit",
            symbol="BTC",
            open=100000000,
            high=101000000,
            low=99000000,
            close=100500000,
            volume=100,
        )
        print(f"  [OK] 데이터 검증 테스트 통과")
        return True
    except Exception as e:
        print(f"  [FAIL] 데이터 모델 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("Kimchi Premium Arbitrage System")
    print("   환경 설정 확인")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_required_packages(),
        check_environment_variables(),
        check_directory_structure(),
        check_configuration(),
        check_data_models(),
    ]

    print("\n" + "=" * 50)
    if all(checks):
        print("[SUCCESS] 모든 확인 완료! 시스템을 시작할 준비가 되었습니다.")
    else:
        print("[WARNING] 일부 설정이 필요합니다. 위의 메시지를 확인하세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()
