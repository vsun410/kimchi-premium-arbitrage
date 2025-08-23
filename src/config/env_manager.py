"""
환경변수 관리 및 API 키 로딩
보안 강화된 환경변수 로더
"""

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# .env 파일 로드
load_dotenv(BASE_DIR / ".env")


@dataclass
class APICredentials:
    """API 인증 정보"""

    access_key: str
    secret_key: str

    def __post_init__(self):
        """초기화 후 검증"""
        if not self.access_key or not self.secret_key:
            raise ValueError("API 키가 비어있습니다")

        if self.access_key.startswith("your_") or self.secret_key.startswith("your_"):
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요")

    def get_signature(self, message: str) -> str:
        """HMAC 서명 생성 (업비트용)"""
        return hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()

    def mask(self) -> str:
        """마스킹된 키 표시"""
        return f"{self.access_key[:4]}...{self.access_key[-4:]}"


class EnvManager:
    """환경변수 관리자"""

    def __init__(self, use_encryption: bool = True):
        """
        초기화

        Args:
            use_encryption: 암호화 사용 여부
        """
        self.use_encryption = use_encryption
        self._credentials_cache = {}

        if use_encryption:
            try:
                from src.utils.crypto_manager import SecureEnvLoader

                self.secure_loader = SecureEnvLoader()
            except ImportError:
                print("[WARN] 암호화 모듈을 찾을 수 없습니다. 일반 모드로 전환")
                self.use_encryption = False
                self.secure_loader = None
        else:
            self.secure_loader = None

    def get_upbit_credentials(self) -> APICredentials:
        """업비트 API 인증 정보"""
        if "upbit" in self._credentials_cache:
            return self._credentials_cache["upbit"]

        if self.use_encryption and self.secure_loader:
            # 암호화된 소스에서 로드
            keys = self.secure_loader.load_secure()
            access_key = keys.get("UPBIT_ACCESS_KEY")
            secret_key = keys.get("UPBIT_SECRET_KEY")
        else:
            # 환경변수에서 로드
            access_key = os.getenv("UPBIT_ACCESS_KEY")
            secret_key = os.getenv("UPBIT_SECRET_KEY")

        if not access_key or not secret_key:
            raise ValueError("업비트 API 키가 설정되지 않았습니다")

        creds = APICredentials(access_key, secret_key)
        self._credentials_cache["upbit"] = creds
        return creds

    def get_binance_credentials(self) -> APICredentials:
        """바이낸스 API 인증 정보"""
        if "binance" in self._credentials_cache:
            return self._credentials_cache["binance"]

        if self.use_encryption and self.secure_loader:
            # 암호화된 소스에서 로드
            keys = self.secure_loader.load_secure()
            api_key = keys.get("BINANCE_API_KEY")
            secret_key = keys.get("BINANCE_SECRET_KEY")
        else:
            # 환경변수에서 로드
            api_key = os.getenv("BINANCE_API_KEY")
            secret_key = os.getenv("BINANCE_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("바이낸스 API 키가 설정되지 않았습니다")

        creds = APICredentials(api_key, secret_key)
        self._credentials_cache["binance"] = creds
        return creds

    def get_exchange_rate_api_key(self) -> str:
        """환율 API 키"""
        if self.use_encryption and self.secure_loader:
            keys = self.secure_loader.load_secure()
            api_key = keys.get("EXCHANGE_RATE_API_KEY")
        else:
            api_key = os.getenv("EXCHANGE_RATE_API_KEY")

        if not api_key or api_key.startswith("your_"):
            # 무료 API 사용 (제한적)
            return "free_tier"

        return api_key

    def get_trading_config(self) -> Dict[str, Any]:
        """거래 설정"""
        return {
            "capital_per_exchange": int(os.getenv("CAPITAL_PER_EXCHANGE", 20000000)),
            "max_position_size_percent": float(os.getenv("MAX_POSITION_SIZE_PERCENT", 1.0)),
            "kimchi_premium_entry": float(os.getenv("KIMCHI_PREMIUM_ENTRY_THRESHOLD", 4.0)),
            "kimchi_premium_exit": float(os.getenv("KIMCHI_PREMIUM_EXIT_THRESHOLD", 2.0)),
        }

    def get_monitoring_config(self) -> Dict[str, Optional[str]]:
        """모니터링 설정"""
        return {
            "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
            "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

    def validate_all(self) -> bool:
        """모든 설정 검증"""
        print("\n[환경변수 검증]")

        errors = []

        # 업비트 검증
        try:
            upbit = self.get_upbit_credentials()
            print(f"  [OK] 업비트 API: {upbit.mask()}")
        except Exception as e:
            errors.append(f"업비트: {e}")
            print(f"  [FAIL] 업비트 API: {e}")

        # 바이낸스 검증
        try:
            binance = self.get_binance_credentials()
            print(f"  [OK] 바이낸스 API: {binance.mask()}")
        except Exception as e:
            errors.append(f"바이낸스: {e}")
            print(f"  [FAIL] 바이낸스 API: {e}")

        # 환율 API 검증
        rate_key = self.get_exchange_rate_api_key()
        if rate_key == "free_tier":
            print(f"  [WARN] 환율 API: 무료 버전 (제한적)")
        else:
            print(f"  [OK] 환율 API: 설정됨")

        # 거래 설정 검증
        trading = self.get_trading_config()
        print(f"  [OK] 거래 설정:")
        print(f"      - 자본금: {trading['capital_per_exchange']:,}원")
        print(f"      - 포지션 크기: {trading['max_position_size_percent']}%")
        print(f"      - 김프 진입: {trading['kimchi_premium_entry']}%")
        print(f"      - 김프 청산: {trading['kimchi_premium_exit']}%")

        # 모니터링 설정
        monitoring = self.get_monitoring_config()
        if monitoring["slack_webhook"]:
            print(f"  [OK] Slack 알림: 설정됨")
        if monitoring["discord_webhook"]:
            print(f"  [OK] Discord 알림: 설정됨")

        if errors:
            print(f"\n[FAIL] 검증 실패: {len(errors)}개 오류")
            return False
        else:
            print(f"\n[SUCCESS] 모든 환경변수 검증 완료")
            return True

    def secure_clear_cache(self) -> None:
        """캐시된 인증 정보 안전하게 삭제"""
        self._credentials_cache.clear()
        print("[OK] 인증 정보 캐시 삭제됨")


# 전역 인스턴스
env_manager = EnvManager()


# CLI 테스트
if __name__ == "__main__":
    import sys

    manager = EnvManager()

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        manager.validate_all()
    else:
        print("환경변수 관리자 테스트")
        print("-" * 40)

        try:
            # 업비트 테스트
            upbit = manager.get_upbit_credentials()
            print(f"업비트: {upbit.mask()}")

            # 바이낸스 테스트
            binance = manager.get_binance_credentials()
            print(f"바이낸스: {binance.mask()}")

            # 전체 검증
            if manager.validate_all():
                print("\n[SUCCESS] 환경 설정 준비 완료!")
            else:
                print("\n[FAIL] 환경 설정을 확인하세요")

        except Exception as e:
            print(f"[ERROR] {e}")
