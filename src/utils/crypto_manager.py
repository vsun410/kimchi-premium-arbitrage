"""
API 키 암호화 및 복호화 관리
Fernet 대칭키 암호화 사용
"""

import base64
import getpass
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CryptoManager:
    """API 키 암호화 관리자"""

    def __init__(self, key_file: str = ".keys/master.key"):
        """
        초기화

        Args:
            key_file: 마스터 키 파일 경로
        """
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = self._get_or_create_cipher()

        # 암호화된 API 키 저장 파일
        self.encrypted_file = Path(".keys/api_keys.enc")

    def _get_or_create_cipher(self) -> Fernet:
        """마스터 키를 가져오거나 생성"""
        if self.key_file.exists():
            # 기존 키 로드
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            # 새 키 생성
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)

            # 파일 권한 설정 (Windows에서는 제한적)
            try:
                os.chmod(self.key_file, 0o600)
            except:
                pass

            print(f"[INFO] 새 마스터 키 생성: {self.key_file}")
            print("[WARN] 이 키를 안전하게 백업하세요! 분실 시 API 키 복구 불가")

        return Fernet(key)

    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """비밀번호로부터 암호화 키 유도"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_api_keys(self, api_keys: Dict[str, str]) -> None:
        """API 키들을 암호화하여 저장"""
        # 메타데이터 추가
        data = {"keys": api_keys, "encrypted_at": datetime.now().isoformat(), "version": "1.0"}

        # JSON 직렬화 후 암호화
        json_data = json.dumps(data)
        encrypted_data = self.cipher.encrypt(json_data.encode())

        # 파일에 저장
        self.encrypted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.encrypted_file, "wb") as f:
            f.write(encrypted_data)

        print(f"[OK] API 키 암호화 완료: {self.encrypted_file}")

    def decrypt_api_keys(self) -> Optional[Dict[str, str]]:
        """암호화된 API 키 복호화"""
        if not self.encrypted_file.exists():
            print("[WARN] 암호화된 API 키 파일이 없습니다")
            return None

        try:
            with open(self.encrypted_file, "rb") as f:
                encrypted_data = f.read()

            # 복호화
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())

            # 암호화 시간 확인
            encrypted_at = datetime.fromisoformat(data["encrypted_at"])
            age = datetime.now() - encrypted_at

            if age > timedelta(days=90):
                print(f"[WARN] API 키가 {age.days}일 전에 암호화됨. 로테이션을 고려하세요.")

            return data["keys"]

        except Exception as e:
            print(f"[FAIL] API 키 복호화 실패: {e}")
            return None

    def add_api_key(self, name: str, value: str) -> None:
        """단일 API 키 추가"""
        # 기존 키 로드
        keys = self.decrypt_api_keys() or {}

        # 새 키 추가
        keys[name] = value

        # 다시 암호화하여 저장
        self.encrypt_api_keys(keys)

    def remove_api_key(self, name: str) -> None:
        """API 키 제거"""
        keys = self.decrypt_api_keys() or {}

        if name in keys:
            del keys[name]
            self.encrypt_api_keys(keys)
            print(f"[OK] API 키 제거됨: {name}")
        else:
            print(f"[WARN] API 키를 찾을 수 없음: {name}")

    def rotate_master_key(self) -> None:
        """마스터 키 로테이션"""
        # 기존 API 키 복호화
        api_keys = self.decrypt_api_keys()
        if not api_keys:
            print("[FAIL] 로테이션할 API 키가 없습니다")
            return

        # 백업
        backup_file = self.key_file.with_suffix(
            f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.rename(self.key_file, backup_file)
        print(f"[OK] 기존 키 백업: {backup_file}")

        # 새 키 생성
        self.cipher = self._get_or_create_cipher()

        # API 키 재암호화
        self.encrypt_api_keys(api_keys)
        print("[OK] 마스터 키 로테이션 완료")

    def verify_keys(self) -> bool:
        """암호화된 키 검증"""
        try:
            keys = self.decrypt_api_keys()
            if keys:
                print(f"[OK] {len(keys)}개의 API 키 확인됨")
                for key_name in keys:
                    print(f"  - {key_name}: ****** (암호화됨)")
                return True
            return False
        except Exception as e:
            print(f"[FAIL] 키 검증 실패: {e}")
            return False


class SecureEnvLoader:
    """안전한 환경변수 로더"""

    def __init__(self, crypto_manager: Optional[CryptoManager] = None):
        """
        초기화

        Args:
            crypto_manager: 암호화 관리자 인스턴스
        """
        self.crypto = crypto_manager or CryptoManager()
        self._cached_keys = {}

    def load_from_env(self) -> Dict[str, str]:
        """환경변수에서 API 키 로드"""
        from dotenv import load_dotenv

        load_dotenv()

        api_keys = {}
        key_names = [
            "UPBIT_ACCESS_KEY",
            "UPBIT_SECRET_KEY",
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
            "EXCHANGE_RATE_API_KEY",
        ]

        for key_name in key_names:
            value = os.getenv(key_name)
            if value and not value.startswith("your_"):
                api_keys[key_name] = value

        return api_keys

    def load_secure(self) -> Dict[str, str]:
        """암호화된 파일 또는 환경변수에서 안전하게 로드"""
        # 1. 먼저 암호화된 파일에서 시도
        encrypted_keys = self.crypto.decrypt_api_keys()

        if encrypted_keys:
            print("[OK] 암호화된 API 키 로드됨")
            return encrypted_keys

        # 2. 환경변수에서 로드
        env_keys = self.load_from_env()

        if env_keys:
            print("[OK] 환경변수에서 API 키 로드됨")
            # 자동으로 암호화하여 저장
            response = input("API 키를 암호화하여 저장하시겠습니까? (y/n): ")
            if response.lower() == "y":
                self.crypto.encrypt_api_keys(env_keys)
            return env_keys

        print("[WARN] API 키를 찾을 수 없습니다")
        return {}

    def get_key(self, key_name: str) -> Optional[str]:
        """특정 API 키 가져오기"""
        if not self._cached_keys:
            self._cached_keys = self.load_secure()

        return self._cached_keys.get(key_name)

    def validate_keys(self) -> bool:
        """모든 필수 API 키 검증"""
        required_keys = [
            "UPBIT_ACCESS_KEY",
            "UPBIT_SECRET_KEY",
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
        ]

        keys = self.load_secure()
        missing = []

        for key_name in required_keys:
            if key_name not in keys or not keys[key_name]:
                missing.append(key_name)

        if missing:
            print(f"[FAIL] 필수 API 키 누락: {', '.join(missing)}")
            return False

        print("[OK] 모든 필수 API 키 확인됨")
        return True


# CLI 도구
if __name__ == "__main__":
    import sys

    crypto = CryptoManager()
    loader = SecureEnvLoader(crypto)

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python crypto_manager.py encrypt  # API 키 암호화")
        print("  python crypto_manager.py decrypt  # API 키 복호화")
        print("  python crypto_manager.py verify   # API 키 검증")
        print("  python crypto_manager.py rotate   # 마스터 키 로테이션")
        sys.exit(1)

    command = sys.argv[1]

    if command == "encrypt":
        # 환경변수에서 키 로드 후 암호화
        keys = loader.load_from_env()
        if keys:
            crypto.encrypt_api_keys(keys)
        else:
            print("[FAIL] 환경변수에서 API 키를 찾을 수 없습니다")

    elif command == "decrypt":
        keys = crypto.decrypt_api_keys()
        if keys:
            print("[OK] 복호화된 API 키:")
            for name, value in keys.items():
                # 일부만 표시 (보안)
                masked = (
                    value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
                )
                print(f"  {name}: {masked}")

    elif command == "verify":
        crypto.verify_keys()

    elif command == "rotate":
        crypto.rotate_master_key()

    else:
        print(f"[FAIL] 알 수 없는 명령: {command}")
