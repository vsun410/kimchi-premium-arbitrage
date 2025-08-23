#!/usr/bin/env python3
"""
API 키 로테이션 스크립트
정기적으로 API 키를 업데이트하고 암호화
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import schedule

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.crypto_manager import CryptoManager, SecureEnvLoader


class KeyRotationManager:
    """API 키 로테이션 관리자"""

    def __init__(self):
        self.crypto = CryptoManager()
        self.loader = SecureEnvLoader(self.crypto)
        self.rotation_log = Path(".keys/rotation_log.json")

    def check_key_age(self) -> Dict[str, int]:
        """API 키 나이 확인 (일 단위)"""
        if not self.rotation_log.exists():
            return {}

        with open(self.rotation_log, "r") as f:
            log = json.load(f)

        key_ages = {}
        for key_name, info in log.items():
            last_rotated = datetime.fromisoformat(info["last_rotated"])
            age = (datetime.now() - last_rotated).days
            key_ages[key_name] = age

        return key_ages

    def log_rotation(self, key_name: str) -> None:
        """로테이션 기록"""
        log = {}
        if self.rotation_log.exists():
            with open(self.rotation_log, "r") as f:
                log = json.load(f)

        log[key_name] = {
            "last_rotated": datetime.now().isoformat(),
            "rotation_count": log.get(key_name, {}).get("rotation_count", 0) + 1,
        }

        self.rotation_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rotation_log, "w") as f:
            json.dump(log, f, indent=2)

    def should_rotate(self, key_name: str, max_age_days: int = 90) -> bool:
        """로테이션 필요 여부 확인"""
        ages = self.check_key_age()

        if key_name not in ages:
            # 첫 기록
            return True

        return ages[key_name] >= max_age_days

    def rotate_exchange_keys(self, exchange: str) -> bool:
        """거래소 API 키 로테이션"""
        print(f"\n[INFO] {exchange} API 키 로테이션 시작...")

        # 여기서는 실제 거래소 API를 호출하여 새 키를 생성해야 함
        # 데모 목적으로 수동 입력 받음

        print(f"[ACTION] {exchange} 거래소에서 새 API 키를 생성하세요")
        print("거래소 웹사이트:")

        if exchange == "UPBIT":
            print("  https://upbit.com/mypage/open_api_management")
            access_key = input("새 Access Key 입력: ").strip()
            secret_key = input("새 Secret Key 입력: ").strip()

            if access_key and secret_key:
                self.crypto.add_api_key("UPBIT_ACCESS_KEY", access_key)
                self.crypto.add_api_key("UPBIT_SECRET_KEY", secret_key)
                self.log_rotation("UPBIT_ACCESS_KEY")
                self.log_rotation("UPBIT_SECRET_KEY")
                return True

        elif exchange == "BINANCE":
            print("  https://www.binance.com/en/my/settings/api-management")
            api_key = input("새 API Key 입력: ").strip()
            secret_key = input("새 Secret Key 입력: ").strip()

            if api_key and secret_key:
                self.crypto.add_api_key("BINANCE_API_KEY", api_key)
                self.crypto.add_api_key("BINANCE_SECRET_KEY", secret_key)
                self.log_rotation("BINANCE_API_KEY")
                self.log_rotation("BINANCE_SECRET_KEY")
                return True

        return False

    def check_all_keys(self) -> None:
        """모든 키 상태 확인"""
        print("\n" + "=" * 50)
        print("API 키 상태 점검")
        print("=" * 50)

        ages = self.check_key_age()

        if not ages:
            print("[WARN] 로테이션 기록이 없습니다")
            print("모든 API 키를 새로 설정하는 것을 권장합니다")
            return

        for key_name, age in ages.items():
            status = "[OK]" if age < 60 else "[WARN]" if age < 90 else "[CRITICAL]"
            print(f"{status} {key_name}: {age}일 경과")

            if age >= 90:
                print(f"  -> 즉시 로테이션 필요!")
            elif age >= 60:
                print(f"  -> 로테이션 권장 (30일 이내)")

    def auto_rotation_check(self) -> None:
        """자동 로테이션 체크"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 자동 체크 실행")

        needs_rotation = []

        # 업비트 키 체크
        if self.should_rotate("UPBIT_ACCESS_KEY", max_age_days=90):
            needs_rotation.append("UPBIT")

        # 바이낸스 키 체크
        if self.should_rotate("BINANCE_API_KEY", max_age_days=90):
            needs_rotation.append("BINANCE")

        if needs_rotation:
            print(f"[ALERT] 로테이션이 필요한 거래소: {', '.join(needs_rotation)}")
            print("수동으로 'python scripts/rotate_keys.py manual' 실행하세요")
        else:
            print("[OK] 모든 API 키가 안전한 상태입니다")

    def schedule_rotation_checks(self) -> None:
        """로테이션 체크 스케줄링"""
        # 매일 오전 9시 체크
        schedule.every().day.at("09:00").do(self.auto_rotation_check)

        print("[INFO] API 키 로테이션 체크 스케줄러 시작")
        print("[INFO] 종료하려면 Ctrl+C를 누르세요")

        # 즉시 한 번 실행
        self.auto_rotation_check()

        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크


def main():
    """메인 실행 함수"""
    manager = KeyRotationManager()

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python rotate_keys.py check     # 키 상태 확인")
        print("  python rotate_keys.py manual    # 수동 로테이션")
        print("  python rotate_keys.py auto      # 자동 체크 시작")
        print("  python rotate_keys.py master    # 마스터 키 로테이션")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        manager.check_all_keys()

    elif command == "manual":
        print("\n어느 거래소 키를 로테이션하시겠습니까?")
        print("1. UPBIT")
        print("2. BINANCE")
        print("3. 모두")

        choice = input("선택 (1/2/3): ").strip()

        if choice == "1":
            if manager.rotate_exchange_keys("UPBIT"):
                print("[OK] UPBIT 키 로테이션 완료")
        elif choice == "2":
            if manager.rotate_exchange_keys("BINANCE"):
                print("[OK] BINANCE 키 로테이션 완료")
        elif choice == "3":
            success = True
            if manager.rotate_exchange_keys("UPBIT"):
                print("[OK] UPBIT 키 로테이션 완료")
            else:
                success = False
            if manager.rotate_exchange_keys("BINANCE"):
                print("[OK] BINANCE 키 로테이션 완료")
            else:
                success = False

            if success:
                print("[OK] 모든 키 로테이션 완료")

    elif command == "auto":
        try:
            manager.schedule_rotation_checks()
        except KeyboardInterrupt:
            print("\n[INFO] 스케줄러 종료")

    elif command == "master":
        print("[WARN] 마스터 키를 로테이션하면 모든 암호화된 데이터를 재암호화합니다")
        confirm = input("계속하시겠습니까? (yes/no): ")

        if confirm.lower() == "yes":
            manager.crypto.rotate_master_key()
        else:
            print("[INFO] 취소되었습니다")

    else:
        print(f"[FAIL] 알 수 없는 명령: {command}")


if __name__ == "__main__":
    main()
