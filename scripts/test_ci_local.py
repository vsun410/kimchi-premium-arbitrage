#!/usr/bin/env python3
"""
로컬에서 CI 파이프라인 테스트
GitHub Actions 실행 전 로컬 검증
"""

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """명령 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[TEST] {description}")
    print(f"Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"[PASS] {description}")
            if result.stdout:
                print(result.stdout[:500])  # 처음 500자만
        else:
            print(f"[FAIL] {description}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("LOCAL CI PIPELINE TEST")
    print("=" * 60)

    tests = [
        # 1. 코드 포맷팅 체크
        ("black --check src/ tests/ scripts/", "Black formatting check"),
        # 2. Import 정렬 체크
        ("isort --check-only src/ tests/ scripts/", "Import sorting check"),
        # 3. Linting
        ("flake8 src/ --count --exit-zero --statistics", "Flake8 linting"),
        # 4. Type checking
        ("mypy src/ --ignore-missing-imports", "MyPy type checking"),
        # 5. Security check
        ("bandit -r src/ -ll -f json", "Bandit security scan"),
        # 6. Test execution
        ("pytest tests/ -v", "Unit tests"),
        # 7. Import verification
        (
            'python -c "import src.utils.logger; import src.data.websocket_manager"',
            "Module imports",
        ),
    ]

    results = []

    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))

    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:40} {status}")

    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] All CI checks passed! Ready for GitHub Actions.")
        return 0
    else:
        print("\n[WARNING] Some checks failed. Fix issues before pushing.")
        print("\nTo fix formatting issues:")
        print("  black src/ tests/ scripts/")
        print("  isort src/ tests/ scripts/")
        return 1


if __name__ == "__main__":
    exit(main())
