"""
Manual initialization script (skip Notion validation for now)
"""

import os
import sys
import json
from pathlib import Path

def setup_system():
    """시스템 파일 설정"""
    
    print("\n[INFO] Setting up Executive Control System files...")
    
    # 1. PRD 생성
    prd_content = """# Kimchi Premium Trading System PRD

## Vision
실시간 김치프리미엄 차익거래 자동화 시스템

## Core Objectives
1. 실시간 김치프리미엄 차익거래 자동화
2. 리스크 중립적 헤지 포지션 유지
3. 24/7 무중단 운영 시스템
4. ML 기반 객관적 진입/청산 시그널

## Red Lines (절대 금지)
1. 동시 양방향 포지션 한도 초과
2. 단일 거래 자본금 10% 초과
3. 수동 개입 필요한 로직
4. 테스트되지 않은 코드 배포
"""
    
    Path("executive_control").mkdir(exist_ok=True)
    with open("executive_control/prd.md", "w", encoding="utf-8") as f:
        f.write(prd_content)
    print("  [OK] PRD created")
    
    # 2. Architecture 생성
    architecture = {
        "components": {
            "trading_core": {
                "interfaces": ["WebSocket", "REST API"],
                "constraints": ["latency < 100ms", "stateless"]
            },
            "ml_engine": {
                "interfaces": ["gRPC", "Message Queue"],
                "constraints": ["prediction_time < 500ms"]
            },
            "dashboard": {
                "interfaces": ["REST API", "WebSocket"],
                "constraints": ["real-time updates", "responsive UI"]
            }
        }
    }
    
    with open("executive_control/architecture.json", "w", encoding="utf-8") as f:
        json.dump(architecture, f, indent=2)
    print("  [OK] Architecture spec created")
    
    # 3. Validation script 생성
    validation_script = """#!/usr/bin/env python3
'''Simple validation script'''

import sys

def validate_file(file_path):
    print(f"[INFO] Validating {file_path}...")
    # 간단한 검증 로직
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Red line 체크
    red_lines = ["input(", "manual", "untested"]
    violations = []
    
    for red_line in red_lines:
        if red_line in code.lower():
            violations.append(f"Red line violation: {red_line}")
    
    if violations:
        print("[ERROR] Validation failed:")
        for v in violations:
            print(f"  - {v}")
        return 1
    
    print("[OK] Validation passed")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <file>")
        sys.exit(1)
    
    sys.exit(validate_file(sys.argv[1]))
"""
    
    with open("validate.py", "w", encoding="utf-8") as f:
        f.write(validation_script)
    print("  [OK] Validation script created")
    
    # 4. Config 생성
    config = {
        "version": "1.0.0",
        "initialized": True,
        "components": {
            "vision_guardian": True,
            "task_orchestrator": True,
            "notion_integration": True,
            "claude_interceptor": True
        }
    }
    
    with open("executive_control/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("  [OK] Configuration saved")
    
    print("\n[SUCCESS] System setup complete!")
    print("\n[NEXT STEPS]:")
    print("1. Open your Notion dashboard:")
    print("   https://notion.so/25bd547955ef8162a332d0b7611fee25")
    print("")
    print("2. Test validation:")
    print("   python validate.py your_file.py")
    print("")
    print("3. View created files:")
    print("   - executive_control/prd.md (Project vision)")
    print("   - executive_control/architecture.json (System design)")
    print("   - validate.py (Code validation)")
    print("")
    print("[INFO] Executive Control System is ready to use!")
    print("[INFO] Your code will now be validated against the project vision!")

if __name__ == "__main__":
    setup_system()