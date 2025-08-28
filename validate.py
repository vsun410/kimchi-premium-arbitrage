#!/usr/bin/env python3
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
