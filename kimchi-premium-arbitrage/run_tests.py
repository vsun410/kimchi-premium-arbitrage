#!/usr/bin/env python
"""
Simple test runner for CI environment
"""
import sys
import os

def run_minimal_tests():
    """Run minimal tests without pytest"""
    print("="*50)
    print("Running minimal tests...")
    print("="*50)
    
    # Add backend to path
    sys.path.insert(0, 'backend')
    
    # Import and run test_minimal
    try:
        from backend.tests.test_minimal import test_always_passes, test_python_works
        
        print("Running test_always_passes...")
        test_always_passes()
        print("[PASS] test_always_passes passed")
        
        print("Running test_python_works...")
        test_python_works()
        print("[PASS] test_python_works passed")
        
        print("\n[SUCCESS] All minimal tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_minimal_tests())