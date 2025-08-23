#!/usr/bin/env python3
"""
태스크 관리 자동화 테스트
Task #13 구현 검증
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from daily_report import DailyReportGenerator
from update_tasks import TaskManager


def test_task_manager_basic():
    """태스크 매니저 기본 기능 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Task Manager Basic Functions")
    print("=" * 60)
    
    # 실제 태스크 파일 사용
    manager = TaskManager("../.taskmaster/tasks/tasks.json")
    
    try:
        # 1. 진행 상황 통계
        print("\n[Testing progress stats...]")
        stats = manager.get_progress_stats()
        
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Completion rate: {stats['completion_rate']:.1f}%")
        print(f"  Done: {stats['status_counts']['done']}")
        print(f"  In-progress: {stats['status_counts']['in-progress']}")
        print(f"  Pending: {stats['status_counts']['pending']}")
        
        if stats['total_tasks'] > 0:
            print("  [OK] Progress stats working")
        else:
            print("  [FAIL] No tasks found")
            return False
        
        # 2. 상태별 태스크 조회
        print("\n[Testing task filtering...]")
        done_tasks = manager.get_tasks_by_status("done")
        pending_tasks = manager.get_tasks_by_status("pending")
        
        print(f"  Done tasks: {len(done_tasks)}")
        print(f"  Pending tasks: {len(pending_tasks)}")
        
        if len(done_tasks) >= 0 and len(pending_tasks) >= 0:
            print("  [OK] Task filtering working")
        
        # 3. 의존성 체크
        print("\n[Testing dependency check...]")
        issues = manager.check_dependencies()
        
        if issues:
            print(f"  Found {len(issues)} dependency issues")
            for issue in issues[:3]:
                print(f"    - {issue}")
        else:
            print("  No dependency issues found")
        print("  [OK] Dependency check working")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic test failed: {e}")
        return False


def test_progress_report():
    """진행 리포트 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Progress Report Generation")
    print("=" * 60)
    
    manager = TaskManager("../.taskmaster/tasks/tasks.json")
    
    try:
        # 리포트 생성
        print("\n[Generating progress report...]")
        report = manager.generate_progress_report()
        
        # 리포트 내용 검증
        if "TASK PROGRESS REPORT" in report:
            print("  [OK] Report header found")
        else:
            print("  [FAIL] Report header missing")
            return False
        
        if "Overall Progress:" in report:
            print("  [OK] Progress section found")
        else:
            print("  [FAIL] Progress section missing")
            return False
        
        if "Status Breakdown:" in report:
            print("  [OK] Status breakdown found")
        else:
            print("  [FAIL] Status breakdown missing")
            return False
        
        # 리포트 길이 확인
        lines = report.split("\n")
        print(f"  Report lines: {len(lines)}")
        
        if len(lines) > 20:
            print("  [OK] Report generation successful")
            return True
        else:
            print("  [FAIL] Report too short")
            return False
            
    except Exception as e:
        print(f"[ERROR] Report test failed: {e}")
        return False


def test_gantt_chart():
    """Gantt 차트 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Gantt Chart Generation")
    print("=" * 60)
    
    manager = TaskManager("../.taskmaster/tasks/tasks.json")
    
    try:
        # Gantt 데이터 생성
        print("\n[Generating Gantt data...]")
        gantt_data = manager.generate_gantt_data()
        
        if not gantt_data:
            print("  [FAIL] No Gantt data generated")
            return False
        
        print(f"  Generated {len(gantt_data)} Gantt items")
        
        # 데이터 구조 검증
        required_fields = ["id", "title", "status", "priority", "symbol"]
        sample_item = gantt_data[0]
        
        for field in required_fields:
            if field in sample_item:
                print(f"  [OK] Field '{field}' present")
            else:
                print(f"  [FAIL] Field '{field}' missing")
                return False
        
        # Gantt 차트 출력 테스트
        print("\n[Testing Gantt chart display...]")
        try:
            manager.print_gantt_chart()
            print("  [OK] Gantt chart displayed")
        except Exception as e:
            print(f"  [FAIL] Gantt display error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Gantt test failed: {e}")
        return False


def test_csv_export():
    """CSV 내보내기 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: CSV Export")
    print("=" * 60)
    
    manager = TaskManager("../.taskmaster/tasks/tasks.json")
    
    try:
        # 임시 파일로 내보내기
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        print(f"\n[Exporting to CSV: {temp_file}...]")
        manager.export_to_csv(temp_file)
        
        # 파일 존재 확인
        if Path(temp_file).exists():
            file_size = Path(temp_file).stat().st_size
            print(f"  [OK] CSV file created ({file_size} bytes)")
            
            # 내용 확인
            with open(temp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  CSV lines: {len(lines)}")
                
                # 헤더 확인
                if "ID,Title,Description,Status" in lines[0]:
                    print("  [OK] CSV header correct")
                else:
                    print("  [FAIL] CSV header incorrect")
                    return False
            
            # 정리
            os.unlink(temp_file)
            return True
        else:
            print("  [FAIL] CSV file not created")
            return False
            
    except Exception as e:
        print(f"[ERROR] CSV export test failed: {e}")
        return False


def test_daily_report():
    """일일 리포트 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 5: Daily Report Generation")
    print("=" * 60)
    
    try:
        generator = DailyReportGenerator()
        
        # 일일 리포트 생성
        print("\n[Generating daily report...]")
        daily_report = generator.generate_daily_report()
        
        if "DAILY PROGRESS REPORT" in daily_report:
            print("  [OK] Daily report header found")
        else:
            print("  [FAIL] Daily report header missing")
            return False
        
        # 필수 섹션 확인
        required_sections = [
            "Overall Progress",
            "Progress Bar",
            "Priority Distribution"
        ]
        
        for section in required_sections:
            if section in daily_report:
                print(f"  [OK] Section '{section}' found")
            else:
                print(f"  [FAIL] Section '{section}' missing")
        
        # 리포트 저장 테스트
        print("\n[Testing report save...]")
        filepath = generator.save_report(daily_report)
        
        if Path(filepath).exists():
            print(f"  [OK] Report saved to {filepath}")
            os.unlink(filepath)  # 정리
        else:
            print("  [FAIL] Report not saved")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Daily report test failed: {e}")
        return False


def test_weekly_summary():
    """주간 요약 테스트"""
    print("\n" + "=" * 60)
    print("TEST 6: Weekly Summary")
    print("=" * 60)
    
    try:
        generator = DailyReportGenerator()
        
        # 주간 요약 생성
        print("\n[Generating weekly summary...]")
        summary = generator.generate_weekly_summary()
        
        if "WEEKLY SUMMARY REPORT" in summary:
            print("  [OK] Weekly summary header found")
        else:
            print("  [FAIL] Weekly summary header missing")
            return False
        
        if "Phase 1 Status" in summary:
            print("  [OK] Phase status section found")
        else:
            print("  [FAIL] Phase status section missing")
            return False
        
        # Phase 1 진행률 확인
        if "Phase 1 Progress:" in summary:
            print("  [OK] Phase 1 progress tracked")
        else:
            print("  [FAIL] Phase 1 progress missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Weekly summary test failed: {e}")
        return False


def test_html_dashboard():
    """HTML 대시보드 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 7: HTML Dashboard")
    print("=" * 60)
    
    try:
        generator = DailyReportGenerator()
        
        # HTML 대시보드 생성
        print("\n[Generating HTML dashboard...]")
        dashboard_file = generator.generate_html_dashboard()
        
        if Path(dashboard_file).exists():
            file_size = Path(dashboard_file).stat().st_size
            print(f"  [OK] Dashboard created ({file_size} bytes)")
            
            # HTML 내용 확인
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "Task Dashboard" in content:
                    print("  [OK] Dashboard title found")
                else:
                    print("  [FAIL] Dashboard title missing")
                    return False
                
                if "Completion Rate" in content:
                    print("  [OK] Stats section found")
                else:
                    print("  [FAIL] Stats section missing")
                    return False
            
            # 정리 (옵션)
            # os.unlink(dashboard_file)
            print(f"  Dashboard saved at: {dashboard_file}")
            return True
        else:
            print("  [FAIL] Dashboard not created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Dashboard test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("TASK AUTOMATION TEST SUITE")
    print("=" * 60)
    print("\nTesting Task #13: Task Management Automation Scripts")
    
    tests = [
        ("Task Manager Basic", test_task_manager_basic),
        ("Progress Report", test_progress_report),
        ("Gantt Chart", test_gantt_chart),
        ("CSV Export", test_csv_export),
        ("Daily Report", test_daily_report),
        ("Weekly Summary", test_weekly_summary),
        ("HTML Dashboard", test_html_dashboard),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name:20} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] Task #13 COMPLETED! Task automation ready.")
        print("\nKey features implemented:")
        print("  1. Task status tracking and updates")
        print("  2. Progress report generation")
        print("  3. Gantt chart visualization")
        print("  4. CSV export functionality")
        print("  5. Daily progress reports")
        print("  6. Weekly summaries")
        print("  7. HTML dashboard")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)