#!/usr/bin/env python3
"""
일일 진행 리포트 자동 생성 스크립트
매일 실행하여 프로젝트 진행 상황을 추적
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from update_tasks import TaskManager


class DailyReportGenerator:
    """일일 리포트 생성기"""
    
    def __init__(self, tasks_file: str = "../.taskmaster/tasks/tasks.json"):
        """초기화"""
        self.manager = TaskManager(tasks_file)
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
    def generate_daily_report(self) -> str:
        """
        일일 리포트 생성
        
        Returns:
            리포트 내용
        """
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        report = []
        report.append("=" * 70)
        report.append("DAILY PROGRESS REPORT")
        report.append("=" * 70)
        report.append(f"Date: {today.strftime('%Y-%m-%d')}")
        report.append(f"Time: {today.strftime('%H:%M:%S')}")
        report.append("")
        
        # 1. 전체 진행 상황
        stats = self.manager.get_progress_stats()
        report.append("Overall Progress")
        report.append("-" * 40)
        report.append(f"Completion Rate: {stats['completion_rate']:.1f}%")
        report.append(f"Total Tasks: {stats['total_tasks']}")
        report.append(f"Completed: {stats['status_counts']['done']}")
        report.append(f"In Progress: {stats['status_counts']['in-progress']}")
        report.append(f"Pending: {stats['status_counts']['pending']}")
        report.append(f"Blocked: {stats['status_counts']['blocked']}")
        report.append("")
        
        # 2. 진행률 그래프
        report.append("Progress Bar")
        report.append("-" * 40)
        progress_percentage = int(stats['completion_rate'])
        filled = int(progress_percentage / 2)  # 50 chars total
        bar = "#" * filled + "-" * (50 - filled)
        report.append(f"[{bar}] {progress_percentage}%")
        report.append("")
        
        # 3. 현재 진행 중인 작업
        in_progress = self.manager.get_tasks_by_status("in-progress")
        if in_progress:
            report.append("Currently In Progress")
            report.append("-" * 40)
            for task in in_progress:
                report.append(f"- Task #{task['id']}: {task['title']}")
                if task.get('description'):
                    report.append(f"    - {task['description']}")
            report.append("")
        
        # 4. 오늘 완료한 작업 (실제로는 최근 완료 표시)
        completed = self.manager.get_tasks_by_status("done")
        if completed:
            recent_completed = completed[-3:] if len(completed) > 3 else completed
            report.append("Recently Completed")
            report.append("-" * 40)
            for task in recent_completed:
                report.append(f"- Task #{task['id']}: {task['title']}")
            report.append("")
        
        # 5. 다음 대기 작업
        pending = self.manager.get_tasks_by_status("pending")
        if pending:
            report.append("Next Pending Tasks")
            report.append("-" * 40)
            for task in pending[:5]:  # 최대 5개
                priority = task.get('priority', 'medium')
                priority_emoji = {
                    'high': '[HIGH]',
                    'medium': '[MED]',
                    'low': '[LOW]'
                }.get(priority, '[N/A]')
                report.append(f"{priority_emoji} Task #{task['id']}: {task['title']}")
                
                # 의존성 표시
                deps = task.get('dependencies', [])
                if deps:
                    report.append(f"    - Depends on: {', '.join(f'#{d}' for d in deps)}")
            report.append("")
        
        # 6. 블록 이슈
        blocked = self.manager.get_tasks_by_status("blocked")
        if blocked:
            report.append("Blocked Tasks")
            report.append("-" * 40)
            for task in blocked:
                report.append(f"- Task #{task['id']}: {task['title']}")
            report.append("")
        
        # 7. 의존성 체크
        issues = self.manager.check_dependencies()
        if issues:
            report.append("Dependency Issues")
            report.append("-" * 40)
            for issue in issues[:5]:  # 최대 5개
                report.append(f"- {issue}")
            report.append("")
        
        # 8. 우선순위별 분포
        report.append("Priority Distribution")
        report.append("-" * 40)
        for priority in ['high', 'medium', 'low']:
            count = stats['priority_counts'].get(priority, 0)
            percentage = (count / stats['total_tasks']) * 100 if stats['total_tasks'] > 0 else 0
            emoji = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}.get(priority, '[N/A]')
            report.append(f"{emoji} {priority.capitalize():8}: {count:3d} tasks ({percentage:5.1f}%)")
        report.append("")
        
        # 9. 예상 완료 시간 (간단한 추정)
        if stats['status_counts']['done'] > 0 and stats['status_counts']['pending'] > 0:
            # 하루에 평균 2개 태스크 완료 가정
            avg_tasks_per_day = 2
            remaining_tasks = stats['status_counts']['pending'] + stats['status_counts']['in-progress']
            estimated_days = remaining_tasks / avg_tasks_per_day
            estimated_date = today + timedelta(days=estimated_days)
            
            report.append("Estimated Completion")
            report.append("-" * 40)
            report.append(f"Remaining Tasks: {remaining_tasks}")
            report.append(f"Estimated Days: {estimated_days:.1f}")
            report.append(f"Estimated Date: {estimated_date.strftime('%Y-%m-%d')}")
            report.append("")
        
        # 10. 권장 사항
        report.append("Recommendations")
        report.append("-" * 40)
        
        if stats['status_counts']['blocked'] > 0:
            report.append("- Resolve blocked tasks to improve progress")
        
        if stats['status_counts']['in-progress'] > 3:
            report.append("- Consider completing in-progress tasks before starting new ones")
        
        if stats['completion_rate'] < 50:
            report.append("- Focus on high-priority tasks to accelerate progress")
        elif stats['completion_rate'] > 80:
            report.append("- Excellent progress! Consider planning next phase")
        
        if not report[-1].startswith("-"):
            report.append("- Keep up the good work!")
        
        report.append("")
        report.append("=" * 70)
        report.append(f"Report generated at {today.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, content: str) -> str:
        """
        리포트 저장
        
        Args:
            content: 리포트 내용
            
        Returns:
            저장된 파일 경로
        """
        today = datetime.now()
        filename = f"daily_report_{today.strftime('%Y%m%d')}.txt"
        filepath = self.report_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return str(filepath)
    
    def generate_weekly_summary(self) -> str:
        """
        주간 요약 생성
        
        Returns:
            주간 요약 리포트
        """
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        
        summary = []
        summary.append("=" * 70)
        summary.append("WEEKLY SUMMARY REPORT")
        summary.append("=" * 70)
        summary.append(f"Week: {week_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
        summary.append("")
        
        stats = self.manager.get_progress_stats()
        
        # 주간 진행률
        summary.append("Week Overview")
        summary.append("-" * 40)
        summary.append(f"Total Completion: {stats['completion_rate']:.1f}%")
        summary.append(f"Tasks Completed: {stats['status_counts']['done']}")
        summary.append(f"Tasks Remaining: {stats['status_counts']['pending']}")
        summary.append("")
        
        # Phase별 상태 (Phase 1 기준)
        summary.append("Phase 1 Status")
        summary.append("-" * 40)
        phase1_total = 13  # Phase 1 총 태스크 수
        phase1_done = min(stats['status_counts']['done'], phase1_total)
        phase1_percentage = (phase1_done / phase1_total) * 100
        summary.append(f"Phase 1 Progress: {phase1_done}/{phase1_total} ({phase1_percentage:.1f}%)")
        
        if phase1_percentage == 100:
            summary.append("Phase 1 COMPLETE! Ready for Phase 2")
        elif phase1_percentage >= 90:
            summary.append("Almost there! Just a few tasks remaining")
        elif phase1_percentage >= 70:
            summary.append("Good progress! Keep pushing")
        else:
            summary.append("Still working through initial phase")
        
        summary.append("")
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def generate_html_dashboard(self) -> str:
        """
        HTML 대시보드 생성
        
        Returns:
            HTML 파일 경로
        """
        stats = self.manager.get_progress_stats()
        gantt_data = self.manager.generate_gantt_data()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Progress Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .task-list {{
            margin: 20px 0;
        }}
        .task-item {{
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #667eea;
            background: #f5f5f5;
            border-radius: 5px;
        }}
        .priority-high {{ border-left-color: #e74c3c; }}
        .priority-medium {{ border-left-color: #f39c12; }}
        .priority-low {{ border-left-color: #27ae60; }}
        .status-done {{ background: #d4edda; }}
        .status-in-progress {{ background: #fff3cd; }}
        .status-blocked {{ background: #f8d7da; }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Kimchi Premium Arbitrage - Task Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Completion Rate</div>
                <div class="stat-value">{stats['completion_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Tasks</div>
                <div class="stat-value">{stats['total_tasks']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Completed</div>
                <div class="stat-value">{stats['status_counts']['done']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">In Progress</div>
                <div class="stat-value">{stats['status_counts']['in-progress']}</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {stats['completion_rate']}%">
                {stats['completion_rate']:.1f}%
            </div>
        </div>
        
        <h2>Task Overview</h2>
        <div class="task-list">
"""
        
        # 태스크 리스트 추가
        for task in gantt_data[:20]:  # 최대 20개 표시
            priority_class = f"priority-{task.get('priority', 'medium')}"
            status_class = f"status-{task['status']}"
            
            html_content += f"""
            <div class="task-item {priority_class} {status_class}">
                <strong>Task #{task['id']}</strong>: {task['title']} 
                <span style="float: right">{task['symbol']} {task['status']}</span>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # HTML 파일 저장
        dashboard_file = self.report_dir / "dashboard.html"
        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return str(dashboard_file)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Report Generator")
    parser.add_argument("--type", choices=["daily", "weekly", "dashboard"],
                       default="daily", help="Report type")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (no output)")
    
    args = parser.parse_args()
    
    generator = DailyReportGenerator()
    
    if args.type == "daily":
        report = generator.generate_daily_report()
        if not args.quiet:
            print(report)
        
        if args.save:
            filepath = generator.save_report(report)
            print(f"\n[SUCCESS] Daily report saved to: {filepath}")
    
    elif args.type == "weekly":
        summary = generator.generate_weekly_summary()
        if not args.quiet:
            print(summary)
        
        if args.save:
            filepath = generator.save_report(summary)
            print(f"\n[SUCCESS] Weekly summary saved to: {filepath}")
    
    elif args.type == "dashboard":
        filepath = generator.generate_html_dashboard()
        print(f"[SUCCESS] HTML dashboard generated: {filepath}")
        
        # 브라우저로 열기 옵션
        import webbrowser
        try:
            webbrowser.open(f"file://{Path(filepath).absolute()}")
            print("[INFO] Dashboard opened in browser")
        except:
            print("[INFO] Please open the dashboard manually")


if __name__ == "__main__":
    main()