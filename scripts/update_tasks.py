#!/usr/bin/env python3
"""
태스크 관리 자동화 스크립트
Task #13: 태스크 상태 업데이트 및 진행상황 추적
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class TaskManager:
    """태스크 관리자"""
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
        """
        초기화
        
        Args:
            tasks_file: 태스크 파일 경로
        """
        self.tasks_file = Path(tasks_file)
        self.tasks_data = self._load_tasks()
        
    def _load_tasks(self) -> Dict:
        """태스크 파일 로드"""
        if not self.tasks_file.exists():
            print(f"[ERROR] Tasks file not found: {self.tasks_file}")
            return {}
        
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save_tasks(self):
        """태스크 파일 저장"""
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(self.tasks_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Tasks saved to {self.tasks_file}")
    
    def update_task_status(self, task_id: int, new_status: str) -> bool:
        """
        태스크 상태 업데이트
        
        Args:
            task_id: 태스크 ID
            new_status: 새 상태 (pending, in-progress, done, blocked)
            
        Returns:
            성공 여부
        """
        valid_statuses = ["pending", "in-progress", "done", "blocked", "cancelled"]
        if new_status not in valid_statuses:
            print(f"[ERROR] Invalid status: {new_status}")
            print(f"Valid statuses: {', '.join(valid_statuses)}")
            return False
        
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        
        for task in tasks:
            if task["id"] == task_id:
                old_status = task["status"]
                task["status"] = new_status
                
                # 메타데이터 업데이트
                self.tasks_data["master"]["metadata"]["updated"] = datetime.now().isoformat() + "Z"
                
                self._save_tasks()
                print(f"[SUCCESS] Task #{task_id} status updated: {old_status} -> {new_status}")
                return True
        
        print(f"[ERROR] Task #{task_id} not found")
        return False
    
    def get_progress_stats(self) -> Dict:
        """
        진행 상황 통계
        
        Returns:
            통계 딕셔너리
        """
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        
        if not tasks:
            return {}
        
        total = len(tasks)
        status_counts = {
            "pending": 0,
            "in-progress": 0,
            "done": 0,
            "blocked": 0,
            "cancelled": 0,
        }
        
        priority_counts = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        
        for task in tasks:
            status = task.get("status", "pending")
            priority = task.get("priority", "medium")
            
            if status in status_counts:
                status_counts[status] += 1
            
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        completion_rate = (status_counts["done"] / total) * 100 if total > 0 else 0
        
        return {
            "total_tasks": total,
            "completion_rate": completion_rate,
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "last_updated": self.tasks_data.get("master", {}).get("metadata", {}).get("updated", "N/A"),
        }
    
    def generate_progress_report(self) -> str:
        """
        진행 상황 리포트 생성
        
        Returns:
            리포트 문자열
        """
        stats = self.get_progress_stats()
        
        if not stats:
            return "No tasks found."
        
        report = []
        report.append("=" * 60)
        report.append("TASK PROGRESS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 전체 진행률
        report.append(f"Overall Progress: {stats['completion_rate']:.1f}%")
        report.append(f"Total Tasks: {stats['total_tasks']}")
        report.append("")
        
        # 상태별 통계
        report.append("Status Breakdown:")
        for status, count in stats['status_counts'].items():
            percentage = (count / stats['total_tasks']) * 100
            bar_length = int(percentage / 2)  # 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report.append(f"  {status:12} [{bar}] {count:3d} ({percentage:5.1f}%)")
        
        report.append("")
        
        # 우선순위별 통계
        report.append("Priority Distribution:")
        for priority, count in stats['priority_counts'].items():
            percentage = (count / stats['total_tasks']) * 100
            report.append(f"  {priority:8} : {count:3d} tasks ({percentage:5.1f}%)")
        
        report.append("")
        
        # 현재 진행 중인 태스크
        in_progress = self.get_tasks_by_status("in-progress")
        if in_progress:
            report.append("Currently In Progress:")
            for task in in_progress:
                report.append(f"  - Task #{task['id']}: {task['title']}")
        
        # 다음 대기 태스크
        pending = self.get_tasks_by_status("pending")
        if pending:
            report.append("")
            report.append(f"Next Pending Tasks (showing first 3):")
            for task in pending[:3]:
                report.append(f"  - Task #{task['id']}: {task['title']} [Priority: {task.get('priority', 'medium')}]")
        
        # 블록된 태스크
        blocked = self.get_tasks_by_status("blocked")
        if blocked:
            report.append("")
            report.append("Blocked Tasks:")
            for task in blocked:
                report.append(f"  - Task #{task['id']}: {task['title']}")
        
        report.append("")
        report.append(f"Last Updated: {stats['last_updated']}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_tasks_by_status(self, status: str) -> List[Dict]:
        """
        상태별 태스크 조회
        
        Args:
            status: 태스크 상태
            
        Returns:
            태스크 리스트
        """
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        return [task for task in tasks if task.get("status") == status]
    
    def generate_gantt_data(self) -> List[Dict]:
        """
        Gantt 차트용 데이터 생성
        
        Returns:
            Gantt 데이터 리스트
        """
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        gantt_data = []
        
        # 각 태스크를 Gantt 형식으로 변환
        for i, task in enumerate(tasks):
            gantt_item = {
                "id": task["id"],
                "title": task["title"],
                "status": task["status"],
                "priority": task.get("priority", "medium"),
                "dependencies": task.get("dependencies", []),
                "start_index": i,
                "duration": 1,  # 기본 1단위
            }
            
            # 우선순위에 따른 색상
            if task.get("priority") == "high":
                gantt_item["color"] = "red"
            elif task.get("priority") == "low":
                gantt_item["color"] = "gray"
            else:
                gantt_item["color"] = "blue"
            
            # 상태에 따른 표시
            if task["status"] == "done":
                gantt_item["symbol"] = "[DONE]"
            elif task["status"] == "in-progress":
                gantt_item["symbol"] = "[PROG]"
            elif task["status"] == "blocked":
                gantt_item["symbol"] = "[BLCK]"
            else:
                gantt_item["symbol"] = "[PEND]"
            
            gantt_data.append(gantt_item)
        
        return gantt_data
    
    def print_gantt_chart(self):
        """Gantt 차트 출력 (텍스트 기반)"""
        gantt_data = self.generate_gantt_data()
        
        if not gantt_data:
            print("No tasks to display")
            return
        
        print("\n" + "=" * 80)
        print("GANTT CHART (Text Visualization)")
        print("=" * 80)
        print(f"{'ID':<5} {'Title':<40} {'Priority':<8} {'Status':<12} Progress")
        print("-" * 80)
        
        for item in gantt_data:
            # 진행 바 생성
            if item["status"] == "done":
                progress_bar = "####################"  # 100%
            elif item["status"] == "in-progress":
                progress_bar = "##########----------"  # 50%
            elif item["status"] == "blocked":
                progress_bar = "####----------------"  # 20%
            else:
                progress_bar = "--------------------"  # 0%
            
            # 제목 줄임
            title = item["title"][:37] + "..." if len(item["title"]) > 40 else item["title"]
            
            print(
                f"{item['id']:<5} "
                f"{title:<40} "
                f"{item['priority']:<8} "
                f"{item['status']:<12} "
                f"{item['symbol']} {progress_bar}"
            )
            
            # 의존성 표시
            if item["dependencies"]:
                deps = ", ".join(f"#{d}" for d in item["dependencies"])
                print(f"      └─ Depends on: {deps}")
        
        print("=" * 80)
        print("\nLegend: [DONE] Complete | [PROG] In Progress | [BLCK] Blocked | [PEND] Pending")
        print("=" * 80)
    
    def export_to_csv(self, output_file: str = "task_report.csv"):
        """
        CSV로 내보내기
        
        Args:
            output_file: 출력 파일명
        """
        import csv
        
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 헤더
            writer.writerow([
                "ID", "Title", "Description", "Status", "Priority",
                "Dependencies", "Details"
            ])
            
            # 데이터
            for task in tasks:
                writer.writerow([
                    task["id"],
                    task["title"],
                    task["description"],
                    task["status"],
                    task.get("priority", "medium"),
                    ", ".join(str(d) for d in task.get("dependencies", [])),
                    task.get("details", "")[:100]  # 상세 내용은 100자까지
                ])
        
        print(f"[SUCCESS] Tasks exported to {output_file}")
    
    def check_dependencies(self) -> List[str]:
        """
        의존성 체크
        
        Returns:
            문제 리스트
        """
        tasks = self.tasks_data.get("master", {}).get("tasks", [])
        issues = []
        
        # 태스크 ID 맵 생성
        task_ids = {task["id"]: task for task in tasks}
        
        for task in tasks:
            task_id = task["id"]
            deps = task.get("dependencies", [])
            
            for dep_id in deps:
                # 숫자로 변환 시도
                try:
                    dep_id = int(dep_id)
                except:
                    pass
                
                # 의존성 존재 확인
                if dep_id not in task_ids:
                    issues.append(f"Task #{task_id} depends on non-existent task #{dep_id}")
                    continue
                
                # 의존성 상태 확인
                dep_task = task_ids[dep_id]
                if task["status"] == "done" and dep_task["status"] != "done":
                    issues.append(
                        f"Task #{task_id} is done but dependency #{dep_id} is {dep_task['status']}"
                    )
                
                if task["status"] == "in-progress" and dep_task["status"] not in ["done", "in-progress"]:
                    issues.append(
                        f"Task #{task_id} is in-progress but dependency #{dep_id} is {dep_task['status']}"
                    )
        
        return issues


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Management Automation")
    parser.add_argument("command", choices=[
        "status", "update", "report", "gantt", "export", "check"
    ], help="Command to execute")
    parser.add_argument("--id", type=int, help="Task ID for update")
    parser.add_argument("--status", choices=[
        "pending", "in-progress", "done", "blocked", "cancelled"
    ], help="New status for update")
    parser.add_argument("--output", default="task_report.csv", help="Output file for export")
    parser.add_argument("--file", default="../.taskmaster/tasks/tasks.json", 
                       help="Tasks file path")
    
    args = parser.parse_args()
    
    # 태스크 매니저 초기화
    manager = TaskManager(args.file)
    
    if args.command == "status":
        # 진행 상황 표시
        stats = manager.get_progress_stats()
        print(f"\n[PROGRESS] {stats['completion_rate']:.1f}% Complete")
        print(f"Done: {stats['status_counts']['done']}/{stats['total_tasks']}")
        
    elif args.command == "update":
        # 태스크 상태 업데이트
        if not args.id or not args.status:
            print("[ERROR] --id and --status required for update")
            sys.exit(1)
        
        success = manager.update_task_status(args.id, args.status)
        if not success:
            sys.exit(1)
    
    elif args.command == "report":
        # 진행 리포트 생성
        report = manager.generate_progress_report()
        print(report)
        
        # 파일로도 저장
        report_file = f"task_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[INFO] Report saved to {report_file}")
    
    elif args.command == "gantt":
        # Gantt 차트 표시
        manager.print_gantt_chart()
    
    elif args.command == "export":
        # CSV 내보내기
        manager.export_to_csv(args.output)
    
    elif args.command == "check":
        # 의존성 체크
        issues = manager.check_dependencies()
        if issues:
            print("\n[WARNING] Dependency Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n[OK] No dependency issues found")


if __name__ == "__main__":
    main()