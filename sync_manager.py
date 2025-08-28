"""
GitHub-Notion 통합 동기화 매니저
Task Master, GitHub, Notion을 연동하여 프로젝트 진행 상황을 자동 동기화
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()


class ProjectSyncManager:
    """프로젝트 동기화 관리자"""
    
    def __init__(self):
        """초기화"""
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.project_root = Path("C:/workshop/kimchi-premium-arbitrage")
        self.config_path = self.project_root / "executive_control" / "kimp_notion_config.json"
        self.load_config()
        
    def load_config(self):
        """Notion 설정 로드"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[ERROR] Config load failed: {e}")
            self.config = {}
    
    def get_taskmaster_status(self) -> Dict:
        """Task Master 상태 가져오기"""
        try:
            # Task Master list 명령 실행
            result = subprocess.run(
                ["task-master", "list", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except Exception as e:
            print(f"[ERROR] Task Master status failed: {e}")
            return {}
    
    def get_git_status(self) -> Dict:
        """Git 상태 가져오기"""
        try:
            # 현재 브랜치
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            # 커밋 수
            commit_count = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            # 변경된 파일 수
            changed_files = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip().split('\n')
            
            return {
                "branch": branch,
                "total_commits": int(commit_count),
                "changed_files": len([f for f in changed_files if f])
            }
        except Exception as e:
            print(f"[ERROR] Git status failed: {e}")
            return {}
    
    def update_notion_task(self, task_id: str, status: str, progress: float):
        """Notion 태스크 업데이트"""
        try:
            # Task ID로 Notion 페이지 검색
            response = self.notion.databases.query(
                database_id=self.config.get('tasks_db'),
                filter={
                    "property": "Task ID",
                    "rich_text": {
                        "equals": task_id
                    }
                }
            )
            
            if response['results']:
                page_id = response['results'][0]['id']
                
                # 상태 및 진행률 업데이트
                self.notion.pages.update(
                    page_id=page_id,
                    properties={
                        "Status": {"select": {"name": self._map_status(status)}},
                        "Progress": {"number": progress}
                    }
                )
                print(f"[OK] Updated Notion task {task_id}: {status} ({progress*100:.0f}%)")
        except Exception as e:
            print(f"[ERROR] Notion update failed for task {task_id}: {e}")
    
    def _map_status(self, taskmaster_status: str) -> str:
        """Task Master 상태를 Notion 상태로 매핑"""
        mapping = {
            "pending": "Todo",
            "in-progress": "In Progress",
            "done": "Done",
            "blocked": "Blocked",
            "deferred": "On Hold",
            "cancelled": "Cancelled"
        }
        return mapping.get(taskmaster_status, "Todo")
    
    def on_task_complete(self, task_id: str):
        """태스크 완료 시 동기화"""
        print(f"\n[INFO] Syncing completed task #{task_id}")
        
        # 1. Task Master 상태 업데이트
        subprocess.run([
            "task-master", "set-status",
            f"--id={task_id}",
            "--status=done"
        ], check=True)
        
        # 2. Git 커밋
        commit_message = f"feat: [Task #{task_id}] Complete implementation"
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"[OK] Git committed: {commit_message}")
        
        # 3. Notion 업데이트
        self.update_notion_task(task_id, "done", 1.0)
        
        # 4. 진행률 계산
        self.update_progress_dashboard()
    
    def update_progress_dashboard(self):
        """전체 진행률 대시보드 업데이트"""
        try:
            # Task Master 전체 상태
            tm_status = self.get_taskmaster_status()
            git_status = self.get_git_status()
            
            # 진행률 계산
            if tm_status:
                total_tasks = tm_status.get('total', 0)
                done_tasks = tm_status.get('done', 0)
                progress = done_tasks / total_tasks if total_tasks > 0 else 0
                
                # Notion 프로젝트 페이지 업데이트
                self.notion.blocks.children.append(
                    self.config.get('project_page'),
                    children=[{
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{
                                "text": {
                                    "content": f"[AUTO-SYNC] {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                                              f"Progress: {progress*100:.1f}% ({done_tasks}/{total_tasks})\n"
                                              f"Branch: {git_status.get('branch', 'unknown')}\n"
                                              f"Commits: {git_status.get('total_commits', 0)}"
                                }
                            }],
                            "icon": {"emoji": "📊"}
                        }
                    }]
                )
                print(f"[OK] Dashboard updated: {progress*100:.1f}% complete")
        except Exception as e:
            print(f"[ERROR] Dashboard update failed: {e}")
    
    def daily_sync(self):
        """일일 동기화 실행"""
        print("="*60)
        print(f"   Daily Sync - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        # 1. Task Master 상태 동기화
        tm_status = self.get_taskmaster_status()
        if tm_status:
            print(f"[INFO] Task Master: {tm_status.get('done', 0)}/{tm_status.get('total', 0)} tasks complete")
        
        # 2. Git 상태 확인
        git_status = self.get_git_status()
        if git_status:
            print(f"[INFO] Git: Branch '{git_status['branch']}' with {git_status['changed_files']} changes")
        
        # 3. Notion 대시보드 업데이트
        self.update_progress_dashboard()
        
        # 4. 보고서 생성
        self.generate_daily_report()
        
        print("\n[SUCCESS] Daily sync completed")
    
    def generate_daily_report(self):
        """일일 보고서 생성"""
        try:
            # 오늘 완료된 태스크 조회
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 보고서 파일 생성
            report_path = self.project_root / "reports" / f"daily_{today}.md"
            report_path.parent.mkdir(exist_ok=True)
            
            report_content = f"""# Daily Report - {today}

## Task Master Status
- Total Tasks: {self.get_taskmaster_status().get('total', 0)}
- Completed: {self.get_taskmaster_status().get('done', 0)}
- In Progress: {self.get_taskmaster_status().get('in_progress', 0)}
- Pending: {self.get_taskmaster_status().get('pending', 0)}

## Git Status
- Current Branch: {self.get_git_status().get('branch', 'unknown')}
- Total Commits: {self.get_git_status().get('total_commits', 0)}
- Changed Files: {self.get_git_status().get('changed_files', 0)}

## Next Actions
- Review pending tasks
- Check blocked dependencies
- Update project roadmap

---
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            print(f"[OK] Daily report saved: {report_path}")
            
        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
    
    def sync_task_to_notion(self, task_id: str):
        """특정 태스크를 Notion과 동기화"""
        try:
            # Task Master에서 태스크 정보 가져오기
            result = subprocess.run(
                ["task-master", "show", task_id, "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            task_data = json.loads(result.stdout)
            
            # Notion 업데이트
            status = task_data.get('status', 'pending')
            progress = 1.0 if status == 'done' else 0.5 if status == 'in-progress' else 0.0
            
            self.update_notion_task(task_id, status, progress)
            
        except Exception as e:
            print(f"[ERROR] Task sync failed for {task_id}: {e}")


def main():
    """메인 실행 함수"""
    import sys
    
    manager = ProjectSyncManager()
    
    if len(sys.argv) < 2:
        print("Usage: python sync_manager.py [command]")
        print("Commands:")
        print("  daily_sync - Run daily synchronization")
        print("  task_complete <id> - Mark task as complete and sync")
        print("  sync_task <id> - Sync specific task to Notion")
        print("  update_dashboard - Update progress dashboard")
        return
    
    command = sys.argv[1]
    
    if command == "daily_sync":
        manager.daily_sync()
    elif command == "task_complete" and len(sys.argv) > 2:
        manager.on_task_complete(sys.argv[2])
    elif command == "sync_task" and len(sys.argv) > 2:
        manager.sync_task_to_notion(sys.argv[2])
    elif command == "update_dashboard":
        manager.update_progress_dashboard()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()