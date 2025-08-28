#!/usr/bin/env python3
"""
Notion Workspace Setup Guide
Helps create the required databases and pages for the project
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime
import json

class NotionWorkspaceSetup:
    """Setup Notion workspace structure"""
    
    def __init__(self):
        # Load environment
        env_notion_path = Path('.env.notion')
        if env_notion_path.exists():
            load_dotenv(env_notion_path)
        
        token = os.getenv('NOTION_TOKEN')
        if not token:
            raise ValueError("NOTION_TOKEN not found")
        
        self.client = Client(auth=token)
        self.database_ids = {}
    
    def create_project_page(self):
        """Create main project page"""
        print("\n[INFO] 메인 프로젝트 페이지를 생성하려면:")
        print("1. Notion에서 새 페이지 생성")
        print("2. 제목: 'Kimchi Premium Trading System'")
        print("3. 페이지 우측 상단 '...' 클릭 → 'Add connections' → 'kimp' Integration 추가")
        print("\n페이지를 생성하고 Integration을 추가하셨나요? (y/n): ", end="")
        
        if input().lower() != 'y':
            return None
        
        # Search for the page
        response = self.client.search(
            query="Kimchi Premium Trading System",
            filter={"value": "page", "property": "object"}
        )
        
        if response['results']:
            page_id = response['results'][0]['id']
            print(f"[SUCCESS] 메인 페이지 발견: {page_id}")
            return page_id
        else:
            print("[WARNING] 페이지를 찾을 수 없습니다. 수동으로 ID를 입력해주세요.")
            return None
    
    def create_database_structure(self, parent_id):
        """Create all required databases"""
        
        databases = {
            "Tasks": {
                "description": "작업 관리 데이터베이스",
                "properties": {
                    "Title": {"title": {}},
                    "Status": {
                        "status": {
                            "options": [
                                {"name": "Not Started", "color": "gray"},
                                {"name": "In Progress", "color": "blue"},
                                {"name": "Review", "color": "yellow"},
                                {"name": "Done", "color": "green"},
                                {"name": "Blocked", "color": "red"}
                            ]
                        }
                    },
                    "Priority": {
                        "select": {
                            "options": [
                                {"name": "P0", "color": "red"},
                                {"name": "P1", "color": "orange"},
                                {"name": "P2", "color": "yellow"},
                                {"name": "P3", "color": "green"}
                            ]
                        }
                    },
                    "Phase": {
                        "select": {
                            "options": [
                                {"name": "Phase 1", "color": "purple"},
                                {"name": "Phase 2", "color": "blue"},
                                {"name": "Phase 3", "color": "green"},
                                {"name": "Phase 4", "color": "yellow"},
                                {"name": "Phase 5", "color": "orange"},
                                {"name": "Phase 6", "color": "red"}
                            ]
                        }
                    },
                    "Assignee": {"people": {}},
                    "Due Date": {"date": {}},
                    "Dependencies": {"relation": {}},
                    "Acceptance Criteria": {"rich_text": {}},
                    "Technical Spec": {"rich_text": {}},
                    "Implementation Notes": {"rich_text": {}}
                }
            },
            "Architecture": {
                "description": "시스템 아키텍처 문서",
                "properties": {
                    "Component": {"title": {}},
                    "Type": {
                        "select": {
                            "options": [
                                {"name": "Service", "color": "blue"},
                                {"name": "Database", "color": "green"},
                                {"name": "API", "color": "purple"},
                                {"name": "Library", "color": "orange"}
                            ]
                        }
                    },
                    "Description": {"rich_text": {}},
                    "Dependencies": {"multi_select": {}},
                    "Mermaid": {"rich_text": {}},
                    "Diagram": {"url": {}},
                    "Owner": {"people": {}}
                }
            },
            "Test Reports": {
                "description": "테스트 결과 리포트",
                "properties": {
                    "Title": {"title": {}},
                    "Task": {"relation": {}},
                    "Date": {"date": {}},
                    "Coverage": {"number": {"format": "percent"}},
                    "Passed": {"number": {}},
                    "Failed": {"number": {}},
                    "Status": {
                        "select": {
                            "options": [
                                {"name": "Pass", "color": "green"},
                                {"name": "Fail", "color": "red"},
                                {"name": "Partial", "color": "yellow"}
                            ]
                        }
                    },
                    "Report": {"rich_text": {}}
                }
            },
            "Performance Metrics": {
                "description": "성능 지표 추적",
                "properties": {
                    "Date": {"title": {}},
                    "Sharpe Ratio": {"number": {}},
                    "Win Rate": {"number": {"format": "percent"}},
                    "Max Drawdown": {"number": {"format": "percent"}},
                    "Total PnL": {"number": {"format": "won"}},
                    "Total Trades": {"number": {}},
                    "Average Return": {"number": {"format": "percent"}},
                    "Notes": {"rich_text": {}}
                }
            },
            "Risk Incidents": {
                "description": "리스크 이벤트 로그",
                "properties": {
                    "Incident": {"title": {}},
                    "Severity": {
                        "select": {
                            "options": [
                                {"name": "Critical", "color": "red"},
                                {"name": "High", "color": "orange"},
                                {"name": "Medium", "color": "yellow"},
                                {"name": "Low", "color": "green"}
                            ]
                        }
                    },
                    "Type": {
                        "select": {
                            "options": [
                                {"name": "Position Limit", "color": "blue"},
                                {"name": "Drawdown", "color": "red"},
                                {"name": "Connection", "color": "purple"},
                                {"name": "Data Quality", "color": "orange"}
                            ]
                        }
                    },
                    "Timestamp": {"date": {}},
                    "Resolution": {"rich_text": {}},
                    "Impact": {"rich_text": {}}
                }
            },
            "Decision Log": {
                "description": "프로젝트 결정 사항 기록",
                "properties": {
                    "Decision": {"title": {}},
                    "Category": {
                        "select": {
                            "options": [
                                {"name": "Architecture", "color": "purple"},
                                {"name": "Strategy", "color": "blue"},
                                {"name": "Risk", "color": "red"},
                                {"name": "Process", "color": "green"}
                            ]
                        }
                    },
                    "Date": {"date": {}},
                    "Status": {
                        "select": {
                            "options": [
                                {"name": "Proposed", "color": "gray"},
                                {"name": "Approved", "color": "green"},
                                {"name": "Rejected", "color": "red"},
                                {"name": "Deferred", "color": "yellow"}
                            ]
                        }
                    },
                    "Rationale": {"rich_text": {}},
                    "Impact": {"rich_text": {}},
                    "Alternatives": {"rich_text": {}}
                }
            },
            "Execution Logs": {
                "description": "명령 실행 로그",
                "properties": {
                    "Command": {"title": {}},
                    "Status": {
                        "select": {
                            "options": [
                                {"name": "Success", "color": "green"},
                                {"name": "Failed", "color": "red"},
                                {"name": "Timeout", "color": "orange"}
                            ]
                        }
                    },
                    "Timestamp": {"date": {}},
                    "Duration": {"number": {}},
                    "Output": {"rich_text": {}},
                    "Error": {"rich_text": {}}
                }
            },
            "Pull Requests": {
                "description": "PR 추적",
                "properties": {
                    "Title": {"title": {}},
                    "PR Number": {"number": {}},
                    "Branch": {"rich_text": {}},
                    "Task": {"relation": {}},
                    "Status": {
                        "select": {
                            "options": [
                                {"name": "Open", "color": "blue"},
                                {"name": "Merged", "color": "green"},
                                {"name": "Closed", "color": "red"}
                            ]
                        }
                    },
                    "Created": {"date": {}},
                    "Files Changed": {"number": {}},
                    "Review Status": {"rich_text": {}}
                }
            }
        }
        
        created_dbs = {}
        
        for db_name, config in databases.items():
            print(f"\n[CREATE] {db_name} 데이터베이스 생성 중...")
            
            try:
                # Create database
                new_db = self.client.databases.create(
                    parent={"page_id": parent_id},
                    title=[{"text": {"content": db_name}}],
                    properties=config["properties"]
                )
                
                db_id = new_db['id']
                created_dbs[db_name] = db_id
                print(f"[SUCCESS] {db_name}: {db_id}")
                
                # Add description as content
                self.client.blocks.children.append(
                    block_id=db_id,
                    children=[
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{
                                    "text": {"content": config["description"]}
                                }]
                            }
                        }
                    ]
                )
                
            except Exception as e:
                print(f"[ERROR] {db_name} 생성 실패: {str(e)}")
        
        return created_dbs
    
    def save_database_ids(self, database_ids):
        """Save database IDs to .env file"""
        
        env_mapping = {
            "Tasks": "NOTION_TASKS_DB",
            "Architecture": "NOTION_ARCHITECTURE_DB",
            "Test Reports": "NOTION_TEST_REPORTS_DB",
            "Performance Metrics": "NOTION_METRICS_DB",
            "Risk Incidents": "NOTION_RISKS_DB",
            "Decision Log": "NOTION_DECISIONS_DB",
            "Execution Logs": "NOTION_EXECUTION_LOGS_DB",
            "Pull Requests": "NOTION_PR_DB"
        }
        
        # Read existing .env.notion
        env_path = Path('.env.notion')
        lines = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()
        
        # Update with new IDs
        for db_name, db_id in database_ids.items():
            env_var = env_mapping.get(db_name)
            if env_var:
                # Check if variable exists
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{env_var}="):
                        lines[i] = f"{env_var}={db_id}\n"
                        found = True
                        break
                
                if not found:
                    # Append if not found
                    lines.append(f"{env_var}={db_id}\n")
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        print("\n[SUCCESS] Database IDs saved to .env.notion")
    
    def create_sample_tasks(self, tasks_db_id):
        """Create initial tasks from Phase 1"""
        
        phase1_tasks = [
            {
                "title": "P1.1: 프로젝트 구조 및 개발 환경 설정",
                "status": "Done",
                "priority": "P0",
                "acceptance": "- 프로젝트 디렉토리 구조 생성\n- 가상환경 설정\n- Git 리포지토리 초기화"
            },
            {
                "title": "P1.2: CCXT Pro WebSocket 설정",
                "status": "Done",
                "priority": "P0",
                "acceptance": "- Upbit/Binance WebSocket 연결\n- 실시간 가격 스트림\n- 오더북 데이터 수집"
            },
            {
                "title": "P1.3: BTC 1년치 히스토리컬 데이터 수집",
                "status": "In Progress",
                "priority": "P1",
                "acceptance": "- 1시간 봉 데이터 수집\n- CSV 저장\n- 데이터 검증"
            },
            {
                "title": "P1.4: 오더북 15초 간격 수집 파이프라인",
                "status": "Not Started",
                "priority": "P2",
                "acceptance": "- 15초 간격 스냅샷\n- 유동성 분석\n- 데이터베이스 저장"
            }
        ]
        
        for task in phase1_tasks:
            try:
                self.client.pages.create(
                    parent={"database_id": tasks_db_id},
                    properties={
                        "Title": {"title": [{"text": {"content": task["title"]}}]},
                        "Status": {"status": {"name": task["status"]}},
                        "Priority": {"select": {"name": task["priority"]}},
                        "Phase": {"select": {"name": "Phase 1"}},
                        "Acceptance Criteria": {
                            "rich_text": [{"text": {"content": task["acceptance"]}}]
                        }
                    }
                )
                print(f"[CREATED] Task: {task['title']}")
            except Exception as e:
                print(f"[ERROR] Failed to create task: {e}")
    
    def setup_workspace(self):
        """Complete workspace setup"""
        
        print("\n" + "="*60)
        print("Notion Workspace Setup for Kimchi Premium Trading System")
        print("="*60)
        
        # Step 1: Create main page
        print("\n[STEP 1] 메인 프로젝트 페이지 생성")
        parent_id = self.create_project_page()
        
        if not parent_id:
            print("\n메인 페이지 ID를 수동으로 입력해주세요:")
            print("(Notion에서 페이지 URL 복사 → ID는 URL의 마지막 부분)")
            parent_id = input("Page ID: ").strip()
        
        # Step 2: Create databases
        print("\n[STEP 2] 데이터베이스 구조 생성")
        database_ids = self.create_database_structure(parent_id)
        
        if database_ids:
            # Step 3: Save IDs
            print("\n[STEP 3] 환경 변수 저장")
            self.save_database_ids(database_ids)
            
            # Step 4: Create sample tasks
            if "Tasks" in database_ids:
                print("\n[STEP 4] 샘플 태스크 생성")
                response = input("샘플 태스크를 생성하시겠습니까? (y/n): ")
                if response.lower() == 'y':
                    self.create_sample_tasks(database_ids["Tasks"])
            
            print("\n" + "="*60)
            print("[COMPLETE] Notion 워크스페이스 설정 완료!")
            print("="*60)
            print("\n생성된 데이터베이스:")
            for name, db_id in database_ids.items():
                print(f"  - {name}: {db_id}")
            
            print("\n다음 단계:")
            print("1. Notion에서 생성된 데이터베이스 확인")
            print("2. 필요한 경우 View와 Filter 설정")
            print("3. Claude Code에서 notion_integration_example.py 참고하여 통합 시작")
        else:
            print("\n[WARNING] 데이터베이스 생성 실패. 수동으로 생성해주세요.")


if __name__ == "__main__":
    try:
        setup = NotionWorkspaceSetup()
        setup.setup_workspace()
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        print("\n문제 해결:")
        print("1. NOTION_TOKEN이 올바른지 확인")
        print("2. Integration이 워크스페이스에 추가되었는지 확인")
        print("3. 네트워크 연결 확인")