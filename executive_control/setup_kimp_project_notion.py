"""
김치프리미엄 프로젝트를 Notion에 통합 관리 시스템 구축
현재 구현 상태를 Notion 태스크로 자동 변환
"""

import os
import json
import asyncio
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class KimpProjectNotionSetup:
    """김치프리미엄 프로젝트 Notion 통합"""
    
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.parent_page = os.getenv("NOTION_DASHBOARD_PAGE")
        
        # 프로젝트 구조 정의
        self.project_structure = {
            "name": "🚀 김치프리미엄 차익거래 시스템",
            "description": "BTC 김치프리미엄 + 추세돌파 하이브리드 전략",
            "modules": {
                "data_collection": {
                    "name": "📡 데이터 수집 시스템",
                    "progress": 100,
                    "status": "completed",
                    "components": [
                        {"name": "WebSocket 연결 관리", "status": "done", "progress": 100},
                        {"name": "API Manager", "status": "done", "progress": 100},
                        {"name": "데이터 정규화", "status": "done", "progress": 100},
                        {"name": "재연결 메커니즘", "status": "done", "progress": 100}
                    ]
                },
                "backtesting": {
                    "name": "📊 백테스팅 시스템",
                    "progress": 100,
                    "status": "completed",
                    "components": [
                        {"name": "백테스팅 엔진", "status": "done", "progress": 100},
                        {"name": "성과 분석기", "status": "done", "progress": 100},
                        {"name": "리포트 생성", "status": "done", "progress": 100},
                        {"name": "전략 시뮬레이터", "status": "done", "progress": 100}
                    ]
                },
                "dynamic_hedge": {
                    "name": "🔄 동적 헤지 시스템",
                    "progress": 100,
                    "status": "completed",
                    "components": [
                        {"name": "추세 분석", "status": "done", "progress": 100},
                        {"name": "포지션 관리", "status": "done", "progress": 100},
                        {"name": "패턴 인식", "status": "done", "progress": 100},
                        {"name": "역프리미엄 대응", "status": "done", "progress": 100}
                    ]
                },
                "ml_models": {
                    "name": "🤖 ML 모델",
                    "progress": 70,
                    "status": "in_progress",
                    "components": [
                        {"name": "LSTM 모델", "status": "done", "progress": 100},
                        {"name": "피처 엔지니어링", "status": "done", "progress": 100},
                        {"name": "XGBoost 앙상블", "status": "in_progress", "progress": 60},
                        {"name": "강화학습 (PPO/DQN)", "status": "todo", "progress": 0}
                    ]
                },
                "strategies": {
                    "name": "📈 전략 구현",
                    "progress": 60,
                    "status": "in_progress",
                    "components": [
                        {"name": "김프 기본 전략", "status": "done", "progress": 100},
                        {"name": "추세 추종 전략", "status": "done", "progress": 100},
                        {"name": "하이브리드 전략", "status": "in_progress", "progress": 50},
                        {"name": "ML 기반 전략", "status": "todo", "progress": 0}
                    ]
                },
                "live_trading": {
                    "name": "💹 실시간 거래",
                    "progress": 90,
                    "status": "testing",
                    "components": [
                        {"name": "주문 실행 시스템", "status": "done", "progress": 100},
                        {"name": "포지션 트래킹", "status": "done", "progress": 100},
                        {"name": "리스크 모니터링", "status": "testing", "progress": 80},
                        {"name": "Paper Trading", "status": "testing", "progress": 70}
                    ]
                },
                "production": {
                    "name": "🚢 Production 배포",
                    "progress": 0,
                    "status": "todo",
                    "components": [
                        {"name": "Docker 컨테이너화", "status": "todo", "progress": 0},
                        {"name": "Kubernetes 설정", "status": "todo", "progress": 0},
                        {"name": "AWS/GCP 인프라", "status": "todo", "progress": 0},
                        {"name": "모니터링 대시보드", "status": "todo", "progress": 0}
                    ]
                },
                "ui_dashboard": {
                    "name": "🎨 UI Dashboard",
                    "progress": 0,
                    "status": "todo",
                    "components": [
                        {"name": "React 프론트엔드", "status": "todo", "progress": 0},
                        {"name": "실시간 차트", "status": "todo", "progress": 0},
                        {"name": "모바일 앱", "status": "todo", "progress": 0},
                        {"name": "PWA 지원", "status": "todo", "progress": 0}
                    ]
                }
            }
        }
        
    async def create_project_workspace(self):
        """김프 프로젝트 전용 워크스페이스 생성"""
        
        print("="*60)
        print("   김치프리미엄 프로젝트 Notion 통합 시작")
        print("="*60)
        print()
        
        # 1. 메인 프로젝트 페이지 생성
        project_page = await self._create_main_page()
        
        # 2. 태스크 데이터베이스 생성
        tasks_db = await self._create_tasks_database(project_page['id'])
        
        # 3. 마일스톤 데이터베이스 생성
        milestones_db = await self._create_milestones_database(project_page['id'])
        
        # 4. 버그/이슈 트래커 생성
        issues_db = await self._create_issues_database(project_page['id'])
        
        # 5. 현재 구현 상태를 태스크로 변환
        await self._populate_tasks(tasks_db)
        
        # 6. 마일스톤 설정
        await self._create_milestones(milestones_db)
        
        # 7. 대시보드 위젯 추가
        await self._add_dashboard_widgets(project_page['id'])
        
        # 설정 저장
        config = {
            "project_page": project_page['id'],
            "tasks_db": tasks_db,
            "milestones_db": milestones_db,
            "issues_db": issues_db,
            "created_at": datetime.now().isoformat()
        }
        
        with open("kimp_notion_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print()
        print("="*60)
        print("[SUCCESS] 김치프리미엄 프로젝트 Notion 통합 완료!")
        print("="*60)
        print()
        print(f"[PROJECT] 프로젝트 페이지: https://notion.so/{project_page['id'].replace('-', '')}")
        print(f"[TASKS] 태스크 DB: {tasks_db}")
        print(f"[MILESTONES] 마일스톤 DB: {milestones_db}")
        print(f"[ISSUES] 이슈 트래커: {issues_db}")
        
        return config
    
    async def _create_main_page(self):
        """메인 프로젝트 페이지 생성"""
        
        page = self.notion.pages.create(
            parent={"page_id": self.parent_page},
            icon={"emoji": "🚀"},
            properties={
                "title": [{"text": {"content": self.project_structure["name"]}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": "김치프리미엄 차익거래 시스템"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{
                            "text": {"content": "BTC 김치프리미엄과 추세돌파를 결합한 하이브리드 전략 시스템"}
                        }],
                        "color": "blue_background"
                    }
                },
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{
                            "text": {"content": f"전체 진행률: 65% | 목표 수익률: 월 8-15% | Sharpe Ratio: > 2.5"}
                        }],
                        "icon": {"emoji": "📊"},
                        "color": "green_background"
                    }
                }
            ]
        )
        
        print("[OK] 메인 프로젝트 페이지 생성 완료")
        return page
    
    async def _create_tasks_database(self, parent_page_id):
        """태스크 관리 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_page_id},
            title=[{"text": {"content": "📋 태스크 관리"}}],
            properties={
                "Task": {"title": {}},
                "Module": {
                    "select": {
                        "options": [
                            {"name": "데이터 수집", "color": "blue"},
                            {"name": "백테스팅", "color": "green"},
                            {"name": "동적 헤지", "color": "purple"},
                            {"name": "ML 모델", "color": "orange"},
                            {"name": "전략", "color": "yellow"},
                            {"name": "실시간 거래", "color": "pink"},
                            {"name": "Production", "color": "red"},
                            {"name": "UI Dashboard", "color": "brown"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Todo", "color": "gray"},
                            {"name": "In Progress", "color": "yellow"},
                            {"name": "Testing", "color": "orange"},
                            {"name": "Done", "color": "green"},
                            {"name": "Blocked", "color": "red"}
                        ]
                    }
                },
                "Priority": {
                    "select": {
                        "options": [
                            {"name": "긴급", "color": "red"},
                            {"name": "높음", "color": "orange"},
                            {"name": "보통", "color": "yellow"},
                            {"name": "낮음", "color": "gray"}
                        ]
                    }
                },
                "Progress": {"number": {"format": "percent"}},
                "Assignee": {"people": {}},
                "Due Date": {"date": {}},
                "Sprint": {"select": {}},
                "Story Points": {"number": {"format": "number"}},
                "Created": {"created_time": {}},
                "Updated": {"last_edited_time": {}}
            }
        )
        
        print("[OK] 태스크 데이터베이스 생성 완료")
        return database['id']
    
    async def _create_milestones_database(self, parent_page_id):
        """마일스톤 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_page_id},
            title=[{"text": {"content": "🎯 마일스톤"}}],
            properties={
                "Milestone": {"title": {}},
                "Target Date": {"date": {}},
                "Progress": {"number": {"format": "percent"}},
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Planning", "color": "gray"},
                            {"name": "Active", "color": "blue"},
                            {"name": "Completed", "color": "green"},
                            {"name": "Delayed", "color": "red"}
                        ]
                    }
                },
                "Description": {"rich_text": {}}
            }
        )
        
        print("[OK] 마일스톤 데이터베이스 생성 완료")
        return database['id']
    
    async def _create_issues_database(self, parent_page_id):
        """버그/이슈 트래커 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_page_id},
            title=[{"text": {"content": "🐛 이슈 트래커"}}],
            properties={
                "Issue": {"title": {}},
                "Type": {
                    "select": {
                        "options": [
                            {"name": "Bug", "color": "red"},
                            {"name": "Feature", "color": "green"},
                            {"name": "Enhancement", "color": "blue"},
                            {"name": "Documentation", "color": "gray"}
                        ]
                    }
                },
                "Severity": {
                    "select": {
                        "options": [
                            {"name": "Critical", "color": "red"},
                            {"name": "High", "color": "orange"},
                            {"name": "Medium", "color": "yellow"},
                            {"name": "Low", "color": "gray"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Open", "color": "red"},
                            {"name": "In Progress", "color": "yellow"},
                            {"name": "Resolved", "color": "green"},
                            {"name": "Closed", "color": "gray"}
                        ]
                    }
                },
                "Reporter": {"people": {}},
                "Created": {"created_time": {}}
            }
        )
        
        print("[OK] 이슈 트래커 데이터베이스 생성 완료")
        return database['id']
    
    async def _populate_tasks(self, tasks_db_id):
        """현재 구현 상태를 태스크로 변환"""
        
        print("[INFO] 태스크 데이터 입력 중...")
        
        for module_key, module_data in self.project_structure["modules"].items():
            for component in module_data["components"]:
                # 태스크 생성
                task = {
                    "Task": {"title": [{"text": {"content": component["name"]}}]},
                    "Module": {"select": {"name": module_data["name"].split()[1]}},
                    "Status": {"select": {"name": self._get_status(component["status"])}},
                    "Progress": {"number": component["progress"] / 100}
                }
                
                # 우선순위 설정
                if component["status"] == "todo":
                    task["Priority"] = {"select": {"name": "높음"}}
                elif component["status"] == "in_progress":
                    task["Priority"] = {"select": {"name": "긴급"}}
                else:
                    task["Priority"] = {"select": {"name": "보통"}}
                
                try:
                    self.notion.pages.create(
                        parent={"database_id": tasks_db_id},
                        properties=task
                    )
                except Exception as e:
                    print(f"  [WARNING] 태스크 생성 실패: {e}")
        
        print("[OK] 태스크 데이터 입력 완료")
    
    async def _create_milestones(self, milestones_db_id):
        """프로젝트 마일스톤 생성"""
        
        milestones = [
            {
                "name": "MVP 완성",
                "date": "2025-09-15",
                "progress": 0.65,
                "status": "Active",
                "description": "기본 김프 전략 + ML 모델 통합"
            },
            {
                "name": "Paper Trading 안정화",
                "date": "2025-09-30",
                "progress": 0.30,
                "status": "Active",
                "description": "1개월 Paper Trading 테스트"
            },
            {
                "name": "Production 배포",
                "date": "2025-10-15",
                "progress": 0,
                "status": "Planning",
                "description": "AWS/GCP 클라우드 배포"
            },
            {
                "name": "실거래 시작",
                "date": "2025-10-30",
                "progress": 0,
                "status": "Planning",
                "description": "소액 실거래 테스트"
            }
        ]
        
        for milestone in milestones:
            try:
                self.notion.pages.create(
                    parent={"database_id": milestones_db_id},
                    properties={
                        "Milestone": {"title": [{"text": {"content": milestone["name"]}}]},
                        "Target Date": {"date": {"start": milestone["date"]}},
                        "Progress": {"number": milestone["progress"]},
                        "Status": {"select": {"name": milestone["status"]}},
                        "Description": {"rich_text": [{"text": {"content": milestone["description"]}}]}
                    }
                )
            except Exception as e:
                print(f"  [WARNING] 마일스톤 생성 실패: {e}")
        
        print("[OK] 마일스톤 생성 완료")
    
    async def _add_dashboard_widgets(self, page_id):
        """대시보드 위젯 추가"""
        
        widgets = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📊 프로젝트 현황"}}]
                }
            },
            {
                "object": "block",
                "type": "column_list",
                "column_list": {
                    "children": [
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [
                                                {"text": {"content": "완료된 모듈\n"}},
                                                {"text": {"content": "• 데이터 수집 ✅\n• 백테스팅 ✅\n• 동적 헤지 ✅"}}
                                            ],
                                            "icon": {"emoji": "✅"},
                                            "color": "green_background"
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [
                                                {"text": {"content": "진행 중\n"}},
                                                {"text": {"content": "• ML 모델 (70%)\n• 전략 구현 (60%)\n• 실시간 거래 (90%)"}}
                                            ],
                                            "icon": {"emoji": "🔄"},
                                            "color": "yellow_background"
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [
                                                {"text": {"content": "예정\n"}},
                                                {"text": {"content": "• Production 배포\n• UI Dashboard\n• 멀티 에셋"}}
                                            ],
                                            "icon": {"emoji": "📅"},
                                            "color": "gray_background"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
        
        for widget in widgets:
            try:
                self.notion.blocks.children.append(page_id, children=[widget])
            except Exception as e:
                print(f"  [WARNING] 위젯 추가 실패: {e}")
        
        print("[OK] 대시보드 위젯 추가 완료")
    
    def _get_status(self, status):
        """상태 매핑"""
        status_map = {
            "done": "Done",
            "in_progress": "In Progress",
            "testing": "Testing",
            "todo": "Todo"
        }
        return status_map.get(status, "Todo")


async def main():
    """메인 실행 함수"""
    setup = KimpProjectNotionSetup()
    await setup.create_project_workspace()


if __name__ == "__main__":
    asyncio.run(main())