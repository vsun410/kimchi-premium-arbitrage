"""
Setup Notion Executive Dashboard
Notion에 Executive Control System을 위한 데이터베이스와 대시보드 생성
"""

import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from notion_client import Client
import json


class NotionExecutiveDashboardSetup:
    """
    Notion에 Executive Control Dashboard 설정
    필요한 모든 데이터베이스와 페이지를 자동 생성
    """
    
    def __init__(self, notion_token: str):
        self.notion = Client(auth=notion_token)
        self.created_resources = {}
    
    async def setup_complete_dashboard(self, parent_page_id: str) -> Dict:
        """
        완전한 Executive Dashboard 설정
        
        Args:
            parent_page_id: 대시보드를 생성할 부모 페이지 ID
            
        Returns:
            생성된 리소스 ID 딕셔너리
        """
        
        print("[INFO] Setting up Notion Executive Dashboard...")
        
        # 1. 메인 대시보드 페이지 생성
        dashboard_page = await self._create_main_dashboard(parent_page_id)
        self.created_resources['dashboard_page'] = dashboard_page['id']
        
        # 2. 프로젝트 비전 데이터베이스 생성
        vision_db = await self._create_vision_database(dashboard_page['id'])
        self.created_resources['vision_db'] = vision_db['id']
        
        # 3. 작업 관리 데이터베이스 생성
        tasks_db = await self._create_tasks_database(dashboard_page['id'])
        self.created_resources['tasks_db'] = tasks_db['id']
        
        # 4. 검증 결과 데이터베이스 생성
        validation_db = await self._create_validation_database(dashboard_page['id'])
        self.created_resources['validation_db'] = validation_db['id']
        
        # 5. 코드 블록 라이브러리 생성
        blocks_db = await self._create_code_blocks_database(dashboard_page['id'])
        self.created_resources['blocks_db'] = blocks_db['id']
        
        # 6. 대시보드에 위젯 추가
        await self._add_dashboard_widgets(dashboard_page['id'])
        
        # 7. 초기 비전 문서 생성
        await self._create_initial_vision_document(vision_db['id'])
        
        # 8. 환경 변수 파일 생성
        await self._create_env_file()
        
        print("\n[SUCCESS] Executive Dashboard Setup Complete!")
        print(f"\n[DASHBOARD URL]: https://notion.so/{dashboard_page['id'].replace('-', '')}")
        
        return self.created_resources
    
    async def _create_main_dashboard(self, parent_id: str) -> Dict:
        """메인 대시보드 페이지 생성"""
        
        page = self.notion.pages.create(
            parent={"page_id": parent_id},
            properties={
                "title": [{"text": {"content": "🎯 Executive Control Dashboard"}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": "Executive Control System"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": "프로젝트 비전을 지키며 코드 품질을 관리하는 Executive Board"
                            }
                        }]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{
                            "text": {
                                "content": "이 시스템은 LLM이 생성하는 코드가 원래 비전에서 벗어나지 않도록 자동으로 감시하고 검증합니다."
                            }
                        }],
                        "icon": {"emoji": "🛡️"},
                        "color": "blue_background"
                    }
                }
            ]
        )
        
        print("[OK] Created main dashboard page")
        return page
    
    async def _create_vision_database(self, parent_id: str) -> Dict:
        """프로젝트 비전 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_id},
            title=[{"text": {"content": "📋 Project Vision & Architecture"}}],
            properties={
                "Title": {"title": {}},
                "Type": {
                    "select": {
                        "options": [
                            {"name": "Core Vision", "color": "red"},
                            {"name": "Architecture", "color": "blue"},
                            {"name": "Red Lines", "color": "orange"},
                            {"name": "Success Metrics", "color": "green"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Active", "color": "green"},
                            {"name": "Draft", "color": "gray"},
                            {"name": "Archived", "color": "default"}
                        ]
                    }
                },
                "Last Edited": {"last_edited_time": {}},
                "Version": {"rich_text": {}},
                "Author": {"people": {}}
            }
        )
        
        print("[OK] Created vision database")
        return database
    
    async def _create_tasks_database(self, parent_id: str) -> Dict:
        """작업 관리 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_id},
            title=[{"text": {"content": "📝 Tasks & Requirements"}}],
            properties={
                "Task ID": {"title": {}},
                "Title": {"rich_text": {}},
                "Component": {
                    "select": {
                        "options": [
                            {"name": "trading_core", "color": "purple"},
                            {"name": "ml_engine", "color": "pink"},
                            {"name": "dashboard", "color": "blue"},
                            {"name": "risk_manager", "color": "red"},
                            {"name": "data_pipeline", "color": "green"},
                            {"name": "general", "color": "gray"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Pending", "color": "default"},
                            {"name": "In Progress", "color": "yellow"},
                            {"name": "In Review", "color": "orange"},
                            {"name": "Approved", "color": "green"},
                            {"name": "Rejected", "color": "red"},
                            {"name": "Completed", "color": "blue"}
                        ]
                    }
                },
                "Priority": {
                    "select": {
                        "options": [
                            {"name": "High", "color": "red"},
                            {"name": "Medium", "color": "yellow"},
                            {"name": "Low", "color": "gray"}
                        ]
                    }
                },
                "Effort": {"number": {"format": "number"}},
                "Dependencies": {"multi_select": {}},
                "Assigned To": {"people": {}},
                "Created": {"created_time": {}},
                "Due Date": {"date": {}},
                "Business Score": {"number": {"format": "percent"}},
                "Architecture Score": {"number": {"format": "percent"}},
                "Drift Score": {"number": {"format": "percent"}}
            }
        )
        
        # 데이터베이스 뷰 추가는 생략 (데이터베이스 ID에는 직접 블록 추가 불가)
        
        print("[OK] Created tasks database")
        return database
    
    async def _create_validation_database(self, parent_id: str) -> Dict:
        """검증 결과 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_id},
            title=[{"text": {"content": "🔍 Validation Results"}}],
            properties={
                "Task ID": {"title": {}},
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Approved", "color": "green"},
                            {"name": "Rejected", "color": "red"},
                            {"name": "Warning", "color": "yellow"},
                            {"name": "Block Replaced", "color": "blue"}
                        ]
                    }
                },
                "Validator": {
                    "select": {
                        "options": [
                            {"name": "Vision Guardian", "color": "purple"},
                            {"name": "File Monitor", "color": "orange"},
                            {"name": "Git Hook", "color": "pink"},
                            {"name": "CI/CD", "color": "blue"}
                        ]
                    }
                },
                "Timestamp": {"date": {}},
                "Business Score": {"number": {"format": "percent"}},
                "Architecture Score": {"number": {"format": "percent"}},
                "Drift Score": {"number": {"format": "percent"}},
                "Violations": {"multi_select": {}},
                "File Path": {"rich_text": {}},
                "Commit Hash": {"rich_text": {}}
            }
        )
        
        print("[OK] Created validation database")
        return database
    
    async def _create_code_blocks_database(self, parent_id: str) -> Dict:
        """코드 블록 라이브러리 데이터베이스 생성"""
        
        database = self.notion.databases.create(
            parent={"page_id": parent_id},
            title=[{"text": {"content": "🧩 Code Block Library"}}],
            properties={
                "Block ID": {"title": {}},
                "Component": {
                    "select": {
                        "options": [
                            {"name": "trading_core", "color": "purple"},
                            {"name": "ml_engine", "color": "pink"},
                            {"name": "dashboard", "color": "blue"},
                            {"name": "risk_manager", "color": "red"},
                            {"name": "data_pipeline", "color": "green"}
                        ]
                    }
                },
                "Function Name": {"rich_text": {}},
                "Description": {"rich_text": {}},
                "Version": {"rich_text": {}},
                "Replaceable": {"checkbox": {}},
                "Last Modified": {"last_edited_time": {}},
                "Performance Score": {"number": {"format": "percent"}},
                "Test Coverage": {"number": {"format": "percent"}},
                "Dependencies": {"multi_select": {}},
                "Tags": {"multi_select": {}}
            }
        )
        
        print("[OK] Created code blocks database")
        return database
    
    async def _add_dashboard_widgets(self, dashboard_id: str):
        """대시보드에 실시간 위젯 추가"""
        
        widgets = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📈 Real-time Metrics"}}]
                }
            },
            {
                "object": "block",
                "type": "synced_block",
                "synced_block": {
                    "synced_from": None,
                    "children": [
                        {
                            "object": "block",
                            "type": "table",
                            "table": {
                                "table_width": 4,
                                "has_column_header": True,
                                "has_row_header": False,
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "table_row",
                                        "table_row": {
                                            "cells": [
                                                [{"text": {"content": "Metric"}}],
                                                [{"text": {"content": "Current"}}],
                                                [{"text": {"content": "Target"}}],
                                                [{"text": {"content": "Status"}}]
                                            ]
                                        }
                                    },
                                    {
                                        "object": "block",
                                        "type": "table_row",
                                        "table_row": {
                                            "cells": [
                                                [{"text": {"content": "Code Drift"}}],
                                                [{"text": {"content": "5.2%"}}],
                                                [{"text": {"content": "< 10%"}}],
                                                [{"text": {"content": "✅ Good"}}]
                                            ]
                                        }
                                    },
                                    {
                                        "object": "block",
                                        "type": "table_row",
                                        "table_row": {
                                            "cells": [
                                                [{"text": {"content": "Vision Alignment"}}],
                                                [{"text": {"content": "92%"}}],
                                                [{"text": {"content": "> 85%"}}],
                                                [{"text": {"content": "✅ Good"}}]
                                            ]
                                        }
                                    },
                                    {
                                        "object": "block",
                                        "type": "table_row",
                                        "table_row": {
                                            "cells": [
                                                [{"text": {"content": "Tasks Completed"}}],
                                                [{"text": {"content": "42/100"}}],
                                                [{"text": {"content": "100"}}],
                                                [{"text": {"content": "🔄 Progress"}}]
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🚨 Recent Alerts"}}]
                }
            },
            {
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"text": {"content": "Code Validation Failures (0)"}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"text": {"content": "No recent failures"}}]
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🔄 Component Health"}}]
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
                                            "rich_text": [{"text": {"content": "Trading Core\n✅ Healthy"}}],
                                            "icon": {"emoji": "💹"}
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
                                            "rich_text": [{"text": {"content": "ML Engine\n✅ Healthy"}}],
                                            "icon": {"emoji": "🤖"}
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
                                            "rich_text": [{"text": {"content": "Dashboard\n🔄 Building"}}],
                                            "icon": {"emoji": "📊"}
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
            self.notion.blocks.children.append(dashboard_id, children=[widget])
        
        print("[OK] Added dashboard widgets")
    
    async def _create_initial_vision_document(self, vision_db_id: str):
        """초기 비전 문서 생성"""
        
        vision_page = self.notion.pages.create(
            parent={"database_id": vision_db_id},
            properties={
                "Title": {"title": [{"text": {"content": "Kimchi Premium Trading System Vision"}}]},
                "Type": {"select": {"name": "Core Vision"}},
                "Status": {"select": {"name": "Active"}},
                "Version": {"rich_text": [{"text": {"content": "1.0.0"}}]}
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": "🎯 Project Vision"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Core Objectives"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": "실시간 김치프리미엄 차익거래 자동화"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": "리스크 중립적 헤지 포지션 유지"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": "24/7 무중단 운영 시스템"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": "ML 기반 객관적 진입/청산 시그널"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "🚫 Red Lines (절대 금지)"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "동시 양방향 포지션 한도 초과 금지"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "단일 거래 자본금 10% 초과 금지"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "수동 개입 필요한 로직 금지"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": "테스트되지 않은 코드 배포 금지"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "📊 Success Metrics"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "table",
                    "table": {
                        "table_width": 2,
                        "has_column_header": True,
                        "has_row_header": False,
                        "children": [
                            {
                                "object": "block",
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "Metric"}}],
                                        [{"text": {"content": "Target"}}]
                                    ]
                                }
                            },
                            {
                                "object": "block",
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "Sharpe Ratio"}}],
                                        [{"text": {"content": "> 1.5"}}]
                                    ]
                                }
                            },
                            {
                                "object": "block",
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "Max Drawdown"}}],
                                        [{"text": {"content": "< 15%"}}]
                                    ]
                                }
                            },
                            {
                                "object": "block",
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "Win Rate"}}],
                                        [{"text": {"content": "> 60%"}}]
                                    ]
                                }
                            },
                            {
                                "object": "block",
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "System Uptime"}}],
                                        [{"text": {"content": "> 99%"}}]
                                    ]
                                }
                            }
                        ]
                    }
                }
            ]
        )
        
        print("[OK] Created initial vision document")
        return vision_page
    
    async def _create_env_file(self):
        """환경 변수 파일 생성"""
        
        env_content = f"""# Notion Executive Control System Configuration
# Generated by setup_notion_executive_dashboard.py

# Notion API Token
NOTION_TOKEN=your_notion_token_here

# Database IDs (auto-generated)
NOTION_VISION_DB={self.created_resources.get('vision_db', '')}
NOTION_TASKS_DB={self.created_resources.get('tasks_db', '')}
NOTION_VALIDATION_DB={self.created_resources.get('validation_db', '')}
NOTION_BLOCKS_DB={self.created_resources.get('blocks_db', '')}
NOTION_DASHBOARD_PAGE={self.created_resources.get('dashboard_page', '')}

# Additional Settings
VALIDATION_THRESHOLD_BUSINESS=0.8
VALIDATION_THRESHOLD_ARCHITECTURE=0.8
VALIDATION_THRESHOLD_DRIFT=0.2
"""
        
        env_path = '.env.executive'
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"[OK] Created environment file: {env_path}")
        
        # Instructions 파일 생성
        instructions = f"""
# Executive Control System Setup Complete!

## Your Dashboard URLs:
- Main Dashboard: https://notion.so/{self.created_resources.get('dashboard_page', '').replace('-', '')}
- Vision Database: https://notion.so/{self.created_resources.get('vision_db', '').replace('-', '')}
- Tasks Database: https://notion.so/{self.created_resources.get('tasks_db', '').replace('-', '')}

## Next Steps:

1. **Update .env file**:
   - Copy the NOTION_TOKEN from your actual .env file
   - Replace 'your_notion_token_here' in .env.executive

2. **Initialize the system**:
   ```python
   python -m executive_control.initialize
   ```

3. **Set up validation hooks**:
   ```python
   python -m executive_control.claude_code_interceptor
   ```

4. **Start monitoring**:
   ```python
   python -m executive_control.monitor
   ```

## How to Use:

### Submit a new requirement:
```python
from executive_control.notion_governance_integration import NotionGovernanceIntegration

governance = NotionGovernanceIntegration(token, config)
await governance.submit_requirement("Your requirement here")
```

### Validate code:
```python
python validate.py your_file.py
```

### View dashboard:
Open Notion and navigate to the Executive Control Dashboard

## Database IDs (save these):
{json.dumps(self.created_resources, indent=2)}
"""
        
        with open('EXECUTIVE_SETUP_COMPLETE.md', 'w') as f:
            f.write(instructions)
        
        print("\n[INFO] Instructions saved to: EXECUTIVE_SETUP_COMPLETE.md")


async def main():
    """메인 설정 함수"""
    
    # Notion 토큰 확인
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found in environment variables")
        print("Please set: export NOTION_TOKEN=your_token_here")
        return
    
    # 부모 페이지 ID 입력 받기
    print("\n[INFO] Notion Executive Dashboard Setup")
    print("="*50)
    parent_page = input("Enter the parent page ID or URL where dashboard should be created: ").strip()
    
    # URL에서 ID 추출
    if "notion.so" in parent_page:
        # URL 형식: https://www.notion.so/Page-Name-xxxxx
        parent_page = parent_page.split("/")[-1].split("-")[-1]
    
    # 하이픈 추가 (필요한 경우)
    if len(parent_page) == 32 and "-" not in parent_page:
        parent_page = f"{parent_page[:8]}-{parent_page[8:12]}-{parent_page[12:16]}-{parent_page[16:20]}-{parent_page[20:]}"
    
    print(f"\nUsing parent page ID: {parent_page}")
    confirm = input("Proceed with setup? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Setup cancelled.")
        return
    
    # 대시보드 설정
    setup = NotionExecutiveDashboardSetup(notion_token)
    resources = await setup.setup_complete_dashboard(parent_page)
    
    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("="*50)
    print("\n📋 Created Resources:")
    for key, value in resources.items():
        print(f"  - {key}: {value}")
    
    print("\n👉 Check EXECUTIVE_SETUP_COMPLETE.md for next steps!")


if __name__ == "__main__":
    asyncio.run(main())