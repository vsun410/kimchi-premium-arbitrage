"""
Notion Dashboard 자동 업데이트 스크립트
페이지 구조 재편성 및 디자인 적용
"""

import os
import asyncio
from notion_client import Client
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()


class NotionDashboardUpdater:
    """Notion Dashboard를 예쁘게 꾸미는 업데이터"""
    
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.dashboard_page_id = os.getenv("NOTION_DASHBOARD_PAGE")
        
    async def restructure_dashboard(self):
        """대시보드 구조 재편성"""
        
        print("[INFO] Restructuring Notion Dashboard...")
        
        # 1. 메인 대시보드 페이지 업데이트
        new_structure = await self._create_main_structure()
        
        # 2. 3단 레이아웃 생성
        await self._create_three_column_layout()
        
        # 3. 위젯 섹션 추가
        await self._add_widget_sections()
        
        # 4. 색상 테마 적용 안내
        await self._add_theme_guide()
        
        print("[SUCCESS] Dashboard restructuring complete!")
        
    async def _create_main_structure(self):
        """메인 구조 생성"""
        
        # 기존 콘텐츠 삭제 (주의: 백업 필수)
        # 실제로는 기존 콘텐츠를 보존하면서 재구성
        
        main_blocks = [
            # Welcome Section
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{
                        "text": {"content": "Executive Control Center"},
                        "annotations": {"bold": True}
                    }]
                }
            },
            {
                "object": "block",
                "type": "quote",
                "quote": {
                    "rich_text": [{
                        "text": {
                            "content": "First, solve the problem. Then, write the code. - John Johnson"
                        }
                    }],
                    "color": "blue_background"
                }
            },
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            },
            # Navigation Bar
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{
                        "text": {"content": "Quick Navigation: "},
                        "annotations": {"bold": True}
                    }, {
                        "text": {"content": "Dashboard | Projects | Second Brain | Calendar | Settings"},
                        "annotations": {"code": True}
                    }],
                    "icon": {"emoji": "🧭"},
                    "color": "gray_background"
                }
            }
        ]
        
        # 블록 추가
        for block in main_blocks:
            try:
                self.notion.blocks.children.append(
                    self.dashboard_page_id,
                    children=[block]
                )
            except Exception as e:
                print(f"[WARNING] Could not add block: {e}")
        
        return main_blocks
    
    async def _create_three_column_layout(self):
        """3단 레이아웃 생성"""
        
        columns_block = {
            "object": "block",
            "type": "column_list",
            "column_list": {
                "children": [
                    # Column 1: Calendar & Time
                    {
                        "object": "block",
                        "type": "column",
                        "column": {
                            "children": [
                                {
                                    "object": "block",
                                    "type": "heading_3",
                                    "heading_3": {
                                        "rich_text": [{"text": {"content": "📅 Today's Schedule"}}]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "to_do",
                                    "to_do": {
                                        "rich_text": [{"text": {"content": "Morning standup"}}],
                                        "checked": False
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "to_do",
                                    "to_do": {
                                        "rich_text": [{"text": {"content": "Code review"}}],
                                        "checked": False
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "to_do",
                                    "to_do": {
                                        "rich_text": [{"text": {"content": "Sprint planning"}}],
                                        "checked": False
                                    }
                                }
                            ]
                        }
                    },
                    # Column 2: Metrics & Progress
                    {
                        "object": "block",
                        "type": "column",
                        "column": {
                            "children": [
                                {
                                    "object": "block",
                                    "type": "heading_3",
                                    "heading_3": {
                                        "rich_text": [{"text": {"content": "📊 Key Metrics"}}]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "bulleted_list_item",
                                    "bulleted_list_item": {
                                        "rich_text": [{"text": {"content": "Code Quality: 92%"}}]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "bulleted_list_item",
                                    "bulleted_list_item": {
                                        "rich_text": [{"text": {"content": "Code Drift: 5.2%"}}]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "bulleted_list_item",
                                    "bulleted_list_item": {
                                        "rich_text": [{"text": {"content": "Tasks Complete: 42/100"}}]
                                    }
                                }
                            ]
                        }
                    },
                    # Column 3: Quick Actions
                    {
                        "object": "block",
                        "type": "column",
                        "column": {
                            "children": [
                                {
                                    "object": "block",
                                    "type": "heading_3",
                                    "heading_3": {
                                        "rich_text": [{"text": {"content": "⚡ Quick Actions"}}]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [{
                                            "text": {"content": "[New Task] "},
                                            "annotations": {"bold": True}
                                        }, {
                                            "text": {"content": "[New Note] "},
                                            "annotations": {"bold": True}
                                        }, {
                                            "text": {"content": "[Review]"},
                                            "annotations": {"bold": True}
                                        }]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        try:
            self.notion.blocks.children.append(
                self.dashboard_page_id,
                children=[columns_block]
            )
            print("[OK] 3-column layout created")
        except Exception as e:
            print(f"[WARNING] Could not create columns: {e}")
    
    async def _add_widget_sections(self):
        """위젯 섹션 추가"""
        
        widget_sections = [
            # Progress Bar Section
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📈 Project Progress"}}]
                }
            },
            {
                "object": "block",
                "type": "table",
                "table": {
                    "table_width": 3,
                    "has_column_header": True,
                    "has_row_header": False,
                    "children": [
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {
                                "cells": [
                                    [{"text": {"content": "Project"}}],
                                    [{"text": {"content": "Progress"}}],
                                    [{"text": {"content": "Status"}}]
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {
                                "cells": [
                                    [{"text": {"content": "Trading Core"}}],
                                    [{"text": {"content": "████████░░ 80%"}}],
                                    [{"text": {"content": "🟢 On Track"}}]
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {
                                "cells": [
                                    [{"text": {"content": "ML Engine"}}],
                                    [{"text": {"content": "██████░░░░ 60%"}}],
                                    [{"text": {"content": "🟡 In Progress"}}]
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {
                                "cells": [
                                    [{"text": {"content": "Dashboard"}}],
                                    [{"text": {"content": "███░░░░░░░ 30%"}}],
                                    [{"text": {"content": "🔵 Planning"}}]
                                ]
                            }
                        }
                    ]
                }
            },
            # System Status
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🔄 System Status"}}]
                }
            },
            {
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"text": {"content": "Component Health"}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"text": {"content": "🟢 Trading Core: Online"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"text": {"content": "🟢 ML Engine: Online"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"text": {"content": "🟡 Dashboard: Building"}}]
                            }
                        }
                    ]
                }
            }
        ]
        
        for block in widget_sections:
            try:
                self.notion.blocks.children.append(
                    self.dashboard_page_id,
                    children=[block]
                )
            except Exception as e:
                print(f"[WARNING] Could not add widget: {e}")
        
        print("[OK] Widget sections added")
    
    async def _add_theme_guide(self):
        """테마 가이드 추가"""
        
        theme_guide = {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "text": {"content": "Dark Mode Setup: "},
                    "annotations": {"bold": True}
                }, {
                    "text": {"content": "Press Cmd+Shift+L (Mac) or Ctrl+Shift+L (Windows) to toggle dark mode"}
                }],
                "icon": {"emoji": "🌙"},
                "color": "purple_background"
            }
        }
        
        try:
            self.notion.blocks.children.append(
                self.dashboard_page_id,
                children=[theme_guide]
            )
            print("[OK] Theme guide added")
        except Exception as e:
            print(f"[WARNING] Could not add theme guide: {e}")
    
    async def create_kanban_board(self):
        """칸반 보드 데이터베이스 생성"""
        
        print("[INFO] Creating Kanban Board...")
        
        # Tasks 데이터베이스에 뷰 추가
        tasks_db_id = os.getenv("NOTION_TASKS_DB")
        
        if not tasks_db_id:
            print("[ERROR] Tasks database ID not found")
            return
        
        # 칸반 보드 뷰는 Notion UI에서 직접 설정해야 함
        # 여기서는 안내만 제공
        
        guide_text = """
        ## Kanban Board Setup Guide:
        
        1. Go to Tasks & Requirements database
        2. Click "..." menu -> "Layout" -> "Board"
        3. "Group by" -> "Status"
        4. Create these groups:
           - Backlog
           - To Do
           - In Progress
           - Review
           - Done
        """
        
        print(guide_text)
        return guide_text
    
    async def add_custom_widgets(self):
        """커스텀 위젯 임베드 안내"""
        
        widget_guide = """
        ## Custom Widget Setup:
        
        1. Upload notion_widgets.html to GitHub Pages:
           - Create repository: notion-widgets
           - Settings -> Pages -> Source: Deploy from main branch
           - URL: https://[username].github.io/notion-widgets/
        
        2. Embed in Notion:
           - Type /embed
           - Paste the URL
           - Adjust size
        
        3. Recommended Widgets:
           - Indify.co for free widgets
           - Clock, Weather, Counter, etc.
        """
        
        print(widget_guide)
        return widget_guide


async def main():
    """메인 실행 함수"""
    
    print("\n" + "="*50)
    print("[INFO] Notion Dashboard Design Update")
    print("="*50)
    
    updater = NotionDashboardUpdater()
    
    # 1. 대시보드 구조 재편성
    await updater.restructure_dashboard()
    
    # 2. 칸반 보드 안내
    kanban_guide = await updater.create_kanban_board()
    
    # 3. 위젯 추가 안내
    widget_guide = await updater.add_custom_widgets()
    
    # 가이드 저장
    with open("notion_setup_guide.md", "w", encoding="utf-8") as f:
        f.write("# Notion Dashboard Setup Guide\n\n")
        f.write(kanban_guide)
        f.write("\n\n")
        f.write(widget_guide)
    
    print("\n" + "="*50)
    print("[SUCCESS] Dashboard update complete!")
    print("="*50)
    print("\n[NEXT STEPS]:")
    print("1. Open Notion Dashboard")
    print("2. Enable Dark Mode (Cmd+Shift+L)")
    print("3. Set up Kanban Board view")
    print("4. Add custom widgets")
    print("\nDashboard URL: https://notion.so/" + os.getenv("NOTION_DASHBOARD_PAGE", "").replace("-", ""))


if __name__ == "__main__":
    asyncio.run(main())