#!/usr/bin/env python3
"""
Notion API 생성 및 수정 테스트
페이지를 만들고 수정하여 API가 제대로 작동하는지 확인
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime
import time
import json

# Load environment
env_notion_path = Path('.env.notion')
if env_notion_path.exists():
    load_dotenv(env_notion_path)

# Initialize client
token = os.getenv('NOTION_TOKEN')
if not token:
    print("[ERROR] NOTION_TOKEN not found")
    sys.exit(1)

client = Client(auth=token)

def find_or_create_test_page():
    """테스트 페이지 찾기 또는 생성"""
    
    print("\n" + "="*60)
    print("Notion API 테스트 페이지 생성/수정 테스트")
    print("="*60)
    
    # 1. 먼저 기존 테스트 페이지 검색
    print("\n[1] 기존 테스트 페이지 검색 중...")
    search_response = client.search(
        query="Claude Code Test Page",
        filter={"value": "page", "property": "object"}
    )
    
    if search_response['results']:
        page_id = search_response['results'][0]['id']
        print(f"[FOUND] 기존 테스트 페이지 발견: {page_id}")
        return page_id
    
    # 2. 없으면 새로 생성 (루트에)
    print("[CREATE] 새 테스트 페이지 생성 중...")
    
    try:
        # 먼저 사용자가 접근 가능한 페이지 찾기
        print("[SEARCH] 접근 가능한 페이지 검색 중...")
        all_pages = client.search(filter={"value": "page", "property": "object"})
        
        if not all_pages['results']:
            print("\n[INFO] Notion에서 먼저 수동으로 페이지를 생성하고 Integration을 추가해주세요:")
            print("  1. Notion에서 새 페이지 생성")
            print("  2. 페이지 우측 상단 '...' → 'Add connections' → 'kimp' 추가")
            print("  3. 이 스크립트를 다시 실행")
            return None
        
        # 첫 번째 접근 가능한 페이지를 부모로 사용
        parent_page = all_pages['results'][0]
        parent_id = parent_page['id']
        parent_title = parent_page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Unknown')
        
        print(f"[INFO] '{parent_title}' 페이지 아래에 테스트 페이지를 생성합니다.")
        
        new_page = client.pages.create(
            parent={"page_id": parent_id},
            properties={
                "title": {
                    "title": [{
                        "text": {
                            "content": "Claude Code Test Page"
                        }
                    }]
                }
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{
                            "text": {
                                "content": "Claude Code Notion Integration Test"
                            }
                        }]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        }]
                    }
                }
            ]
        )
        
        page_id = new_page['id']
        print(f"[SUCCESS] 테스트 페이지 생성 완료: {page_id}")
        return page_id
        
    except Exception as e:
        print(f"[ERROR] 페이지 생성 실패: {str(e)}")
        print("\n[HELP] 먼저 Notion에서:")
        print("  1. 아무 페이지나 생성하세요")
        print("  2. 해당 페이지에 'kimp' Integration을 추가하세요")
        print("  3. 이 스크립트를 다시 실행하세요")
        return None

def modify_page(page_id):
    """페이지 내용 수정"""
    
    print(f"\n[2] 페이지 수정 테스트")
    
    try:
        # 1. 페이지에 새 블록 추가
        print("[APPEND] 새 콘텐츠 추가 중...")
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 여러 종류의 블록 추가
        new_blocks = [
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "text": {
                            "content": f"Test Update - {current_time}"
                        }
                    }]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "text": {
                            "content": "API 연결 상태: Active"
                        }
                    }]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "text": {
                            "content": f"마지막 테스트: {current_time}"
                        }
                    }]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "text": {
                            "content": "Integration 이름: kimp"
                        }
                    }]
                }
            },
            {
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": [{
                        "text": {
                            "content": json.dumps({
                                "status": "connected",
                                "timestamp": current_time,
                                "bot_name": "kimp",
                                "test": "successful"
                            }, indent=2)
                        }
                    }],
                    "language": "json"
                }
            },
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{
                        "text": {
                            "content": "This page is automatically updated by Claude Code via Notion API"
                        }
                    }],
                    "icon": {
                        "emoji": "🤖"
                    }
                }
            }
        ]
        
        client.blocks.children.append(
            block_id=page_id,
            children=new_blocks
        )
        
        print("[SUCCESS] 콘텐츠 추가 완료!")
        
        # 2. 페이지 제목 업데이트
        print("[UPDATE] 페이지 제목 업데이트 중...")
        
        client.pages.update(
            page_id=page_id,
            properties={
                "title": {
                    "title": [{
                        "text": {
                            "content": f"Claude Code Test Page (Updated: {datetime.now().strftime('%H:%M:%S')})"
                        }
                    }]
                }
            }
        )
        
        print("[SUCCESS] 제목 업데이트 완료!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 페이지 수정 실패: {str(e)}")
        return False

def create_test_database(parent_id):
    """테스트 데이터베이스 생성"""
    
    print(f"\n[3] 테스트 데이터베이스 생성")
    
    try:
        # 데이터베이스 생성
        test_db = client.databases.create(
            parent={"page_id": parent_id},
            title=[{
                "text": {
                    "content": "Test Tasks Database"
                }
            }],
            properties={
                "Name": {
                    "title": {}
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "To Do", "color": "gray"},
                            {"name": "In Progress", "color": "blue"},
                            {"name": "Done", "color": "green"}
                        ]
                    }
                },
                "Priority": {
                    "select": {
                        "options": [
                            {"name": "High", "color": "red"},
                            {"name": "Medium", "color": "yellow"},
                            {"name": "Low", "color": "green"}
                        ]
                    }
                },
                "Created": {
                    "date": {}
                },
                "Tags": {
                    "multi_select": {
                        "options": [
                            {"name": "API", "color": "purple"},
                            {"name": "Test", "color": "blue"},
                            {"name": "Integration", "color": "orange"}
                        ]
                    }
                }
            }
        )
        
        db_id = test_db['id']
        print(f"[SUCCESS] 데이터베이스 생성 완료: {db_id}")
        
        # 샘플 항목 추가
        print("[ADD] 샘플 태스크 추가 중...")
        
        sample_tasks = [
            {
                "name": "Test Notion API Connection",
                "status": "Done",
                "priority": "High",
                "tags": ["API", "Test"]
            },
            {
                "name": "Create Workspace Structure",
                "status": "In Progress",
                "priority": "High",
                "tags": ["Integration"]
            },
            {
                "name": "Implement Auto-sync",
                "status": "To Do",
                "priority": "Medium",
                "tags": ["API", "Integration"]
            }
        ]
        
        for task in sample_tasks:
            client.pages.create(
                parent={"database_id": db_id},
                properties={
                    "Name": {
                        "title": [{
                            "text": {
                                "content": task["name"]
                            }
                        }]
                    },
                    "Status": {
                        "select": {
                            "name": task["status"]
                        }
                    },
                    "Priority": {
                        "select": {
                            "name": task["priority"]
                        }
                    },
                    "Created": {
                        "date": {
                            "start": datetime.now().isoformat()
                        }
                    },
                    "Tags": {
                        "multi_select": [{"name": tag} for tag in task["tags"]]
                    }
                }
            )
            print(f"  - Added: {task['name']}")
        
        print("[SUCCESS] 샘플 태스크 추가 완료!")
        return db_id
        
    except Exception as e:
        print(f"[ERROR] 데이터베이스 생성 실패: {str(e)}")
        return None

def main():
    """메인 실행"""
    
    # 1. 테스트 페이지 찾기/생성
    page_id = find_or_create_test_page()
    
    if not page_id:
        print("\n[STOP] 페이지를 생성할 수 없습니다.")
        return
    
    # 2. 페이지 수정
    modify_success = modify_page(page_id)
    
    # 3. 데이터베이스 생성
    if modify_success:
        db_id = create_test_database(page_id)
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    
    print(f"\n[PAGE] 테스트 페이지 ID: {page_id}")
    print(f"[URL] https://notion.so/{page_id.replace('-', '')}")
    
    if modify_success:
        print("\n[SUCCESS] 모든 테스트 성공!")
        print("\nNotion에서 확인해보세요:")
        print("1. 'Claude Code Test Page' 페이지가 생성/업데이트됨")
        print("2. 새로운 콘텐츠가 추가됨")
        print("3. Test Tasks Database가 생성됨")
        print("\n다음 단계:")
        print("- NOTION_SETUP_GUIDE.md 참고하여 실제 프로젝트 워크스페이스 구성")
        print("- notion_integration_example.py로 자동화 구현")
    else:
        print("\n[WARNING] 일부 테스트 실패. 위의 오류 확인 필요")

if __name__ == "__main__":
    main()