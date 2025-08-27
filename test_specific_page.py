#!/usr/bin/env python3
"""
특정 Notion 페이지 접근 테스트
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime

# Load environment
env_notion_path = Path('.env.notion')
if env_notion_path.exists():
    load_dotenv(env_notion_path)

token = os.getenv('NOTION_TOKEN')
if not token:
    print("[ERROR] NOTION_TOKEN not found")
    sys.exit(1)

client = Client(auth=token)

def test_specific_page():
    """특정 페이지 접근 테스트"""
    
    # URL에서 추출한 페이지 ID (하이픈 추가 필요)
    page_id_raw = "25bd547955ef80fab225da25da498940"
    # Notion API는 하이픈이 있는 형식을 선호
    page_id = f"{page_id_raw[:8]}-{page_id_raw[8:12]}-{page_id_raw[12:16]}-{page_id_raw[16:20]}-{page_id_raw[20:]}"
    
    print("="*60)
    print("특정 페이지 접근 테스트")
    print("="*60)
    print(f"\n[INFO] 페이지 ID (원본): {page_id_raw}")
    print(f"[INFO] 페이지 ID (포맷): {page_id}")
    
    # 1. 페이지 정보 가져오기
    print("\n[1] 페이지 정보 조회 시도...")
    try:
        page = client.pages.retrieve(page_id=page_id)
        print("[SUCCESS] 페이지 접근 성공!")
        
        # 페이지 제목 추출
        title = "제목 없음"
        if 'properties' in page and 'title' in page['properties']:
            title_prop = page['properties']['title']
            if 'title' in title_prop and len(title_prop['title']) > 0:
                title = title_prop['title'][0]['plain_text']
        
        print(f"  - 페이지 제목: {title}")
        print(f"  - 생성 시간: {page['created_time']}")
        print(f"  - 마지막 수정: {page['last_edited_time']}")
        
    except Exception as e:
        print(f"[ERROR] 페이지 접근 실패: {str(e)}")
        print("\n가능한 원인:")
        print("1. Integration이 페이지에 추가되지 않음")
        print("2. 페이지 ID가 잘못됨")
        print("3. 권한 문제")
        return None
    
    # 2. 페이지에 콘텐츠 추가
    print("\n[2] 페이지에 콘텐츠 추가 시도...")
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
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
                            "content": f"Claude Code Test - {current_time}"
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
                            "content": "API 연결 테스트 성공! 이 내용은 Claude Code가 자동으로 추가한 것입니다."
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
                            "content": f"테스트 시간: {current_time}"
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
                            "content": "Integration: kimp"
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
                            "content": "상태: Connected"
                        }
                    }]
                }
            },
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{
                        "text": {
                            "content": "Success! Notion API와 Claude Code가 성공적으로 연결되었습니다. 이제 자동화를 구현할 수 있습니다."
                        }
                    }],
                    "icon": {
                        "emoji": "✅"
                    }
                }
            }
        ]
        
        client.blocks.children.append(
            block_id=page_id,
            children=new_blocks
        )
        
        print("[SUCCESS] 콘텐츠 추가 성공!")
        print("\nNotion 페이지를 확인해보세요:")
        print(f"https://notion.so/{page_id_raw}")
        
    except Exception as e:
        print(f"[ERROR] 콘텐츠 추가 실패: {str(e)}")
    
    # 3. 하위에 데이터베이스 생성 시도
    print("\n[3] 테스트 데이터베이스 생성 시도...")
    try:
        test_db = client.databases.create(
            parent={"page_id": page_id},
            title=[{
                "text": {
                    "content": "Claude Code Test Database"
                }
            }],
            properties={
                "Task": {
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
                "Created": {
                    "date": {}
                }
            }
        )
        
        db_id = test_db['id']
        print(f"[SUCCESS] 데이터베이스 생성 성공: {db_id}")
        
        # 샘플 아이템 추가
        client.pages.create(
            parent={"database_id": db_id},
            properties={
                "Task": {
                    "title": [{
                        "text": {
                            "content": "API 연결 테스트"
                        }
                    }]
                },
                "Status": {
                    "select": {
                        "name": "Done"
                    }
                },
                "Created": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                }
            }
        )
        print("[SUCCESS] 샘플 태스크 추가 완료!")
        
    except Exception as e:
        print(f"[ERROR] 데이터베이스 생성 실패: {str(e)}")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    test_specific_page()