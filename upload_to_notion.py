"""
Notion API를 사용하여 CI 수정 보고서 업로드
"""
import requests
import json
from datetime import datetime

# Notion API 설정
NOTION_TOKEN = "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"
NOTION_VERSION = "2022-06-28"

# 헤더 설정
headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION
}

def create_page(parent_id, title, content_blocks):
    """Notion 페이지 생성"""
    url = "https://api.notion.com/v1/pages"
    
    data = {
        "parent": {"page_id": parent_id.replace("-", "")},
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
        },
        "children": content_blocks
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def read_markdown_file(filepath):
    """마크다운 파일 읽기"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def markdown_to_notion_blocks(markdown_text):
    """마크다운을 Notion 블록으로 변환 (간단한 버전)"""
    blocks = []
    lines = markdown_text.split('\n')
    
    for line in lines:
        if line.strip() == "":
            continue
            
        # 헤더 처리
        if line.startswith('# '):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": line[2:].strip()}
                    }]
                }
            })
        elif line.startswith('## '):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": line[3:].strip()}
                    }]
                }
            })
        elif line.startswith('### '):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": line[4:].strip()}
                    }]
                }
            })
        elif line.startswith('```'):
            # 코드 블록 시작/종료는 별도 처리 필요
            continue
        elif line.startswith('- '):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": line[2:].strip()}
                    }]
                }
            })
        else:
            # 일반 텍스트
            if line.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": line.strip()}
                        }]
                    }
                })
    
    return blocks[:100]  # Notion API 제한으로 100개 블록까지만

def main():
    """메인 실행 함수"""
    print("Notion 업로드 시작...")
    
    # 부모 페이지 ID (제공된 Notion 페이지)
    PARENT_PAGE_ID = "25cd547955ef8180adebe531bd5fc9c6"
    
    # 마크다운 파일 읽기
    markdown_content = read_markdown_file("docs/NOTION_COMPLETE_REPORT.md")
    
    # Notion 블록으로 변환
    blocks = markdown_to_notion_blocks(markdown_content)
    
    # 페이지 생성
    title = f"CI Pipeline 수정 보고서 - {datetime.now().strftime('%Y-%m-%d')}"
    
    print(f"페이지 생성 중: {title}")
    print(f"블록 수: {len(blocks)}")
    print(f"대상 페이지: {PARENT_PAGE_ID}")
    
    try:
        # 페이지 생성
        result = create_page(PARENT_PAGE_ID, title, blocks)
        
        if 'id' in result:
            print("SUCCESS: Upload completed!")
            print(f"Page ID: {result['id']}")
            print(f"URL: {result.get('url', 'No URL info')}")
        else:
            print(f"FAILED: Upload failed - {result}")
            if 'message' in result:
                print(f"Error message: {result['message']}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()