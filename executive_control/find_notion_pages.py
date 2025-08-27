"""
Find accessible Notion pages
"""

import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.getenv("NOTION_TOKEN"))

try:
    # Search for pages
    response = notion.search(
        filter={"property": "object", "value": "page"},
        page_size=10
    )
    
    print("[INFO] Found accessible pages:")
    print("="*50)
    
    for page in response.get("results", []):
        page_id = page["id"]
        
        # Get title
        if "properties" in page and "title" in page["properties"]:
            title_prop = page["properties"]["title"]
            if title_prop.get("title"):
                title = title_prop["title"][0]["text"]["content"] if title_prop["title"] else "Untitled"
            else:
                title = "Untitled"
        elif "properties" in page:
            # Try to find any title property
            for prop_name, prop_value in page["properties"].items():
                if prop_value.get("type") == "title" and prop_value.get("title"):
                    title = prop_value["title"][0]["text"]["content"] if prop_value["title"] else "Untitled"
                    break
            else:
                title = "Untitled"
        else:
            title = "Untitled"
        
        url = page.get("url", "")
        
        print(f"\nTitle: {title}")
        print(f"ID: {page_id}")
        print(f"URL: {url}")
        print("-"*30)
    
    # 특정 페이지 찾기 (KimChi Premium)
    search_response = notion.search(
        query="KimChi",
        filter={"property": "object", "value": "page"}
    )
    
    if search_response.get("results"):
        print("\n[INFO] Found KimChi Premium page:")
        page = search_response["results"][0]
        print(f"ID: {page['id']}")
        print(f"URL: {page.get('url', '')}")
        
        # 이 페이지를 부모로 사용
        parent_id = page['id']
    
except Exception as e:
    print(f"[ERROR] Failed to search pages: {e}")

print("\n[TIP] Make sure to share the parent page with your integration in Notion")
print("Go to the page in Notion -> Share -> Invite -> Select your integration")