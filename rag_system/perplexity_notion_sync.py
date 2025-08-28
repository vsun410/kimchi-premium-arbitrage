"""
Perplexity Research to Notion 자동 동기화
외부 리서치를 자동으로 Notion Knowledge Base에 추가
"""

import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from notion_client import Client
import httpx
import json

class PerplexityNotionSync:
    """Perplexity 리서치 결과를 Notion에 자동 저장"""
    
    def __init__(self, notion_token: str, perplexity_api_key: str):
        self.notion = Client(auth=notion_token)
        self.perplexity_key = perplexity_api_key
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        
    async def research_and_save(self, 
                                query: str, 
                                database_id: str,
                                category: str = "Research") -> str:
        """리서치 수행 후 Notion에 저장"""
        
        # 1. Perplexity로 리서치 수행
        research_result = await self._perform_research(query)
        
        # 2. 결과를 구조화
        structured_data = self._structure_research(query, research_result)
        
        # 3. Notion에 페이지 생성
        page_id = self._create_notion_page(database_id, structured_data, category)
        
        print(f"✅ Research saved to Notion: {page_id}")
        return page_id
    
    async def _perform_research(self, query: str) -> Dict:
        """Perplexity API로 리서치 수행"""
        
        # 구조화된 프롬프트
        structured_query = f"""
        Research the following topic for a trading system architecture:
        
        Query: {query}
        
        Please provide:
        1. Current industry best practices
        2. Technical implementation details
        3. Code examples if available
        4. Comparison of different approaches
        5. Specific recommendations for a kimchi premium trading system
        
        Format the response with clear sections and bullet points.
        """
        
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {"role": "user", "content": structured_query}
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "search_domain_filter": ["technical", "academic"],
            "return_citations": True,
            "search_recency_filter": "year"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.perplexity_url,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
        return response.json()
    
    def _structure_research(self, query: str, result: Dict) -> Dict:
        """리서치 결과를 구조화"""
        
        # 응답에서 콘텐츠 추출
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        citations = result.get('citations', [])
        
        # 섹션별로 파싱
        sections = self._parse_sections(content)
        
        return {
            'title': f"Research: {query}",
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'content': content,
            'sections': sections,
            'citations': citations,
            'metadata': {
                'model': result.get('model', 'unknown'),
                'search_results': result.get('search_results', 0)
            }
        }
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """콘텐츠를 섹션별로 파싱"""
        sections = {
            'summary': '',
            'best_practices': '',
            'implementation': '',
            'code_examples': '',
            'recommendations': ''
        }
        
        # 간단한 섹션 파싱 (실제로는 더 정교하게)
        current_section = 'summary'
        lines = content.split('\n')
        
        for line in lines:
            lower_line = line.lower()
            if 'best practice' in lower_line:
                current_section = 'best_practices'
            elif 'implement' in lower_line:
                current_section = 'implementation'
            elif 'code' in lower_line or '```' in line:
                current_section = 'code_examples'
            elif 'recommend' in lower_line:
                current_section = 'recommendations'
            
            sections[current_section] += line + '\n'
        
        return sections
    
    def _create_notion_page(self, 
                           database_id: str, 
                           data: Dict,
                           category: str) -> str:
        """Notion에 페이지 생성"""
        
        # 페이지 속성 설정
        properties = {
            "Title": {
                "title": [{"text": {"content": data['title']}}]
            },
            "Category": {
                "multi_select": [{"name": category}]
            },
            "Source Type": {
                "select": {"name": "Perplexity Research"}
            },
            "Tags": {
                "multi_select": [
                    {"name": "automated"},
                    {"name": "research"}
                ]
            }
        }
        
        # 페이지 콘텐츠 블록 생성
        children = self._create_content_blocks(data)
        
        # 페이지 생성
        response = self.notion.pages.create(
            parent={"database_id": database_id},
            properties=properties,
            children=children
        )
        
        return response['id']
    
    def _create_content_blocks(self, data: Dict) -> List[Dict]:
        """Notion 블록 생성"""
        blocks = []
        
        # 메타데이터 섹션
        blocks.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"text": {"content": "📊 Research Metadata"}}]
            }
        })
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "text": {
                        "content": f"Query: {data['query']}\nTimestamp: {data['timestamp']}"
                    }
                }]
            }
        })
        
        # 요약 섹션
        if data['sections']['summary']:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📝 Summary"}}]
                }
            })
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "text": {"content": data['sections']['summary'][:2000]}
                    }]
                }
            })
        
        # Best Practices 섹션
        if data['sections']['best_practices']:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "✅ Best Practices"}}]
                }
            })
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "text": {"content": data['sections']['best_practices'][:2000]}
                    }]
                }
            })
        
        # 코드 예제 섹션
        if data['sections']['code_examples']:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "💻 Code Examples"}}]
                }
            })
            
            # 코드 블록 추출 및 추가
            code_content = data['sections']['code_examples']
            if '```' in code_content:
                # 실제 코드 블록 파싱 (간단한 예시)
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{
                            "text": {"content": code_content[:2000]}
                        }],
                        "language": "python"
                    }
                })
        
        # 참고 자료 섹션
        if data.get('citations'):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📚 References"}}]
                }
            })
            
            for citation in data['citations'][:10]:  # 최대 10개
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "text": {"content": str(citation)}
                        }]
                    }
                })
        
        return blocks


class ResearchAutomation:
    """리서치 자동화 워크플로우"""
    
    def __init__(self, sync_client: PerplexityNotionSync):
        self.sync = sync_client
        self.research_queries = []
        
    async def batch_research(self, 
                            queries: List[str], 
                            database_id: str) -> List[str]:
        """여러 쿼리를 배치로 리서치"""
        
        page_ids = []
        
        for query in queries:
            try:
                page_id = await self.sync.research_and_save(
                    query=query,
                    database_id=database_id,
                    category="Architecture"
                )
                page_ids.append(page_id)
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error researching '{query}': {e}")
                
        return page_ids
    
    def generate_research_topics(self, base_topic: str) -> List[str]:
        """기본 주제에서 리서치 토픽 생성"""
        
        topics = [
            f"{base_topic} best practices",
            f"{base_topic} architecture patterns",
            f"{base_topic} performance optimization",
            f"{base_topic} security considerations",
            f"{base_topic} scaling strategies",
            f"{base_topic} monitoring and observability",
            f"{base_topic} testing strategies",
            f"{base_topic} deployment patterns"
        ]
        
        return topics


# 사용 예시
async def main():
    # 초기화
    notion_token = os.getenv("NOTION_TOKEN")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    knowledge_db = os.getenv("NOTION_KNOWLEDGE_DB")
    
    sync = PerplexityNotionSync(notion_token, perplexity_key)
    automation = ResearchAutomation(sync)
    
    # 1. 단일 리서치
    await sync.research_and_save(
        query="Real-time cryptocurrency arbitrage system architecture",
        database_id=knowledge_db,
        category="Trading Systems"
    )
    
    # 2. 배치 리서치
    topics = automation.generate_research_topics(
        "WebSocket streaming for trading"
    )
    
    page_ids = await automation.batch_research(
        queries=topics[:3],  # 처음 3개만
        database_id=knowledge_db
    )
    
    print(f"Created {len(page_ids)} research pages")


if __name__ == "__main__":
    asyncio.run(main())