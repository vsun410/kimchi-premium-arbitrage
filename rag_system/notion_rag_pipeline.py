"""
Notion to RAG Pipeline
Notion 데이터를 Vector DB로 자동 동기화하고 RAG 시스템 구축
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from notion_client import Client
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import json
from pathlib import Path

class NotionRAGPipeline:
    """Notion 기반 RAG 시스템 파이프라인"""
    
    def __init__(self, notion_token: str):
        self.notion = Client(auth=notion_token)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self._init_collection()
        
    def _init_collection(self):
        """ChromaDB 컬렉션 초기화"""
        return self.chroma_client.get_or_create_collection(
            name="notion_knowledge_base",
            metadata={"description": "Trading system knowledge base"}
        )
    
    def sync_from_notion(self, database_id: str):
        """Notion 데이터베이스에서 데이터 동기화"""
        
        # 1. Notion에서 모든 페이지 가져오기
        pages = self._fetch_all_pages(database_id)
        
        # 2. 각 페이지를 처리하여 벡터화
        for page in pages:
            self._process_page(page)
            
        print(f"✅ Synced {len(pages)} pages to vector database")
        
    def _fetch_all_pages(self, database_id: str) -> List[Dict]:
        """모든 페이지 가져오기"""
        pages = []
        has_more = True
        start_cursor = None
        
        while has_more:
            response = self.notion.databases.query(
                database_id=database_id,
                start_cursor=start_cursor
            )
            pages.extend(response['results'])
            has_more = response['has_more']
            start_cursor = response.get('next_cursor')
            
        return pages
    
    def _process_page(self, page: Dict):
        """페이지 처리 및 벡터화"""
        
        # 페이지 ID와 콘텐츠 추출
        page_id = page['id']
        content = self._extract_content(page)
        metadata = self._extract_metadata(page)
        
        # 콘텐츠 청킹
        chunks = self._chunk_content(content, chunk_size=500)
        
        # 각 청크를 임베딩하고 저장
        for i, chunk in enumerate(chunks):
            chunk_id = f"{page_id}_{i}"
            embedding = self.embedder.encode(chunk).tolist()
            
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "page_id": page_id,
                    "synced_at": datetime.now().isoformat()
                }]
            )
    
    def _extract_content(self, page: Dict) -> str:
        """페이지에서 텍스트 콘텐츠 추출"""
        # 페이지 블록 콘텐츠 가져오기
        blocks = self.notion.blocks.children.list(page['id'])
        
        content_parts = []
        for block in blocks['results']:
            text = self._extract_text_from_block(block)
            if text:
                content_parts.append(text)
                
        return "\n\n".join(content_parts)
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """블록에서 텍스트 추출"""
        block_type = block['type']
        
        if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3']:
            texts = block[block_type].get('rich_text', [])
            return ' '.join([t['plain_text'] for t in texts])
        elif block_type == 'code':
            code = block['code']
            language = code.get('language', 'python')
            texts = code.get('rich_text', [])
            code_text = ' '.join([t['plain_text'] for t in texts])
            return f"```{language}\n{code_text}\n```"
        elif block_type == 'bulleted_list_item':
            texts = block['bulleted_list_item'].get('rich_text', [])
            return '• ' + ' '.join([t['plain_text'] for t in texts])
        
        return ""
    
    def _extract_metadata(self, page: Dict) -> Dict:
        """페이지 메타데이터 추출"""
        props = page.get('properties', {})
        
        metadata = {
            "title": self._get_property_value(props.get('Title')),
            "category": self._get_property_value(props.get('Category')),
            "source_type": self._get_property_value(props.get('Source Type')),
            "tags": self._get_property_value(props.get('Tags')),
            "created_time": page.get('created_time'),
            "last_edited": page.get('last_edited_time')
        }
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _get_property_value(self, prop: Dict) -> Any:
        """Notion 속성 값 추출"""
        if not prop:
            return None
            
        prop_type = prop.get('type')
        
        if prop_type == 'title':
            texts = prop.get('title', [])
            return ' '.join([t['plain_text'] for t in texts])
        elif prop_type == 'select':
            return prop['select']['name'] if prop.get('select') else None
        elif prop_type == 'multi_select':
            return [s['name'] for s in prop.get('multi_select', [])]
        elif prop_type == 'rich_text':
            texts = prop.get('rich_text', [])
            return ' '.join([t['plain_text'] for t in texts])
            
        return None
    
    def _chunk_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """콘텐츠를 청크로 분할"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks if chunks else [content]
    
    def query(self, query: str, n_results: int = 5) -> List[Dict]:
        """RAG 쿼리 실행"""
        
        # 쿼리 임베딩
        query_embedding = self.embedder.encode(query).tolist()
        
        # 유사도 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
            
        return formatted_results


class NotionLLMIntegration:
    """Notion과 LLM 통합"""
    
    def __init__(self, rag_pipeline: NotionRAGPipeline):
        self.rag = rag_pipeline
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """프롬프트 템플릿 로드"""
        return {
            'prd_generation': """
Based on the following context from our knowledge base, generate a PRD:

Context:
{context}

Requirements:
{requirements}

Generate a comprehensive PRD including:
1. Problem Statement
2. Solution Overview
3. User Stories
4. Technical Requirements
5. Success Metrics
""",
            'architecture_design': """
Based on the following architectural patterns and best practices:

Context:
{context}

System Requirements:
{requirements}

Design a system architecture including:
1. Component Diagram
2. Data Flow
3. Technology Stack
4. Scaling Strategy
5. Security Considerations
""",
            'code_generation': """
Based on the following examples and patterns:

Context:
{context}

Task:
{task}

Generate production-ready code with:
1. Proper error handling
2. Type hints
3. Documentation
4. Unit tests
"""
        }
    
    def generate_prd(self, requirements: str) -> str:
        """PRD 생성"""
        # 관련 지식 검색
        context_results = self.rag.query(requirements, n_results=10)
        context = "\n\n".join([r['content'] for r in context_results])
        
        # 프롬프트 생성
        prompt = self.prompt_templates['prd_generation'].format(
            context=context,
            requirements=requirements
        )
        
        # LLM 호출 (여기서는 템플릿만 반환)
        return f"""
# Product Requirements Document

## Context Retrieved:
{context[:500]}...

## Generated PRD Structure:
[LLM will generate PRD based on context]

Prompt used:
{prompt[:200]}...
"""
    
    def design_architecture(self, requirements: str) -> str:
        """아키텍처 설계"""
        context_results = self.rag.query(requirements, n_results=10)
        context = "\n\n".join([r['content'] for r in context_results])
        
        prompt = self.prompt_templates['architecture_design'].format(
            context=context,
            requirements=requirements
        )
        
        return f"""
# System Architecture Design

## Retrieved Knowledge:
{context[:500]}...

## Architecture Proposal:
[LLM will design architecture based on context]
"""
    
    def generate_code(self, task: str) -> str:
        """코드 생성"""
        context_results = self.rag.query(task, n_results=5)
        context = "\n\n".join([r['content'] for r in context_results])
        
        prompt = self.prompt_templates['code_generation'].format(
            context=context,
            task=task
        )
        
        return f"""
# Generated Code

## Context from Knowledge Base:
{context[:500]}...

## Code Implementation:
[LLM will generate code based on patterns]
"""


# 사용 예시
if __name__ == "__main__":
    # 초기화
    notion_token = os.getenv("NOTION_TOKEN")
    pipeline = NotionRAGPipeline(notion_token)
    
    # Notion 동기화
    knowledge_db_id = os.getenv("NOTION_KNOWLEDGE_DB")
    if knowledge_db_id:
        pipeline.sync_from_notion(knowledge_db_id)
    
    # RAG 쿼리 테스트
    results = pipeline.query("마이크로서비스 아키텍처 설계 패턴")
    print(f"Found {len(results)} relevant results")
    
    # LLM 통합 테스트
    llm_integration = NotionLLMIntegration(pipeline)
    
    # PRD 생성
    prd = llm_integration.generate_prd(
        "김치프리미엄 실시간 차익거래 시스템"
    )
    print(prd[:500])
    
    # 아키텍처 설계
    architecture = llm_integration.design_architecture(
        "실시간 WebSocket 기반 트레이딩 시스템"
    )
    print(architecture[:500])