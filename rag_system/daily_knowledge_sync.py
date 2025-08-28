"""
Daily Knowledge Sync
매일 실행하여 Notion → Vector DB 동기화 및 새로운 리서치 수행
"""

import schedule
import time
import asyncio
from datetime import datetime
import os
from notion_rag_pipeline import NotionRAGPipeline
from perplexity_notion_sync import PerplexityNotionSync, ResearchAutomation

class DailyKnowledgeSync:
    """일일 지식 동기화 자동화"""
    
    def __init__(self):
        self.notion_token = os.getenv("NOTION_TOKEN")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.knowledge_db = os.getenv("NOTION_KNOWLEDGE_DB")
        
        self.rag_pipeline = NotionRAGPipeline(self.notion_token)
        self.research_sync = PerplexityNotionSync(
            self.notion_token, 
            self.perplexity_key
        )
        
    def sync_notion_to_vector_db(self):
        """Notion → Vector DB 동기화"""
        print(f"[{datetime.now()}] Starting Notion to Vector DB sync...")
        
        try:
            self.rag_pipeline.sync_from_notion(self.knowledge_db)
            print("✅ Vector DB sync completed")
        except Exception as e:
            print(f"❌ Sync failed: {e}")
    
    async def perform_daily_research(self):
        """일일 리서치 수행"""
        print(f"[{datetime.now()}] Starting daily research...")
        
        # 오늘의 리서치 주제 (동적으로 생성 가능)
        daily_topics = [
            "Latest cryptocurrency arbitrage strategies",
            "Real-time trading system optimizations",
            "Machine learning for price prediction updates"
        ]
        
        automation = ResearchAutomation(self.research_sync)
        
        for topic in daily_topics:
            try:
                await self.research_sync.research_and_save(
                    query=topic,
                    database_id=self.knowledge_db,
                    category="Daily Research"
                )
                await asyncio.sleep(5)  # Rate limiting
            except Exception as e:
                print(f"❌ Research failed for '{topic}': {e}")
    
    def generate_daily_report(self):
        """일일 리포트 생성"""
        print(f"[{datetime.now()}] Generating daily report...")
        
        # RAG 시스템으로 오늘의 인사이트 생성
        insights_query = """
        What are the most important insights from today's research
        for our kimchi premium trading system?
        """
        
        results = self.rag_pipeline.query(insights_query, n_results=10)
        
        report = f"""
# Daily Knowledge Report - {datetime.now().strftime('%Y-%m-%d')}

## 📊 Sync Statistics
- Total documents in vector DB: {len(results)}
- New research topics: 3
- Last sync: {datetime.now()}

## 🔍 Key Insights
"""
        for i, result in enumerate(results[:3], 1):
            report += f"\n{i}. {result['content'][:200]}...\n"
        
        # 리포트를 파일로 저장
        report_path = f"reports/daily_{datetime.now().strftime('%Y%m%d')}.md"
        os.makedirs("reports", exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to {report_path}")
    
    def run_daily_sync(self):
        """일일 동기화 실행"""
        # 1. Notion → Vector DB 동기화
        self.sync_notion_to_vector_db()
        
        # 2. 새로운 리서치 수행
        asyncio.run(self.perform_daily_research())
        
        # 3. 일일 리포트 생성
        self.generate_daily_report()
        
        print(f"✅ Daily sync completed at {datetime.now()}")


def main():
    """메인 스케줄러"""
    sync = DailyKnowledgeSync()
    
    # 매일 오전 9시에 실행
    schedule.every().day.at("09:00").do(sync.run_daily_sync)
    
    # 테스트를 위한 즉시 실행
    print("Running initial sync...")
    sync.run_daily_sync()
    
    # 스케줄러 실행
    print("Scheduler started. Waiting for scheduled tasks...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크


if __name__ == "__main__":
    main()