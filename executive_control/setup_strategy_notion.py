"""
Notion 전략 문서화 시스템 구축
각 전략을 체계적으로 관리하고 구현과 연결
"""

import os
import json
import asyncio
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv
from typing import Dict, List
import time

load_dotenv()


class StrategyNotionSetup:
    """전략 관리 Notion 시스템 구축"""
    
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.load_config()
        
    def load_config(self):
        """기존 설정 로드"""
        try:
            with open("kimp_notion_config.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"[WARNING] Config load error: {e}")
            self.config = {}
    
    def save_config(self):
        """설정 저장"""
        with open("kimp_notion_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def create_strategy_database(self):
        """전략 데이터베이스 생성"""
        print("\n[1/4] 전략 데이터베이스 생성 중...")
        
        # 데이터베이스 생성
        strategy_db = self.notion.databases.create(
            parent={"page_id": self.config['project_page']},
            title=[{
                "type": "text",
                "text": {"content": "[STRATEGIES] 트레이딩 전략 관리"}
            }],
            properties={
                "전략명": {"title": {}},
                "전략 ID": {
                    "rich_text": {}
                },
                "카테고리": {
                    "select": {
                        "options": [
                            {"name": "차익거래", "color": "blue"},
                            {"name": "추세추종", "color": "green"},
                            {"name": "평균회귀", "color": "yellow"},
                            {"name": "ML/AI", "color": "purple"},
                            {"name": "하이브리드", "color": "red"},
                            {"name": "헤지", "color": "gray"},
                            {"name": "실험적", "color": "pink"}
                        ]
                    }
                },
                "상태": {
                    "select": {
                        "options": [
                            {"name": "아이디어", "color": "gray"},
                            {"name": "설계중", "color": "yellow"},
                            {"name": "구현중", "color": "orange"},
                            {"name": "백테스팅", "color": "blue"},
                            {"name": "페이퍼트레이딩", "color": "purple"},
                            {"name": "실거래", "color": "green"},
                            {"name": "중단", "color": "red"}
                        ]
                    }
                },
                "위험도": {
                    "select": {
                        "options": [
                            {"name": "매우낮음", "color": "green"},
                            {"name": "낮음", "color": "blue"},
                            {"name": "중간", "color": "yellow"},
                            {"name": "높음", "color": "orange"},
                            {"name": "매우높음", "color": "red"}
                        ]
                    }
                },
                "예상 수익률": {
                    "rich_text": {}
                },
                "예상 샤프비율": {
                    "number": {
                        "format": "number"
                    }
                },
                "최대 손실": {
                    "rich_text": {}
                },
                "필요 자본": {
                    "rich_text": {}
                },
                "구현 복잡도": {
                    "select": {
                        "options": [
                            {"name": "매우간단", "color": "green"},
                            {"name": "간단", "color": "blue"},
                            {"name": "보통", "color": "yellow"},
                            {"name": "복잡", "color": "orange"},
                            {"name": "매우복잡", "color": "red"}
                        ]
                    }
                },
                "관련 Task": {
                    "rich_text": {}
                },
                "구현 파일": {
                    "files": {}
                },
                "백테스팅 결과": {
                    "rich_text": {}
                },
                "실거래 성과": {
                    "rich_text": {}
                },
                "생성일": {
                    "date": {}
                },
                "업데이트": {
                    "last_edited_time": {}
                },
                "태그": {
                    "multi_select": {
                        "options": [
                            {"name": "김치프리미엄", "color": "red"},
                            {"name": "BTC", "color": "orange"},
                            {"name": "ETH", "color": "purple"},
                            {"name": "선물", "color": "blue"},
                            {"name": "현물", "color": "green"},
                            {"name": "차익거래", "color": "yellow"},
                            {"name": "자동화", "color": "pink"},
                            {"name": "고빈도", "color": "gray"}
                        ]
                    }
                }
            }
        )
        
        self.config['strategy_db'] = strategy_db['id']
        print(f"[SUCCESS] 전략 DB 생성: {strategy_db['id']}")
        return strategy_db['id']
    
    def add_strategy_template(self, strategy_data: Dict):
        """전략 템플릿 추가"""
        # 전략 페이지 생성
        page = self.notion.pages.create(
            parent={"database_id": self.config['strategy_db']},
            properties={
                "전략명": {"title": [{"text": {"content": strategy_data['name']}}]},
                "전략 ID": {"rich_text": [{"text": {"content": strategy_data['id']}}]},
                "카테고리": {"select": {"name": strategy_data['category']}},
                "상태": {"select": {"name": strategy_data['status']}},
                "위험도": {"select": {"name": strategy_data['risk']}},
                "예상 수익률": {"rich_text": [{"text": {"content": strategy_data['expected_return']}}]},
                "예상 샤프비율": {"number": strategy_data['expected_sharpe']},
                "최대 손실": {"rich_text": [{"text": {"content": strategy_data['max_loss']}}]},
                "필요 자본": {"rich_text": [{"text": {"content": strategy_data['capital']}}]},
                "구현 복잡도": {"select": {"name": strategy_data['complexity']}},
                "관련 Task": {"rich_text": [{"text": {"content": strategy_data.get('tasks', '')}}]},
                "생성일": {"date": {"start": datetime.now().isoformat()}},
                "태그": {"multi_select": [{"name": tag} for tag in strategy_data['tags']]}
            },
            children=strategy_data['content']
        )
        
        return page['id']
    
    def create_current_strategies(self):
        """현재 구현된 전략들 문서화"""
        print("\n[2/4] 현재 전략들 문서화 중...")
        
        strategies = [
            {
                "name": "김치프리미엄 + 추세돌파 하이브리드",
                "id": "KIMP_TREND_001",
                "category": "하이브리드",
                "status": "백테스팅",
                "risk": "중간",
                "expected_return": "월 5-10%",
                "expected_sharpe": 2.5,
                "max_loss": "-10% (월간)",
                "capital": "4,000만원 (거래소별 2,000만원)",
                "complexity": "복잡",
                "tasks": "Task #11, #29, #30, #31, #32",
                "tags": ["김치프리미엄", "BTC", "선물", "현물", "차익거래", "자동화"],
                "content": [
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"text": {"content": "전략 개요"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": 
                            "김치프리미엄 차익거래와 추세추종을 결합한 하이브리드 전략입니다. "
                            "기본적으로 델타 중립 헤지를 유지하면서, 시장 추세에 따라 포지션 비율을 동적으로 조정합니다."
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "핵심 메커니즘"}}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "델타 중립 헤지: 업비트 현물 LONG + 바이낸스 선물 SHORT"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "상승 추세: 현물 70% / 선물 30%"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "하락 추세: 현물 30% / 선물 70%"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "진입 조건"}}]}
                    },
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": [{"text": {"content": 
"""김프 > 3%: 기본 진입
김프 > 5%: 포지션 2배
김프 > 7%: 최대 포지션 (3배)

추세 필터:
- MA20 > MA50 > MA200: 상승
- RSI > 50: 강세
- MACD 골든크로스: 매수"""
                            }}],
                            "language": "python"
                        }
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "구현 상태"}}]}
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "기본 김프 전략 구현"}}],
                            "checked": True
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "추세 분석 엔진 구현"}}],
                            "checked": True
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "동적 헤지 시스템 구현"}}],
                            "checked": True
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "백테스팅 완료"}}],
                            "checked": True
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "실시간 실행 엔진"}}],
                            "checked": False
                        }
                    }
                ]
            },
            {
                "name": "PPO 강화학습 자동 트레이딩",
                "id": "PPO_RL_001",
                "category": "ML/AI",
                "status": "구현중",
                "risk": "높음",
                "expected_return": "월 8-15%",
                "expected_sharpe": 1.5,
                "max_loss": "-15% (월간)",
                "capital": "1,000만원",
                "complexity": "매우복잡",
                "tasks": "Task #17 (subtasks 17.1-17.5)",
                "tags": ["ML/AI", "BTC", "자동화", "고빈도"],
                "content": [
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"text": {"content": "전략 개요"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": 
                            "PPO (Proximal Policy Optimization) 강화학습을 사용한 자동 트레이딩 전략입니다. "
                            "에이전트가 시장 상황을 학습하고 최적의 진입/청산 타이밍을 자동으로 결정합니다."
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "기술 스택"}}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "OpenAI Gym 환경: 20차원 상태 공간"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "stable-baselines3 PPO 알고리즘"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": 
                            "Sharpe Ratio 기반 보상 함수"
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "구현 진행률"}}]}
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "거래 환경 클래스 구현 (17.1)"}}],
                            "checked": True
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "보상 함수 설계 (17.2)"}}],
                            "checked": False
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "PPO 에이전트 구현 (17.3)"}}],
                            "checked": False
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "경험 재생 버퍼 (17.4)"}}],
                            "checked": False
                        }
                    },
                    {
                        "object": "block",
                        "type": "to_do",
                        "to_do": {
                            "rich_text": [{"text": {"content": "학습 파이프라인 (17.5)"}}],
                            "checked": False
                        }
                    }
                ]
            },
            {
                "name": "삼각 차익거래 (Triangular Arbitrage)",
                "id": "TRI_ARB_001",
                "category": "차익거래",
                "status": "아이디어",
                "risk": "낮음",
                "expected_return": "월 2-4%",
                "expected_sharpe": 3.0,
                "max_loss": "-2% (월간)",
                "capital": "500만원",
                "complexity": "보통",
                "tasks": "미정",
                "tags": ["차익거래", "고빈도", "자동화"],
                "content": [
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"text": {"content": "전략 개요"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": 
                            "3개 통화쌍 간의 가격 불일치를 활용한 무위험 차익거래 전략입니다. "
                            "BTC/USDT → USDT/KRW → KRW/BTC 순환 거래로 수익을 창출합니다."
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "구현 계획"}}]}
                    },
                    {
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{"text": {"content": 
                                "아직 구현되지 않은 전략입니다. 향후 개발 예정입니다."
                            }}],
                            "icon": {"emoji": "💡"}
                        }
                    }
                ]
            },
            {
                "name": "역프리미엄 활용 전략",
                "id": "REV_PREM_001",
                "category": "차익거래",
                "status": "백테스팅",
                "risk": "중간",
                "expected_return": "월 3-5%",
                "expected_sharpe": 2.0,
                "max_loss": "-5% (월간)",
                "capital": "2,000만원",
                "complexity": "보통",
                "tasks": "Task #32",
                "tags": ["김치프리미엄", "차익거래", "BTC"],
                "content": [
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"text": {"content": "전략 개요"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": 
                            "김치프리미엄이 음수(역프리미엄)로 전환될 때를 활용하는 전략입니다. "
                            "한국 시장이 해외보다 저평가될 때 반대 포지션을 구축합니다."
                        }}]}
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "핵심 로직"}}]}
                    },
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": [{"text": {"content": 
"""if 김프율 < -2%:
    # 역프리미엄 진입
    바이낸스 현물: LONG
    업비트 선물: SHORT (있다면)
    
if 김프율 > 0:
    # 포지션 청산
    이익 실현"""
                            }}],
                            "language": "python"
                        }
                    }
                ]
            }
        ]
        
        # 각 전략 추가
        for strategy in strategies:
            try:
                page_id = self.add_strategy_template(strategy)
                print(f"  [OK] {strategy['name']} 추가 완료")
                time.sleep(0.5)  # API 제한 방지
            except Exception as e:
                print(f"  [ERROR] {strategy['name']} 추가 실패: {e}")
    
    def create_strategy_template_page(self):
        """새 전략 작성용 템플릿 페이지 생성"""
        print("\n[3/4] 전략 템플릿 페이지 생성 중...")
        
        template_page = self.notion.pages.create(
            parent={"page_id": self.config['project_page']},
            properties={
                "title": [{
                    "text": {"content": "[TEMPLATE] 새 전략 작성 가이드"}
                }]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"text": {"content": "새 전략 작성 템플릿"}}]}
                },
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"text": {"content": 
                            "이 템플릿을 복사하여 새로운 전략을 작성하세요. "
                            "각 섹션을 빠짐없이 채워주시면 자동으로 구현 태스크가 생성됩니다."
                        }}],
                        "icon": {"emoji": "📝"}
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "1. 전략 기본 정보"}}]}
                },
                {
                    "object": "block",
                    "type": "table",
                    "table": {
                        "table_width": 2,
                        "has_column_header": False,
                        "has_row_header": False,
                        "children": [
                            {
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "전략명"}}],
                                        [{"text": {"content": "[여기에 작성]"}}]
                                    ]
                                }
                            },
                            {
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "카테고리"}}],
                                        [{"text": {"content": "차익거래/추세추종/평균회귀/ML/하이브리드"}}]
                                    ]
                                }
                            },
                            {
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "예상 수익률"}}],
                                        [{"text": {"content": "월 X-Y%"}}]
                                    ]
                                }
                            },
                            {
                                "type": "table_row",
                                "table_row": {
                                    "cells": [
                                        [{"text": {"content": "위험도"}}],
                                        [{"text": {"content": "낮음/중간/높음"}}]
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "2. 전략 설명"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": 
                        "[전략의 핵심 아이디어와 작동 원리를 설명하세요]"
                    }}]}
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "3. 진입 조건"}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "조건 1: "}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "조건 2: "}}]}
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "4. 청산 조건"}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "이익 실현: "}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "손절매: "}}]}
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "5. 리스크 관리"}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "최대 포지션 크기: "}}]}
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": "최대 손실 한도: "}}]}
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "6. 구현 요구사항"}}]}
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": "데이터 수집"}}],
                        "checked": False
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": "신호 생성 로직"}}],
                        "checked": False
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": "백테스팅"}}],
                        "checked": False
                    }
                },
                {
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": "실시간 실행"}}],
                        "checked": False
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "7. 예제 코드"}}]}
                },
                {
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{"text": {"content": 
"""# 전략 의사코드
def strategy_logic():
    if entry_condition:
        enter_position()
    elif exit_condition:
        exit_position()
    else:
        hold()"""
                        }}],
                        "language": "python"
                    }
                }
            ]
        )
        
        self.config['strategy_template'] = template_page['id']
        print(f"[SUCCESS] 템플릿 페이지 생성: {template_page['id']}")
    
    def add_strategy_link_to_main_page(self):
        """메인 프로젝트 페이지에 전략 링크 추가"""
        print("\n[4/4] 메인 페이지에 전략 섹션 추가 중...")
        
        # 전략 섹션 추가
        self.notion.blocks.children.append(
            self.config['project_page'],
            children=[
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"text": {"content": "📈 트레이딩 전략 관리"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": 
                        "모든 트레이딩 전략을 체계적으로 문서화하고 관리합니다. "
                        "새 전략을 추가하면 자동으로 구현 태스크가 생성됩니다."
                    }}]}
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
                                            "type": "link_to_page",
                                            "link_to_page": {
                                                "type": "database_id",
                                                "database_id": self.config['strategy_db']
                                            }
                                        },
                                        {
                                            "object": "block",
                                            "type": "paragraph",
                                            "paragraph": {"rich_text": [{"text": {"content": 
                                                "전략 데이터베이스에서 모든 전략을 관리합니다"
                                            }}]}
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
                                            "type": "link_to_page",
                                            "link_to_page": {
                                                "type": "page_id",
                                                "page_id": self.config['strategy_template']
                                            }
                                        },
                                        {
                                            "object": "block",
                                            "type": "paragraph",
                                            "paragraph": {"rich_text": [{"text": {"content": 
                                                "새 전략 작성 템플릿을 사용하세요"
                                            }}]}
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"text": {"content": 
                            "전략 상태: 아이디어 → 설계중 → 구현중 → 백테스팅 → 페이퍼트레이딩 → 실거래"
                        }}],
                        "icon": {"emoji": "🔄"}
                    }
                }
            ]
        )
        
        print("[SUCCESS] 메인 페이지 업데이트 완료")
    
    def run_setup(self):
        """전체 설정 실행"""
        print("="*60)
        print("   Notion 전략 관리 시스템 구축")
        print("="*60)
        
        try:
            # 1. 전략 데이터베이스 생성
            self.create_strategy_database()
            
            # 2. 현재 전략들 추가
            self.create_current_strategies()
            
            # 3. 템플릿 페이지 생성
            self.create_strategy_template_page()
            
            # 4. 메인 페이지 업데이트
            self.add_strategy_link_to_main_page()
            
            # 설정 저장
            self.save_config()
            
            print("\n" + "="*60)
            print("   [SUCCESS] 전략 관리 시스템 구축 완료!")
            print("="*60)
            print(f"\n전략 DB: https://notion.so/{self.config['strategy_db'].replace('-', '')}")
            print(f"템플릿: https://notion.so/{self.config['strategy_template'].replace('-', '')}")
            print(f"프로젝트: https://notion.so/{self.config['project_page'].replace('-', '')}")
            
        except Exception as e:
            print(f"\n[ERROR] 설정 실패: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    setup = StrategyNotionSetup()
    setup.run_setup()