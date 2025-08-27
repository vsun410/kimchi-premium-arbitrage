"""
Vision Guardian - 프로젝트 비전 수호 시스템
코드가 원래 비전과 설계에서 벗어나지 않도록 감시하고 검증
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

@dataclass
class ProjectVision:
    """프로젝트의 불변 비전"""
    core_objectives: List[str]
    red_lines: List[str]  # 절대 넘으면 안되는 선
    success_metrics: Dict[str, float]
    architecture_principles: List[str]
    
@dataclass
class ValidationResult:
    """코드 검증 결과"""
    task_id: str
    approved: bool
    business_score: float
    architecture_score: float
    drift_score: float
    violations: List[str]
    recommendations: List[str]
    timestamp: datetime


class VisionGuardian:
    """
    프로젝트 비전 수호자
    모든 코드 변경사항이 원래 비전과 일치하는지 검증
    """
    
    def __init__(self, prd_path: str, architecture_path: str):
        """
        Args:
            prd_path: PRD 문서 경로
            architecture_path: 아키텍처 문서 경로
        """
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = self._initialize_vector_db()
        
        # 핵심 비전 로드
        self.vision = self._load_project_vision(prd_path)
        self.architecture = self._load_architecture_spec(architecture_path)
        
        # 비전과 아키텍처를 벡터화하여 저장
        self._index_core_documents()
        
    def _initialize_vector_db(self) -> chromadb.Collection:
        """벡터 DB 초기화"""
        client = chromadb.PersistentClient(path="./executive_control/chroma_db")
        
        return client.get_or_create_collection(
            name="project_vision",
            metadata={"description": "Core project vision and architecture"}
        )
    
    def _load_project_vision(self, prd_path: str) -> ProjectVision:
        """PRD에서 프로젝트 비전 추출"""
        with open(prd_path, 'r', encoding='utf-8') as f:
            prd_content = f.read()
        
        # 실제로는 PRD 파싱 로직이 들어가야 함
        # 여기서는 예시로 하드코딩
        return ProjectVision(
            core_objectives=[
                "실시간 김치프리미엄 차익거래 자동화",
                "리스크 중립적 헤지 포지션 유지",
                "24/7 무중단 운영 시스템",
                "ML 기반 객관적 진입/청산 시그널"
            ],
            red_lines=[
                "동시 양방향 포지션 한도 초과 금지",
                "단일 거래 자본금 10% 초과 금지",
                "수동 개입 필요한 로직 금지",
                "테스트되지 않은 코드 배포 금지"
            ],
            success_metrics={
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.15,
                "win_rate": 0.6,
                "system_uptime": 0.99
            },
            architecture_principles=[
                "마이크로서비스 아키텍처",
                "이벤트 드리븐 설계",
                "무상태 컴포넌트",
                "장애 격리 원칙"
            ]
        )
    
    def _load_architecture_spec(self, arch_path: str) -> Dict:
        """아키텍처 명세 로드"""
        if Path(arch_path).exists():
            with open(arch_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 기본 아키텍처 명세
        return {
            "components": {
                "trading_core": {
                    "interfaces": ["WebSocket", "REST API"],
                    "dependencies": ["market_data", "risk_manager"],
                    "constraints": ["latency < 100ms", "stateless"]
                },
                "ml_engine": {
                    "interfaces": ["gRPC", "Message Queue"],
                    "dependencies": ["historical_data", "feature_store"],
                    "constraints": ["prediction_time < 500ms"]
                },
                "dashboard": {
                    "interfaces": ["REST API", "WebSocket"],
                    "dependencies": ["trading_core", "ml_engine"],
                    "constraints": ["real-time updates", "responsive UI"]
                }
            },
            "data_flow": {
                "market_data": "trading_core",
                "trading_core": "risk_manager",
                "ml_engine": "trading_core",
                "all": "dashboard"
            }
        }
    
    def _index_core_documents(self):
        """핵심 문서를 벡터화하여 인덱싱"""
        documents = []
        metadatas = []
        ids = []
        
        # 비전 문서 인덱싱
        for i, objective in enumerate(self.vision.core_objectives):
            documents.append(objective)
            metadatas.append({"type": "core_objective", "index": i})
            ids.append(f"objective_{i}")
        
        # Red lines 인덱싱
        for i, red_line in enumerate(self.vision.red_lines):
            documents.append(red_line)
            metadatas.append({"type": "red_line", "index": i})
            ids.append(f"red_line_{i}")
        
        # 아키텍처 원칙 인덱싱
        for i, principle in enumerate(self.vision.architecture_principles):
            documents.append(principle)
            metadatas.append({"type": "architecture_principle", "index": i})
            ids.append(f"principle_{i}")
        
        # 벡터화 및 저장
        if documents:
            embeddings = self.embedder.encode(documents).tolist()
            self.vector_db.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
    
    def validate_code(self, code: str, task_spec: Dict) -> ValidationResult:
        """
        제출된 코드 검증
        
        Args:
            code: 검증할 코드
            task_spec: 작업 명세
            
        Returns:
            ValidationResult: 검증 결과
        """
        violations = []
        recommendations = []
        
        # 1. 비즈니스 로직 준수도 검사
        business_score = self._check_business_compliance(code, task_spec)
        
        # 2. 아키텍처 준수도 검사
        arch_score = self._check_architecture_compliance(code, task_spec)
        
        # 3. 코드 드리프트 계산
        drift_score = self._calculate_drift(code, task_spec)
        
        # 4. Red line 위반 검사
        red_line_violations = self._check_red_lines(code)
        violations.extend(red_line_violations)
        
        # 5. 승인 여부 결정
        approved = (
            business_score >= 0.8 and
            arch_score >= 0.8 and
            drift_score < 0.2 and
            len(red_line_violations) == 0
        )
        
        # 6. 개선 권고사항 생성
        if business_score < 0.9:
            recommendations.append(f"비즈니스 로직 준수도 개선 필요 (현재: {business_score:.2f})")
        if arch_score < 0.9:
            recommendations.append(f"아키텍처 패턴 준수 필요 (현재: {arch_score:.2f})")
        if drift_score > 0.1:
            recommendations.append(f"원본 명세와의 차이 줄이기 필요 (드리프트: {drift_score:.2f})")
        
        return ValidationResult(
            task_id=task_spec.get('task_id', 'unknown'),
            approved=approved,
            business_score=business_score,
            architecture_score=arch_score,
            drift_score=drift_score,
            violations=violations,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _check_business_compliance(self, code: str, task_spec: Dict) -> float:
        """비즈니스 로직 준수도 검사"""
        # 코드를 임베딩하고 비전과 비교
        code_embedding = self.embedder.encode(code)
        
        # 핵심 목표와의 유사도 계산
        results = self.vector_db.query(
            query_embeddings=[code_embedding.tolist()],
            where={"type": "core_objective"},
            n_results=len(self.vision.core_objectives)
        )
        
        if results['distances'][0]:
            # 거리를 유사도로 변환 (1 - normalized_distance)
            similarities = [1 - (d / 2) for d in results['distances'][0]]
            return np.mean(similarities)
        
        return 0.5  # 기본값
    
    def _check_architecture_compliance(self, code: str, task_spec: Dict) -> float:
        """아키텍처 준수도 검사"""
        compliance_checks = []
        
        # 컴포넌트 확인
        component = task_spec.get('component', '')
        if component in self.architecture['components']:
            component_spec = self.architecture['components'][component]
            
            # 인터페이스 체크
            for interface in component_spec['interfaces']:
                if interface.lower() in code.lower():
                    compliance_checks.append(1.0)
                else:
                    compliance_checks.append(0.5)
            
            # 제약사항 체크
            for constraint in component_spec['constraints']:
                # 간단한 키워드 체크 (실제로는 더 정교해야 함)
                if 'latency' in constraint and 'async' in code:
                    compliance_checks.append(1.0)
                elif 'stateless' in constraint and 'self.state' not in code:
                    compliance_checks.append(1.0)
                else:
                    compliance_checks.append(0.7)
        
        return np.mean(compliance_checks) if compliance_checks else 0.8
    
    def _calculate_drift(self, code: str, task_spec: Dict) -> float:
        """코드 드리프트 계산"""
        # 원본 명세와 현재 코드의 차이 계산
        spec_text = json.dumps(task_spec)
        
        spec_embedding = self.embedder.encode(spec_text)
        code_embedding = self.embedder.encode(code)
        
        # 코사인 유사도 계산
        similarity = np.dot(spec_embedding, code_embedding) / (
            np.linalg.norm(spec_embedding) * np.linalg.norm(code_embedding)
        )
        
        # 드리프트는 1 - 유사도
        return max(0, 1 - similarity)
    
    def _check_red_lines(self, code: str) -> List[str]:
        """Red line 위반 검사"""
        violations = []
        
        # 각 red line에 대해 검사
        red_line_checks = {
            "동시 양방향 포지션": ["long_position", "short_position", "simultaneous"],
            "자본금 10% 초과": ["position_size", "capital", "0.1"],
            "수동 개입": ["input(", "manual", "user_input"],
            "테스트되지 않은": ["# TODO", "# FIXME", "raise NotImplementedError"]
        }
        
        for red_line, keywords in red_line_checks.items():
            if all(keyword in code.lower() for keyword in keywords):
                violations.append(f"Red line 위반 가능성: {red_line}")
        
        return violations
    
    def generate_feedback(self, result: ValidationResult) -> str:
        """개발자를 위한 피드백 생성"""
        feedback = f"""
# Code Validation Report
## Task: {result.task_id}
## Status: {'✅ APPROVED' if result.approved else '❌ NEEDS REVISION'}

### Scores
- Business Logic Compliance: {result.business_score:.1%}
- Architecture Compliance: {result.architecture_score:.1%}
- Code Drift: {result.drift_score:.1%}

### Violations
"""
        for violation in result.violations:
            feedback += f"- ⚠️ {violation}\n"
        
        feedback += "\n### Recommendations\n"
        for rec in result.recommendations:
            feedback += f"- 💡 {rec}\n"
        
        return feedback
    
    def get_compliance_report(self) -> Dict:
        """전체 준수도 리포트 생성"""
        # 최근 검증 결과들을 집계하여 리포트 생성
        return {
            "total_validations": 0,  # DB에서 조회
            "approval_rate": 0.0,
            "average_business_score": 0.0,
            "average_architecture_score": 0.0,
            "average_drift": 0.0,
            "common_violations": [],
            "timestamp": datetime.now().isoformat()
        }


# 사용 예시
if __name__ == "__main__":
    # Vision Guardian 초기화
    guardian = VisionGuardian(
        prd_path="./prd.md",
        architecture_path="./architecture.json"
    )
    
    # 샘플 코드 검증
    sample_code = """
    async def execute_trade(self, signal):
        # Check risk limits
        if self.position_size > self.capital * 0.09:  # Within 10% limit
            return False
        
        # Execute via WebSocket
        async with self.ws_client as ws:
            result = await ws.send_order(signal)
            
        return result
    """
    
    task_spec = {
        "task_id": "TR-001",
        "component": "trading_core",
        "description": "Implement trade execution"
    }
    
    # 검증 수행
    result = guardian.validate_code(sample_code, task_spec)
    
    # 피드백 출력
    print(guardian.generate_feedback(result))