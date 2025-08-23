# CLAUDE.md - Kimchi Premium Futures Hedge Arbitrage System
*Updated by Claude on 2025-08-24*

## 🎯 프로젝트 개요

### 프로젝트명: Kimchi Premium Futures Hedge Arbitrage System
**목표**: 업비트 현물과 바이낸스 선물을 활용한 김치프리미엄 차익거래 자동화
**전략**: 델타 중립 헤지 (업비트 현물 매수 + 바이낸스 선물 숏)
**모델**: LSTM + XGBoost + RL (강화학습) 트리플 하이브리드
**자본금**: 각 거래소 2,000만원 (총 4,000만원)

### 핵심 가치
- ✅ ML 기반 객관적 진입/청산 신호
- ✅ 선물 헤지를 통한 리스크 중립화
- ✅ 24/7 완전 자동화 시스템

## 0) 프로젝트 운영 모드

### 역할 정의
- **Product Manager (사용자)**: PRD 정의, 태스크 승인/거절, 테스트 검증
- **Claude Code (AI)**: 코드 생성, 태스크 구현, 문서화
- **Task Master**: 작업 관리, 진행 추적, 의존성 관리

### 작업 규칙
1. **모든 작업은 승인 후 진행**: "이 태스크를 진행할까요? (Y/N)"
2. **비개발자 친화적 설명**: 기술 용어 최소화, 주석 상세화
3. **단계별 진행**: Phase 1부터 순차적 완료
4. **매 응답 마지막**: "다음 단계 제안: [간단 설명]"

### 저장소 원칙
- main은 보호 브랜치. 직접 푸시 금지
- 모든 변경은 feature/* 브랜치 → PR → 자동검사 통과 → 승인 후 병합

## 1) SAFETY MODE

### CRITICAL RULES
1. 절대 전체 파일 덮어쓰지 말 것(백업 필수)
2. 기능 추가 시 기존 설계 보존
3. 주요 변경 전 체크포인트 생성(git tag 또는 copy)
4. 스타일(디자인/CSS) 변경 금지(요청 시에만)

### 변경 전 루틴
1. 백업 생성
2. 변경 미리보기(diff, 영향 파일 리스트)
3. 승인 대기(승인 전 적용 금지)

### 백업 규칙
- 파일 단위: `filename.ext` → `filename.ext.bak.YYYYMMDD-HHMM`
- 브랜치 단위: `git switch -c backup/<issue>-<timestamp>`
- 태그 단위: `git tag safety-<issue>-<timestamp>`

## 2) TaskMaster 모드

### 기본 흐름
1. PRD 파싱으로 자동 태스크 생성
2. 태스크 검토 및 승인 받기
3. 각 단계 실행 후 상태 업데이트

### 토큰 최적화
- 계획 단계: 최소 맥락만(요구사항/파일 목록/변경 범위)
- 실행 단계: 관련 파일만 로드
- 전체 코드베이스 로드 금지

## 📋 작업 분할 목록 (Task Breakdown)

### Phase 1: Data Infrastructure & Basic Strategy
- [ ] P1.1: 프로젝트 구조 및 개발 환경 설정
- [ ] P1.2: CCXT Pro WebSocket 설정 (가격 + 오더북)
- [ ] P1.3: BTC 1년치 히스토리컬 데이터 수집
- [ ] P1.4: 오더북 15초 간격 수집 파이프라인
- [ ] P1.5: 김프율 계산 및 유동성 분석 모듈
- [ ] P1.6: 환율(USD/KRW) 데이터 통합
- [ ] P1.7: 단순 임계값 기반 진입/청산 로직
- [ ] P1.8: CSV 기반 데이터 저장 시스템

### Phase 2: ML Model Development
- [ ] P2.1: Feature engineering 파이프라인 (온체인 지표 포함)
- [ ] P2.2: ED-LSTM 모델 구현 (HuggingFace 기반)
- [ ] P2.3: XGBoost 앙상블 레이어 추가
- [ ] P2.4: PPO 에이전트 학습 (DQN 앙상블 옵션)
- [ ] P2.5: Optuna 하이퍼파라미터 최적화
- [ ] P2.6: 모델 평가 시스템 (Sharpe, Calmar Ratio)

### Phase 3: Backtesting System
- [ ] P3.1: Walk-forward analysis 구현
- [ ] P3.2: 거래 비용 및 슬리피지 시뮬레이션
- [ ] P3.3: 성과 분석 리포트 생성
- [ ] P3.4: 과적합 검증 로직
- [ ] P3.5: 파라미터 최적화

### Phase 4: Paper Trading & Monitoring
- [ ] P4.1: 실시간 데이터 스트림 연결 (BTC 우선)
- [ ] P4.2: 가상 잔고 관리 시스템
- [ ] P4.3: 실시간 포지션 추적
- [ ] P4.4: Streamlit 대시보드 구축
- [ ] P4.5: PostgreSQL 데이터베이스 설정
- [ ] P4.6: AWS CloudWatch/SNS 알림 시스템

### Phase 5: Advanced Features
- [ ] P5.1: Triangular Arbitrage 모듈 구현
- [ ] P5.2: ETH, XRP 확장 (멀티코인 지원)
- [ ] P5.3: Kelly Criterion + 1% rule 포지션 사이징
- [ ] P5.4: 김프 역전 자동 대응 시스템
- [ ] P5.5: 온체인 지표 통합 (Whale Alert)

### Phase 6: Production Deployment
- [ ] P6.1: AWS/GCP 인프라 설정
- [ ] P6.2: Docker 컨테이너화
- [ ] P6.3: 실거래 API 연동 및 테스트
- [ ] P6.4: 모니터링 및 알림 시스템
- [ ] P6.5: 백업 및 복구 전략

## 📊 진행 로그 (Progress Log)

### 2025-08-24
- ✅ 프로젝트 초기 설정 완료
- ✅ Git 리포지토리 생성 (private)
- ✅ Task #1: 프로젝트 구조 및 환경 설정 완료
- ✅ Task #2: CI/CD 파이프라인 구축 완료 (GitHub Actions)
- ✅ Task #3: API 키 보안 시스템 구현 (Fernet 암호화)
- ✅ Task #4: 로깅 및 모니터링 시스템 구축
- ✅ Task #5: WebSocket 연결 관리자 구현
- ✅ Task #6: 재연결 메커니즘 구현 (exponential backoff)
- ✅ Task #9: 환율 데이터 통합 (99.84% 정확도 달성)
- ✅ Task #10: 김치 프리미엄 계산기 구현
- ✅ CI 파이프라인 문제 해결 (fix/ci-pipeline 브랜치)
- ✅ Git 브랜치 워크플로우 문서화
- 🔄 Task #7: BTC 히스토리컬 데이터 수집 준비 중

## 🎯 최종 결정 항목 (Final Decisions)

### 승인된 항목
- (아직 없음)

### 거부된 항목
- (아직 없음)

### 대기 중인 항목
- Task Master를 통한 PRD 파싱 및 태스크 자동 생성

## 🚀 Success Criteria

### MVP 기준 (Phase 1-3 완료)
- ✅ 3개 코인 데이터 수집 작동
- ✅ LSTM 모델 학습 완료
- ✅ 백테스팅에서 양의 수익률
- ✅ 기본 리스크 관리 구현

### 성과 목표
- Sharpe Ratio > 1.5
- Calmar Ratio > 2.0
- Max Drawdown < 15%
- 월 평균 수익률 > 2%
- Win Rate > 60%

## 🔄 Git 브랜치 워크플로우 (Git Branch Workflow)

### 브랜치 전략
우리 프로젝트는 **GitHub Flow** 방식을 사용합니다:
- `master`: 항상 배포 가능한 상태 유지
- `feature/*`: 새로운 기능 개발
- `fix/*`: 버그 수정
- `experiment/*`: 실험적 기능

### 언제 브랜치를 만들어야 하나요?

#### 1. **새로운 Task 시작할 때** (권장) 
```bash
# 예시: Task #7 히스토리컬 데이터 수집
git checkout -b feature/historical-data
```

#### 2. **CI/CD 실패 수정할 때**
```bash
# 예시: CI 파이프라인 수정
git checkout -b fix/ci-pipeline
```

#### 3. **실험적 기능 테스트할 때**
```bash
# 예시: 새로운 ML 모델 테스트
git checkout -b experiment/new-ml-model
```

### 작업 플로우

#### 1. 브랜치 생성 및 작업
```bash
# 1. 새 브랜치 생성
git checkout -b feature/task-name

# 2. 작업 진행 (Claude가 코드 작성)
# 3. 커밋
git add .
git commit -m "feat: implement feature X"

# 4. GitHub Desktop에서 Push (사용자가 직접)
```

#### 2. Pull Request 및 CI 확인
- GitHub Desktop에서 "Push origin" 클릭
- GitHub 웹사이트에서 "Create Pull Request" 클릭
- CI 체크 자동 실행 (약 2-3분)
- 모든 체크 통과 확인

#### 3. CI 실패 시 대응
```bash
# 1. 실패 원인 확인 (GitHub Actions 로그)
# 2. 수정 작업
# 3. 커밋 및 푸시
git add .
git commit -m "fix: resolve CI issue"
# 4. GitHub Desktop에서 Push
```

#### 4. 머지 및 정리
```bash
# PR 머지 후
git checkout master
git pull origin master
git branch -d feature/task-name  # 로컬 브랜치 삭제
```

### GitHub Desktop 사용 가이드

#### Push 승인 프로세스
1. **Claude가 알림**: "Push 준비 완료! GitHub Desktop에서 push 해주세요"
2. **사용자 확인**: 커밋 내용 검토
3. **Push 실행**: Push origin 버튼 클릭
4. **결과 확인**: CI 성공/실패 확인

#### 롤백 방법 (문제 발생 시)
1. **GitHub Desktop에서**:
   - History 탭 → 문제 커밋 우클릭 → "Revert Changes in Commit"
   
2. **GitHub 웹사이트에서**:
   - PR 페이지 → "Revert" 버튼 클릭

### CI/CD 파이프라인 상태

#### 성공해야 하는 체크들:
- ✅ **Code Quality Check** (Black, Flake8, MyPy)
- ✅ **Run Tests** (Python 3.9, 3.10, 3.11)
- ✅ **Security Scan** (Bandit, Safety)
- ✅ **Build Check** (Import verification)

#### 일반적인 CI 실패 원인:
1. **Import 오류**: PYTHONPATH 설정 또는 잘못된 import
2. **포맷팅**: Black/isort 미적용
3. **의존성**: requirements.txt 버전 충돌
4. **보안**: 하드코딩된 키나 취약점

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md