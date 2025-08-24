# 브랜치 전략 가이드

## 🌳 현재 브랜치 구조

```
master (main)
├── feature/ml-models (Phase 2 완료) ✅
│   ├── 데이터 수집 완료
│   ├── 특징 엔지니어링 완료
│   └── XGBoost 학습 완료
│
└── phase3-backtesting (Phase 3 시작) 🚀 <- 현재 위치
    └── 백테스팅 시스템 구축 예정
```

## 📌 Phase별 브랜치 관리

### Phase 2: ML Models (완료)
- **브랜치**: `feature/ml-models`
- **태그**: `phase2-ml-training-complete`
- **상태**: 완료, 언제든 롤백 가능
- **포함 내용**:
  - ✅ 1년치 데이터 수집
  - ✅ 147개 특징 엔지니어링
  - ✅ XGBoost 모델 학습

### Phase 3: Backtesting (진행중)
- **브랜치**: `phase3-backtesting`
- **목표**: 백테스팅 시스템 구축
- **예정 작업**:
  - [ ] Walk-forward analysis
  - [ ] 거래 비용 시뮬레이션
  - [ ] 성과 분석 리포트
  - [ ] 파라미터 최적화

### Phase 4: Paper Trading (예정)
- **브랜치**: `phase4-paper-trading` (생성 예정)
- **목표**: 실시간 시뮬레이션

## 🔄 GitHub Desktop 작업 흐름

### 1. Phase 2 Push (백업)
```
1. GitHub Desktop 열기
2. Current Branch: feature/ml-models 선택
3. Commit 확인
4. "Push origin" 클릭
```

### 2. Phase 3 작업
```
1. Current Branch: phase3-backtesting 선택
2. 작업 진행
3. 주기적으로 Commit
4. 문제 없으면 Push
```

### 3. 문제 발생 시 롤백
```
옵션 1: 이전 커밋으로
- History → 원하는 커밋 우클릭 → "Revert"

옵션 2: Phase 2로 완전 롤백
- Branch → feature/ml-models 선택
```

## 🛡️ 안전 장치

### 태그로 체크포인트 관리
```bash
# Phase 2 완료 시점
git checkout phase2-ml-training-complete

# 데이터 수집 완료 시점
git checkout phase2-data-collection-complete
```

### 브랜치별 독립성
- 각 Phase는 독립된 브랜치
- 한 Phase 실패해도 다른 Phase 영향 없음
- 언제든 이전 Phase로 돌아갈 수 있음

## 📝 권장 사항

1. **각 Phase 완료 시**:
   - GitHub Desktop에서 Push
   - 태그 생성 (checkpoint)
   - 다음 Phase 브랜치 생성

2. **매일 작업 종료 시**:
   - Commit & Push
   - 진행 상황 기록

3. **문제 발생 시**:
   - 즉시 이전 브랜치로 전환
   - 문제 분석 후 새 브랜치로 재시도

## 🎯 현재 상태

- **완료**: Phase 1, Phase 2
- **진행중**: Phase 3 (phase3-backtesting)
- **대기**: Phase 4, 5, 6

이 전략으로 각 Phase를 안전하게 독립적으로 관리할 수 있습니다!