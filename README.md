# Kimchi Premium Futures Hedge Arbitrage System

[![CI Pipeline](https://github.com/yourusername/kimchi-premium-arbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/kimchi-premium-arbitrage/actions/workflows/ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

업비트 현물과 바이낸스 선물을 활용한 김치프리미엄 차익거래 자동화 시스템

## 🎯 프로젝트 개요

LSTM과 강화학습(RL)을 결합하여 최적의 김프 진입 타이밍을 포착하고, 델타 중립 헤지를 통해 리스크를 최소화하면서 안정적인 수익을 추구하는 알고리즘 트레이딩 플랫폼입니다.

### 핵심 전략
- **델타 중립**: 업비트 현물 매수 + 바이낸스 선물 숏
- **ML 기반 신호**: LSTM + XGBoost + RL 트리플 하이브리드
- **리스크 관리**: Kelly Criterion + 1% rule

## 📋 주요 기능

- ✅ 실시간 김치프리미엄 모니터링
- ✅ ML 기반 진입/청산 신호 생성
- ✅ 자동 헤지 포지션 관리
- ✅ 24/7 자동 거래 실행
- ✅ 실시간 리스크 모니터링

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/kimchi-premium-arbitrage.git
cd kimchi-premium-arbitrage

# Python 가상환경 생성 (Python 3.9+ 필요)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일을 열어 API 키 입력
# - UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
# - BINANCE_API_KEY, BINANCE_SECRET_KEY
# - 기타 필요한 설정값
```

### 3. 환경 확인

```bash
# 환경 설정 확인 스크립트 실행
python scripts/check_environment.py
```

## 📁 프로젝트 구조

```
kimchi-premium-arbitrage/
├── data/                 # 데이터 저장
│   ├── raw/             # 원본 데이터
│   ├── processed/       # 전처리된 데이터
│   └── cache/           # 임시 캐시
├── models/              # ML 모델
│   ├── lstm/           # LSTM 모델
│   ├── xgboost/        # XGBoost 모델
│   └── rl/             # 강화학습 모델
├── src/                 # 소스 코드
│   ├── data_collectors/ # 데이터 수집
│   ├── strategies/     # 거래 전략
│   ├── utils/          # 유틸리티
│   └── config/         # 설정 파일
├── tests/              # 테스트 코드
├── logs/               # 로그 파일
├── configs/            # 설정 파일
├── scripts/            # 실행 스크립트
└── docs/               # 문서
```

## 🛠 기술 스택

### Core
- **Python 3.9+**
- **CCXT Pro**: 거래소 API 통합
- **Pandas/NumPy**: 데이터 처리

### Machine Learning
- **PyTorch**: LSTM 모델
- **XGBoost**: 앙상블 학습
- **Stable-Baselines3**: 강화학습 (PPO/DQN)
- **Optuna**: 하이퍼파라미터 최적화

### Infrastructure
- **Docker**: 컨테이너화
- **PostgreSQL**: 데이터베이스 (Phase 4)
- **AWS**: 클라우드 배포 (Phase 6)

## 📊 성과 목표

- Sharpe Ratio > 1.5
- Calmar Ratio > 2.0
- Max Drawdown < 15%
- 월 평균 수익률 > 2%
- Win Rate > 60%

## ⚠️ 리스크 관리

- **포지션 크기**: 자본금의 1% (Kelly Criterion)
- **최대 노출**: 총 자본의 30%
- **손절**: 2 * ATR
- **김프 진입**: > 4%
- **김프 청산**: < 2% 또는 역전

## 🔄 개발 로드맵

### Phase 1: Data Infrastructure ✅
- [x] 프로젝트 구조 설정
- [ ] WebSocket 데이터 수집
- [ ] 히스토리컬 데이터 다운로드
- [ ] 김프율 계산 모듈

### Phase 2: ML Models
- [ ] Feature Engineering
- [ ] LSTM 모델 구현
- [ ] XGBoost 앙상블
- [ ] RL 에이전트

### Phase 3: Backtesting
- [ ] Walk-forward analysis
- [ ] 성과 평가 시스템

### Phase 4: Paper Trading
- [ ] 실시간 시뮬레이션
- [ ] 모니터링 대시보드

### Phase 5: Advanced Features
- [ ] Triangular Arbitrage
- [ ] Multi-coin 지원

### Phase 6: Production
- [ ] 클라우드 배포
- [ ] 24/7 자동 운영

## 📝 라이선스

이 프로젝트는 비공개 프로젝트입니다.

## 👥 기여

내부 개발팀만 기여 가능합니다.

## 📞 문의

프로젝트 관련 문의: [your-email@example.com]

---

*Last Updated: 2025-08-24*