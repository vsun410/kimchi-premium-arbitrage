# 프로젝트 구조 (Modular Architecture)
*Updated: 2025-08-25*

## 📁 디렉토리 구조

```
kimchi-premium-arbitrage/
│
├── backend/                    # 백엔드 핵심 모듈
│   ├── api/                   # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── trading_api.py     # 거래 API
│   │   ├── data_api.py        # 데이터 API
│   │   └── monitoring_api.py  # 모니터링 API
│   │
│   ├── core/                  # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── exchange_manager.py    # 거래소 관리
│   │   ├── position_manager.py    # 포지션 관리
│   │   ├── risk_manager.py        # 리스크 관리
│   │   └── order_executor.py      # 주문 실행
│   │
│   ├── data/                  # 데이터 처리
│   │   ├── __init__.py
│   │   ├── collectors/        # 데이터 수집기
│   │   │   ├── price_collector.py
│   │   │   ├── orderbook_collector.py
│   │   │   └── rate_collector.py
│   │   ├── processors/        # 데이터 처리기
│   │   │   ├── kimchi_calculator.py
│   │   │   └── feature_extractor.py
│   │   └── storage/           # 데이터 저장
│   │       ├── database.py
│   │       └── cache.py
│   │
│   └── utils/                 # 유틸리티
│       ├── __init__.py
│       ├── logger.py
│       ├── config.py
│       └── helpers.py
│
├── strategies/                # 거래 전략 (독립 모듈)
│   ├── __init__.py
│   ├── base_strategy.py      # 전략 베이스 클래스
│   │
│   ├── mean_reversion/        # 평균회귀 전략
│   │   ├── __init__.py
│   │   ├── strategy.py        # 전략 구현
│   │   ├── config.py          # 설정
│   │   └── backtest.py        # 백테스트
│   │
│   ├── arbitrage/             # 차익거래 전략
│   │   ├── __init__.py
│   │   ├── kimchi_arbitrage.py
│   │   ├── triangular_arbitrage.py
│   │   └── config.py
│   │
│   └── ml_models/             # ML 기반 전략
│       ├── __init__.py
│       ├── lstm_predictor.py
│       ├── xgboost_model.py
│       └── rl_agent.py
│
├── frontend/                  # 프론트엔드 (독립 모듈)
│   ├── dashboard/             # 대시보드
│   │   ├── __init__.py
│   │   ├── realtime_monitor.py
│   │   ├── charts.py
│   │   └── templates/
│   │
│   ├── api_client/            # API 클라이언트
│   │   ├── __init__.py
│   │   └── client.py
│   │
│   └── components/            # UI 컴포넌트
│       ├── __init__.py
│       ├── position_panel.py
│       ├── trade_history.py
│       └── performance_metrics.py
│
├── paper_trading/             # Paper Trading 모듈
│   ├── __init__.py
│   ├── engine.py              # Paper Trading 엔진
│   ├── simulator.py           # 시뮬레이터
│   └── analyzer.py            # 성과 분석
│
├── tests/                     # 테스트
│   ├── backend/               # 백엔드 테스트
│   ├── strategies/            # 전략 테스트
│   └── integration/           # 통합 테스트
│
├── scripts/                   # 실행 스크립트
│   ├── start_paper_trading.py
│   ├── start_live_trading.py
│   ├── test_connections.py
│   └── monitor_dashboard.py
│
├── configs/                   # 설정 파일
│   ├── trading_config.yaml
│   ├── strategy_config.yaml
│   └── api_config.yaml
│
├── logs/                      # 로그
│   ├── trading/
│   ├── errors/
│   └── performance/
│
└── data/                      # 데이터 저장소
    ├── historical/
    ├── realtime/
    └── reports/
```

## 🎯 모듈 설명

### 1. Backend (백엔드)
- **독립성**: 전략과 프론트엔드에 독립적
- **역할**: 거래소 연결, 데이터 수집, 주문 실행
- **API**: RESTful/WebSocket API 제공

### 2. Strategies (전략)
- **독립성**: 완전히 독립적인 전략 모듈
- **플러그인 방식**: 새 전략 추가 용이
- **인터페이스**: BaseStrategy 상속

### 3. Frontend (프론트엔드)
- **독립성**: 백엔드 API를 통해서만 통신
- **확장성**: 웹/모바일 앱으로 확장 가능
- **기술**: Streamlit → React/Vue 마이그레이션 가능

## 🔄 모듈 간 통신

```python
# 예시: 전략 모듈 사용
from strategies.mean_reversion import MeanReversionStrategy
from backend.core import ExchangeManager

# 전략 초기화
strategy = MeanReversionStrategy(config_file='configs/strategy_config.yaml')

# 백엔드와 연결
exchange_manager = ExchangeManager()

# 전략 실행
signal = strategy.generate_signal(market_data)
if signal:
    exchange_manager.execute_order(signal)
```

## 🚀 실행 방법

### Paper Trading
```bash
# 백엔드 시작
python backend/api/trading_api.py

# 전략 시작
python strategies/mean_reversion/strategy.py --mode=paper

# 대시보드 시작
python frontend/dashboard/realtime_monitor.py
```

### Live Trading
```bash
# 백엔드 시작
python backend/api/trading_api.py --live

# 전략 시작
python strategies/mean_reversion/strategy.py --mode=live

# 모니터링
python scripts/monitor_dashboard.py
```

## 📦 패키지 구조

각 모듈은 독립적인 패키지로 관리:

```python
# backend/setup.py
setup(
    name='kimchi-backend',
    version='1.0.0',
    packages=['backend'],
    install_requires=[
        'ccxt>=4.0.0',
        'pandas>=2.0.0',
        'aiohttp>=3.9.0'
    ]
)

# strategies/setup.py
setup(
    name='kimchi-strategies',
    version='1.0.0',
    packages=['strategies'],
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0'
    ]
)

# frontend/setup.py
setup(
    name='kimchi-frontend',
    version='1.0.0',
    packages=['frontend'],
    install_requires=[
        'streamlit>=1.30.0',
        'plotly>=5.18.0'
    ]
)
```

## 🔧 환경 설정

### 개발 환경
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 개발 모드 설치
pip install -e backend/
pip install -e strategies/
pip install -e frontend/
```

### 프로덕션 환경
```bash
# Docker Compose
docker-compose up -d backend
docker-compose up -d strategy-mean-reversion
docker-compose up -d frontend
```

## 📊 모니터링

### Grafana Dashboard
- 실시간 김프 차트
- 포지션 상태
- 손익 추적
- 시스템 메트릭

### Prometheus Metrics
- API 응답 시간
- 주문 체결률
- 에러율
- 리소스 사용량

## 🔒 보안

### API 키 관리
- 환경 변수 사용
- HashiCorp Vault 연동
- 키 로테이션

### 접근 제어
- JWT 인증
- Rate Limiting
- IP Whitelist

## 📈 확장 계획

### Phase 1 (현재)
- Mean Reversion 전략
- Paper Trading
- 기본 대시보드

### Phase 2
- ML 모델 통합
- 멀티 전략 실행
- 웹 프론트엔드

### Phase 3
- 모바일 앱
- Cloud 배포
- 자동 스케일링

## 💡 개발 가이드

### 새 전략 추가
1. `strategies/` 디렉토리에 새 폴더 생성
2. `BaseStrategy` 클래스 상속
3. `generate_signal()` 메서드 구현
4. 설정 파일 추가

### API 엔드포인트 추가
1. `backend/api/` 에 새 라우트 추가
2. 비즈니스 로직은 `backend/core/` 에 구현
3. 테스트 작성
4. 문서 업데이트

### 프론트엔드 컴포넌트 추가
1. `frontend/components/` 에 새 컴포넌트 생성
2. API 클라이언트 사용
3. 대시보드에 통합
4. 스타일링 적용