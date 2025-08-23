# API 키 보안 가이드

## 개요

이 문서는 Kimchi Premium Arbitrage System의 API 키 보안 관리 방법을 설명합니다.

## 보안 기능

### 1. API 키 암호화
- Fernet 대칭키 암호화 사용
- 마스터 키는 `.keys/master.key`에 저장 (Git 제외)
- API 키는 암호화되어 `.keys/api_keys.enc`에 저장

### 2. 환경변수 관리
- `.env` 파일은 Git에서 제외
- `.env.example` 템플릿 제공
- 암호화된 저장소 우선 사용

### 3. 키 로테이션
- 90일마다 API 키 로테이션 권장
- 자동 체크 스케줄러 제공
- 마스터 키 로테이션 지원

## 초기 설정

### 1. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# UPBIT_ACCESS_KEY=your_actual_key_here
# UPBIT_SECRET_KEY=your_actual_secret_here
# BINANCE_API_KEY=your_actual_key_here
# BINANCE_SECRET_KEY=your_actual_secret_here
```

### 2. API 키 암호화

```bash
# 환경변수의 API 키를 암호화하여 저장
python src/utils/crypto_manager.py encrypt

# 암호화된 키 확인
python src/utils/crypto_manager.py verify
```

### 3. API 연결 테스트

```bash
# 모든 API 연결 테스트
python scripts/test_api_connection.py

# 개별 테스트
python scripts/test_api_connection.py upbit
python scripts/test_api_connection.py binance
python scripts/test_api_connection.py rate
```

## 키 로테이션

### 수동 로테이션

```bash
# 키 상태 확인
python scripts/rotate_keys.py check

# 수동 로테이션
python scripts/rotate_keys.py manual
```

### 자동 체크

```bash
# 자동 체크 스케줄러 실행 (매일 오전 9시)
python scripts/rotate_keys.py auto
```

### 마스터 키 로테이션

```bash
# 마스터 키 로테이션 (모든 데이터 재암호화)
python scripts/rotate_keys.py master
```

## 보안 주의사항

### DO ✅
- API 키는 항상 암호화하여 저장
- 정기적으로 키 로테이션 수행
- `.keys/master.key` 파일 안전하게 백업
- 거래소에서 최소 권한만 부여
  - 업비트: 조회 권한만
  - 바이낸스: 선물 거래, 읽기 권한만

### DON'T ❌
- `.env` 파일을 Git에 커밋하지 마세요
- API 키를 코드에 하드코딩하지 마세요
- 마스터 키를 분실하지 마세요 (복구 불가)
- API 키를 평문으로 로그에 남기지 마세요

## 거래소별 API 설정

### 업비트 (Upbit)
1. https://upbit.com/mypage/open_api_management 접속
2. Open API 신청
3. IP 주소 화이트리스트 설정
4. 권한: 자산 조회, 주문 조회만 체크

### 바이낸스 (Binance)
1. https://www.binance.com/en/my/settings/api-management 접속
2. Create API 클릭
3. API restrictions 설정:
   - Enable Reading ✅
   - Enable Futures ✅
   - Restrict access to trusted IPs only ✅

### 환율 API
- 무료: https://exchangerate-api.com (제한: 1500회/월)
- 유료: 필요시 업그레이드

## 문제 해결

### API 키가 작동하지 않을 때
1. 환경변수 확인: `python src/config/env_manager.py validate`
2. API 연결 테스트: `python scripts/test_api_connection.py`
3. 거래소 API 권한 확인
4. IP 화이트리스트 확인

### 마스터 키를 분실했을 때
- 백업이 없다면 복구 불가능
- 새로 설정 필요:
  1. `.keys` 디렉토리 삭제
  2. API 키 재설정
  3. 다시 암호화

### 암호화 오류
```bash
# 암호화 상태 확인
python src/utils/crypto_manager.py verify

# 필요시 재암호화
python src/utils/crypto_manager.py encrypt
```

## 응급 상황 대응

API 키가 노출되었을 경우:
1. **즉시** 거래소에서 해당 API 키 삭제
2. 새 API 키 생성
3. 시스템에 새 키 설정
4. 키 로테이션 수행
5. 거래소 활동 로그 확인

---

*보안은 시스템의 가장 중요한 부분입니다. 항상 주의하세요!*