# 🚀 Notion API 테스트 - 빠른 시작 가이드

## 현재 상황
✅ API 토큰은 유효함 (연결 성공!)
❌ 하지만 아직 Integration이 어떤 페이지에도 추가되지 않음

## 📝 필수 단계 (5분 소요)

### 1️⃣ Notion에서 테스트 페이지 생성 (1분)

1. **Notion 열기** (브라우저 또는 앱)
2. **새 페이지 만들기** (아무 곳이나)
   - 제목: "API Test" (아무거나 OK)
3. **페이지 생성 완료**

### 2️⃣ Integration 추가하기 (중요!) (30초)

1. 방금 만든 페이지에서 **우측 상단 `...` (점 3개)** 클릭
2. **"Connections"** 또는 **"Add connections"** 찾기
3. **"kimp"** 선택 (우리가 만든 Integration)
4. **"Confirm"** 또는 **"Add"** 클릭

### 3️⃣ 테스트 실행 (30초)

```bash
# 테스트 스크립트 실행
python test_notion_create_modify.py
```

## ✅ 성공 확인 방법

테스트가 성공하면:
- Notion에 "Claude Code Test Page" 생성됨
- 페이지에 자동으로 내용이 추가됨
- "Test Tasks Database" 생성됨
- 샘플 태스크 3개 추가됨

## ❌ 실패하는 경우

### "접근 가능한 페이지를 찾을 수 없습니다"
→ Integration이 추가되지 않음. 2번 단계 다시 확인

### "Invalid token" 오류
→ 토큰이 잘못됨. `.env.notion` 확인

### "Unauthorized" 오류  
→ Integration 권한 문제. Notion에서 다시 추가

## 🎯 빠른 체크리스트

- [ ] Notion에 아무 페이지나 생성했나요?
- [ ] 그 페이지에 "kimp" Integration을 추가했나요?
- [ ] 추가할 때 "Confirm" 버튼을 눌렀나요?
- [ ] `test_notion_create_modify.py` 실행했나요?

## 💡 팁

- Integration은 **각 페이지마다** 추가해야 함
- 한 번 추가하면 하위 페이지는 자동으로 접근 가능
- 데이터베이스도 마찬가지로 개별 추가 필요

## 📊 다음 단계

테스트 성공 후:
1. `NOTION_SETUP_GUIDE.md` 참고하여 실제 워크스페이스 구성
2. 프로젝트 태스크 데이터베이스 생성
3. Claude Code와 자동 동기화 구현

---

**문제가 있나요?**
- API 토큰 확인: `.env.notion` 파일의 `NOTION_TOKEN`
- Integration 이름 확인: Notion 설정에서 "kimp"인지 확인
- 네트워크 확인: VPN이나 프록시 끄고 재시도