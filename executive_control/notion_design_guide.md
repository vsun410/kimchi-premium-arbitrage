# 🎨 Notion Executive Dashboard 디자인 가이드

## 📐 레이아웃 구조

### 1. **메인 대시보드 (Home)**
```
┌─────────────────────────────────────┐
│      🌅 Daily Quote / Welcome       │
├─────────────┬───────────────────────┤
│   📅 Today  │    📊 Key Metrics     │
│  Calendar   │  • Code Drift: 5.2%  │
│             │  • Tasks: 42/100      │
├─────────────┼───────────────────────┤
│ 🎯 Focus    │   🔥 In Progress      │
│  (3 items)  │     (5 tasks)         │
├─────────────┴───────────────────────┤
│        📈 Weekly Progress Graph     │
└─────────────────────────────────────┘
```

### 2. **프로젝트 뷰 (Projects)**
```
┌─────────────────────────────────────┐
│    Toggle View: 📋 List | 🗂️ Kanban │
├─────────────────────────────────────┤
│  🚀 Active  │  🎯 Planning │ ✅ Done │
│             │              │         │
│  Project 1  │  Project A   │ Proj X  │
│  Project 2  │  Project B   │ Proj Y  │
└─────────────────────────────────────┘
```

## 🎨 색상 팔레트 (Dark Mode)

### Primary Colors
- **Background**: #191919 (거의 검정)
- **Card Background**: #2F3437 (다크 그레이)
- **Accent**: #5B9CF6 (부드러운 파란색)

### Status Colors
- **Success**: #4CAF50 (초록)
- **Warning**: #FFA726 (오렌지)
- **Error**: #EF5350 (빨강)
- **Info**: #29B6F6 (하늘색)

### Text Colors
- **Primary Text**: #E0E0E0
- **Secondary Text**: #9E9E9E
- **Muted Text**: #616161

## 🧩 위젯 구성

### 1. **시계 위젯**
```html
<!-- Notion에 임베드할 수 있는 HTML -->
<iframe src="https://indify.co/widgets/live/clock/..." 
        style="width: 100%; height: 150px;">
</iframe>
```

### 2. **날씨 위젯**
```html
<iframe src="https://indify.co/widgets/live/weather/..." 
        style="width: 100%; height: 100px;">
</iframe>
```

### 3. **깃허브 Stats**
```html
<iframe src="https://github-readme-stats.vercel.app/api?username=YOUR_USERNAME&theme=dark" 
        style="width: 100%; height: 200px;">
</iframe>
```

## 📚 페이지 구조

### Level 1: Main Dashboard
```
🏠 Executive Control Center
├── 📊 Dashboard
├── 📋 Projects & Tasks  
├── 🧠 Second Brain
├── 📅 Calendar
└── ⚙️ Settings
```

### Level 2: Project Structure
```
📋 Projects & Tasks
├── 🚀 Active Projects
│   ├── Kimchi Premium Trading
│   ├── ML Model Development
│   └── Infrastructure Setup
├── 📝 Task Inbox
├── 🎯 This Week
└── 📈 Progress Reports
```

### Level 3: Knowledge Base
```
🧠 Second Brain
├── 📖 Documentation
│   ├── Architecture
│   ├── API Reference
│   └── Deployment Guide
├── 💡 Ideas & Research
├── 🔗 Resources
└── 📚 Learning Notes
```

## 🎯 디자인 원칙

### 1. **Visual Hierarchy**
- **제목**: 24px, Bold
- **부제목**: 18px, Medium
- **본문**: 14px, Regular
- **캡션**: 12px, Light

### 2. **Spacing**
- **섹션 간격**: 32px
- **카드 패딩**: 16px
- **아이템 간격**: 8px

### 3. **Icons**
- **프로젝트**: 🚀 🎯 ⚡ 🔧
- **상태**: ✅ 🔄 ⏸️ ❌
- **우선순위**: 🔴 🟡 🟢
- **카테고리**: 📊 📈 📉 📋

## 💫 애니메이션 & 인터랙션

### Hover Effects
```css
/* Notion 커스텀 CSS (브라우저 확장 사용) */
.notion-collection-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}
```

### Progress Bars
```
완료율: ████████░░ 80%
코드 품질: ██████████ 100%
테스트 커버리지: ███████░░░ 70%
```

## 🗓️ 캘린더 뷰 디자인

### Weekly View
```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Mon │ Tue │ Wed │ Thu │ Fri │ Sat │ Sun │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 🔵  │     │ 🔴  │ 🔵  │     │     │     │
│ 9am │     │ 2pm │ 10am│     │     │     │
│Meeting    │Sprint│Review│     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

## 🎨 Notion 꾸미기 도구

### 1. **무료 위젯 사이트**
- [Indify](https://indify.co) - 시계, 날씨, 카운터
- [WidgetBox](https://widgetbox.app) - 차트, 프로그레스 바
- [Apption](https://apption.co) - 다양한 임베드 위젯

### 2. **아이콘 & 커버**
- [Notion Icons](https://notionicons.com) - 무료 아이콘
- [Unsplash](https://unsplash.com) - 무료 커버 이미지
- [Icons8](https://icons8.com) - 일관된 스타일 아이콘

### 3. **폰트 & 타이포그래피**
```
Title: Inter Bold
Body: Inter Regular
Code: JetBrains Mono
```

## 📱 모바일 최적화

### Mobile Layout
```
┌─────────────┐
│   📊 Stats  │
├─────────────┤
│  Today's    │
│   Tasks     │
├─────────────┤
│   Quick     │
│   Actions   │
└─────────────┘
```

## 🔗 유용한 임베드

### 1. **GitHub Activity**
```html
<img src="https://github-readme-activity-graph.vercel.app/graph?username=USERNAME&theme=github-dark">
```

### 2. **Spotify Now Playing**
```html
<img src="https://spotify-github-profile.vercel.app/api/view?uid=YOUR_SPOTIFY_ID">
```

### 3. **Pomodoro Timer**
```html
<iframe src="https://pomofocus.io/" width="100%" height="400"></iframe>
```

## 🚀 구현 체크리스트

- [ ] 다크모드 색상 팔레트 적용
- [ ] 메인 대시보드 레이아웃 구성
- [ ] 캘린더 위젯 추가
- [ ] GitHub Stats 임베드
- [ ] 프로젝트 칸반 보드 생성
- [ ] Second Brain 구조 구축
- [ ] 모바일 뷰 최적화
- [ ] 커스텀 아이콘 적용
- [ ] Progress 차트 추가
- [ ] Daily/Weekly 템플릿 생성

---

## 💡 Pro Tips

1. **일관성**: 같은 종류의 정보는 같은 아이콘 사용
2. **여백**: 충분한 여백으로 숨 쉴 공간 제공
3. **대비**: 다크모드에서도 읽기 쉬운 대비율 유지
4. **계층**: 최대 3단계 깊이까지만 구조화
5. **반응형**: 모바일에서도 사용 가능하게 설계