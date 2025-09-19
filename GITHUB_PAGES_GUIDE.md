# GitHub Pages 배포 가이드

RV Exoplanet Simulator를 GitHub Pages로 배포하는 단계별 가이드입니다.

## 📋 준비사항

- GitHub 계정
- 생성된 파일들:
  - `index.html`
  - `rv_simulator.js`
  - `README.md`

## 🚀 배포 단계

### 1단계: GitHub 저장소 생성

1. [GitHub](https://github.com)에 로그인
2. 우측 상단의 "+" 버튼 클릭 → "New repository" 선택
3. 저장소 설정:
   - **Repository name**: `rv-exoplanet-simulator` (또는 원하는 이름)
   - **Description**: "웹 기반 외계행성 시선속도 시뮬레이터"
   - **Visibility**: Public (GitHub Pages 무료 사용을 위해)
   - **Initialize**: 체크하지 않음 (기존 파일들이 있으므로)
4. "Create repository" 클릭

### 2단계: 파일 업로드

#### 방법 A: 웹 인터페이스 사용 (간단)

1. 생성된 저장소 페이지에서 "uploading an existing file" 클릭
2. 파일들을 드래그 앤 드롭하거나 "choose your files" 클릭
3. 업로드할 파일들:
   - `index.html`
   - `rv_simulator.js`
   - `README.md`
4. "Commit changes" 클릭

#### 방법 B: Git 명령어 사용 (고급)

```bash
# 저장소 클론
git clone https://github.com/[사용자명]/rv-exoplanet-simulator.git
cd rv-exoplanet-simulator

# 파일 복사
cp /Users/wonchae/Desktop/index.html .
cp /Users/wonchae/Desktop/rv_simulator.js .
cp /Users/wonchae/Desktop/README.md .

# Git에 추가 및 커밋
git add .
git commit -m "Initial commit: RV Exoplanet Simulator web version"
git push origin main
```

### 3단계: GitHub Pages 활성화

1. 저장소 페이지에서 "Settings" 탭 클릭
2. 왼쪽 메뉴에서 "Pages" 클릭
3. "Source" 섹션에서:
   - "Deploy from a branch" 선택
   - "Branch"를 "main" 선택
   - "Folder"를 "/ (root)" 선택
4. "Save" 클릭

### 4단계: 배포 확인

1. 몇 분 후 저장소 페이지로 돌아가기
2. "Settings" → "Pages"에서 녹색 체크 표시와 함께 URL 확인
3. URL은 다음과 같은 형태: `https://[사용자명].github.io/[저장소명]`
4. URL을 클릭하여 웹사이트 접속 확인

## 🔧 추가 설정 (선택사항)

### 커스텀 도메인 설정

1. 도메인을 구매한 경우
2. "Settings" → "Pages"에서 "Custom domain" 입력
3. 도메인 DNS 설정에서 CNAME 레코드 추가

### HTTPS 강제 설정

1. "Settings" → "Pages"에서 "Enforce HTTPS" 체크
2. 보안을 위해 HTTPS 사용 강제

## 🐛 문제 해결

### 페이지가 로드되지 않는 경우

1. **파일명 확인**: `index.html`이 루트 디렉토리에 있는지 확인
2. **대소문자**: 파일명이 정확한지 확인 (GitHub는 대소문자 구분)
3. **캐시**: 브라우저 캐시 삭제 후 재시도
4. **배포 시간**: 최대 10분 정도 소요될 수 있음

### JavaScript 오류가 발생하는 경우

1. 브라우저 개발자 도구(F12)에서 콘솔 확인
2. Chart.js CDN 로딩 확인
3. 파일 경로가 올바른지 확인

### 차트가 표시되지 않는 경우

1. Chart.js 라이브러리 로딩 확인
2. Canvas 요소 크기 확인
3. JavaScript 오류 확인

## 📱 모바일 최적화

현재 웹사이트는 반응형 디자인으로 모바일에서도 사용 가능합니다:

- 작은 화면에서는 세로 레이아웃으로 변경
- 터치 인터페이스 지원
- 적절한 폰트 크기와 버튼 크기

## 🔄 업데이트 방법

코드를 수정한 후:

```bash
# 변경사항 커밋
git add .
git commit -m "Update: [변경 내용 설명]"
git push origin main
```

GitHub Pages는 자동으로 업데이트됩니다 (몇 분 소요).

## 📊 성능 최적화

### 이미지 최적화
- 필요시 이미지 파일 압축
- WebP 형식 사용 고려

### JavaScript 최적화
- 코드 압축 (minification)
- 불필요한 라이브러리 제거

### CDN 사용
- Chart.js는 이미 CDN 사용 중
- 다른 라이브러리도 CDN 고려

## 🎯 다음 단계

웹사이트가 성공적으로 배포되면:

1. **사용자 피드백 수집**: GitHub Issues 활용
2. **기능 추가**: 새로운 시뮬레이션 기능
3. **성능 개선**: 로딩 속도 최적화
4. **문서화**: 사용법 가이드 추가
5. **SEO 최적화**: 검색 엔진 최적화

## 📞 지원

문제가 발생하면:

1. GitHub Issues에 버그 리포트 작성
2. README.md 파일 확인
3. 브라우저 개발자 도구에서 오류 확인

---

**축하합니다!** 🎉 이제 RV Exoplanet Simulator가 전 세계에서 접근 가능한 웹사이트로 운영됩니다!
