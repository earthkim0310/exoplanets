# RV (Doppler) Exoplanet Simulator — Web (PyScript)

이 저장소는 Tkinter 기반 데스크톱 앱을 **브라우저에서 실행 가능한** 웹 버전으로 단순화한 템플릿입니다.
Python 코드(모델)와 matplotlib를 **PyScript (Pyodide)** 를 통해 브라우저에서 바로 실행합니다.

## 폴더 구조
```
.
├─ index.html       # GitHub Pages 진입점
├─ rv_model.py      # 순수 파이썬 물리 모델 (GUI 없음)
├─ main.py          # HTML 슬라이더 ↔ matplotlib 바인딩
└─ assets/
   └─ style.css     # 간단한 스타일
```

## 사용 방법 (GitHub Pages)
1. GitHub에서 새 저장소를 만들고 위 파일들을 그대로 업로드합니다.
2. Settings → Pages → **Source: Deploy from a branch** → Branch: `main` (또는 `master`), 폴더는 `/root` 선택 → Save
3. 수십 초 후 `https://<계정명>.github.io/<저장소명>/` 로 접속하면 실행됩니다.

## 로컬 테스트
- 별도 빌드가 필요 없습니다. 단, PyScript는 CDN을 통해 로드되므로 **오프라인 환경에서는 동작하지 않습니다.**
- 로컬에서 확인하려면 간단한 HTTP 서버로 열어도 됩니다.
  ```bash
  python3 -m http.server 8000
  # 브라우저에서 http://localhost:8000 접속
  ```

## 주의
- 원본 Tkinter UI는 브라우저에서 사용할 수 없으므로, 슬라이더·버튼을 HTML로 바꾸고,
  matplotlib 그림을 브라우저에 직접 렌더링하도록 구성했습니다.
- 필요한 경우 `rv_model.py`에 있는 함수를 확장하거나, `main.py`에서 추가 슬라이더/플롯을 바인딩하세요.
