# 오뚝이 (Ottugi) — 리볼빙 조기경보 & AI 상환 코칭

리볼빙(일부결제금액이월약정) 사용자의 다음 달 상태를 예측(XGBoost)하고, SHAP로
예측 근거를 설명하며, 위험 단계(관찰/주의/경고/심화)를 판정하고, Claude Haiku 4.5로
근거가 태깅된 상환 코칭 메시지를 생성하는 Streamlit MVP.

## 실행

### Streamlit Community Cloud (배포)

- Main file path: `app/streamlit_app.py`
- 의존성: `requirements.txt` (표준 pip)
- Secrets: 앱 대시보드 **Settings → Secrets** 에 아래 형식으로 입력
  (`.streamlit/secrets.toml.example` 참고)

  ```toml
  ANTHROPIC_API_KEY = "sk-ant-..."
  USE_MOCK_COACHING = "true"   # 또는 "false"
  ```

  `USE_MOCK_COACHING = "true"` 이면 API 키 없이도 동작한다(mock 코칭).

  **배포 기본값은 `"true"`(mock).** 실제 Claude Haiku 4.5 코칭으로 전환하려면
  Secrets에서 `ANTHROPIC_API_KEY`에 유효한 키를 넣고 `USE_MOCK_COACHING = "false"`로
  바꾸기만 하면 된다(앱 자동 재시작, 코드 변경 불필요). 근거 태깅 JSON 검증
  로직은 mock/real 동일하게 적용된다.

### 로컬 — 표준 pip venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

로컬에서 실 API 코칭을 테스트하려면 `.streamlit/secrets.toml.example` 을
`.streamlit/secrets.toml` 로 복사해 값을 채운다(이 파일은 커밋되지 않음).

### 로컬 — macOS micromamba (기존 개발 환경)

macOS에서 XGBoost가 요구하는 libomp(OpenMP) 의존성 때문에 개발은 conda-forge
환경을 써 왔다. 리눅스인 Streamlit Cloud에서는 불필요하다.

```bash
./.micromamba/bin/micromamba create -y -p ./.micromamba/envs/ottugi -c conda-forge \
    python=3.11 numpy pandas scikit-learn xgboost shap streamlit plotly jsonschema pip
./.micromamba/envs/ottugi/bin/python -m pip install anthropic
./.micromamba/envs/ottugi/bin/streamlit run app/streamlit_app.py
```

## 구조

| 경로 | 역할 |
|---|---|
| `app/streamlit_app.py` | 화면 구성 · 페이지 라우팅 (계산식 없음) |
| `app/theme.py`, `app/charts.py` | 스타일 · 차트 |
| `app/forecast_utils.py` | 다개월 전망 / 최소 개입액 오케스트레이션 (model.py 재호출) |
| `src/model.py` | 예측 모델(f1=S, f2=r) · 재귀 예측 · 시뮬레이션 회계식 |
| `src/risk.py` | 위험 단계 판정 규칙 |
| `src/shap_utils.py` | SHAP TreeExplainer 연동 |
| `src/coaching.py` | Claude Haiku 4.5 코칭 메시지 생성 + 근거 태깅 검증 |
| `src/config.py` | 시뮬레이션/모델 파라미터 |
| `data/`, `models/`, `outputs/` | 합성 데이터 · 학습된 모델 · 검증 지표 (배포에 필요, 커밋됨) |
