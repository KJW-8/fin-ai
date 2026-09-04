"""오뚝이(Ottugi) — Streamlit 앱 (UI/UX 전면 개편판).

이 파일은 화면 구성(페이지 라우팅·레이아웃·문구·스타일)만 다룬다. 데이터 처리,
예측 모델(f1=S, f2=r), SHAP 계산, 위험도 판정 로직, 시뮬레이션 회계식은 전부
src/model.py, src/risk.py, src/coaching.py, src/shap_utils.py의 기존 함수를
그대로 호출하며, 이 파일 안에서 새로운 계산식을 만들지 않는다. 유일한 예외는
app/forecast_utils.py의 다개월 전망/최소 개입액 탐색인데, 그마저도 내부적으로
model.py의 recursive_forecast()·simulate_extra_payment()를 그대로 호출한다
(자세한 내용은 forecast_utils.py 모듈 docstring 참고).

predicted_carryover_share는 확률도 위험도 점수(0~100점)도 아니다. 화면에서는
risk_indicator(관찰/주의/경고/심화) 4단계를 주된 시각 요소로 쓰고, carryover_share
숫자는 보조 정보로만 표시한다.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for p in (SRC_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import INTEREST_RATE_FIXED, PAYMENT_RATIO_GAP_WARN_CUTOFF, RISK_LEVEL_THRESHOLD_DEFAULT  # noqa: E402
from coaching import FEATURE_EXPLANATIONS, FEATURE_LABELS, generate_coaching_message  # noqa: E402
from model import build_feature_row, build_feature_table, deterministic_recursion_step, recursive_forecast, simulate_extra_payment  # noqa: E402
from risk import classify_risk_indicator  # noqa: E402
from shap_utils import build_explainers, explain_row, load_models  # noqa: E402

import charts  # noqa: E402
import theme  # noqa: E402
import mascot  # noqa: E402
from coaching import _hydrate_env_from_st_secrets  # noqa: E402
from forecast_utils import find_first_escalation, find_minimum_intervention, multi_month_outlook, simulate_intervention_trajectory  # noqa: E402

# 이산시간 위험모형(보완 모델) — XGBoost 파이프라인과 별도. "언제 위험 단계로 넘어갈
# 가능성이 있는가"만 담당한다. 계산은 전부 src/hazard.py 가 하고 이 파일은 표시만 한다.
import hazard as hz  # noqa: E402
import recovery as rec  # noqa: E402

# Streamlit Cloud에서 Secrets로 넣은 ANTHROPIC_API_KEY / USE_MOCK_COACHING 값을
# os.environ에 미리 반영해 둔다 (이 파일의 os.environ 조회와 coaching.py가 일관되게
# 같은 값을 보도록). 로컬 쉘 환경변수가 있으면 그쪽이 우선한다.
_hydrate_env_from_st_secrets()

PAGES = [
    ("home", "내 금융 상태"),
    ("risk", "위험 분석"),
    ("coaching", "AI 상환 코칭"),
    ("simulator", "상환 시뮬레이션"),
    ("trust", "모델 신뢰도"),
]
PAGE_LABELS = dict(PAGES)

# 화면에 노출되는 각 지표가 "무엇을 기준으로 계산됐는지" 설명하는 문구.
# 실제 계좌 데이터(청구액·결제액·이월원금 등 monthly_transaction/derived_features)를
# 근거로 계산된다는 점을 사용자가 바로 아래에서 확인할 수 있게 한다.
METRIC_DEFINITIONS = {
    "리볼빙 의존도": "이번 달 카드 사용액과 이월된 원금 중, 갚지 못하고 다음 달로 넘어가는 금액의 비율이에요. 실제 이번 달 청구액·이월원금으로 계산됩니다.",
    "최근 3개월 변화(의존도)": "3개월 전 리볼빙 의존도와 비교해 얼마나 달라졌는지예요. 지난 3개월간의 실제 결제 이력을 비교해서 계산합니다.",
    "상승 추세": "리볼빙 의존도가 3개월 전보다 올라가고 있는지예요. 지난 3개월간의 실제 결제 이력을 비교해서 계산합니다.",
    "결제여유": "이번 달 약정한 결제 비율이 카드사가 정한 최소 결제 비율보다 얼마나 여유 있는지예요. 0%에 가까울수록 최소한만 겨우 갚는 상태에 가깝습니다.",
    "연속 최소결제": "결제여유가 거의 없는(최소 수준에 가까운) 상태가 몇 개월째 이어지고 있는지예요. 매달 실제 결제 이력을 이어서 추적한 값입니다.",
    "최소결제 반복": "결제여유가 거의 없는(최소 수준에 가까운) 상태가 몇 개월째 이어지고 있는지예요. 매달 실제 결제 이력을 이어서 추적한 값입니다.",
}


def definitions_for(*labels: str) -> str:
    items = [(label, METRIC_DEFINITIONS[label]) for label in labels if label in METRIC_DEFINITIONS]
    return theme.definitions_panel(items)


def go_to(page: str) -> None:
    # "nav_page"는 사이드바 라디오 위젯의 key이기도 하다. 그 위젯은 페이지 본문(이 함수가
    # 호출되는 시점)보다 먼저 렌더링되므로, 이 시점에 st.session_state["nav_page"]를 직접
    # 덮어쓰면 "위젯 인스턴스화 이후에는 그 key를 수정할 수 없다"는 Streamlit 예외가 난다.
    # 그래서 다음 rerun의 맨 앞(사이드바 렌더링 전)에서 반영할 "대기 값"만 남겨 둔다.
    st.session_state["_pending_nav"] = page
    st.rerun()


# ---------------------------------------------------------------------------
# 데이터/모델 로딩 (캐시) — 기존과 동일
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    account_master = pd.read_csv(DATA_DIR / "account_master.csv")
    monthly_transaction = pd.read_csv(DATA_DIR / "monthly_transaction.csv")
    derived_features = pd.read_csv(DATA_DIR / "derived_features.csv")
    feature_table = build_feature_table(monthly_transaction, derived_features, account_master)
    return account_master, monthly_transaction, derived_features, feature_table


@st.cache_resource
def load_models_and_explainers():
    model_S, model_r, feature_cols = load_models()
    explainer_S, explainer_r = build_explainers(model_S, model_r)
    return model_S, model_r, feature_cols, explainer_S, explainer_r


@st.cache_resource
def load_hazard():
    """이산시간 위험모형 번들. 없으면 None (앱은 hazard 기능만 빼고 정상 동작)."""
    try:
        b = hz.load_bundle()
        return b["model"], b["feats"]
    except Exception:
        return None, None


@st.cache_data
def load_hazard_metrics():
    p = OUTPUTS_DIR / "hazard_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_outputs():
    with open(OUTPUTS_DIR / "model_metrics.json", encoding="utf-8") as f:
        model_metrics = json.load(f)
    with open(OUTPUTS_DIR / "risk_threshold_sensitivity_summary.json", encoding="utf-8") as f:
        risk_sensitivity = json.load(f)
    with open(OUTPUTS_DIR / "shap_feature_importance.json", encoding="utf-8") as f:
        shap_importance = json.load(f)
    return model_metrics, risk_sensitivity, shap_importance


# ---------------------------------------------------------------------------
# 직접 입력 -> 앵커 행(anchor row) 구성 — 기존과 동일
# ---------------------------------------------------------------------------
def build_manual_anchor_row(inputs: dict) -> pd.DataFrame:
    m = inputs["m"]
    L = inputs["L"]
    S_t = inputs["S_t"]
    r_t = max(inputs["r_t"], m)
    B_prev = inputs["B_prev"]
    month_index = inputs["month_index"]
    i = INTEREST_RATE_FIXED

    calc = deterministic_recursion_step(B_prev, S_t, r_t, m, i)
    gap = r_t - m
    revolving_streak_months = 1 if calc["B_t"] > 0 else 0
    minimum_payment_streak = 1 if gap <= PAYMENT_RATIO_GAP_WARN_CUTOFF else 0
    delinquency_count_6m = inputs.get("delinquency_count_6m") or 0

    cs_t3 = inputs.get("carryover_share_t3")
    r_t3 = inputs.get("r_t3")
    carryover_share_delta_3m = (calc["carryover_share"] - cs_t3) if cs_t3 not in (None, "") else np.nan
    committed_ratio_delta_3m = (r_t - r_t3) if r_t3 not in (None, "") else np.nan

    row = {
        "account_id": f"manual-{uuid.uuid4()}",
        "month_index": month_index,
        "feature_month_index": month_index,
        "billing_amount": S_t,
        "committed_payment_ratio": r_t,
        "revolving_principal_before_payment": calc["P_t"],
        "scheduled_principal_payment": calc["A_t"],
        "revolving_fee": calc["I_t"],
        "ending_carryover_principal": calc["B_t"],
        "total_payment_amount": calc["total_payment_amount"],
        "minimum_principal_required": calc["minimum_principal_required"],
        "actual_principal_paid": calc["A_t"],
        "revolving_active": 1 if (B_prev > 0 or r_t < 1.0) else 0,
        "payment_status_정상": 1,
        "payment_status_최소결제": 0,
        "payment_status_연체": 0,
        "carryover_share": calc["carryover_share"],
        "carryover_share_delta_3m": carryover_share_delta_3m,
        "carryover_share_slope_3m": np.nan,
        "committed_ratio_delta_3m": committed_ratio_delta_3m,
        "payment_ratio_gap": gap,
        "revolving_streak_months": revolving_streak_months,
        "minimum_payment_streak": minimum_payment_streak,
        "delinquency_count_6m": delinquency_count_6m,
        "limit_utilization_ratio": float(np.clip(S_t / L, 0.0, 1.0)) if L > 0 else 0.0,
        "minimum_payment_ratio": m,
        "card_limit": L,
        "interest_rate": i,
    }
    return pd.DataFrame([row])


def get_anchor_row_for_account(account_id: str, feature_table: pd.DataFrame, account_master: pd.DataFrame) -> pd.DataFrame:
    rows = feature_table[feature_table["account_id"] == account_id]
    latest = rows.loc[[rows["month_index"].idxmax()]].copy()
    latest = latest.merge(account_master[["account_id", "interest_rate"]], on="account_id", how="left")
    return latest


# 데모 고객 선택 시, 각 티어에서 "그 티어다운" 최신월 risk_indicator를 가진 계좌를
# 우선 고른다. 예전엔 그냥 tier의 첫 계좌를 썼는데, medium tier 800명 중 다수가
# 12개월차에 우연히 "심화"에 몰려 있어(무작위 드리프트 특성상) 중위험 데모 고객이
# 고위험과 똑같이 "심화"로 보이는 문제가 있었다 — 실제 데이터로 확인해보니 medium
# tier 중 214명은 "경고" 상태였으므로, 그중 하나를 우선 선택하도록 바꿨다.
DEMO_PREFERRED_LEVELS = {
    "low": ("관찰",),
    "medium": ("경고", "주의"),
    "high": ("심화",),
}


def pick_demo_account(tier: str, account_master: pd.DataFrame, derived_features: pd.DataFrame) -> str:
    tier_account_ids = account_master.loc[account_master["persona_tier"] == tier, "account_id"]
    tier_df = derived_features[derived_features["account_id"].isin(tier_account_ids)]
    if not tier_df.empty:
        latest = tier_df.loc[tier_df.groupby("account_id")["month_index"].idxmax()]
        for level in DEMO_PREFERRED_LEVELS.get(tier, ()):
            match = latest.loc[latest["risk_indicator"] == level, "account_id"]
            if not match.empty:
                return match.iloc[0]
    return tier_account_ids.iloc[0]


# ---------------------------------------------------------------------------
# 예측 + SHAP + 위험판정 번들 계산 — 기존과 동일
# ---------------------------------------------------------------------------
def build_prediction_bundle(anchor_row: pd.DataFrame, monthly_transaction, derived_features, model_S, model_r, feature_cols, explainer_S, explainer_r):
    row = anchor_row.iloc[0]

    current_risk = classify_risk_indicator(
        carryover_share=row["carryover_share"],
        carryover_share_delta_3m=row["carryover_share_delta_3m"],
        payment_ratio_gap=row["payment_ratio_gap"],
        minimum_payment_streak=row["minimum_payment_streak"],
        warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT,
    )

    forecast = recursive_forecast(
        model_S, model_r, feature_cols, anchor_row, monthly_transaction, derived_features, horizon=1
    )
    pred = forecast.iloc[0]

    predicted_risk = classify_risk_indicator(
        carryover_share=pred["predicted_carryover_share"],
        carryover_share_delta_3m=pred["carryover_share_delta_3m"],
        payment_ratio_gap=pred["payment_ratio_gap"],
        minimum_payment_streak=pred["minimum_payment_streak"],
        warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT,
    )

    X_row = build_feature_row({c: row.get(c, np.nan) for c in feature_cols}, feature_cols)
    shap_S, base_S = explain_row(explainer_S, X_row, feature_cols)
    shap_r, base_r = explain_row(explainer_r, X_row, feature_cols)

    return {
        "account_id": row["account_id"],
        "month_index": int(row["month_index"]),
        "m": row["minimum_payment_ratio"],
        "L": row["card_limit"],
        "i": row.get("interest_rate", INTEREST_RATE_FIXED),
        "B_current": row["ending_carryover_principal"],
        "current_carryover_share": row["carryover_share"],
        "current_delta_3m": row["carryover_share_delta_3m"],
        "current_gap": row["payment_ratio_gap"],
        "current_streak": int(row["minimum_payment_streak"]),
        "current_risk": current_risk,
        "pred_S": pred["pred_S"],
        "pred_r": pred["pred_r"],
        "pred_B": pred["pred_B"],
        "predicted_carryover_share": pred["predicted_carryover_share"],
        "predicted_delta_3m": pred["carryover_share_delta_3m"],
        "predicted_gap": pred["payment_ratio_gap"],
        "predicted_streak": int(pred["minimum_payment_streak"]),
        "predicted_risk": predicted_risk,
        "shap_S": shap_S,
        "shap_r": shap_r,
        "base_S": base_S,
        "base_r": base_r,
    }


# ---------------------------------------------------------------------------
# Hazard Model + 회복 게이지 — 계산은 전부 src/hazard.py, src/recovery.py 가 담당.
# 이 파일은 anchor_row 에서 필요한 피처만 추려 넘기고 결과를 표시할 뿐이다.
# ---------------------------------------------------------------------------
def _hazard_state_from_row(row, risk_level: str) -> dict:
    st_dict = {}
    for c in hz.HAZARD_FEATURES:
        v = row.get(c) if hasattr(row, "get") else (row[c] if c in row else None)
        st_dict[c] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)
    st_dict["current_risk"] = risk_level
    return st_dict


@st.cache_data(show_spinner=False)
def _cached_hazard_bundle(state_items: tuple, current_month: int):
    model, feats = load_hazard()
    if model is None:
        return None
    return hz.build_hazard_bundle(dict(state_items), int(current_month), model=model, feats=feats)


def compute_hazard_bundle(row, risk_level: str, current_month: int) -> dict | None:
    state = _hazard_state_from_row(row, risk_level)
    # dict 는 캐시 키가 안 되므로 정렬된 튜플로
    key = tuple(sorted((k, (round(v, 6) if isinstance(v, float) else v)) for k, v in state.items()))
    return _cached_hazard_bundle(key, current_month)


def _simulated_hazard_bundle(anchor_row, bundle, sim_calc, new_risk, new_delta_3m, new_streak):
    """What-if 시뮬레이션 상태에 대한 hazard 재계산. 시뮬레이터가 바꾸는 값(의존도/결제여유/
    연속최소결제/3개월변화)만 갱신하고 나머지 피처는 anchor 유지. 계산은 src/hazard.py."""
    model, feats = load_hazard()
    if model is None:
        return None
    arow = anchor_row.iloc[0]
    state = _hazard_state_from_row(arow, new_risk)
    state["carryover_share"] = float(sim_calc["carryover_share"])
    state["payment_ratio_gap"] = float(sim_calc["payment_ratio_gap"])
    state["minimum_payment_streak"] = float(new_streak)
    if pd.notna(new_delta_3m):
        state["carryover_share_delta_3m"] = float(new_delta_3m)
    state["current_risk"] = new_risk
    return hz.build_hazard_bundle(state, int(bundle["month_index"]), model=model, feats=feats)


# ---------------------------------------------------------------------------
# 화면 텍스트 유틸 (표현만 담당, 계산 없음)
# ---------------------------------------------------------------------------
def fmt_pct(x: float, signed: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:+.1f}%p" if signed else f"{x*100:.1f}%"


def fmt_won(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}원"


def top_signals(bundle: dict, k: int = 3) -> list[dict]:
    """핵심 위험 신호: S/r 두 모델의 SHAP 기여도를 합쳐 |기여도| 상위 k개."""
    combined = []
    for feat, val in bundle["shap_S"].items():
        combined.append({"feature": feat, "value": val, "model": "S"})
    for feat, val in bundle["shap_r"].items():
        combined.append({"feature": feat, "value": val, "model": "r"})
    combined.sort(key=lambda d: abs(d["value"]), reverse=True)
    return combined[:k]


def risk_level_action_text(level: str) -> str:
    return {
        "관찰": "지금은 특별한 조치가 필요하지 않습니다. 이대로 유지해보세요.",
        "주의": "지금 가장 효과적인 방법은 결제 비율을 조금 더 높이는 것입니다.",
        "경고": "지금 가장 효과적인 방법은 추가 상환입니다.",
        "심화": "지금 가장 효과적인 방법은 추가 상환이며, 전문 상담도 함께 고려해보세요.",
    }.get(level, "")


# ---------------------------------------------------------------------------
# Hazard Model 표시 블록 — "언제 위험 단계로 넘어갈 가능성이 있는가"
# XGBoost(다음 달 값이 어떻게 움직일까)와 역할이 다르다는 점을 명시한다.
# ---------------------------------------------------------------------------
def render_hazard_block(bundle: dict, context: str = "home") -> None:
    hzb = bundle.get("hazard")
    mascot.section_with_accent(
        "언제 위험 단계로 넘어갈 가능성이 있나요?",
        "XGBoost 전망이 '값이 어떻게 움직일까'라면, 이 이산시간 위험모형은 '언제 경고 단계로 넘어갈 "
        "가능성이 있을까'를 별도로 계산합니다.",
        accent_key="report",
    )
    if hzb is None:
        st.info("위험 전환 모형이 아직 학습되지 않았습니다 (models/hazard_model.joblib 없음).")
        return
    if not hzb.get("applicable"):
        st.markdown(
            theme.alert_card(
                "ℹ️", "이미 경고/심화 단계예요",
                "위험 전환 모형은 아직 경고 단계로 넘어가지 않은 분의 '전환 시점'을 예측하는 모형이에요. "
                "지금은 아래 회복 미션과 상환 시뮬레이션에서 개선 방향을 확인하시는 게 더 도움이 됩니다.",
                tone="주의",
            ),
            unsafe_allow_html=True,
        )
        return

    tp3 = hzb.get("transition_probability_3m")
    mtw = hzb.get("median_time_to_warning")
    tiles = [
        theme.metric_tile("향후 3개월 경고 전환 가능성", fmt_pct(tp3) if tp3 is not None else "—",
                          note="현재 패턴 유지 가정 · 확정값 아님"),
        theme.metric_tile(
            "예상 전환 시점",
            (f"약 {mtw}개월 후" if mtw else "관측기간 내 낮음"),
            note="생존확률이 50% 아래로 내려가는 시점",
        ),
    ]
    st.markdown(theme.card_open() + theme.metric_row(tiles) + theme.card_close(), unsafe_allow_html=True)

    traj = hzb.get("trajectory") or []
    if traj:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t["month_offset"] for t in traj],
            y=[t["survival"] * 100 for t in traj],
            mode="lines+markers", name="유지 확률",
            line=dict(color=theme.BRAND, width=3),
        ))
        fig.add_hline(y=50, line_dash="dot", line_color=theme.SUBTLE,
                      annotation_text="50% (예상 전환 시점 기준선)", annotation_position="bottom right")
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="개월 후", yaxis_title="경고 단계로 안 넘어갈 확률(%)",
                          yaxis=dict(range=[0, 100]))
        with st.container(border=True):
            st.markdown(
                f'<div style="color:{theme.SUBTLE};font-size:0.85rem;margin-bottom:0.3rem;">'
                "선이 아래로 내려갈수록, 지금 패턴이 유지될 때 그 시점까지 경고 단계로 넘어가지 않을 확률이 낮아진다는 뜻이에요.</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("이 수치는 별도의 이산시간 위험모형(로지스틱 회귀)이 계산한 추정치이며, 확정된 결과가 아닙니다. "
               "모형 검증 지표는 '모델 신뢰도' 화면에서 확인할 수 있어요.")


# ---------------------------------------------------------------------------
# 사이드바: 네비게이션 + Demo Mode
# ---------------------------------------------------------------------------
def render_sidebar(account_master: pd.DataFrame, feature_table: pd.DataFrame, derived_features: pd.DataFrame):
    with st.sidebar:
        # 제목 + 문구(왼쪽) / 마스코트(오른쪽) 나란히
        _g = mascot.accent("greeting", size_px=96)
        st.markdown(
            theme.compact_html(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0 1rem 0;">'
                f'<div style="flex:1;min-width:0;">'
                f'<div class="ottugi-wordmark" style="font-size:2.4rem;color:{theme.SIDEBAR_TEXT};line-height:1.05;">오뚝이</div>'
                f'<div style="font-size:0.82rem;color:{theme.SIDEBAR_TEXT_MUTED};font-weight:500;line-height:1.45;margin-top:4px;">'
                f'당신의 리밸런싱도,<br>쓰러져도 스스로 중심을 되찾는 오뚝이처럼</div></div>'
                f'<div style="flex-shrink:0;">{_g}</div></div>'
            ),
            unsafe_allow_html=True,
        )

        # 위젯 key를 "nav_page" 하나로 통일한다. key와 index를 동시에 쓰면(예전 버전) 위젯이
        # 이미 가지고 있는 session_state[key] 값이 index보다 우선시되어, go_to()에서
        # 프로그래밍적으로 설정한 페이지 이동이 라디오 위젯에 의해 즉시 덮어써지는 버그가
        # 있었다. key만 남기고 최초 1회만 기본값을 시드하는 방식으로 고쳤다.
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "home"

        nav_keys = [k for k, _ in PAGES]
        st.radio("메뉴", nav_keys, format_func=lambda k: PAGE_LABELS[k], label_visibility="collapsed", key="nav_page")

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.78rem;font-weight:800;color:{theme.SIDEBAR_TEXT_MUTED};letter-spacing:0.04em;'
            f'margin-bottom:0.5rem;">DEMO MODE</div>',
            unsafe_allow_html=True,
        )

        demo_options = ["저위험 고객", "중위험 고객", "고위험 고객", "직접 입력"]
        demo_choice = st.radio("데모 고객", demo_options, label_visibility="collapsed", key="demo_choice")

        tier_map = {"저위험 고객": "low", "중위험 고객": "medium", "고위험 고객": "high"}

        if demo_choice in tier_map:
            tier = tier_map[demo_choice]
            account_id = pick_demo_account(tier, account_master, derived_features)
            selection_key = f"sample::{account_id}"
            anchor_row = get_anchor_row_for_account(account_id, feature_table, account_master)
            demo_label = demo_choice
        else:
            with st.expander("고객 정보 입력", expanded=True):
                st.caption("잘 모르는 항목은 기본값을 그대로 두셔도 괜찮아요 — 각 항목의 ⓘ 아이콘에 확인 방법을 안내해 두었습니다.")

                L = st.number_input(
                    "카드 한도 (원)", min_value=500_000, max_value=20_000_000, value=4_000_000, step=100_000,
                    help="본인 카드의 이용한도예요. 카드사 앱 > 카드 정보/한도 조회 메뉴에서 확인할 수 있어요.",
                )
                st.caption(f"= {fmt_won(L)}")

                m_pct = st.slider(
                    "최소결제비율 m (%)", 10, 30, 20,
                    help="카드사가 신용 상태에 따라 부여하는 하한선이에요(보통 10~30%). 카드사 앱의 리볼빙 안내 메뉴에서 "
                         "확인할 수 있어요. 잘 모르시면 기본값(20%)을 그대로 두셔도 괜찮아요.",
                )
                r_pct = st.slider(
                    "이번 달 약정결제비율 r (%)", int(m_pct), 100, max(int(m_pct), 70),
                    help="리볼빙 이용 시 본인이 직접 선택한 결제 비율이에요. 카드사 앱의 '리볼빙 결제비율 설정' 메뉴나 "
                         "최근 청구서에서 확인할 수 있어요.",
                )
                S_t = st.number_input(
                    "이번 달 카드 사용액 (원)", min_value=0, max_value=int(L), value=min(1_000_000, int(L)), step=50_000,
                    help="이번 달 카드로 결제한 금액 합계예요. 카드사 앱의 이번 달 이용내역 합계를 참고하세요.",
                )
                st.caption(f"= {fmt_won(S_t)}")

                B_prev = st.number_input(
                    "전월 이월원금 (원, 처음이면 0)", min_value=0, value=0, step=50_000,
                    help="지난달에 다 갚지 못하고 이번 달로 넘어온 금액이에요. 리볼빙을 처음 이용하신다면 0으로 두시면 "
                         "됩니다. 카드 명세서의 '이월잔액'/'전월 이월원금' 항목에서 확인할 수 있어요.",
                )
                st.caption(f"= {fmt_won(B_prev)}")

                month_index = st.number_input(
                    "가입 후 경과 개월수", min_value=1, max_value=60, value=1, step=1,
                    help="리볼빙을 이용하기 시작한 지 몇 개월째인지예요. 정확히 모르신다면 대략적인 개월수를 "
                         "입력하셔도 괜찮아요.",
                )

                st.caption("고급 입력 (선택, 예측 정확도 향상)")
                has_hist = st.checkbox("3개월 전 데이터 입력하기", help="입력하면 최근 추세를 반영해 '주의/경고' 판정 정확도가 올라가요. 모르면 비워두셔도 됩니다.")
                cs_t3 = st.number_input("3개월 전 리볼빙 의존도 (%)", 0.0, 100.0, 10.0, step=1.0) / 100 if has_hist else None
                r_t3 = st.number_input("3개월 전 약정결제비율 (%)", 0.0, 100.0, 80.0, step=1.0) / 100 if has_hist else None
                delinquency_count_6m = st.number_input(
                    "최근 6개월 연체 횟수", 0, 6, 0,
                    help="최근 6개월간 결제일을 넘겨 연체한 횟수예요. 모르시면 0으로 두셔도 됩니다.",
                )

            inputs = dict(
                L=L, m=m_pct / 100, r_t=r_pct / 100, S_t=S_t, B_prev=B_prev, month_index=int(month_index),
                carryover_share_t3=cs_t3, r_t3=r_t3, delinquency_count_6m=delinquency_count_6m,
            )
            selection_key = f"manual::{L}:{m_pct}:{r_pct}:{S_t}:{B_prev}:{month_index}:{cs_t3}:{r_t3}:{delinquency_count_6m}"
            anchor_row = build_manual_anchor_row(inputs)
            demo_label = "직접 입력"

        if not anchor_row["minimum_payment_ratio"].notna().all() or not anchor_row["card_limit"].notna().all():
            st.error("선택한 고객의 계좌 정보(m, 한도)를 찾을 수 없습니다.")
            st.stop()

        use_mock = os.environ.get("USE_MOCK_COACHING", "true").strip().lower() != "false"
        st.markdown(
            f'<div style="margin-top:1rem;">{theme.demo_mode_badge(("MOCK 코칭" if use_mock else "실 API 코칭"))}</div>',
            unsafe_allow_html=True,
        )

    return anchor_row, selection_key, demo_label


# ---------------------------------------------------------------------------
# 페이지 1: 내 금융 상태
# ---------------------------------------------------------------------------
def render_home(bundle: dict, outlook: list[dict]):
    st.markdown(theme.section_header("안녕하세요, 고객님.", "지금 내 리볼빙 상태를 한눈에 확인해보세요.").strip(), unsafe_allow_html=True)

    # ① 지금 나는 어떤 상태인가 — 오뚝이 마스코트 + 회복 게이지
    # 두 카드를 하나의 flex row로 묶어(align-items:stretch) 고객·상태와 무관하게 항상
    # 같은 높이가 되고 상/하단 가로선이 정렬되도록 한다. 텍스트가 길어지면 두 카드가
    # 함께 늘어난다.
    hzb = bundle.get("hazard")
    rscore = bundle.get("recovery_score", 50.0)
    hint = rec.recovery_hint(
        bundle["current_risk"],
        transition_probability_3m=(hzb.get("transition_probability_3m") if hzb else None),
    )
    st.markdown(
        theme.compact_html(
            '<div style="display:flex;gap:0.9rem;align-items:stretch;flex-wrap:wrap;margin:0.2rem 0 0.6rem 0;">'
            + mascot.card_html(bundle["current_risk"], recovery_score=rscore, size_px=100,
                               key="home", pad="1rem 1.2rem", fill_height=True)
            + theme.recovery_gauge_html(rscore, hint=hint, fill_height=True)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )

    # 각 지표가 어떻게 계산되는지 설명을 별도 패널로 떼어놓지 않고, 타일 안 라벨과
    # 수치 사이에 작은 글씨로 넣는다.
    sub_metrics = theme.metric_row(
        [
            theme.metric_tile("리볼빙 의존도", fmt_pct(bundle["current_carryover_share"]),
                              desc=METRIC_DEFINITIONS["리볼빙 의존도"]),
            theme.metric_tile("최근 3개월 변화(의존도)", fmt_pct(bundle["current_delta_3m"], signed=True),
                              desc=METRIC_DEFINITIONS["최근 3개월 변화(의존도)"]),
            theme.metric_tile("결제여유", fmt_pct(bundle["current_gap"]),
                              desc=METRIC_DEFINITIONS["결제여유"]),
            theme.metric_tile("연속 최소결제", f"{bundle['current_streak']}개월",
                              desc=METRIC_DEFINITIONS["연속 최소결제"]),
        ]
    )
    st.markdown(
        theme.risk_hero_card(
            level=bundle["current_risk"],
            headline="현재 결제 패턴을 기준으로 판정된 위험 단계입니다.",
            sub_metrics_html=sub_metrics,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(theme.card_open() + theme.risk_stepper_html(bundle["current_risk"]) + theme.card_close(), unsafe_allow_html=True)

    # 조기경보
    escalation = find_first_escalation(outlook, target_level="경고")
    current_above_target = theme.RISK_ORDER[bundle["current_risk"]] >= theme.RISK_ORDER["경고"]
    if current_above_target:
        st.markdown(
            theme.alert_card("🚨", "이미 주의가 필요한 단계입니다", "아래 AI 코칭과 상환 시뮬레이션에서 구체적인 개선 방법을 확인해보세요.", tone=bundle["current_risk"]),
            unsafe_allow_html=True,
        )
    elif escalation:
        st.markdown(
            theme.alert_card(
                "⚠️", "조기경보",
                f"현재 추세가 유지되면 약 {escalation['month_offset']}개월 후 '경고' 단계로 전환될 것으로 예측됩니다.",
                tone="경고",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            theme.alert_card("✅", "안정적인 흐름", "향후 3개월간 현재 추세가 유지된다면 경고 단계로 전환되지 않을 것으로 예측됩니다.", tone="관찰"),
            unsafe_allow_html=True,
        )

    mascot.section_with_accent("향후 위험 궤적", "지금 패턴이 유지될 경우 예상되는 변화예요.", accent_key="focus")
    with st.container(border=True):
        st.markdown(
            f'<div style="color:{theme.SUBTLE};font-size:0.9rem;margin-bottom:0.4rem;">'
            "이 선은 지금 패턴이 그대로 유지될 경우 예상되는 리볼빙 의존도 변화예요. "
            "배경 색이 바뀌는 지점이 위험 단계가 전환되는 시점입니다.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.risk_trajectory_chart(outlook), width="stretch", config={"displayModeBar": False})

    # ④ 언제 위험 단계로 넘어갈 가능성이 있는가 — 이산시간 위험모형(XGBoost와 역할 분리)
    render_hazard_block(bundle, context="home")

    mascot.section_with_accent("핵심 위험 신호", "다음 달 전망에 가장 크게 영향을 준 요인입니다.", accent_key="analyze")
    signal_lines = []
    for sig in top_signals(bundle, k=3):
        label = FEATURE_LABELS.get(sig["feature"], sig["feature"])
        target = "사용액" if sig["model"] == "S" else "약정결제비율"
        direction = "높이는" if sig["value"] > 0 else "낮추는"
        signal_lines.append(f"<li style='margin-bottom:6px;'><b>{label}</b> — 다음 달 {target} 전망을 {direction} 방향으로 작용</li>")
    st.markdown(
        theme.card_open() + f"<ul style='margin:0;padding-left:1.2rem;'>{''.join(signal_lines)}</ul>" + theme.card_close(),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.alert_card("💡", "추천 행동", risk_level_action_text(bundle["predicted_risk"]), tone=bundle["predicted_risk"]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("위험 원인 확인하기", type="primary", width="stretch"):
            go_to("risk")
    with c2:
        if st.button("AI 코칭 받기", width="stretch"):
            go_to("coaching")


# ---------------------------------------------------------------------------
# 페이지 2: 위험 분석
# ---------------------------------------------------------------------------
def render_risk(bundle: dict):
    def _risk_tile(label: str, state: str) -> str:
        c = theme.RISK_COLORS.get(state, theme.RISK_COLORS["관찰"])["main"]
        img = mascot.state_img(state, size_px=74)
        return (
            f'<div style="flex:1;min-width:150px;background:{theme.SURFACE};border:1px solid {theme.LINE};'
            f'border-radius:14px;padding:1rem 1.2rem;">'
            f'<div style="color:{theme.SUBTLE};font-size:0.82rem;font-weight:600;">{label}</div>'
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-top:4px;">'
            f'{img}<div style="color:{c};font-size:1.5rem;font-weight:900;">{state}</div></div></div>'
        )

    st.markdown(
        theme.compact_html(
            f'<div style="display:flex;gap:0.9rem;flex-wrap:wrap;background:{theme.SURFACE};border:1px solid {theme.LINE};'
            f'border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem;">'
            f'{_risk_tile("현재 위험도", bundle["current_risk"])}'
            f'{_risk_tile("다음 달 예측 위험도", bundle["predicted_risk"])}</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(theme.section_header("위험도 구성", "네 가지 신호를 종합해 위험 단계를 판정합니다.").strip(), unsafe_allow_html=True)
    rising = "상승 중" if pd.notna(bundle["current_delta_3m"]) and bundle["current_delta_3m"] > 0 else "안정적"
    st.markdown(
        theme.metric_row(
            [
                theme.metric_tile("리볼빙 의존도", fmt_pct(bundle["current_carryover_share"]),
                                  desc=METRIC_DEFINITIONS["리볼빙 의존도"]),
                theme.metric_tile("상승 추세", rising, note=fmt_pct(bundle["current_delta_3m"], signed=True),
                                  desc=METRIC_DEFINITIONS["상승 추세"]),
                theme.metric_tile("결제여유", fmt_pct(bundle["current_gap"]), note="약정결제비율 − 최소결제비율",
                                  desc=METRIC_DEFINITIONS["결제여유"]),
                theme.metric_tile("최소결제 반복", f"{bundle['current_streak']}개월 연속" if bundle["current_streak"] > 0 else "없음",
                                  desc=METRIC_DEFINITIONS["최소결제 반복"]),
            ]
        ),
        unsafe_allow_html=True,
    )

    mascot.section_with_accent(
        "예측에 영향을 준 주요 요인",
        "막대가 길수록 영향이 큽니다. 주황색은 위험을 높이는 방향, 청록색은 낮추는 방향이에요.",
        accent_key="report",
    )

    def _factor_explain_block(shap_dict: dict) -> str:
        """상위 3개 요인의 '왜 영향을 주는지' 설명을, 느슨한 텍스트가 아니라 한 덩어리
        블록(라벨 + 설명 목록)으로 묶어서 보여준다."""
        top_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        rows = []
        for i, (feat, _v) in enumerate(top_items):
            label = FEATURE_LABELS.get(feat, feat)
            expl = FEATURE_EXPLANATIONS.get(feat, "")
            sep = "" if i == 0 else f"border-top:1px dashed {theme.LINE};"
            rows.append(
                f'<div style="padding:0.55rem 0;{sep}">'
                f'<div style="font-weight:800;color:{theme.INK};font-size:0.9rem;">{label}</div>'
                f'<div style="color:{theme.SUBTLE};font-size:0.83rem;line-height:1.5;margin-top:2px;">{expl}</div>'
                f'</div>'
            )
        return theme.compact_html(
            f'<div style="background:{theme.PAGE_BG};border:1px solid {theme.LINE};border-radius:10px;'
            f'padding:0.15rem 0.9rem 0.55rem;margin-top:0.7rem;">'
            f'<div style="font-size:0.72rem;font-weight:800;color:{theme.SUBTLE};letter-spacing:0.03em;'
            f'padding:0.6rem 0 0.15rem;">이 요인들이 왜 영향을 주나요?</div>'
            + "".join(rows) + "</div>"
        )

    def shap_section(shap_dict: dict, title: str, k: int = 5):
        # 두 박스 높이를 고정해 좌우 정렬을 맞춘다(차트 높이는 동일, 설명 글자 수만 달라서
        # 예전엔 박스 높이가 어긋났음).
        with st.container(border=True, height=700):
            st.markdown(
                f'<div style="font-weight:800;color:{theme.INK};font-size:1rem;margin-bottom:0.2rem;">{title}</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(charts.shap_bar_chart(shap_dict, FEATURE_LABELS, k=k), width="stretch", config={"displayModeBar": False})
            st.markdown(_factor_explain_block(shap_dict), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        shap_section(bundle["shap_S"], "다음 달 사용액 예측")
    with c2:
        shap_section(bundle["shap_r"], "다음 달 상환 비율 예측")

    with st.expander("전문 분석 보기 (원본 피처명 · SHAP 수치)"):
        def shap_bar(shap_dict: dict, title: str):
            items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            labels = [f for f, _ in items]
            values = [v for _, v in items]
            colors = ["#c62828" if v > 0 else "#1565c0" for v in values]
            fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
            fig.update_layout(title=title, height=380, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            shap_bar(bundle["shap_S"], f"S(t+1) 예측 기여도 (base={bundle['base_S']:,.0f}원)")
        with cc2:
            shap_bar(bundle["shap_r"], f"r(t+1) 예측 기여도 (base={bundle['base_r']*100:.1f}%)")
        st.caption("빨간 막대 = 값을 높이는 방향, 파란 막대 = 낮추는 방향으로 기여")

    if st.button("AI 코칭 받기", type="primary"):
        go_to("coaching")


# ---------------------------------------------------------------------------
# 페이지 3: AI 상환 코칭
# ---------------------------------------------------------------------------
def render_coaching(bundle: dict, anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols, outlook: list[dict]):
    st.markdown(
        theme.compact_html(
            f'<div style="background:{theme.SURFACE};border:1px solid {theme.LINE};border-radius:16px;'
            f'padding:1.1rem 1.4rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.8rem;">'
            f'{mascot.accent("smile", size_px=56)}'
            f'<div><div style="font-weight:900;font-size:1.58rem;color:{theme.BRAND};letter-spacing:-0.01em;">AI 상환 코칭</div>'
            f'<div style="color:{theme.SUBTLE};margin-top:4px;">현재까지의 결제 패턴과 앞으로의 예측 결과를 모두 종합해서 알려드릴게요.</div></div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )

    # --- 시각 요약: 위험도 배지 + 3개월 궤적 타임라인 (텍스트만 있으면 눈에 잘 안 들어오니
    #     코칭 문장을 읽기 전에 무엇을 근거로 한 코칭인지 그림으로 먼저 보여준다) ---
    badges_html = (
        '<div style="display:flex;gap:2.2rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem;">'
        f'<div><div style="color:{theme.SUBTLE};font-size:0.8rem;font-weight:700;margin-bottom:4px;">현재</div>{theme.risk_badge_html(bundle["current_risk"])}</div>'
        f'<div style="color:{theme.SUBTLE};font-size:1.4rem;">→</div>'
        f'<div><div style="color:{theme.SUBTLE};font-size:0.8rem;font-weight:700;margin-bottom:4px;">다음 달 예측</div>{theme.risk_badge_html(bundle["predicted_risk"])}</div>'
        "</div>"
    )
    steps = [
        {"label": "현재" if s["month_offset"] == 0 else f"{s['month_offset']}개월 후", "level": s["level"], "value": fmt_pct(s["carryover_share"])}
        for s in outlook
    ]
    st.markdown(
        theme.card_open() + badges_html + theme.forecast_timeline_html(steps) + theme.card_close(),
        unsafe_allow_html=True,
    )

    # 회복 서사: 흔들림 → 중심 잡기 (오뚝이의 핵심 메시지). 마스코트 2컷.
    _shaky, _steady = mascot.accent("shaky", size_px=76), mascot.accent("steady", size_px=76)
    if _shaky and _steady:
        st.markdown(
            theme.compact_html(f"""
            <div style="display:flex;gap:1rem;align-items:center;justify-content:center;flex-wrap:wrap;
                        background:{theme.BRAND_SOFT};border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
                <div style="text-align:center;">{_shaky}
                    <div style="font-size:0.78rem;color:{theme.SUBTLE};font-weight:700;margin-top:2px;">지금 (흔들리는 중)</div></div>
                <div style="color:{theme.SUBTLE};font-size:1.5rem;">→</div>
                <div style="text-align:center;">{_steady}
                    <div style="font-size:0.78rem;color:{theme.BRAND};font-weight:800;margin-top:2px;">행동을 바꾸면 (중심 회복)</div></div>
                <div style="flex:1;min-width:180px;color:{theme.INK};font-size:0.9rem;line-height:1.5;">
                    아래 코칭과 시뮬레이션에서, 어떤 행동을 가정하면 오뚝이가 다시 중심을 잡는지 확인할 수 있어요.</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # --- 최소 개입액을 먼저 계산해서, 사용자가 시뮬레이터를 직접 안 돌려봤어도
    #     "얼마를 더 갚으면 되는지"를 코칭 컨텍스트의 simulation 근거로 자동 포함시킨다.
    #     (사용자가 직접 시뮬레이터를 돌려본 결과가 있으면 그걸 우선한다) ---
    with st.spinner("현재 상태·SHAP 요인·3개월 전망·추가 상환 시나리오를 종합하는 중..."):
        min_intervention = find_minimum_intervention(
            model_S, model_r, feature_cols, anchor_row, monthly_transaction, derived_features, horizon=3, target_max_level="경고"
        )

    manual_sim = st.session_state.get("simulation_result")
    auto_intervention = min_intervention["achieved"] and min_intervention["extra_payment"] and min_intervention["extra_payment"] > 0
    if manual_sim:
        simulation_ctx = manual_sim
    elif auto_intervention:
        first_step = min_intervention["trajectory"][0]
        simulation_ctx = {
            "extra_payment": min_intervention["extra_payment"],
            "new_predicted_carryover_share": first_step["carryover_share"],
            "new_risk_indicator": first_step["level"],
        }
    else:
        simulation_ctx = None

    # --- 코칭 컨텍스트: 현재 상태 + 다음 달 예측 + SHAP 상위 요인(복수) + 3개월 전망 +
    #     (있다면) 시뮬레이션 결과까지 전부 종합해서 LLM에 전달한다. LLM이 이걸 바탕으로
    #     세그먼트 개수를 유연하게 정해 풍부하게 설명하도록, 개수를 이쪽에서 고정하지 않는다. ---
    top_signal_list = top_signals(bundle, k=4)
    top_shap_features_ctx = [{"feature": s["feature"], "contribution": s["value"]} for s in top_signal_list]

    coaching_context = {
        "risk_indicator": bundle["predicted_risk"],
        "carryover_share": bundle["predicted_carryover_share"],
        "carryover_share_delta_3m": bundle["predicted_delta_3m"],
        "payment_ratio_gap": bundle["predicted_gap"],
        "current_risk_indicator": bundle["current_risk"],
        "current_carryover_share": bundle["current_carryover_share"],
        "current_streak": bundle["current_streak"],
        "top_shap_features": top_shap_features_ctx,
        "outlook": [{"month_offset": s["month_offset"], "level": s["level"], "carryover_share": s["carryover_share"]} for s in outlook],
        "hazard": bundle.get("hazard"),  # 이산시간 위험모형: 전환확률/예상 전환 시점 (source: 'hazard')
        "simulation": simulation_ctx,
    }

    try:
        message = generate_coaching_message(coaching_context)
        segments = message["segments"]
    except Exception as e:
        st.error(f"코칭 메시지 생성/검증 실패: {e}")
        return

    # --- 세그먼트를 카드 여러 개로 잘게 쪼개지 않고, "상황·이유·시점"(raw_data+shap+hazard)을
    #     하나로 묶어 한 흐름으로 읽히게 한다. simulation 근거만 "행동" 카드로 분리한다. ---
    story_segs = [s for s in segments if s["source"] in ("raw_data", "shap", "hazard")]
    sim_segs = [s for s in segments if s["source"] == "simulation"]

    if message.get("summary"):
        st.markdown(
            theme.alert_card("🧭", "한 줄 요약", message["summary"], tone=bundle["predicted_risk"]),
            unsafe_allow_html=True,
        )

    if story_segs:
        story_html = "".join(f'<p style="margin:0 0 0.85rem 0;">{theme.highlight_text(seg["text"])}</p>' for seg in story_segs)
        mascot.section_with_accent("지금 상황과 이유, 그리고 시점", accent_key="report")
        st.markdown(theme.coaching_card(story_html, accent=theme.BRAND), unsafe_allow_html=True)

    # --- 최소 개입액(또는 사용자가 직접 돌려본 시뮬레이션 결과)을 "지금 할 수 있는 행동"
    #     하나의 카드 안에 숫자 + LLM 설명 문장을 함께 묶어서 보여준다. ---
    mascot.section_with_accent("지금 할 수 있는 행동", accent_key="strategy")
    sim_narrative = "".join(f'<p style="margin:0.6rem 0 0 0;">{theme.highlight_text(seg["text"])}</p>' for seg in sim_segs)
    if auto_intervention:
        action_html = (
            f'<div style="font-size:1.7rem;font-weight:900;color:{theme.BRAND};margin-bottom:6px;">'
            f'월 {min_intervention["extra_payment"]:,.0f}원 추가 상환</div>'
            f'<div style="color:{theme.SUBTLE};">이 금액을 추가로 상환하면 향후 <b style="color:{theme.BRAND};">3개월</b> 동안 '
            f'위험 단계가 <b style="color:{theme.RISK_COLORS["경고"]["main"]};">경고</b> 이상으로 상승하지 않을 것으로 계산됩니다.</div>'
            + sim_narrative
        )
        accent = theme.BRAND
    elif min_intervention["achieved"]:
        action_html = '현재 패턴을 유지해도 향후 3개월간 "경고" 단계로 상승하지 않을 것으로 예측됩니다. 추가 상환이 꼭 필요하지는 않아요.' + sim_narrative
        accent = theme.RISK_COLORS["관찰"]["main"]
    else:
        action_html = "카드 한도 내 추가 상환만으로는 향후 3개월간 위험 단계 상승을 완전히 막기 어려운 것으로 계산됩니다. 전문 상담을 함께 고려해보세요." + sim_narrative
        accent = theme.RISK_COLORS["경고"]["main"]
    st.markdown(theme.coaching_card(action_html, accent=accent), unsafe_allow_html=True)

    if st.button("상환 시뮬레이션 해보기", type="primary"):
        if auto_intervention:
            st.session_state["prefill_extra_payment"] = int(min_intervention["extra_payment"])
        go_to("simulator")

    with st.expander("근거 보기 — 왜 이런 조언을 했나요?"):
        for seg in segments:
            st.markdown(theme.evidence_line(seg["source"], seg["text"]), unsafe_allow_html=True)

    st.caption("모든 조언 문장은 JSON Schema 검증과 근거(raw_data/shap/simulation) 일치성 검증을 통과한 것만 표시됩니다.")


# ---------------------------------------------------------------------------
# 페이지 4: 상환 시뮬레이션
# ---------------------------------------------------------------------------
def render_simulator(bundle: dict, anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols):
    st.markdown(
        theme.card_open()
        + '<div style="font-weight:800;font-size:1.15rem;">내가 조금 더 갚으면 어떻게 달라질까요?</div>'
        + theme.card_close(),
        unsafe_allow_html=True,
    )

    L = int(bundle["L"])

    # --- 🎯 오뚝이 회복 미션 (경량) : 포인트/뱃지 없음. 기존 simulate_extra_payment() 재사용 진입점일 뿐. ---
    P_t = float(bundle["B_current"]) + float(bundle["pred_S"])
    mission_extra = int(round(min(0.05 * P_t, L) / 10_000) * 10_000)  # 약정결제비율 +5%p 에 해당하는 추가 상환액
    _acc_ap = theme.RISK_COLORS["주의"]["main"]
    st.markdown(
        theme.compact_html(
            f'<div style="background:{theme.SURFACE};border:1px dashed {_acc_ap}80;border-left:4px solid {_acc_ap};'
            f'border-radius:14px;padding:1.1rem 1.3rem;margin:0.6rem 0;display:flex;gap:0.8rem;align-items:flex-start;">'
            f'{mascot.accent("applaud", size_px=58)}'
            f'<div><div style="font-weight:900;color:{theme.BRAND};font-size:1.53rem;margin-bottom:0.4rem;letter-spacing:-0.01em;">🎯 이번 달 오뚝이 미션</div>'
            f'<div style="color:{theme.INK};font-size:0.93rem;line-height:1.55;">'
            f'약정결제비율을 지금보다 <b>약 5%p</b> 높여보는 시나리오예요 (추가 상환 약 <b>{mission_extra:,.0f}원</b>에 해당). '
            f'아래 버튼을 누르면 이 값이 시뮬레이션에 적용돼요 — 미션 달성이 아니라, 행동을 바꿨을 때 '
            f'<b>모델 출력이 실제로 어떻게 변하는지</b> 확인하는 게 목적이에요.</div></div></div>'
        ),
        unsafe_allow_html=True,
    )
    if st.button("이 미션 시나리오 적용해보기", key="apply_mission"):
        st.session_state["prefill_extra_payment"] = mission_extra
        st.rerun()
    SLIDER_KEY, NUMBER_KEY = "extra_payment_slider_widget", "extra_payment_number_widget"

    # 슬라이더/숫자입력 두 위젯을 같은 값으로 동기화한다. 위젯에 key가 이미 지정돼 있으면
    # value= 인자는 (그 key가 session_state에 처음 생기는 시점 이후로는) 무시되므로,
    # 값을 프로그래밍적으로 바꿀 때는 반드시 위젯의 key 자체를 직접 덮어써야 한다
    # (nav_page 라디오와 동일한 종류의 함정 — sidebar 쪽 수정 코멘트 참고).
    if SLIDER_KEY not in st.session_state:
        st.session_state[SLIDER_KEY] = 0
    if NUMBER_KEY not in st.session_state:
        st.session_state[NUMBER_KEY] = 0
    if "prefill_extra_payment" in st.session_state:
        prefill_val = min(st.session_state.pop("prefill_extra_payment"), L)
        st.session_state[SLIDER_KEY] = prefill_val
        st.session_state[NUMBER_KEY] = prefill_val
    # 고객이 바뀌어 한도(L)가 줄어든 경우, 이전 값이 새 한도를 넘지 않도록 클램프
    st.session_state[SLIDER_KEY] = min(st.session_state[SLIDER_KEY], L)
    st.session_state[NUMBER_KEY] = min(st.session_state[NUMBER_KEY], L)

    def _sync_from_slider():
        st.session_state[NUMBER_KEY] = st.session_state[SLIDER_KEY]

    def _sync_from_number():
        st.session_state[SLIDER_KEY] = st.session_state[NUMBER_KEY]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.slider("추가 상환액", 0, L, step=10_000, key=SLIDER_KEY, on_change=_sync_from_slider)
    with col_r:
        st.number_input("정확한 금액", 0, L, step=10_000, key=NUMBER_KEY, on_change=_sync_from_number)

    extra_payment = st.session_state[SLIDER_KEY]

    sim_calc = simulate_extra_payment(
        B_prev=bundle["B_current"], S_pred=bundle["pred_S"], r_pred=bundle["pred_r"],
        m=bundle["m"], i=bundle["i"], extra_payment=extra_payment,
    )
    new_delta_3m = bundle["predicted_delta_3m"] + (sim_calc["carryover_share"] - bundle["predicted_carryover_share"]) \
        if pd.notna(bundle["predicted_delta_3m"]) else np.nan
    new_streak = (bundle["current_streak"] + 1) if sim_calc["payment_ratio_gap"] <= PAYMENT_RATIO_GAP_WARN_CUTOFF else 0
    new_risk = classify_risk_indicator(
        carryover_share=sim_calc["carryover_share"],
        carryover_share_delta_3m=new_delta_3m,
        payment_ratio_gap=sim_calc["payment_ratio_gap"],
        minimum_payment_streak=new_streak,
        warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT,
    )
    _sim_recovery = rec.recovery_score(new_risk, sim_calc["carryover_share"])
    st.session_state["simulation_result"] = {
        "extra_payment": extra_payment,
        "new_predicted_carryover_share": sim_calc["carryover_share"],
        "new_risk_indicator": new_risk,
        "r_effective": sim_calc["r_effective"],
    }

    mt_idx = monthly_transaction.set_index(["account_id", "month_index"])
    df_idx = derived_features.set_index(["account_id", "month_index"])
    traj = simulate_intervention_trajectory(model_S, model_r, feature_cols, anchor_row, mt_idx, df_idx, extra_payment, horizon=3)
    level_3m = traj[-1]["level"] if traj else new_risk

    st.markdown(theme.section_header("변경 결과"), unsafe_allow_html=True)
    # 주의: st.markdown(unsafe_allow_html=True)는 HTML 블록 안에 빈 줄이 있으면 그 지점에서
    # HTML 인식을 멈추고 이후 내용을 그대로 텍스트로 노출한다. theme.compact_html()로
    # 빈 줄을 제거해서 반환한다 (theme.py의 다른 컴포넌트들과 동일한 방식).
    compare_html = theme.compact_html(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;border:1px solid {theme.LINE};border-radius:14px;overflow:hidden;">
        <div style="padding:0.9rem;font-weight:700;color:{theme.SUBTLE};background:{theme.BRAND_SOFT};">구분</div>
        <div style="padding:0.9rem;font-weight:700;text-align:center;background:{theme.BRAND_SOFT};">현재 예측(개입 없음)</div>
        <div style="padding:0.9rem;font-weight:700;text-align:center;background:{theme.BRAND_SOFT};">추가 상환 시뮬레이션</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};">다음 달 위험도</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">{theme.risk_badge_html(bundle['predicted_risk'], 'sm')}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">{theme.risk_badge_html(new_risk, 'sm')}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};">리볼빙 의존도</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">{fmt_pct(bundle['predicted_carryover_share'])}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;font-weight:800;color:{theme.BRAND};">{fmt_pct(sim_calc['carryover_share'])}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};">3개월 후 위험도</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">—</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">{theme.risk_badge_html(level_3m, 'sm')}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};">오뚝이 회복 게이지</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;">{bundle.get('predicted_recovery_score', 0):.0f}</div>
        <div style="padding:0.9rem;border-top:1px solid {theme.LINE};text-align:center;font-weight:800;color:{theme.BRAND};">{_sim_recovery:.0f}</div>
    </div>
    """)
    st.markdown(compare_html, unsafe_allow_html=True)

    # 회복 게이지 + 마스코트 갱신 (시뮬레이션 상태 반영) + hazard 재계산
    _dscore = _sim_recovery - bundle.get("predicted_recovery_score", _sim_recovery)
    _mc1, _mc2 = st.columns([1, 2])
    with _mc1:
        mascot.render(new_risk, recovery_score=_sim_recovery, size_px=170, key="sim",
                      caption=("시나리오를 적용하면 오뚝이 상태가 이렇게 바뀌는 것으로 계산돼요."
                               if extra_payment > 0 else None))
    with _mc2:
        st.markdown(
            theme.recovery_gauge_html(
                _sim_recovery,
                delta=_dscore,
                hint=rec.recovery_hint(new_risk, simulation_delta_score=_dscore),
            ),
            unsafe_allow_html=True,
        )
    _sim_hzb = _simulated_hazard_bundle(anchor_row, bundle, sim_calc, new_risk, new_delta_3m, new_streak)
    if bundle.get("hazard") and bundle["hazard"].get("applicable") and _sim_hzb and _sim_hzb.get("applicable"):
        _b = bundle["hazard"].get("transition_probability_3m")
        _a = _sim_hzb.get("transition_probability_3m")
        if _b is not None and _a is not None:
            st.caption(
                f"위험 전환 모형 기준 향후 3개월 경고 전환 가능성: {_b*100:.0f}% → 시나리오 적용 시 {_a*100:.0f}% "
                f"(입력한 가정에 따른 추정치)"
            )

    if extra_payment > 0:
        st.markdown(
            theme.alert_card(
                "📌", "요약",
                f"월 {extra_payment:,.0f}원 추가 상환 시, 향후 위험 단계 상승을 완화할 수 있는 것으로 계산됩니다.",
                tone=new_risk,
            ),
            unsafe_allow_html=True,
        )

    mascot.section_with_accent("🎯 최소 개입액", "위험 단계를 '경고' 미만으로 유지하기 위한 최소 금액입니다.", accent_key="strategy")
    with st.spinner("계산 중..."):
        min_intervention = find_minimum_intervention(
            model_S, model_r, feature_cols, anchor_row, monthly_transaction, derived_features, horizon=3, target_max_level="경고"
        )
    if min_intervention["achieved"] and min_intervention["extra_payment"] and min_intervention["extra_payment"] > 0:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                theme.card_open()
                + f'<div style="font-size:1.8rem;font-weight:900;color:{theme.BRAND};">월 {min_intervention["extra_payment"]:,.0f}원</div>'
                + f'<div style="color:{theme.SUBTLE};margin-top:4px;">이 금액을 추가 상환하면 향후 3개월 동안 위험도가 "경고" 이상으로 상승하지 않는 것으로 계산됩니다.</div>'
                + theme.card_close(),
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("이 금액 적용하기", width="stretch"):
                st.session_state["prefill_extra_payment"] = int(min_intervention["extra_payment"])
                st.rerun()
    elif min_intervention["achieved"]:
        st.info("추가 상환 없이도 향후 3개월간 '경고' 단계로 상승하지 않을 것으로 예측됩니다.")
    else:
        st.warning("카드 한도 내 추가 상환만으로는 향후 3개월간 위험 단계 상승을 완전히 막기 어려운 것으로 계산됩니다.")

    st.markdown(
        theme.section_header(
            "📊 추가 상환액별 비교", "매달 얼마를 더 갚느냐에 따라 3개월 뒤 위험 단계가 어떻게 달라지는지 미리 비교해볼 수 있어요."
        ).strip(),
        unsafe_allow_html=True,
    )
    # 시나리오: 현재 유지 / 정액 추가상환 3종 / 약정결제비율 +5%p·+10%p (스펙 10번)
    _r5 = int(round(min(0.05 * P_t, L) / 10_000) * 10_000)
    _r10 = int(round(min(0.10 * P_t, L) / 10_000) * 10_000)
    preset_specs = [
        (0, "현재 유지"), (50_000, "월 +5만"), (100_000, "월 +10만"),
        (_r5, "결제비율 +5%p"), (_r10, "결제비율 +10%p"),
    ]
    with st.spinner("금액별 시나리오를 계산하는 중..."):
        scenario_results = []
        for amt, lab in preset_specs:
            amt_capped = min(amt, L)
            preset_traj = simulate_intervention_trajectory(model_S, model_r, feature_cols, anchor_row, mt_idx, df_idx, amt_capped, horizon=3)
            final_step = preset_traj[-1] if preset_traj else {"level": bundle["predicted_risk"], "carryover_share": bundle["predicted_carryover_share"]}
            scenario_results.append({
                "extra_payment": amt_capped, "label": lab, "level": final_step["level"],
                "carryover_share": final_step["carryover_share"],
                "recovery_score": rec.recovery_score(final_step["level"], final_step["carryover_share"]),
            })
    preset_amounts = [s["extra_payment"] for s in scenario_results]

    with st.container(border=True):
        st.plotly_chart(charts.intervention_comparison_chart(scenario_results), width="stretch", config={"displayModeBar": False})
        st.caption("막대 색은 3개월 뒤 예측되는 위험 단계를 나타냅니다.")

    baseline_level = scenario_results[0]["level"]
    base_rscore = scenario_results[0]["recovery_score"]
    cmp_cols = st.columns(len(scenario_results))
    for col, sc in zip(cmp_cols, scenario_results):
        with col:
            gauge_line = (
                f'<div style="margin-top:6px;font-size:0.8rem;color:{theme.SUBTLE};">회복 게이지 '
                f'<b style="color:{theme.INK};">{sc["recovery_score"]:.0f}</b>'
                + (f' <span style="color:{theme.RISK_COLORS["관찰"]["main"] if sc["recovery_score"]>=base_rscore else theme.RISK_COLORS["경고"]["main"]};">'
                   f'({sc["recovery_score"]-base_rscore:+.0f})</span>' if sc is not scenario_results[0] else "")
                + "</div>"
            )
            if sc["extra_payment"] == 0:
                body = f'<div style="text-align:center;"><b>{sc["label"]}</b><br><br>{theme.risk_badge_html(sc["level"], "sm")}{gauge_line}</div>'
                accent = theme.SUBTLE
            else:
                improved = theme.RISK_ORDER[sc["level"]] < theme.RISK_ORDER[baseline_level]
                verb = "낮아집니다" if improved else "유지됩니다"
                sentence = f'{sc["label"]}(추가 상환 약 {sc["extra_payment"]:,.0f}원) 가정 시, 3개월 뒤 위험 단계가 {baseline_level}에서 {sc["level"]}로 {verb}.'
                body = (
                    f'<div style="text-align:center;font-weight:800;">{sc["label"]}</div>'
                    f'<div style="text-align:center;margin-top:4px;">{theme.risk_badge_html(baseline_level, "sm")} → {theme.risk_badge_html(sc["level"], "sm")}</div>'
                    f'<div style="margin-top:8px;font-size:0.8rem;">{theme.highlight_text(sentence)}</div>{gauge_line}'
                )
                accent = theme.BRAND if improved else theme.SUBTLE
            st.markdown(theme.coaching_card(body, accent=accent), unsafe_allow_html=True)

    if st.button("AI 코칭에서 결과 확인하기 →", type="primary"):
        go_to("coaching")


# ---------------------------------------------------------------------------
# 페이지 5: 모델 신뢰도
# ---------------------------------------------------------------------------
def render_trust():
    model_metrics, risk_sensitivity, shap_importance = load_metrics_outputs()

    _acc = mascot.accent("analyze", size_px=56)
    st.markdown(
        theme.compact_html(f"""
        <div style="display:flex;align-items:center;gap:0.8rem;background:{theme.SURFACE};
                    border:1px solid {theme.LINE};border-radius:16px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
            {_acc}
            <div><div style="font-weight:900;font-size:1.65rem;color:{theme.BRAND};letter-spacing:-0.01em;">모델 신뢰도</div>
                 <div style="color:{theme.SUBTLE};margin-top:4px;">이 서비스의 예측 결과가 어떻게 검증되었는지 공개합니다.</div></div>
        </div>"""),
        unsafe_allow_html=True,
    )

    one_step = model_metrics["one_step"]
    metrics_info = [
        ("다음 달 청구액 예측 오차", fmt_won(one_step["S_next_month"]["mae"]), f"다음 달 청구액 예측은 평균적으로 약 {one_step['S_next_month']['mae']/10000:.1f}만원의 오차가 있습니다."),
        ("약정결제비율 예측 오차", fmt_pct(one_step["r_next_month"]["mae"]), "다음 달 약정결제비율 예측은 평균적으로 이 정도의 오차가 있습니다."),
        ("리볼빙 의존도 예측 오차", fmt_pct(one_step["predicted_carryover_share_next_month"]["mae"]), "예측된 S/r을 재귀식에 대입해 계산한 값의 오차입니다."),
    ]
    cols = st.columns(3)
    for col, (label, value, desc) in zip(cols, metrics_info):
        with col:
            st.markdown(theme.metric_tile(label, value, note=desc), unsafe_allow_html=True)

    st.markdown(theme.section_header("검증 방법").strip(), unsafe_allow_html=True)
    st.markdown(
        theme.card_open()
        + "<b>시간 기준 Train / Test 분할</b><br>"
        + "Train: 초기 9개월 · Test: 이후 3개월<br>"
        + f'<div style="color:{theme.SUBTLE};margin-top:6px;">무작위로 섞어 나누면 미래 정보가 학습에 섞여 들어가는 '
        + '데이터 누출이 발생할 수 있어, 실제 서비스 운영 방식과 동일하게 "과거로 미래를 예측"하는 시간 순서를 그대로 지켰습니다.</div>'
        + theme.card_close(),
        unsafe_allow_html=True,
    )
    st.caption(model_metrics["notes"]["one_step"])

    st.markdown(theme.section_header("다개월 예측 성능", "예측 기간이 늘어날수록 오차가 어떻게 쌓이는지 보여줍니다.").strip(), unsafe_allow_html=True)
    rec = model_metrics["recursive_multistep"]
    horizons = ["horizon_1", "horizon_2", "horizon_3"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[rec[h]["carryover_share"]["mae"] * 100 for h in horizons], mode="lines+markers", name="리볼빙 의존도 오차(%p)"))
    fig.update_layout(height=320, xaxis_title="예측 개월수", yaxis_title="MAE (%p)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(model_metrics["notes"]["recursive_multistep_horizon_1"])

    # ------------------------------------------------------------------
    # Hazard Model (보완 모델) 성능 — XGBoost와 별도
    # ------------------------------------------------------------------
    hm = load_hazard_metrics()
    st.markdown(
        theme.section_header(
            "위험 전환 모형 (Discrete-time Hazard Model)",
            "XGBoost와 역할이 다른 보완 모델입니다: '언제 경고/심화 단계로 넘어갈 가능성이 있는가'.",
        ).strip(),
        unsafe_allow_html=True,
    )
    if hm is None:
        st.info("outputs/hazard_metrics.json 이 없습니다. `python scripts/train_hazard.py` 를 먼저 실행하세요.")
    else:
        pp = hm["person_period"]
        cv = hm["account_5fold_cv"]
        ts = hm["temporal_split"]
        cvb = hm["c_index_vs_baseline"]
        cols = st.columns(4)
        with cols[0]:
            st.markdown(theme.metric_tile("person-period 행 / 전환 이벤트",
                        f"{pp['n_person_period_rows']:,} / {pp['n_events_total']}",
                        note=f"censored {pp['n_censored']}계좌 · already_high_risk {pp['n_already_high_risk_excluded']}계좌"),
                        unsafe_allow_html=True)
        with cols[1]:
            st.markdown(theme.metric_tile("C-index (계좌 5-fold CV)",
                        f"{cv['account_3m_c_index_mean']}",
                        note=f"std {cv['account_3m_c_index_std']} · KM baseline 0.5 대비 +{cvb['relative_improvement_pct_over_baseline']}%"),
                        unsafe_allow_html=True)
        with cols[2]:
            st.markdown(theme.metric_tile("C-index (시간분할 test)",
                        f"{ts['three_month_transition_eval']['c_index']}",
                        note=f"전환 이벤트 {ts['test_events']}건" + (" ⚠️ <30, CI 넓음" if ts["small_sample_warning"] else "")),
                        unsafe_allow_html=True)
        with cols[3]:
            st.markdown(theme.metric_tile("person-period Brier (CV)",
                        f"{cv['person_period_brier_mean']}",
                        note="naive ≈ 0.10 · hazard 5분위 보정 양호"),
                        unsafe_allow_html=True)
        if ts.get("small_sample_note"):
            st.warning(ts["small_sample_note"])
        st.caption(
            "calibration 주의: " + hm.get("calibration_caveat", {}).get("three_month_transition_probability", "")
        )

        with st.expander("위험 전환 모형 상세 (KM baseline · calibration · 위험군별 이벤트율 · 한계)"):
            km = hm["kaplan_meier_baseline"]
            st.markdown("**Kaplan-Meier baseline 생존함수 (개인화 없음)**")
            st.table(pd.DataFrame({"개월(t)": list(km.keys()), "S(t)": list(km.values())}).set_index("개월(t)"))
            st.markdown("**예측 hazard 5분위별 실제 이벤트율 (calibration)**")
            st.table(pd.DataFrame(hm["risk_group_event_rates_hazard_quintile"]).set_index("hazard_quintile"))
            st.markdown("**계좌 5-fold CV — 3개월 전환확률 calibration (pooled)**")
            st.table(pd.DataFrame(cv["account_3m_calibration_pooled"]))
            st.markdown("**duration 항 비교**")
            st.json(hm["duration_spec_comparison"])
            st.markdown("**features / leakage 통제 / 알려진 한계**")
            st.write("features:", hm["features"])
            for n in hm["leakage_controls"]:
                st.caption("• " + n)
            for n in hm["known_limitations"]:
                st.caption("⚠ " + n)
            st.markdown("**전환확률/예상 전환 시점 예시**")
            st.table(pd.DataFrame(hm["examples_transition_probability"]))

        # 마스코트 상태 매핑 상태 (전문가/심사위원 확인용)
        ms = mascot.mapping_status()
        st.markdown(theme.section_header("오뚝이 마스코트 상태 매핑").strip(), unsafe_allow_html=True)
        if ms.get("needs_confirmation"):
            st.warning("state_mapping.json 은 위치 기반 추정 매핑입니다 — 각 이미지를 열어 최종 확인 필요. "
                       + ms.get("confidence_note", ""))
        st.caption("risk_indicator → 캐릭터 이미지 (state_mapping.json)")
        st.table(pd.DataFrame([{"상태": k, "파일": v["file"], "파일존재": v["exists"]} for k, v in ms["states"].items()]))
        if ms.get("accents"):
            st.caption("화면 곳곳의 표정/몸짓 accent (state_mapping.json → accents)")
            st.table(pd.DataFrame([{"accent": k, "파일": v["file"], "파일존재": v["exists"], "추정 의미": v.get("추정", "")}
                                   for k, v in ms["accents"].items()]))

    st.markdown(theme.section_header("모델의 한계").strip(), unsafe_allow_html=True)
    st.markdown(
        theme.alert_card(
            "ℹ️", "개념검증(PoC) 단계",
            "현재 MVP는 공시 통계로 파라미터를 보정한 합성 금융데이터로 검증되었습니다. "
            "실서비스 전환 시에는 실제 금융데이터를 통한 추가 검증과 재학습이 필요합니다.",
            tone="주의",
        ),
        unsafe_allow_html=True,
    )

    with st.expander("전문가용 지표 보기 (원본 변수명 · 임계치 민감도 · 재귀 예측 표)"):
        st.markdown("**3개월 재귀 예측 누적오차 (원본 지표)**")
        rows = [
            {"horizon": h, "S MAE(원)": f"{rec[h]['S']['mae']:,.0f}", "r MAE(%p)": f"{rec[h]['r']['mae']*100:.2f}", "carryover_share MAE(%p)": f"{rec[h]['carryover_share']['mae']*100:.2f}"}
            for h in horizons
        ]
        st.table(pd.DataFrame(rows).set_index("horizon"))

        st.markdown("**임계치(35%/40%/45%) 민감도 분석**")
        st.json(risk_sensitivity["summary"])
        st.caption("중위험 대표 계좌 예시: " + json.dumps(risk_sensitivity["medium_persona_example"], ensure_ascii=False))

        st.markdown("**전역 피처 중요도 (|SHAP| 평균)**")
        c1, c2 = st.columns(2)
        with c1:
            df_imp_S = pd.DataFrame(shap_importance["model_S"][:10]).set_index("feature")
            st.bar_chart(df_imp_S)
            st.caption("모델 f1 (S 예측)")
        with c2:
            df_imp_r = pd.DataFrame(shap_importance["model_r"][:10]).set_index("feature")
            st.bar_chart(df_imp_r)
            st.caption("모델 f2 (r 예측)")


# ---------------------------------------------------------------------------
# 앱 본체
# ---------------------------------------------------------------------------
_favicon = BASE_DIR / "assets" / "mascot" / "favicon_face.png"
st.set_page_config(
    page_title="오뚝이 | 리볼빙 조기경보",
    layout="wide",
    page_icon=str(_favicon) if _favicon.exists() else "🪆",
)
theme.inject_global_css()

account_master, monthly_transaction, derived_features, feature_table = load_data()
model_S, model_r, feature_cols, explainer_S, explainer_r = load_models_and_explainers()

# go_to()가 남겨둔 "대기 중인 페이지 이동"을, nav_page 라디오 위젯이 생성되기 전인
# 지금 반영한다 (반영 순서에 대한 설명은 go_to() 정의부 주석 참고).
if "_pending_nav" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_pending_nav")

anchor_row, selection_key, demo_label = render_sidebar(account_master, feature_table, derived_features)

if st.session_state.get("selection_key") != selection_key:
    st.session_state["selection_key"] = selection_key
    st.session_state.pop("simulation_result", None)
    st.session_state.pop("extra_payment_slider_widget", None)
    st.session_state.pop("extra_payment_number_widget", None)

bundle = build_prediction_bundle(
    anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols, explainer_S, explainer_r
)
outlook = multi_month_outlook(bundle, anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols, horizon=3)

# --- Hazard Model(보완) + 회복 게이지 : 계산은 src/hazard.py, src/recovery.py ---
_arow = anchor_row.iloc[0]
bundle["hazard"] = compute_hazard_bundle(_arow, bundle["current_risk"], bundle["month_index"])
bundle["recovery_score"] = rec.recovery_score(bundle["current_risk"], bundle["current_carryover_share"])
bundle["predicted_recovery_score"] = rec.recovery_score(bundle["predicted_risk"], bundle["predicted_carryover_share"])

st.markdown(
    theme.page_header("오뚝이", "리볼빙 조기경보 & AI 상환 코칭", right_html=theme.demo_mode_badge(demo_label)),
    unsafe_allow_html=True,
)

page = st.session_state.get("nav_page", "home")
if page == "home":
    render_home(bundle, outlook)
elif page == "risk":
    render_risk(bundle)
elif page == "coaching":
    render_coaching(bundle, anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols, outlook)
elif page == "simulator":
    render_simulator(bundle, anchor_row, monthly_transaction, derived_features, model_S, model_r, feature_cols)
elif page == "trust":
    render_trust()

st.markdown(theme.footer_badge(), unsafe_allow_html=True)
