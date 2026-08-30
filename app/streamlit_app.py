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
from coaching import _hydrate_env_from_st_secrets  # noqa: E402
from forecast_utils import find_first_escalation, find_minimum_intervention, multi_month_outlook, simulate_intervention_trajectory  # noqa: E402

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
# 사이드바: 네비게이션 + Demo Mode
# ---------------------------------------------------------------------------
def render_sidebar(account_master: pd.DataFrame, feature_table: pd.DataFrame, derived_features: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:2.1rem;font-weight:900;color:{theme.SIDEBAR_TEXT};padding:0.4rem 0 0.2rem 0;">'
            f'{theme.yoga_icon_svg("#ffffff", size=30)} 오뚝이</div>'
            f'<div style="font-size:0.88rem;color:{theme.SIDEBAR_TEXT_MUTED};font-weight:500;line-height:1.5;padding-bottom:1.1rem;">'
            f'당신의 리밸런싱도,<br>쓰러져도 스스로 중심을 되찾는 오뚝이처럼</div>',
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

    sub_metrics = theme.metric_row(
        [
            theme.metric_tile("리볼빙 의존도", fmt_pct(bundle["current_carryover_share"])),
            theme.metric_tile("최근 3개월 변화(의존도)", fmt_pct(bundle["current_delta_3m"], signed=True)),
            theme.metric_tile("결제여유", fmt_pct(bundle["current_gap"])),
            theme.metric_tile("연속 최소결제", f"{bundle['current_streak']}개월"),
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
    st.markdown(
        definitions_for("리볼빙 의존도", "최근 3개월 변화(의존도)", "결제여유", "연속 최소결제"),
        unsafe_allow_html=True,
    )

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

    st.markdown(theme.section_header("향후 위험 궤적").strip(), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f'<div style="color:{theme.SUBTLE};font-size:0.9rem;margin-bottom:0.4rem;">'
            "이 선은 지금 패턴이 그대로 유지될 경우 예상되는 리볼빙 의존도 변화예요. "
            "배경 색이 바뀌는 지점이 위험 단계가 전환되는 시점입니다.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.risk_trajectory_chart(outlook), width="stretch", config={"displayModeBar": False})

    st.markdown(theme.section_header("핵심 위험 신호", "다음 달 전망에 가장 크게 영향을 준 요인입니다.").strip(), unsafe_allow_html=True)
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
    top = theme.metric_row(
        [
            theme.metric_tile("현재 위험도", bundle["current_risk"]),
            theme.metric_tile("다음 달 예측 위험도", bundle["predicted_risk"]),
        ]
    )
    st.markdown(theme.card_open() + top + theme.card_close(), unsafe_allow_html=True)

    st.markdown(theme.section_header("위험도 구성", "네 가지 신호를 종합해 위험 단계를 판정합니다.").strip(), unsafe_allow_html=True)
    rising = "상승 중" if pd.notna(bundle["current_delta_3m"]) and bundle["current_delta_3m"] > 0 else "안정적"
    st.markdown(
        theme.metric_row(
            [
                theme.metric_tile("리볼빙 의존도", fmt_pct(bundle["current_carryover_share"])),
                theme.metric_tile("상승 추세", rising, note=fmt_pct(bundle["current_delta_3m"], signed=True)),
                theme.metric_tile("결제여유", fmt_pct(bundle["current_gap"]), note="약정결제비율 − 최소결제비율"),
                theme.metric_tile("최소결제 반복", f"{bundle['current_streak']}개월 연속" if bundle["current_streak"] > 0 else "없음"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        definitions_for("리볼빙 의존도", "상승 추세", "결제여유", "최소결제 반복"),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.section_header("예측에 영향을 준 주요 요인", "막대가 길수록 영향이 큽니다. 주황색은 위험을 높이는 방향, 청록색은 낮추는 방향이에요.").strip(),
        unsafe_allow_html=True,
    )

    def shap_section(shap_dict: dict, title: str, k: int = 5):
        with st.container(border=True):
            st.markdown(f"<b>{title}</b>", unsafe_allow_html=True)
            st.plotly_chart(charts.shap_bar_chart(shap_dict, FEATURE_LABELS, k=k), width="stretch", config={"displayModeBar": False})
            top_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            expl_rows = []
            for feat, _ in top_items:
                label = FEATURE_LABELS.get(feat, feat)
                expl = FEATURE_EXPLANATIONS.get(feat, "")
                expl_rows.append(f'<div style="margin-top:4px;font-size:0.85rem;"><b>{label}</b> — <span style="color:{theme.SUBTLE};">{expl}</span></div>')
            st.markdown("".join(expl_rows), unsafe_allow_html=True)

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
        theme.card_open()
        + '<div style="font-weight:800;font-size:1.05rem;">AI 상환 코칭</div>'
        + f'<div style="color:{theme.SUBTLE};margin-top:4px;">현재까지의 결제 패턴과 앞으로의 예측 결과를 모두 종합해서 알려드릴게요.</div>'
        + theme.card_close(),
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
        "simulation": simulation_ctx,
    }

    try:
        message = generate_coaching_message(coaching_context)
        segments = message["segments"]
    except Exception as e:
        st.error(f"코칭 메시지 생성/검증 실패: {e}")
        return

    # --- 세그먼트를 카드 여러 개로 잘게 쪼개지 않고, "상황과 이유"(raw_data+shap) 하나로
    #     묶어 한 흐름으로 읽히게 한다. (예전엔 segment 1개 = 카드 1개라 박스가 쭉 나열돼
    #     보였고, "상황"과 "원인"이 서로 분리된 별개 카드처럼 느껴진다는 피드백을 반영) ---
    story_segs = [s for s in segments if s["source"] in ("raw_data", "shap")]
    sim_segs = [s for s in segments if s["source"] == "simulation"]

    if story_segs:
        story_html = "".join(f'<p style="margin:0 0 0.85rem 0;">{theme.highlight_text(seg["text"])}</p>' for seg in story_segs)
        st.markdown(theme.big_section_title("📊 지금 상황과 이유", accent=theme.BRAND), unsafe_allow_html=True)
        st.markdown(theme.coaching_card(story_html, accent=theme.BRAND), unsafe_allow_html=True)

    # --- 최소 개입액(또는 사용자가 직접 돌려본 시뮬레이션 결과)을 "지금 할 수 있는 행동"
    #     하나의 카드 안에 숫자 + LLM 설명 문장을 함께 묶어서 보여준다. ---
    st.markdown(theme.big_section_title("🎯 지금 할 수 있는 행동", accent=theme.RISK_COLORS["관찰"]["main"]), unsafe_allow_html=True)
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
    </div>
    """)
    st.markdown(compare_html, unsafe_allow_html=True)

    if extra_payment > 0:
        st.markdown(
            theme.alert_card(
                "📌", "요약",
                f"월 {extra_payment:,.0f}원 추가 상환 시, 향후 위험 단계 상승을 완화할 수 있는 것으로 계산됩니다.",
                tone=new_risk,
            ),
            unsafe_allow_html=True,
        )

    st.markdown(theme.section_header("🎯 최소 개입액", "위험 단계를 '경고' 미만으로 유지하기 위한 최소 금액입니다.").strip(), unsafe_allow_html=True)
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
    preset_amounts = [0, 50_000, 100_000, 150_000]
    with st.spinner("금액별 시나리오를 계산하는 중..."):
        scenario_results = []
        for amt in preset_amounts:
            amt_capped = min(amt, L)
            preset_traj = simulate_intervention_trajectory(model_S, model_r, feature_cols, anchor_row, mt_idx, df_idx, amt_capped, horizon=3)
            final_step = preset_traj[-1] if preset_traj else {"level": bundle["predicted_risk"], "carryover_share": bundle["predicted_carryover_share"]}
            scenario_results.append({"extra_payment": amt_capped, "level": final_step["level"], "carryover_share": final_step["carryover_share"]})

    with st.container(border=True):
        st.plotly_chart(charts.intervention_comparison_chart(scenario_results), width="stretch", config={"displayModeBar": False})
        st.caption("막대 색은 3개월 뒤 예측되는 위험 단계를 나타냅니다.")

    baseline_level = scenario_results[0]["level"]
    cmp_cols = st.columns(len(preset_amounts))
    for col, sc in zip(cmp_cols, scenario_results):
        with col:
            if sc["extra_payment"] == 0:
                body = f'<div style="text-align:center;"><b>현재 그대로</b><br><br>{theme.risk_badge_html(sc["level"], "sm")}</div>'
                accent = theme.SUBTLE
            else:
                improved = theme.RISK_ORDER[sc["level"]] < theme.RISK_ORDER[baseline_level]
                verb = "낮아집니다" if improved else "유지됩니다"
                sentence = f'월 {sc["extra_payment"]:,.0f}원을 더 갚으면, 3개월 뒤 위험 단계가 {baseline_level}에서 {sc["level"]}로 {verb}.'
                body = (
                    f'<div style="text-align:center;">{theme.risk_badge_html(baseline_level, "sm")} → {theme.risk_badge_html(sc["level"], "sm")}</div>'
                    f'<div style="margin-top:8px;font-size:0.82rem;">{theme.highlight_text(sentence)}</div>'
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

    st.markdown(
        theme.card_open()
        + '<div style="font-weight:800;font-size:1.1rem;">모델 신뢰도</div>'
        + f'<div style="color:{theme.SUBTLE};margin-top:4px;">이 서비스의 예측 결과가 어떻게 검증되었는지 공개합니다.</div>'
        + theme.card_close(),
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
st.set_page_config(page_title="오뚝이 | 리볼빙 조기경보", layout="wide", page_icon="🪆")
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
