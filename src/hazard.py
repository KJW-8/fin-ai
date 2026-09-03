"""오뚝이(Ottugi) — 이산시간 위험모형 (Discrete-time Hazard Model).

기존 XGBoost 파이프라인(S(t+1)/r(t+1) 예측 -> 재귀식 -> predicted_carryover_share ->
risk_indicator)을 **대체하지 않고 보완**하는 별도 모델이다.

  [XGBoost]      "앞으로 리볼빙 의존도가 어떻게 움직일까?"  (다음 달 값, 궤적)
  [Hazard Model] "현재 상태에서 '경고/심화' 단계로 언제 넘어갈 가능성이 높아지는가?"
                 (hazard, 생존확률, 향후 3개월 전환확률, 예상 전환 시점)

모델 형태: Logistic Regression 기반 discrete-time hazard.
  h(t | X, d) = P(이번 기간 t에 처음으로 '경고/심화'로 전환 | 직전 기간까지 전환 안 됨)
  = sigmoid( b0 + b·X_{t-1} + g(d=elapsed_month) )

핵심 설계 (leakage 방지):
  - person-period 행의 feature는 **직전 달(t-1)** 관측값을 쓴다. risk_indicator_t 는
    carryover_share_t 등으로부터 규칙(risk.classify_risk_indicator)으로 결정론적으로
    도출되므로, 같은 달 feature로 같은 달 전환을 예측하면 규칙을 그대로 되학습하는
    순환(circularity)이 된다. 한 달 시차를 두어 "지금(t-1) 상태에서 다음 달(t) 전환
    가능성"을 학습한다.
  - 이벤트 발생 계좌는 첫 전환 시점 이후 위험집합(risk set)에서 제외한다(single-event).
  - 12개월 내 전환하지 않은 계좌는 right censoring 으로 그대로 둔다(제거 금지).

LLM은 이 파일의 어떤 수치도 계산하지 않는다. 여기서 나온 값(전환확률/생존확률/예상
시점/recovery_score)을 문장으로 "설명"만 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

HIGH_RISK_LEVELS = ("경고", "심화")

# XGBoost(model.py TRAIN_TARGET_MAX/TEST_TARGET_MONTHS)와 동일한 시간 기준 분할점.
# period(=elapsed_month) t <= TRAIN_PERIOD_MAX -> train, 그 이후 -> test.
TRAIN_PERIOD_MAX = 9
TEST_PERIODS = (10, 11, 12)
N_MONTHS = 12

# person-period feature: 전부 기존 파생변수/거래 컬럼 재사용 (직전 달 t-1 관측값).
# revolving_payment_to_income_ratio 는 소득 미확보로 전량 결측이라 제외(model.py와 동일).
HAZARD_FEATURES = [
    "carryover_share",              # 리볼빙 의존도 수준
    "carryover_share_delta_3m",     # 최근 3개월 의존도 변화
    "carryover_share_slope_3m",     # 최근 3개월 악화 속도
    "committed_ratio_delta_3m",     # 최근 3개월 약정결제비율 변화
    "payment_ratio_gap",           # 결제여유(약정-최소)
    "revolving_streak_months",      # 연속 리볼빙 개월수
    "minimum_payment_streak",       # 연속 최소결제(gap<=cutoff) 개월수
    "delinquency_count_6m",         # 최근 6개월 연체 횟수
    "limit_utilization_ratio",      # 한도 대비 사용률
    "billing_amount",              # 최근(직전 달) 청구액
    "committed_payment_ratio",      # 최근(직전 달) 약정결제비율
]
# 3개월 델타/기울기는 초기 몇 달 결측(정의상 t-3 필요). "추세 정보 없음 = 0(변화 없음)"
# 으로 대치한다. 이것은 leakage가 아니라 관측 시작부 결측 처리 규칙임을 명시한다.
_FILL_ZERO = ("carryover_share_delta_3m", "carryover_share_slope_3m", "committed_ratio_delta_3m")

# duration(경과 개월수) 항. 이산시간 hazard에서 baseline hazard 모양을 학습하는 부분.
DURATION_COL = "elapsed_month"


# ===========================================================================
# 1. person-period(long) 데이터 변환
# ===========================================================================
@dataclass
class PersonPeriodResult:
    df: pd.DataFrame                       # person-period long 테이블
    n_accounts_total: int
    n_already_high_risk: int               # 관측 시작부터 이미 경고/심화
    already_high_risk_ids: list[str]
    n_accounts_used: int                   # person-period 변환에 실제 사용된 계좌 수
    n_events: int                          # event_occurred == 1 행 수 (= 첫 전환 계좌 수)
    n_censored: int                        # 12개월 내 전환하지 않은 계좌 수
    first_event_month_counts: dict         # 첫 전환 월 분포
    notes: list[str] = field(default_factory=list)


def build_person_period(
    derived_features: pd.DataFrame,
    monthly_transaction: pd.DataFrame | None = None,
) -> PersonPeriodResult:
    """계좌 x 월 패널 -> person-period(long). 각 행: (account_id, period=t,
    elapsed_month=t, event_occurred, feature...(t-1 관측값)).

    period t 의 feature 는 t-1 달 관측값을 쓴다(위 모듈 docstring의 leakage 설계).
    모든 계좌는 month 1 에 '관찰'로 시작하므로 첫 전환 가능 시점은 period 2 이다.
    """
    d = derived_features.sort_values(["account_id", "month_index"]).copy()

    feat_source = d[["account_id", "month_index"] + [c for c in HAZARD_FEATURES if c in d.columns]].copy()
    if monthly_transaction is not None:
        mt_cols = [c for c in ("billing_amount", "committed_payment_ratio") if c in monthly_transaction.columns]
        if mt_cols:
            feat_source = feat_source.merge(
                monthly_transaction[["account_id", "month_index"] + mt_cols],
                on=["account_id", "month_index"], how="left", suffixes=("", "_mt"),
            )
            for c in mt_cols:
                if f"{c}_mt" in feat_source.columns:
                    feat_source[c] = feat_source[c].fillna(feat_source[f"{c}_mt"]) if c in feat_source.columns else feat_source[f"{c}_mt"]
                    feat_source = feat_source.drop(columns=[f"{c}_mt"])

    feat_by_key = feat_source.set_index(["account_id", "month_index"])

    rows = []
    already_high_ids: list[str] = []
    first_event_months: list[int] = []
    n_censored = 0

    for account_id, g in d.groupby("account_id", sort=False):
        g = g.sort_values("month_index")
        levels = dict(zip(g["month_index"], g["risk_indicator"]))
        months = sorted(levels)
        first_month = months[0]

        if levels.get(first_month) in HIGH_RISK_LEVELS:
            # 관측 시작부터 이미 경고/심화 -> '조기경보 대상'이 아닌 '즉시 위험군'.
            already_high_ids.append(account_id)
            continue

        high_months = [mm for mm in months if levels[mm] in HIGH_RISK_LEVELS]
        first_event_month = high_months[0] if high_months else None
        last_period = first_event_month if first_event_month is not None else months[-1]

        if first_event_month is not None:
            first_event_months.append(first_event_month)
        else:
            n_censored += 1

        for t in range(first_month + 1, last_period + 1):
            feat_month = t - 1  # 직전 달 관측값
            if (account_id, feat_month) not in feat_by_key.index:
                continue
            fr = feat_by_key.loc[(account_id, feat_month)]
            row = {
                "account_id": account_id,
                "period": int(t),
                DURATION_COL: int(t),
                "feature_month_index": int(feat_month),
                "event_occurred": int(first_event_month is not None and t == first_event_month),
            }
            for c in HAZARD_FEATURES:
                row[c] = float(fr[c]) if c in fr and pd.notna(fr[c]) else np.nan
            rows.append(row)

    pp = pd.DataFrame(rows)
    for c in _FILL_ZERO:
        if c in pp.columns:
            pp[c] = pp[c].fillna(0.0)
    # 그 외 잔여 결측은 컬럼 중앙값으로 대치(관측 시작부 예외 방어).
    for c in HAZARD_FEATURES:
        if c in pp.columns and pp[c].isna().any():
            pp[c] = pp[c].fillna(pp[c].median())

    # duration 파생: 이산시간 hazard에서 baseline hazard 모양 후보들.
    # - quadratic: elapsed_month + elapsed_month^2 (부드러운 험프)
    # - dummies: period one-hot (t=3..12). t=2 는 이벤트 0건이라 기준 범주.
    #   raw hazard 가 t=3 에 크게 스파이크 후 감소하는 형태라 dummies 가 baseline hazard 를
    #   정확히 학습한다. 서비스 예측 범위(month 2~12)를 전부 커버하므로 외삽 문제 없음.
    pp["elapsed_month_sq"] = pp[DURATION_COL] ** 2
    for t in range(3, N_MONTHS + 1):
        pp[f"dur_{t}"] = (pp[DURATION_COL] == t).astype(float)

    result = PersonPeriodResult(
        df=pp,
        n_accounts_total=int(d["account_id"].nunique()),
        n_already_high_risk=len(already_high_ids),
        already_high_risk_ids=already_high_ids,
        n_accounts_used=int(pp["account_id"].nunique()),
        n_events=int(pp["event_occurred"].sum()),
        n_censored=n_censored,
        first_event_month_counts={int(k): int(v) for k, v in pd.Series(first_event_months).value_counts().sort_index().items()},
        notes=[
            "person-period 행의 feature 는 직전 달(t-1) 관측값 (leakage 방지 — risk_indicator_t 는 "
            "carryover_share_t 등으로부터 규칙 도출되므로 같은 달 feature 사용 시 순환).",
            "모든 계좌가 month 1 에 '관찰'로 시작 -> 첫 전환 가능 period 는 2.",
            "이벤트 발생 계좌는 첫 전환 시점 이후 위험집합에서 제외(single-event).",
            "3개월 델타/기울기 초기 결측은 0(변화 정보 없음)으로 대치.",
        ],
    )
    return result


# ===========================================================================
# 2. 분할 + 학습
# ===========================================================================
def temporal_split(pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """XGBoost와 동일한 시간 기준 분할: period <= TRAIN_PERIOD_MAX -> train, 그 이후 -> test.

    주의: 패널 자료라 한 계좌의 person-period 행이 train/test 로 갈릴 수 있다(초반 period 는
    train, 후반 period 는 test). 시간적 leakage 는 (a) 직전 달 feature (b) single-event 제외
    로 이미 차단된다. 계좌 단위로 완전 분리한 평가는 account_cv_metrics() 로 별도 보고한다.
    """
    train = pp[pp["period"] <= TRAIN_PERIOD_MAX].copy()
    test = pp[pp["period"].isin(TEST_PERIODS)].copy()
    return train, test


def _duration_terms(spec: str) -> list[str]:
    return {
        "linear": [DURATION_COL],
        "quadratic": [DURATION_COL, "elapsed_month_sq"],
        "dummies": [f"dur_{t}" for t in range(3, N_MONTHS + 1)],
        "none": [],
    }[spec]


def make_model(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, C=C, class_weight=None, random_state=42)),
    ])


def fit_hazard_model(
    train: pd.DataFrame, duration_spec: str = "quadratic", C: float = 1.0
) -> tuple[Pipeline, list[str]]:
    feats = HAZARD_FEATURES + _duration_terms(duration_spec)
    model = make_model(C=C)
    model.fit(train[feats].to_numpy(dtype=float), train["event_occurred"].to_numpy(dtype=int))
    return model, feats


def predict_hazard_rows(model: Pipeline, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[feats].to_numpy(dtype=float))[:, 1]


# ===========================================================================
# 3. 생존확률 / 전환확률 / 예상 전환 시점  (서비스 추론용)
# ===========================================================================
def _feature_vector(state: dict, feats: list[str], elapsed_month: int) -> np.ndarray:
    row = []
    for c in feats:
        if c == DURATION_COL:
            row.append(float(elapsed_month))
        elif c == "elapsed_month_sq":
            row.append(float(elapsed_month) ** 2)
        elif c.startswith("dur_"):
            row.append(1.0 if int(c.split("_")[1]) == int(elapsed_month) else 0.0)
        else:
            v = state.get(c, np.nan)
            row.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
    return np.array([row], dtype=float)


def survival_trajectory(
    model: Pipeline, feats: list[str], state: dict, current_month: int, horizon: int = 6
) -> list[dict]:
    """현재 상태(state, 직전 달 관측값과 동일 형식)를 '그대로 유지한다'고 가정하고
    current_month 이후 horizon 개월의 hazard/생존확률 궤적을 계산한다.

    feature 는 고정하고 elapsed_month(duration)만 전진시킨다 -> "현재 패턴이 유지되면"
    이라는 XGBoost recursive_forecast 와 동일한 가정.
    S(t) = Π_{j=1..t} (1 - h_j)

    학습은 t<=12 person-period 로만 이뤄졌다. current_month 가 이미 12 인 사용자(관측
    마지막 달)에 대해서도 몇 달을 내다볼 수 있도록, t>12 는 duration 항을 12 로 고정해
    (dummies 는 전부 0 = 기준 baseline) 외삽한다 -> 'extrapolated' 플래그로 표시.
    """
    out = []
    surv = 1.0
    for k, m in enumerate(range(current_month + 1, current_month + horizon + 1), start=1):
        m_eff = min(m, N_MONTHS)  # t>12 는 baseline duration 으로 고정 외삽
        h = float(model.predict_proba(_feature_vector(state, feats, m_eff))[0, 1])
        surv *= (1.0 - h)
        out.append({"month_offset": k, "elapsed_month": m, "hazard": h,
                    "survival": surv, "extrapolated": m > N_MONTHS})
    return out


def transition_probability_3m(model: Pipeline, feats: list[str], state: dict, current_month: int) -> float | None:
    """향후 3개월 내 '경고/심화' 단계로 전환될 확률 = 1 - S(current+3).
    관측기간(12개월)이 3개월 미만 남았으면 남은 개월만 사용하고 None 대신 부분값을 준다.
    """
    traj = survival_trajectory(model, feats, state, current_month, horizon=3)
    if not traj:
        return None
    return float(1.0 - traj[-1]["survival"])


def median_time_to_warning(
    model: Pipeline, feats: list[str], state: dict, current_month: int, max_horizon: int = 6
) -> int | None:
    """S(t)가 처음으로 0.5 미만이 되는 시점(개월 후). 그 안에 없으면 None."""
    for step in survival_trajectory(model, feats, state, current_month, horizon=max_horizon):
        if step["survival"] < 0.5:
            return int(step["month_offset"])
    return None


# ===========================================================================
# 4. 검증
# ===========================================================================
def harrell_c_index(risk_score: np.ndarray, event_time: np.ndarray, event_observed: np.ndarray) -> tuple[float, int]:
    """계좌 단위 Harrell's C-index. risk_score 가 클수록 위험(=빨리 전환)이라고 가정.
    반환: (c_index, 비교 가능 쌍 수). 쌍이 0이면 (nan, 0).
    """
    risk_score = np.asarray(risk_score, float)
    event_time = np.asarray(event_time, float)
    event_observed = np.asarray(event_observed, int)
    n = len(risk_score)
    conc = disc = tie = 0
    for i in range(n):
        if event_observed[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            # j 가 i 보다 나중까지 생존(또는 더 긴 관측) -> 비교 가능 쌍
            if event_time[j] > event_time[i] or (event_time[j] == event_time[i] and event_observed[j] == 0):
                if risk_score[i] > risk_score[j]:
                    conc += 1
                elif risk_score[i] < risk_score[j]:
                    disc += 1
                else:
                    tie += 1
    total = conc + disc + tie
    if total == 0:
        return float("nan"), 0
    return (conc + 0.5 * tie) / total, total


def kaplan_meier(event_time: np.ndarray, event_observed: np.ndarray, max_t: int = N_MONTHS) -> dict[int, float]:
    """계좌 단위 KM 생존함수 추정 (개인화 없음, baseline)."""
    event_time = np.asarray(event_time, float)
    event_observed = np.asarray(event_observed, int)
    surv = 1.0
    curve = {}
    for t in range(1, max_t + 1):
        at_risk = int((event_time >= t).sum())
        d = int(((event_time == t) & (event_observed == 1)).sum())
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
        curve[t] = surv
    return curve


def calibration_bins(pred_prob: np.ndarray, actual: np.ndarray, n_bins: int = 5) -> list[dict]:
    """예측 전환확률 vs 실제 전환율 (동일 빈도 구간)."""
    pred_prob = np.asarray(pred_prob, float)
    actual = np.asarray(actual, int)
    if len(pred_prob) < n_bins:
        n_bins = max(2, len(pred_prob) // 5 or 2)
    order = np.argsort(pred_prob)
    bins = np.array_split(order, n_bins)
    out = []
    for b in bins:
        if len(b) == 0:
            continue
        out.append({
            "n": int(len(b)),
            "mean_predicted": float(pred_prob[b].mean()),
            "observed_rate": float(actual[b].mean()),
        })
    return out


def brier_score(pred_prob: np.ndarray, actual: np.ndarray) -> float:
    pred_prob = np.asarray(pred_prob, float)
    actual = np.asarray(actual, int)
    return float(np.mean((pred_prob - actual) ** 2))


# ===========================================================================
# 4-b. 서비스 추론 번들 (앱에서 호출하는 단일 진입점)
# ===========================================================================
def build_hazard_bundle(
    bundle_or_state: dict,
    current_month: int,
    model: Pipeline | None = None,
    feats: list[str] | None = None,
    horizon: int = 6,
) -> dict:
    """앱이 쓰는 hazard 산출물 한 번에.

    bundle_or_state: HAZARD_FEATURES 키를 가진 dict (streamlit 앱의 anchor row 또는
      build_prediction_bundle 결과에서 필요한 값만 추려서 넘긴다).
    반환:
      {
        "applicable": bool,          # 이미 경고/심화면 False (이벤트 이미 발생)
        "current_month": int,
        "transition_probability_3m": float | None,
        "median_time_to_warning": int | None,
        "trajectory": [{month_offset, elapsed_month, hazard, survival}, ...],
        "duration_spec": str,
      }
    """
    dur_spec = "quadratic"
    if model is None or feats is None:
        b = load_bundle()
        model, feats = b["model"], b["feats"]
        dur_spec = b.get("meta", {}).get("duration_spec", dur_spec)

    state = {c: bundle_or_state.get(c) for c in HAZARD_FEATURES}
    # 이미 경고/심화 단계면 "언제 전환되는가" 질문 자체가 성립하지 않음.
    current_level = bundle_or_state.get("current_risk") or bundle_or_state.get("risk_indicator")
    applicable = current_level not in HIGH_RISK_LEVELS

    traj = survival_trajectory(model, feats, state, current_month, horizon=horizon)
    if applicable:
        tp3 = transition_probability_3m(model, feats, state, current_month)
        mtw = median_time_to_warning(model, feats, state, current_month, max_horizon=horizon)
    else:
        tp3, mtw = None, None

    return {
        "applicable": applicable,
        "current_month": int(current_month),
        "transition_probability_3m": tp3,
        "median_time_to_warning": mtw,
        "trajectory": traj,
        "duration_spec": dur_spec,
    }


# ===========================================================================
# 5. 저장/로드
# ===========================================================================
def save_bundle(model: Pipeline, feats: list[str], meta: dict, path: Path | None = None) -> Path:
    import joblib
    path = path or (MODELS_DIR / "hazard_model.joblib")
    joblib.dump({"model": model, "feats": feats, "meta": meta}, path)
    return path


def load_bundle(path: Path | None = None) -> dict:
    import joblib
    path = path or (MODELS_DIR / "hazard_model.joblib")
    return joblib.load(path)
