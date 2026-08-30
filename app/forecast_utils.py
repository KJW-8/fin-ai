"""오뚝이(Ottugi) 신규 UI 기능을 위한 오케스트레이션 레이어.

이 모듈은 새로운 계산식을 만들지 않는다. src/model.py의 recursive_forecast(),
simulate_extra_payment(), deterministic_recursion_step()와 src/risk.py의
classify_risk_indicator()를 "그대로" 반복 호출해 아래 두 가지를 조합할 뿐이다.

    1) multi_month_outlook(): 향후 N개월 자연 추세 궤적(조기경보 타임라인)
    2) find_minimum_intervention(): 목표 위험 단계 이하를 유지하기 위한
       최소 월 추가 상환액 탐색 (이분탐색으로 simulate_extra_payment()를 반복 호출)

기존 recursive_forecast()는 "추가 상환액"이라는 개념을 지원하지 않으므로(원래
설계에 없음), 2)는 recursive_forecast()의 재귀 루프와 동일한 부기(history 이월,
streak 갱신, delta_3m/slope_3m 재계산) 방식을 이 파일에서 별도로 얇게 구현해
매 스텝 deterministic_recursion_step() 대신 simulate_extra_payment()를 대입한다.
재귀식·계산 공식 자체는 model.py의 함수를 그대로 호출하는 것이므로 변경되지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PAYMENT_RATIO_GAP_WARN_CUTOFF, RISK_LEVEL_THRESHOLD_DEFAULT  # noqa: E402
from model import build_feature_row, recursive_forecast, simulate_extra_payment  # noqa: E402
from risk import classify_risk_indicator  # noqa: E402

RISK_ORDER = {"관찰": 0, "주의": 1, "경고": 2, "심화": 3}


# ---------------------------------------------------------------------------
# 1) 자연 추세 다개월 전망 (조기경보 타임라인) — recursive_forecast() 그대로 재사용
# ---------------------------------------------------------------------------
def multi_month_outlook(
    bundle: dict,
    anchor_row: pd.DataFrame,
    monthly_transaction: pd.DataFrame,
    derived_features: pd.DataFrame,
    model_S,
    model_r,
    feature_cols: list[str],
    horizon: int = 3,
) -> list[dict]:
    """[{month_offset, level, carryover_share}] — offset 0은 현재(bundle 값), 1..horizon은 예측."""
    outlook = [
        {
            "month_offset": 0,
            "level": bundle["current_risk"],
            "carryover_share": bundle["current_carryover_share"],
        }
    ]

    forecast = recursive_forecast(
        model_S, model_r, feature_cols, anchor_row, monthly_transaction, derived_features, horizon=horizon
    )
    for row in forecast.sort_values("horizon").itertuples(index=False):
        level = classify_risk_indicator(
            carryover_share=row.predicted_carryover_share,
            carryover_share_delta_3m=row.carryover_share_delta_3m,
            payment_ratio_gap=row.payment_ratio_gap,
            minimum_payment_streak=row.minimum_payment_streak,
            warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT,
        )
        outlook.append(
            {"month_offset": int(row.horizon), "level": level, "carryover_share": row.predicted_carryover_share}
        )
    return outlook


def find_first_escalation(outlook: list[dict], target_level: str = "경고") -> dict | None:
    """현재(offset 0)는 target_level 미만인데, 이후 어느 시점부터 target_level 이상이 되는지."""
    current = outlook[0]
    if RISK_ORDER[current["level"]] >= RISK_ORDER[target_level]:
        return None  # 이미 목표 단계 이상 — "조기경보(향후 전환)" 문구 대상이 아님
    for step in outlook[1:]:
        if RISK_ORDER[step["level"]] >= RISK_ORDER[target_level]:
            return step
    return None


# ---------------------------------------------------------------------------
# 2) 추가 상환액 개입 궤적 시뮬레이션 (recursive_forecast 부기 방식을 그대로 따르되
#    매 스텝 deterministic_recursion_step() 대신 simulate_extra_payment()를 사용)
# ---------------------------------------------------------------------------
def _seed_history(account_id: str, t0: int, mt_idx: pd.DataFrame, df_idx: pd.DataFrame) -> tuple[dict, dict]:
    cs_hist, r_hist = {}, {}
    for lag in range(0, 3):
        mm = t0 - lag
        if (account_id, mm) in df_idx.index:
            cs_hist[mm] = df_idx.loc[(account_id, mm), "carryover_share"]
        if (account_id, mm) in mt_idx.index:
            r_hist[mm] = mt_idx.loc[(account_id, mm), "committed_payment_ratio"]
    return cs_hist, r_hist


def simulate_intervention_trajectory(
    model_S,
    model_r,
    feature_cols: list[str],
    anchor_row: pd.DataFrame,
    mt_idx: pd.DataFrame,
    df_idx: pd.DataFrame,
    extra_payment: float,
    horizon: int,
) -> list[dict]:
    row = anchor_row.iloc[0]
    account_id = row["account_id"]
    t0 = int(row["feature_month_index"])
    m = float(row["minimum_payment_ratio"])
    L = float(row["card_limit"])
    i = float(row["interest_rate"])

    cs_hist, r_hist = _seed_history(account_id, t0, mt_idx, df_idx)
    B_prev = float(row["ending_carryover_principal"])
    revolving_streak = int(row["revolving_streak_months"])
    min_pay_streak = int(row["minimum_payment_streak"])
    delinquency_6m = float(row["delinquency_count_6m"])
    state = {c: row.get(c, np.nan) for c in feature_cols}

    trajectory = []
    for h in range(1, horizon + 1):
        t_next = t0 + h
        X = build_feature_row(state, feature_cols)
        S_pred = float(np.clip(model_S.predict(X)[0], 0.0, L))
        r_pred = float(np.clip(model_r.predict(X)[0], m, 1.0))

        # 핵심 재계산은 기존 simulate_extra_payment()를 그대로 호출한다 (로직 변경 없음)
        calc = simulate_extra_payment(B_prev=B_prev, S_pred=S_pred, r_pred=r_pred, m=m, i=i, extra_payment=extra_payment)

        cs_hist[t_next] = calc["carryover_share"]
        r_hist[t_next] = calc["r_effective"]

        gap = calc["payment_ratio_gap"]
        revolving_streak = revolving_streak + 1 if calc["B_t"] > 0 else 0
        min_pay_streak = min_pay_streak + 1 if gap <= PAYMENT_RATIO_GAP_WARN_CUTOFF else 0

        cs_delta_3m = cs_hist.get(t_next) - cs_hist.get(t_next - 3, np.nan)
        xs = [cs_hist.get(t_next - 2, np.nan), cs_hist.get(t_next - 1, np.nan), cs_hist.get(t_next, np.nan)]
        if all(pd.notna(v) for v in xs):
            cs_slope_3m = float(np.polyfit(np.array([0, 1, 2]), xs, 1)[0])
        else:
            cs_slope_3m = np.nan

        level = classify_risk_indicator(
            carryover_share=calc["carryover_share"],
            carryover_share_delta_3m=cs_delta_3m,
            payment_ratio_gap=gap,
            minimum_payment_streak=min_pay_streak,
            warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT,
        )
        trajectory.append(
            {"month_offset": h, "level": level, "carryover_share": calc["carryover_share"], "r_effective": calc["r_effective"]}
        )

        state = {
            "billing_amount": S_pred,
            "committed_payment_ratio": calc["r_effective"],
            "revolving_principal_before_payment": calc["P_t"],
            "scheduled_principal_payment": calc["A_t"],
            "revolving_fee": calc["I_t"],
            "ending_carryover_principal": calc["B_t"],
            "total_payment_amount": calc["total_payment_amount"],
            "minimum_principal_required": calc["minimum_principal_required"],
            "actual_principal_paid": calc["A_t"],
            "revolving_active": 1 if (B_prev > 0 or calc["r_effective"] < 1.0) else 0,
            "month_index": t_next,
            "payment_status_정상": 1,
            "payment_status_최소결제": 0,
            "payment_status_연체": 0,
            "carryover_share": calc["carryover_share"],
            "carryover_share_delta_3m": cs_delta_3m,
            "carryover_share_slope_3m": cs_slope_3m,
            "committed_ratio_delta_3m": r_hist.get(t_next) - r_hist.get(t_next - 3, np.nan),
            "payment_ratio_gap": gap,
            "revolving_streak_months": revolving_streak,
            "minimum_payment_streak": min_pay_streak,
            "delinquency_count_6m": delinquency_6m,
            "limit_utilization_ratio": float(np.clip(S_pred / L, 0.0, 1.0)) if L > 0 else 0.0,
            "minimum_payment_ratio": m,
            "card_limit": L,
            "interest_rate": i,
        }
        B_prev = calc["B_t"]

    return trajectory


def find_minimum_intervention(
    model_S,
    model_r,
    feature_cols: list[str],
    anchor_row: pd.DataFrame,
    monthly_transaction: pd.DataFrame,
    derived_features: pd.DataFrame,
    horizon: int = 3,
    target_max_level: str = "경고",
    tolerance: float = 1_000.0,
) -> dict:
    """향후 horizon개월 동안 위험도가 target_max_level "미만"으로 유지되는 최소 월 추가
    상환액을 이분탐색으로 찾는다. 카드 한도까지 넣어도 목표를 못 맞추면 achieved=False.
    """
    row = anchor_row.iloc[0]
    L = float(row["card_limit"])
    mt_idx = monthly_transaction.set_index(["account_id", "month_index"])
    df_idx = derived_features.set_index(["account_id", "month_index"])

    def max_level_reached(extra_payment: float) -> tuple[str, list[dict]]:
        traj = simulate_intervention_trajectory(model_S, model_r, feature_cols, anchor_row, mt_idx, df_idx, extra_payment, horizon)
        worst = max(traj, key=lambda s: RISK_ORDER[s["level"]])
        return worst["level"], traj

    # 한도 전액을 넣어도 목표를 못 맞추면 탐색 불가
    worst_at_max, traj_at_max = max_level_reached(L)
    if RISK_ORDER[worst_at_max] >= RISK_ORDER[target_max_level]:
        return {"achieved": False, "extra_payment": None, "trajectory": traj_at_max}

    lo, hi = 0.0, L
    best_traj = traj_at_max
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        worst, traj = max_level_reached(mid)
        if RISK_ORDER[worst] < RISK_ORDER[target_max_level]:
            hi = mid
            best_traj = traj
        else:
            lo = mid

    return {"achieved": True, "extra_payment": round(hi / 1000) * 1000, "trajectory": best_traj}
