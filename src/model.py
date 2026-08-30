"""오뚝이(Ottugi) 3단계 — 예측 모델 (4장 아키텍처).

핵심 원칙 (문서 4-1): carryover_share를 직접 회귀 예측하지 않는다.
    f1(과거 피처) -> S_(t+1) 예측
    f2(과거 피처) -> r_(t+1) 예측
    위 두 예측치를 1단계(simulator.py)와 동일한 재귀식에 대입해
    P, A, I, B, predicted_carryover_share 를 "계산"한다 (ML 아님, 회계 항등식).

Train/Test는 무작위 분할이 아니라 시간 기준 분할(계좌별 초반 N개월 -> train,
이후 M개월 -> test)을 사용해 데이터 누출을 방지한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import DAYS_APPROX, DAYS_IN_YEAR, MIN_PRINCIPAL_FLOOR, PAYMENT_RATIO_GAP_WARN_CUTOFF

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# 시간 기준 분할 지점: target_month_index(=예측 대상 월) 기준
#   train: target_month_index in [2, TRAIN_TARGET_MAX]
#   test : target_month_index in TEST_TARGET_MONTHS (모두 TRAIN_TARGET_MAX 이후 시점)
TRAIN_TARGET_MAX = 9
TEST_TARGET_MONTHS = (10, 11, 12)

# 모델 입력에서 제외: 성별/고용형태(문서 3-1 근거로 핵심 피처 미사용),
# persona_tier(생성 메타데이터, 실서비스에서는 존재하지 않는 값),
# revolving_payment_to_income_ratio(소득 미확보로 전량 결측, 문서 3-4 근거로 제외)
EXCLUDED_FEATURES = {
    "revolving_payment_to_income_ratio",
    "risk_indicator",  # 5단계 규칙 기반 후처리 결과(범주형 문자열). ML 입력 피처가 아니라
    # carryover_share 등 다른 파생변수로부터 규칙으로 도출되는 하류(downstream) 산출물이므로 제외.
}


# ---------------------------------------------------------------------------
# 1. 피처 테이블 / 패널(학습용 테이블) 구성
# ---------------------------------------------------------------------------
def build_feature_table(
    monthly_transaction: pd.DataFrame, derived_features: pd.DataFrame, account_master: pd.DataFrame
) -> pd.DataFrame:
    """계좌 x month_index(1..12) 전체에 대한 피처 테이블. target 유무와 무관하게 모든 월을
    포함한다 (12월차처럼 다음 달 target이 없는 "가장 최근 관측" 행도 포함 — SHAP 설명 등
    현재 시점 추론에 필요). 학습용 패널은 build_model_panel()에서 여기에 target을 붙여 만든다.
    """
    mt = monthly_transaction.copy()
    mt = pd.get_dummies(mt, columns=["payment_status"], prefix="payment_status")
    mt["revolving_active"] = mt["revolving_active"].astype(int)

    payment_status_cols = [c for c in mt.columns if c.startswith("payment_status_")]
    mt_feature_cols = [
        "billing_amount",
        "committed_payment_ratio",
        "revolving_principal_before_payment",
        "scheduled_principal_payment",
        "revolving_fee",
        "ending_carryover_principal",
        "total_payment_amount",
        "minimum_principal_required",
        "actual_principal_paid",
        "revolving_active",
    ] + payment_status_cols

    df = mt[["account_id", "month_index"] + mt_feature_cols].copy()

    derived_cols = [c for c in derived_features.columns if c not in ("customer_id", "year_month")]
    derived_cols = [c for c in derived_cols if c not in EXCLUDED_FEATURES]
    df = df.merge(derived_features[derived_cols], on=["account_id", "month_index"], how="left")

    df = df.merge(account_master[["account_id", "minimum_payment_ratio", "card_limit"]], on="account_id", how="left")

    df = df.sort_values(["account_id", "month_index"]).reset_index(drop=True)
    # feature_month_index: 분할/조인 전용 식별자(북키핑). month_index 자체는 계좌 내
    # 상대적 가입 경과월을 나타내는 실제 모델 피처(계절성/추세 학습용)로 그대로 남긴다.
    df["feature_month_index"] = df["month_index"]
    return df


def build_model_panel(
    monthly_transaction: pd.DataFrame, derived_features: pd.DataFrame, account_master: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    df = build_feature_table(monthly_transaction, derived_features, account_master)

    # target: 다음 달 S, r (같은 계좌, month_index+1 행에서 가져옴)
    target = monthly_transaction[["account_id", "month_index", "billing_amount", "committed_payment_ratio"]].copy()
    target["feature_month_index"] = target["month_index"] - 1
    target = target.rename(columns={"billing_amount": "target_S", "committed_payment_ratio": "target_r"})
    target = target[["account_id", "feature_month_index", "target_S", "target_r"]]

    # inner join: month_index=12는 target(month 13)이 없으므로 자동 제외됨
    panel = df.merge(target, on=["account_id", "feature_month_index"], how="inner")
    panel["target_month_index"] = panel["feature_month_index"] + 1

    feature_cols = [c for c in df.columns if c not in ("account_id", "feature_month_index")]
    return panel, feature_cols


def time_based_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간 기준 분할: 초반 target_month_index -> train, 이후 -> test (무작위 분할 아님)."""
    train = panel[panel["target_month_index"] <= TRAIN_TARGET_MAX].copy()
    test = panel[panel["target_month_index"].isin(TEST_TARGET_MONTHS)].copy()
    return train, test


# ---------------------------------------------------------------------------
# 2. 모델 학습
# ---------------------------------------------------------------------------
def _xgb_params() -> dict:
    return dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )


def train_models(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[XGBRegressor, XGBRegressor]:
    X = train_df[feature_cols]
    model_S = XGBRegressor(**_xgb_params())
    model_S.fit(X, train_df["target_S"])

    model_r = XGBRegressor(**_xgb_params())
    model_r.fit(X, train_df["target_r"])

    return model_S, model_r


# ---------------------------------------------------------------------------
# 3. 1개월 예측 성능 평가
# ---------------------------------------------------------------------------
def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    return {"mae": mae, "rmse": rmse}


def evaluate_one_step(
    model_S: XGBRegressor, model_r: XGBRegressor, test_df: pd.DataFrame, feature_cols: list[str], derived_features: pd.DataFrame
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    X = test_df[feature_cols]
    # recursive_forecast()와 동일하게 S 예측치를 [0, card_limit]로, r 예측치를 [m, 1.0]로
    # 클리핑한다. 두 평가 함수가 서로 다른 후처리를 쓰면 방법론이 갈라지므로 통일해 둔다
    # (이번 데이터에서는 clip 대상 행이 0건이라 수치 영향은 없었지만, 데이터가 바뀌면
    # 달라질 수 있어 로직 자체를 맞춰 둔다).
    pred_S = np.clip(model_S.predict(X), 0.0, test_df["card_limit"].values)
    pred_r = np.clip(model_r.predict(X), test_df["minimum_payment_ratio"].values, 1.0)

    metrics = {
        "S_next_month": _mae_rmse(test_df["target_S"].values, pred_S),
        "r_next_month": _mae_rmse(test_df["target_r"].values, pred_r),
        "n_test_rows": int(len(test_df)),
    }

    # 결정론적 계산 레이어(ML 아님)로 predicted_carryover_share(t+1)까지 계산해
    # 실제 carryover_share(t+1)과 비교 (참고 지표 — S/r 오차가 재귀식을 거쳐
    # 서비스 최종 출력 predicted_carryover_share에 어떻게 전파되는지 확인하는 용도)
    B_prev = test_df["ending_carryover_principal"].values.astype(float)
    P = B_prev + pred_S
    A = P * pred_r
    B_next = P - A
    denom = B_next + pred_S
    pred_carryover_share = np.where(denom > 0, B_next / np.where(denom > 0, denom, 1.0), 0.0)

    actual_cs = derived_features.rename(
        columns={"month_index": "target_month_index", "carryover_share": "actual_carryover_share_next"}
    )[["account_id", "target_month_index", "actual_carryover_share_next"]]
    merged = test_df[["account_id", "target_month_index"]].merge(actual_cs, on=["account_id", "target_month_index"], how="left")
    metrics["predicted_carryover_share_next_month"] = _mae_rmse(merged["actual_carryover_share_next"].values, pred_carryover_share)

    return metrics, pred_S, pred_r, pred_carryover_share


# ---------------------------------------------------------------------------
# 4. 결정론적 재귀 계산 레이어 (simulator.py 3장 공식과 동일)
# ---------------------------------------------------------------------------
def deterministic_recursion_step(B_prev: float, S_t: float, r_t: float, m: float, i: float) -> dict:
    P_t = B_prev + S_t
    A_t = P_t * r_t
    I_t = B_prev * i * (DAYS_APPROX / DAYS_IN_YEAR)
    B_t = P_t - A_t
    total_payment_amount = A_t + I_t
    minimum_principal_required = max(P_t * m, MIN_PRINCIPAL_FLOOR)
    denom = B_t + S_t
    carryover_share = (B_t / denom) if denom > 0 else 0.0
    return dict(
        P_t=P_t,
        A_t=A_t,
        I_t=I_t,
        B_t=B_t,
        total_payment_amount=total_payment_amount,
        minimum_principal_required=minimum_principal_required,
        carryover_share=carryover_share,
    )


def simulate_extra_payment(B_prev: float, S_pred: float, r_pred: float, m: float, i: float, extra_payment: float) -> dict:
    """상환 시뮬레이터: 추가 상환액을 입력하면 r_(t+1)을 대체해 재계산한다 (4-1장 설계 원칙).

    모델이 예측한 다음 달 소비(S_pred)는 그대로 두고, 사용자가 추가로 상환하겠다고 입력한
    금액만큼 다음 달 약정결제비율(r)을 끌어올린 시나리오로 재귀식을 다시 계산한다.
    A_t_new = P_t*r_pred + extra_payment = P_t*r_new  ->  r_new = r_pred + extra_payment/P_t
    (r_new은 1.0을 넘을 수 없다 — 원금을 초과 상환할 수는 없으므로 클리핑)
    """
    P_t = B_prev + S_pred
    r_effective = r_pred + (extra_payment / P_t if P_t > 0 else 0.0)
    r_effective = float(np.clip(r_effective, m, 1.0))
    calc = deterministic_recursion_step(B_prev, S_pred, r_effective, m, i)
    calc["r_effective"] = r_effective
    calc["payment_ratio_gap"] = r_effective - m
    return calc


# ---------------------------------------------------------------------------
# 5. 다개월 재귀 예측 (Recursive Multi-step Forecasting) + 누적오차 평가
# ---------------------------------------------------------------------------
def build_feature_row(state: dict, feature_cols: list[str]) -> pd.DataFrame:
    row = {c: state.get(c, np.nan) for c in feature_cols}
    return pd.DataFrame([row])[feature_cols]


def recursive_forecast(
    model_S: XGBRegressor,
    model_r: XGBRegressor,
    feature_cols: list[str],
    anchor_df: pd.DataFrame,
    monthly_transaction: pd.DataFrame,
    derived_features: pd.DataFrame,
    horizon: int = 3,
) -> pd.DataFrame:
    """anchor_df: feature_month_index == TRAIN_TARGET_MAX+? 시점(관측 마지막 real 데이터)의
    실제 피처 행들 (계좌별 1행). 이 시점부터 horizon개월(예: 10,11,12월)을 재귀적으로 예측한다.

    각 스텝에서 m, i는 계좌 고정값을 그대로 사용하고, S와 r만 f1/f2로 재예측한다.
    파생변수(carryover_share_delta_3m, slope_3m, committed_ratio_delta_3m, payment_ratio_gap,
    revolving_streak_months, minimum_payment_streak, limit_utilization_ratio)는 예측된 S/r과
    고정된 m을 3장/3-4장과 동일한 산식으로 재계산한다.

    delinquency_count_6m만 예외: 미래의 연체 이벤트 발생 여부는 f1/f2(S,r 예측 모델)의
    예측 범위 밖이므로, 마지막 관측된 실제값을 그대로 이월(last-observation-carried-forward)
    한다. 이는 이 재귀 예측 레이어의 알려진 한계로 별도 명시한다.
    """
    mt_idx = monthly_transaction.set_index(["account_id", "month_index"])
    df_idx = derived_features.set_index(["account_id", "month_index"])

    records = []
    for row in anchor_df.itertuples(index=False):
        account_id = row.account_id
        t0 = row.feature_month_index
        m = row.minimum_payment_ratio
        L = row.card_limit
        i = row.interest_rate if hasattr(row, "interest_rate") else None

        # carryover_share / r 히스토리 시드 (t0-2, t0-1, t0 실제값)
        cs_hist = {}
        r_hist = {}
        for lag in range(0, 3):
            mm = t0 - lag
            if (account_id, mm) in df_idx.index:
                cs_hist[mm] = df_idx.loc[(account_id, mm), "carryover_share"]
            if (account_id, mm) in mt_idx.index:
                r_hist[mm] = mt_idx.loc[(account_id, mm), "committed_payment_ratio"]

        B_prev = float(row.ending_carryover_principal)
        revolving_streak = int(row.revolving_streak_months)
        min_pay_streak = int(row.minimum_payment_streak)
        delinquency_6m = float(row.delinquency_count_6m)

        state = {c: getattr(row, c) for c in feature_cols if hasattr(row, c)}

        for h in range(1, horizon + 1):
            t_next = t0 + h

            X = build_feature_row(state, feature_cols)
            S_pred = float(model_S.predict(X)[0])
            r_pred = float(model_r.predict(X)[0])
            S_pred = float(np.clip(S_pred, 0.0, L))
            r_pred = float(np.clip(r_pred, m, 1.0))

            calc = deterministic_recursion_step(B_prev, S_pred, r_pred, m, i)

            cs_hist[t_next] = calc["carryover_share"]
            r_hist[t_next] = r_pred

            gap = r_pred - m
            revolving_streak = revolving_streak + 1 if calc["B_t"] > 0 else 0
            min_pay_streak = min_pay_streak + 1 if gap <= PAYMENT_RATIO_GAP_WARN_CUTOFF else 0

            cs_delta_3m = cs_hist.get(t_next) - cs_hist.get(t_next - 3, np.nan)
            xs = [cs_hist.get(t_next - 2, np.nan), cs_hist.get(t_next - 1, np.nan), cs_hist.get(t_next, np.nan)]
            if all(pd.notna(v) for v in xs):
                x_idx = np.array([0, 1, 2])
                cs_slope_3m = float(np.polyfit(x_idx, xs, 1)[0])
            else:
                cs_slope_3m = np.nan
            r_delta_3m = r_hist.get(t_next) - r_hist.get(t_next - 3, np.nan)

            # 다음 스텝 입력 피처 구성 (예측 경로는 "정상" 결제만 가정 — 4-1장 결정론적 계산 레이어 정의)
            state = {
                "billing_amount": S_pred,
                "committed_payment_ratio": r_pred,
                "revolving_principal_before_payment": calc["P_t"],
                "scheduled_principal_payment": calc["A_t"],
                "revolving_fee": calc["I_t"],
                "ending_carryover_principal": calc["B_t"],
                "total_payment_amount": calc["total_payment_amount"],
                "minimum_principal_required": calc["minimum_principal_required"],
                "actual_principal_paid": calc["A_t"],
                "revolving_active": 1 if (B_prev > 0 or r_pred < 1.0) else 0,
                "month_index": t_next,
                "payment_status_정상": 1,
                "payment_status_최소결제": 0,
                "payment_status_연체": 0,
                "carryover_share": calc["carryover_share"],
                "carryover_share_delta_3m": cs_delta_3m,
                "carryover_share_slope_3m": cs_slope_3m,
                "committed_ratio_delta_3m": r_delta_3m,
                "payment_ratio_gap": gap,
                "revolving_streak_months": revolving_streak,
                "minimum_payment_streak": min_pay_streak,
                "delinquency_count_6m": delinquency_6m,  # 이월 (한계 명시)
                "limit_utilization_ratio": float(np.clip(S_pred / L, 0.0, 1.0)),
                "minimum_payment_ratio": m,
                "card_limit": L,
            }
            if i is not None:
                state["interest_rate"] = i

            B_prev = calc["B_t"]

            records.append(
                {
                    "account_id": account_id,
                    "anchor_month_index": t0,
                    "horizon": h,
                    "target_month_index": t_next,
                    "pred_S": S_pred,
                    "pred_r": r_pred,
                    "pred_B": calc["B_t"],
                    "predicted_carryover_share": calc["carryover_share"],
                    # risk.classify_risk_indicator()가 그대로 받아 쓸 수 있도록 재귀 계산 중
                    # 이미 구했던 파생값도 함께 반환한다 (Streamlit 앱의 "다음 달 예측" 판정용).
                    "payment_ratio_gap": gap,
                    "minimum_payment_streak": min_pay_streak,
                    "carryover_share_delta_3m": cs_delta_3m,
                }
            )

    return pd.DataFrame(records)


def attach_actuals(forecast_df: pd.DataFrame, monthly_transaction: pd.DataFrame, derived_features: pd.DataFrame) -> pd.DataFrame:
    actual_mt = monthly_transaction[["account_id", "month_index", "billing_amount", "committed_payment_ratio", "ending_carryover_principal"]]
    actual_mt = actual_mt.rename(
        columns={
            "month_index": "target_month_index",
            "billing_amount": "actual_S",
            "committed_payment_ratio": "actual_r",
            "ending_carryover_principal": "actual_B",
        }
    )
    actual_cs = derived_features[["account_id", "month_index", "carryover_share"]].rename(
        columns={"month_index": "target_month_index", "carryover_share": "actual_carryover_share"}
    )
    out = forecast_df.merge(actual_mt, on=["account_id", "target_month_index"], how="left")
    out = out.merge(actual_cs, on=["account_id", "target_month_index"], how="left")
    return out


def evaluate_recursive(forecast_with_actuals: pd.DataFrame) -> dict:
    metrics = {}
    for h in sorted(forecast_with_actuals["horizon"].unique()):
        sub = forecast_with_actuals[forecast_with_actuals["horizon"] == h]
        metrics[f"horizon_{h}"] = {
            "S": _mae_rmse(sub["actual_S"], sub["pred_S"]),
            "r": _mae_rmse(sub["actual_r"], sub["pred_r"]),
            "carryover_share": _mae_rmse(sub["actual_carryover_share"], sub["predicted_carryover_share"]),
            "n": int(len(sub)),
        }
    metrics["cumulative_3m"] = {
        "S": _mae_rmse(forecast_with_actuals["actual_S"], forecast_with_actuals["pred_S"]),
        "r": _mae_rmse(forecast_with_actuals["actual_r"], forecast_with_actuals["pred_r"]),
        "carryover_share": _mae_rmse(
            forecast_with_actuals["actual_carryover_share"], forecast_with_actuals["predicted_carryover_share"]
        ),
        "n": int(len(forecast_with_actuals)),
    }
    return metrics


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    customer_master = pd.read_csv(DATA_DIR / "customer_master.csv")
    account_master = pd.read_csv(DATA_DIR / "account_master.csv")
    monthly_transaction = pd.read_csv(DATA_DIR / "monthly_transaction.csv")
    derived_features = pd.read_csv(DATA_DIR / "derived_features.csv")

    panel, feature_cols = build_model_panel(monthly_transaction, derived_features, account_master)
    train_df, test_df = time_based_split(panel)
    print(f"panel: {panel.shape}, train: {train_df.shape}, test: {test_df.shape}")
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")

    model_S, model_r = train_models(train_df, feature_cols)

    one_step_metrics, pred_S, pred_r, pred_cs = evaluate_one_step(model_S, model_r, test_df, feature_cols, derived_features)
    print("\n[1개월 예측 성능]")
    print(json.dumps(one_step_metrics, indent=2, ensure_ascii=False))

    # 재귀 3개월 예측: anchor = TRAIN_TARGET_MAX 시점(9월차)의 실제 피처
    anchor_df = panel[panel["target_month_index"] == TRAIN_TARGET_MAX + 1].copy()
    # interest_rate 컬럼이 feature_cols에 없을 수 있으므로 account_master에서 명시적으로 join
    if "interest_rate" not in anchor_df.columns:
        anchor_df = anchor_df.merge(account_master[["account_id", "interest_rate"]], on="account_id", how="left")

    forecast = recursive_forecast(
        model_S, model_r, feature_cols, anchor_df, monthly_transaction, derived_features, horizon=len(TEST_TARGET_MONTHS)
    )
    forecast = attach_actuals(forecast, monthly_transaction, derived_features)
    recursive_metrics = evaluate_recursive(forecast)
    print("\n[3개월 재귀 예측 누적오차]")
    print(json.dumps(recursive_metrics, indent=2, ensure_ascii=False))

    all_metrics = {
        "one_step": one_step_metrics,
        "recursive_multistep": recursive_metrics,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_target_months": f"2-{TRAIN_TARGET_MAX}",
        "test_target_months": list(TEST_TARGET_MONTHS),
        "notes": {
            "one_step": (
                f"test set 전체(target_month_index={list(TEST_TARGET_MONTHS)}, 각 800행씩 "
                f"총 {int(len(test_df))}행)에 대한 평균 MAE/RMSE입니다. 서로 다른 3개 시점의 "
                "1개월 전->1개월 후 전환을 함께 평균한 값이라, 표본이 커서 recursive_multistep의 "
                "horizon_1(anchor 1개 시점만 사용, 800행)보다 더 안정적인 지표입니다."
            ),
            "recursive_multistep_horizon_1": (
                f"horizon_1은 재귀 예측의 시작점(anchor=t{TRAIN_TARGET_MAX})만 사용한 값(800행)으로, "
                f"one_step 지표 중 target_month_index={TRAIN_TARGET_MAX + 1} 행 하나와 동일한 대상입니다. "
                "one_step의 전체 평균과 다르게 나오는 것은 계산 방식 차이가 아니라 평균낸 표본 범위 차이입니다."
            ),
        },
    }
    with open(OUTPUTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    model_S.save_model(MODELS_DIR / "model_S.json")
    model_r.save_model(MODELS_DIR / "model_r.json")
    with open(MODELS_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    forecast.to_csv(OUTPUTS_DIR / "recursive_forecast_sample.csv", index=False)
    print("\nSaved models/metrics to models/ and outputs/")


if __name__ == "__main__":
    main()
