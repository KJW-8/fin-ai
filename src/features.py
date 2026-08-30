"""오뚝이(Ottugi) 2단계 — 파생변수(derived_features) 계산.

설계 문서 3-4장의 산출식을 그대로 구현한다. 모든 파생변수는 계좌(account_id) 단위
시계열 내에서, 시간순(month_index 오름차순) 정렬을 전제로 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MINIMUM_PAYMENT_STREAK_SEVERE, PAYMENT_RATIO_GAP_WARN_CUTOFF  # noqa: F401 (재사용 임계치 근거 명시용)


def _consecutive_streak_by_group(flag: pd.Series, group_key: pd.Series) -> pd.Series:
    """flag(불리언, 시간순 정렬)가 True로 연속된 길이를 group_key 경계마다 리셋하며 계산.

    groupby().apply() 대신 전부 벡터화 연산으로 처리한다 (계좌 수가 많을 때도 빠르고,
    groupby 객체를 미리 만들어 캐싱해 두면 이후 df에 새 컬럼을 추가해도 반영되지 않는
    pandas Copy-on-Write 관련 캐싱 문제를 피할 수 있다).
    """
    new_group = group_key.ne(group_key.shift(1))
    reset_point = (~flag) | new_group
    reset_id = reset_point.cumsum()
    streak = flag.groupby(reset_id).cumcount() + 1
    return streak.where(flag, 0).astype(int)


def _rolling_slope(values: pd.Series, window: int = 3) -> pd.Series:
    """최근 window개 시점에 대한 단순 선형회귀 기울기 (x=0..window-1, y=값)."""
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return values.rolling(window=window, min_periods=window).apply(slope, raw=True)


def compute_derived_features(
    monthly_transaction: pd.DataFrame,
    account_master: pd.DataFrame,
    gap_streak_threshold: float = PAYMENT_RATIO_GAP_WARN_CUTOFF,
) -> pd.DataFrame:
    """monthly_transaction + account_master -> derived_features DataFrame.

    Parameters
    ----------
    gap_streak_threshold: minimum_payment_streak 판정 시 payment_ratio_gap이
        이 값 이하이면 "최소결제 수준에 근접"으로 간주한다. 4-2장 경고단계
        규칙에서 쓰는 payment_ratio_gap <= 0.05 임계치를 그대로 재사용한다
        (동일한 개념이므로 별도 임의값을 새로 도입하지 않음).
    """
    df = monthly_transaction.merge(
        account_master[["account_id", "customer_id", "minimum_payment_ratio", "card_limit"]],
        on="account_id",
        how="left",
    )
    df = df.sort_values(["account_id", "month_index"]).reset_index(drop=True)

    B = df["ending_carryover_principal"].astype(float)
    S = df["billing_amount"].astype(float)
    r = df["committed_payment_ratio"].astype(float)
    m = df["minimum_payment_ratio"].astype(float)
    L = df["card_limit"].astype(float)
    acct = df["account_id"]

    # carryover_share: 분모(B_t+S_t)가 0이면 0으로 예외처리
    denom = B + S
    carryover_share = np.where(denom > 0, B / denom.replace(0, np.nan), 0.0)
    df["carryover_share"] = pd.Series(carryover_share, index=df.index).fillna(0.0)

    # carryover_share_delta_3m = carryover_share_t - carryover_share_{t-3}
    df["carryover_share_delta_3m"] = df["carryover_share"] - df.groupby(acct)["carryover_share"].shift(3)

    # carryover_share_slope_3m: 최근 3개 시점 단순 선형회귀 기울기
    df["carryover_share_slope_3m"] = df.groupby(acct)["carryover_share"].transform(
        lambda s: _rolling_slope(s, 3)
    )

    # committed_ratio_delta_3m = r_t - r_{t-3}
    df["committed_ratio_delta_3m"] = r - df.groupby(acct)["committed_payment_ratio"].shift(3)

    # payment_ratio_gap = r_t - m  (m은 계좌 고정값)
    df["payment_ratio_gap"] = r - m

    # revolving_streak_months: B_t > 0 이 연속된 개월수 (계좌 경계에서 리셋)
    df["revolving_streak_months"] = _consecutive_streak_by_group(B > 0, acct)

    # minimum_payment_streak: gap_t <= threshold(최소결제 수준 근접)이 연속된 개월수
    is_near_minimum = df["payment_ratio_gap"] <= gap_streak_threshold
    df["minimum_payment_streak"] = _consecutive_streak_by_group(is_near_minimum, acct)

    # delinquency_count_6m: 최근 6개월(당월 포함) 연체 발생 횟수
    is_delinquent = (df["payment_status"] == "연체").astype(int)
    df["delinquency_count_6m"] = (
        is_delinquent.groupby(acct).transform(lambda s: s.rolling(window=6, min_periods=1).sum()).astype(int)
    )

    # limit_utilization_ratio = S_t / L, 0~1 캡핑
    df["limit_utilization_ratio"] = np.clip(S / L, 0.0, 1.0)

    # revolving_payment_to_income_ratio: MVP 단계 소득 데이터 미확보 -> 결측 처리, 모델 제외
    df["revolving_payment_to_income_ratio"] = np.nan

    feature_cols = [
        "customer_id",
        "account_id",
        "year_month",
        "month_index",
        "carryover_share",
        "carryover_share_delta_3m",
        "carryover_share_slope_3m",
        "committed_ratio_delta_3m",
        "payment_ratio_gap",
        "revolving_streak_months",
        "minimum_payment_streak",
        "delinquency_count_6m",
        "limit_utilization_ratio",
        "revolving_payment_to_income_ratio",
    ]
    return df[feature_cols]


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    account_master = pd.read_csv(DATA_DIR / "account_master.csv")
    monthly_transaction = pd.read_csv(DATA_DIR / "monthly_transaction.csv")

    derived = compute_derived_features(monthly_transaction, account_master)
    derived.to_csv(DATA_DIR / "derived_features.csv", index=False)

    print("derived_features:", derived.shape)
    print(derived.isna().mean().round(3))

    # 중위험(진입형) 페르소나 궤적 확인 — 문서 8-2 "gap이 0.62->0.27로 지속 축소" 재현 여부
    med_accounts = pd.read_csv(DATA_DIR / "account_master.csv")
    med_id = med_accounts[med_accounts["persona_tier"] == "medium"]["account_id"].iloc[0]
    cols = [
        "month_index",
        "carryover_share",
        "carryover_share_delta_3m",
        "carryover_share_slope_3m",
        "payment_ratio_gap",
        "minimum_payment_streak",
        "delinquency_count_6m",
    ]
    print("\n[중위험 샘플 계좌 파생변수 궤적]")
    print(derived[derived["account_id"] == med_id][cols].to_string(index=False))
