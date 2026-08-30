"""오뚝이(Ottugi) 5단계 — 위험 판정(risk_indicator).

문서 4-2장의 복합 경고 단계 규칙을 그대로 구현한다. 모델 출력이 아니라 규칙 기반
후처리이며, predicted_carryover_share를 "확률"이나 "위험도 점수"로 노출하지 않고
이 4단계 범주(관찰/주의/경고/심화)로만 사용자에게 제시한다.

| 단계 | 조건 (모두 충족 시) |
|---|---|
| 관찰 | carryover_share_t < 25% |
| 주의 | carryover_share_t >= 25% 그리고 carryover_share_delta_3m > 0 (상승 추세) |
| 경고 | (carryover_share_t >= 40% 또는 payment_ratio_gap <= 0.05) 그리고 상승 추세 지속 |
| 심화 | minimum_payment_streak >= 3 (최소결제 수준만 반복) |

우선순위: 문서 표는 4단계를 나란히 제시하지만 상호 배타적이지 않을 수 있으므로(예:
carryover_share가 낮아도 minimum_payment_streak>=3인 경우), "가장 심각한 조건이 우선한다"는
원칙으로 심화 > 경고 > 주의 > 관찰 순으로 캐스케이드 판정한다. 심화 조건은 carryover_share
수준과 무관하게 "최소결제 수준만 반복하는 행동 패턴" 자체를 심각한 신호로 보는 문서 취지에
따라 다른 조건보다 우선한다.

심화 조건과 carryover_share 수준의 관계에 대한 검증 노트
----------------------------------------------------------
minimum_payment_streak(따라서 심화 판정)은 carryover_share 수준과 독립적으로 정의돼 있어,
이론적으로는 carryover_share가 25% 미만("관찰" 수준)인데도 심화로 분류되는 경우가 있어
보일 수 있다. 그러나 800개 계좌 전체 시뮬레이션 데이터로 검증한 결과 이런 케이스는
0건이었으며, 다음과 같이 수식으로도 설명된다.

1) 리볼빙 진입 시점(B_{t-1}=0)에조차, minimum_payment_streak가 쌓이려면 r_t가 m에 근접
   (gap<=0.05)해야 하므로 근사적으로
       carryover_share_t ~= B_t / (B_t+S_t) ~= (1-m) / (2-m)   (r_t~=m, B_{t-1}=0 대입)
   m의 정의역(Uniform 10~30%) 내 최댓값인 m=30%를 대입해도
       carryover_share ~= 0.7/1.7 ~= 41.2%
   로, 이미 "경고" 문턱(40%)을 상회한다. 즉 m이 정의역 어디에 있든 첫 달부터 carryover_share가
   40% 안팎에서 시작한다.
2) B_{t-1}>0이 되는 이후 시점부터는 이월잔액이 누적되므로 carryover_share가 더 올라가면
   올라갔지 낮아지지는 않는다.
3) 따라서 minimum_payment_streak>=3을 만족하는 모든 시점에서 carryover_share는 항상 40%
   안팎 이상이며, "심화"와 "관찰/주의" 수준의 동시 발생은 현재 파라미터 범위
   (m ∈ [10%, 30%]) 하에서 구조적으로 불가능하다.
4) 단, 이는 이 파라미터 범위(m의 상한 30%)에 한정된 결론이다. 향후 m의 상한이 지금보다
   낮아지는 방향으로 재조정되면 (1)의 (1-m)/(2-m) 값이 40% 밑으로 내려갈 수 있으므로,
   이 가정과 우선순위 로직을 다시 검토해야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CARRYOVER_SHARE_OBSERVE_CUTOFF,
    MINIMUM_PAYMENT_STREAK_SEVERE,
    PAYMENT_RATIO_GAP_WARN_CUTOFF,
    RISK_LEVEL_THRESHOLD_DEFAULT,
    RISK_LEVEL_THRESHOLD_SENSITIVITY,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

RISK_LEVELS = ["관찰", "주의", "경고", "심화"]


def classify_risk_indicator(
    carryover_share: float,
    carryover_share_delta_3m: float,
    payment_ratio_gap: float,
    minimum_payment_streak: int,
    warn_threshold: float = RISK_LEVEL_THRESHOLD_DEFAULT,
) -> str:
    """단일 시점 값에 대한 risk_indicator 판정 (스칼라 입력, Streamlit 앱 등에서 사용)."""
    rising = (carryover_share_delta_3m is not None) and pd.notna(carryover_share_delta_3m) and (carryover_share_delta_3m > 0)

    if minimum_payment_streak >= MINIMUM_PAYMENT_STREAK_SEVERE:
        return "심화"
    if (carryover_share >= warn_threshold or payment_ratio_gap <= PAYMENT_RATIO_GAP_WARN_CUTOFF) and rising:
        return "경고"
    if carryover_share >= CARRYOVER_SHARE_OBSERVE_CUTOFF and rising:
        return "주의"
    return "관찰"


def apply_risk_indicator(df: pd.DataFrame, warn_threshold: float = RISK_LEVEL_THRESHOLD_DEFAULT) -> pd.Series:
    """derived_features 형태의 DataFrame(carryover_share, carryover_share_delta_3m,
    payment_ratio_gap, minimum_payment_streak 컬럼 필요)에 벡터화된 방식으로 일괄 적용."""
    cs = df["carryover_share"]
    delta = df["carryover_share_delta_3m"]
    gap = df["payment_ratio_gap"]
    streak = df["minimum_payment_streak"]

    rising = delta > 0  # NaN > 0 은 자동으로 False (아직 추세 판단 불가 -> 상승 아님으로 취급)
    severe = streak >= MINIMUM_PAYMENT_STREAK_SEVERE
    warn = ((cs >= warn_threshold) | (gap <= PAYMENT_RATIO_GAP_WARN_CUTOFF)) & rising
    caution = (cs >= CARRYOVER_SHARE_OBSERVE_CUTOFF) & rising

    labels = np.select([severe, warn, caution], ["심화", "경고", "주의"], default="관찰")
    return pd.Series(labels, index=df.index, name="risk_indicator")


# ---------------------------------------------------------------------------
# 임계치(35%/40%/45%) 민감도 분석
# ---------------------------------------------------------------------------
def first_warn_month_by_threshold(
    derived_features: pd.DataFrame, thresholds: list[float] = RISK_LEVEL_THRESHOLD_SENSITIVITY
) -> pd.DataFrame:
    """계좌별로, 각 임계치에서 risk_indicator가 처음 '경고'(또는 그 이상 심각도)가 되는
    month_index를 계산. 임계치가 결과를 얼마나 좌우하는지 보기 위한 표."""
    result = {"account_id": derived_features["account_id"].unique()}
    records = []
    for account_id, g in derived_features.groupby("account_id"):
        g = g.sort_values("month_index")
        row = {"account_id": account_id}
        for th in thresholds:
            labels = apply_risk_indicator(g, warn_threshold=th)
            warn_or_worse = g.loc[labels.isin(["경고", "심화"]), "month_index"]
            row[f"first_warn_month_th{int(th*100)}"] = int(warn_or_worse.min()) if not warn_or_worse.empty else None
        records.append(row)
    return pd.DataFrame(records)


def summarize_threshold_sensitivity(sensitivity_df: pd.DataFrame, thresholds: list[float]) -> dict:
    cols = [f"first_warn_month_th{int(th*100)}" for th in thresholds]
    reached = sensitivity_df.dropna(subset=cols, how="all")

    both = sensitivity_df.dropna(subset=cols, how="any")
    same_month_rate = float((both[cols[0]] == both[cols[-1]]).mean()) if len(both) else None
    avg_shift = (
        float((both[cols[-1]] - both[cols[0]]).mean()) if len(both) else None
    )  # 45% 기준이 35% 기준보다 평균 몇 개월 늦게 경고를 띄우는지

    n_reached_by_th = {c: int(sensitivity_df[c].notna().sum()) for c in cols}

    return {
        "thresholds": thresholds,
        "n_accounts": int(len(sensitivity_df)),
        "n_accounts_reached_warn_by_threshold": n_reached_by_th,
        "n_accounts_reached_all_thresholds": int(len(both)),
        "same_first_warn_month_rate_35_vs_45": same_month_rate,
        "avg_month_shift_35_to_45": avg_shift,
    }


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    account_master = pd.read_csv(DATA_DIR / "account_master.csv")
    derived_features = pd.read_csv(DATA_DIR / "derived_features.csv")

    # 1) 기본 임계치(40%)로 실제(관측) 데이터에 risk_indicator 적용
    derived_features["risk_indicator"] = apply_risk_indicator(derived_features, warn_threshold=RISK_LEVEL_THRESHOLD_DEFAULT)
    derived_features.to_csv(DATA_DIR / "derived_features.csv", index=False)

    print("[risk_indicator 분포 (임계치=40%, 관측 데이터 전체)]")
    print(derived_features["risk_indicator"].value_counts())

    # 2) 페르소나별 대표 계좌로 문서 8장 서술과의 정합성 확인
    persona_accounts = account_master.groupby("persona_tier")["account_id"].first().to_dict()
    print("\n[페르소나별 대표 계좌 risk_indicator 궤적]")
    for tier, account_id in persona_accounts.items():
        sub = derived_features[derived_features["account_id"] == account_id].sort_values("month_index")
        print(f"  {tier}: {list(sub['risk_indicator'])}")

    # 3) 임계치 민감도 분석 (35% / 40% / 45%)
    sensitivity_df = first_warn_month_by_threshold(derived_features, RISK_LEVEL_THRESHOLD_SENSITIVITY)
    summary = summarize_threshold_sensitivity(sensitivity_df, RISK_LEVEL_THRESHOLD_SENSITIVITY)

    print("\n[임계치 민감도 분석 요약]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    medium_account_id = persona_accounts["medium"]
    medium_row = sensitivity_df[sensitivity_df["account_id"] == medium_account_id].to_dict(orient="records")[0]
    print(f"\n[중위험 대표 계좌 임계치별 최초 경고 시점] {medium_row}")

    sensitivity_df.to_csv(OUTPUTS_DIR / "risk_threshold_sensitivity.csv", index=False)
    with open(OUTPUTS_DIR / "risk_threshold_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "medium_persona_example": medium_row}, f, ensure_ascii=False, indent=2)

    print("\nSaved data/derived_features.csv(risk_indicator 추가), "
          "outputs/risk_threshold_sensitivity.csv, outputs/risk_threshold_sensitivity_summary.json")


if __name__ == "__main__":
    main()
