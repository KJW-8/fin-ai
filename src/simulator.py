"""오뚝이(Ottugi) 1단계 — 데이터 시뮬레이터.

설계 문서(`오뚝이_서비스흐름_데이터설계.md`) 3장의 정정된 재귀식을 그대로 구현한다.

    P_t = B_{t-1} + S_t
    A_t = P_t x r_t
    I_t = B_{t-1} x i x (d_t/365)          # d_t=30 고정 근사치 (실제 결제주기 아님, MVP 한정)
    B_t = P_t - A_t                         # (단, 결제 이행 이벤트에 따라 실제 상환원금이 A_t보다
                                             #  적을 수 있음 — 아래 "결제 이행 이벤트" 설명 참고)
    total_payment_amount_t = A_t + I_t
    minimum_principal_required_t = max(P_t x m, 50000)

m(최소결제비율)은 계좌 개설 시 1회 고정, r(약정결제비율)은 m 이상 100% 이하에서
매월 변하는 행동 변수로 티어(저/중/고위험)별 랜덤워크/드리프트로 생성한다.

결제 이행 이벤트에 대한 설계 노트
----------------------------------
문서 3-3은 payment_status를 "정상/최소결제/연체"로 구분하는데, 이는 실제 납부액이
약정액(A_t+I_t)에 못 미치는 달이 존재함을 전제로 한다. 그러나 B_t의 재귀식은 A_t가
"그 달 실제로 상환하는 원금"이라고 정의하므로, 재귀식 자체는 항상 A_t가 반영된다.
두 정의를 모두 살리기 위해, 매월 소액의 확률로 "실제 상환원금이 A_t에 못 미치는"
이행 이벤트(정상/최소결제만 이행/연체)를 부여하고, 그 실제 상환원금을 B_t 재귀식에
대입한다. 정상 이벤트(가장 높은 확률)에서는 실제 상환원금 = A_t이므로 문서의 재귀식과
정확히 동일하게 동작한다. 이 이벤트가 없으면 payment_status/delinquency_count_6m이
항상 "정상"/0으로 고정되어 파생변수로서 의미가 없어지기 때문에 도입한 최소한의 확장이다.

⚠️ 이벤트 발생 확률(config.PAYMENT_EVENT_PROBS)은 공시 통계 등 외부 근거가 없는
시나리오 가정치다. "고위험군일수록 연체가 잦다"는 방향성만 반영했으며, 실서비스
전환 시 실제 연체율 통계로 재보정이 필요하다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    AGE_GROUP_WEIGHTS,
    CARD_LIMIT_SCENARIOS,
    DAYS_APPROX,
    DAYS_IN_YEAR,
    EMPLOYMENT_TYPE_WEIGHTS,
    GENDER_WEIGHTS,
    INTEREST_RATE_FIXED,
    MIN_PRINCIPAL_FLOOR,
    M_MAX,
    M_MIN,
    MYDATA_CONSENT_RATE,
    N_MONTHS,
    PAYMENT_EVENT_PROBS,
    PAYMENT_EVENTS,
    RANDOM_SEED,
    RISK_TIER_WEIGHTS,
)


def _weighted_choice(rng: np.random.Generator, weights: dict, size: int) -> np.ndarray:
    keys = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=size, p=probs)


# ---------------------------------------------------------------------------
# customer_master
# ---------------------------------------------------------------------------
def generate_customer_master(n_customers: int, rng: np.random.Generator, today: pd.Timestamp) -> pd.DataFrame:
    customer_ids = [str(uuid.uuid4()) for _ in range(n_customers)]
    age_group = _weighted_choice(rng, AGE_GROUP_WEIGHTS, n_customers)
    gender = _weighted_choice(rng, GENDER_WEIGHTS, n_customers)
    employment_type = _weighted_choice(rng, EMPLOYMENT_TYPE_WEIGHTS, n_customers)

    # 서비스 가입일: 관측 종료 시점(today) 기준 최근 24~30개월 이내 임의 시점
    join_offset_days = rng.integers(30, 900, size=n_customers)
    join_date = [today - pd.Timedelta(days=int(d)) for d in join_offset_days]

    mydata_consent = rng.random(n_customers) < MYDATA_CONSENT_RATE

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age_group": age_group,
            "gender": gender,
            "employment_type": employment_type,
            "join_date": join_date,
            "mydata_consent": mydata_consent,
        }
    )


# ---------------------------------------------------------------------------
# account_master
# ---------------------------------------------------------------------------
def generate_account_master(
    customer_master: pd.DataFrame, rng: np.random.Generator, today: pd.Timestamp, n_months: int = N_MONTHS
) -> pd.DataFrame:
    n = len(customer_master)
    account_ids = [str(uuid.uuid4()) for _ in range(n)]

    card_limit = rng.choice(CARD_LIMIT_SCENARIOS, size=n)
    minimum_payment_ratio = rng.uniform(M_MIN, M_MAX, size=n)
    interest_rate = np.full(n, INTEREST_RATE_FIXED)

    # 위험 분포가 고르게 섞이도록 페르소나 티어를 부여 (8장 저/중/고위험 로직 참고).
    persona_tier = _weighted_choice(rng, RISK_TIER_WEIGHTS, n)

    # 리볼빙 최초 등록 시점: today를 관측기간 종료월로 두고, 그로부터 정확히
    # n_months 전 달의 1일을 등록월로 삼는다 (모든 계좌가 today 시점까지 n_months
    # 연속 관측되도록). 등록 이후 시점은 "관찰기간 내 임의 시점" 요건을 월 단위
    # 오프셋(0~11개월 랜덤 시작 지연)으로 반영해 계좌마다 관측 시작 캘린더월이
    # 달라지도록 한다.
    start_delay_months = rng.integers(0, 12, size=n)
    enrolled_month = (today.to_period("M") - n_months + 1) - start_delay_months
    revolving_enrolled_date = [p.to_timestamp() for p in enrolled_month]

    return pd.DataFrame(
        {
            "account_id": account_ids,
            "customer_id": customer_master["customer_id"].values,
            "card_limit": card_limit,
            "minimum_payment_ratio": np.round(minimum_payment_ratio, 4),
            "interest_rate": interest_rate,
            "revolving_enrolled_date": revolving_enrolled_date,
            # 생성 단계 메타데이터. 모델 피처로 사용하지 않음(config.NON_MODEL_FEATURES).
            "persona_tier": persona_tier,
        }
    )


# ---------------------------------------------------------------------------
# monthly_transaction — 계좌별 12개월 재귀 시뮬레이션
# ---------------------------------------------------------------------------
@dataclass
class _TierParams:
    s_base: float
    s_noise_std: float
    s_drift: float | None = None       # low/medium: 소비액 완만한 드리프트
    s_growth: float | None = None      # high: 소비액 매월 성장률
    r_init: float = 0.9
    r_vol: float = 0.01
    r_decay: float | None = None       # medium: r이 m으로 수렴하는 승수 감쇠율


def _draw_tier_params(tier: str, L: float, m: float, rng: np.random.Generator) -> _TierParams:
    if tier == "low":
        utilization = rng.uniform(0.15, 0.40)
        return _TierParams(
            s_base=L * utilization,
            s_noise_std=0.04,
            s_drift=rng.uniform(-0.005, 0.005),
            r_init=rng.uniform(0.85, 0.98),
            r_vol=rng.uniform(0.005, 0.02),
        )
    if tier == "medium":
        utilization = rng.uniform(0.20, 0.45)
        return _TierParams(
            s_base=L * utilization,
            s_noise_std=0.05,
            s_drift=rng.uniform(-0.01, 0.01),
            r_init=rng.uniform(0.75, 0.90),
            r_vol=rng.uniform(0.01, 0.03),
            r_decay=rng.uniform(0.88, 0.95),
        )
    # high
    utilization = rng.uniform(0.25, 0.55)
    return _TierParams(
        s_base=L * utilization,
        s_noise_std=0.04,
        s_growth=rng.uniform(0.01, 0.04),
        r_init=m + rng.uniform(0.0, 0.03),
        r_vol=rng.uniform(0.0, 0.01),
    )


def _simulate_account_series(
    account_id: str,
    tier: str,
    L: float,
    m: float,
    i: float,
    entry_date: pd.Timestamp,
    rng: np.random.Generator,
    n_months: int,
) -> list[dict]:
    params = _draw_tier_params(tier, L, m, rng)
    event_probs = PAYMENT_EVENT_PROBS[tier]

    B_prev = 0.0
    r_prev = params.r_init
    s_prev = params.s_base
    rows: list[dict] = []

    for t in range(1, n_months + 1):
        # --- S_t: 당월 청구액 (한도 대비 캡핑) ---
        if tier == "high":
            s_prev = s_prev * (1 + params.s_growth) if t > 1 else params.s_base
            S_t = s_prev * (1 + rng.normal(0, params.s_noise_std))
        else:
            s_prev = params.s_base * ((1 + params.s_drift) ** (t - 1))
            S_t = s_prev * (1 + rng.normal(0, params.s_noise_std))
        S_t = float(np.clip(S_t, L * 0.02, L * 0.97))
        S_t = round(S_t)

        # --- r_t: 약정결제비율 (m 이상 100% 이하 랜덤워크/드리프트, r_t >= m 강제) ---
        if t == 1:
            r_t = params.r_init
        elif tier == "medium":
            r_t = r_prev * params.r_decay + rng.normal(0, params.r_vol)
        elif tier == "low":
            r_t = r_prev + rng.normal(0, params.r_vol)
        else:  # high: m 근방에서 미세하게만 변동
            r_t = m + abs(rng.normal(0, params.r_vol))
        r_t = float(np.clip(r_t, m, 1.0))
        r_prev = r_t

        # --- 3장 정정된 재귀식 ---
        P_t = B_prev + S_t
        A_t = P_t * r_t
        I_t = B_prev * i * (DAYS_APPROX / DAYS_IN_YEAR)
        minimum_principal_required = max(P_t * m, MIN_PRINCIPAL_FLOOR)
        total_payment_amount = A_t + I_t

        # --- 결제 이행 이벤트 (payment_status/연체 파생변수를 위한 최소 확장, 모듈 docstring 참고) ---
        event = rng.choice(PAYMENT_EVENTS, p=event_probs)
        if event == "normal":
            actual_principal_paid = A_t
        elif event == "minimum_only":
            actual_principal_paid = min(A_t, minimum_principal_required)
        else:  # delinquent
            actual_principal_paid = 0.0

        B_t = P_t - actual_principal_paid
        actual_total_paid = actual_principal_paid + I_t

        if actual_total_paid >= total_payment_amount - 1:
            payment_status = "정상"
        elif actual_total_paid >= minimum_principal_required:
            payment_status = "최소결제"
        else:
            payment_status = "연체"

        revolving_active = bool(B_prev > 0 or r_t < 1.0)

        year_month = (entry_date.to_period("M") + (t - 1)).strftime("%Y-%m")

        rows.append(
            {
                "account_id": account_id,
                "year_month": year_month,
                "month_index": t,
                "billing_amount": S_t,
                "committed_payment_ratio": round(r_t, 4),
                "revolving_principal_before_payment": round(P_t),
                "scheduled_principal_payment": round(A_t),
                "revolving_fee": round(I_t),
                "ending_carryover_principal": round(B_t),
                "total_payment_amount": round(total_payment_amount),
                "minimum_principal_required": round(minimum_principal_required),
                "actual_principal_paid": round(actual_principal_paid),
                "revolving_active": revolving_active,
                "payment_status": payment_status,
                "payment_event": event,
            }
        )

        B_prev = round(B_t)

    return rows


def generate_monthly_transaction(
    account_master: pd.DataFrame, rng: np.random.Generator, n_months: int = N_MONTHS
) -> pd.DataFrame:
    all_rows: list[dict] = []
    for row in account_master.itertuples(index=False):
        rows = _simulate_account_series(
            account_id=row.account_id,
            tier=row.persona_tier,
            L=row.card_limit,
            m=row.minimum_payment_ratio,
            i=row.interest_rate,
            entry_date=pd.Timestamp(row.revolving_enrolled_date),
            rng=rng,
            n_months=n_months,
        )
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------
def run_simulation(n_customers: int = 800, seed: int = RANDOM_SEED, n_months: int = N_MONTHS):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize().replace(day=1)

    customer_master = generate_customer_master(n_customers, rng, today)
    account_master = generate_account_master(customer_master, rng, today, n_months)
    monthly_transaction = generate_monthly_transaction(account_master, rng, n_months)

    return customer_master, account_master, monthly_transaction


if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

    customer_master, account_master, monthly_transaction = run_simulation(n_customers=800)

    customer_master.to_csv(DATA_DIR / "customer_master.csv", index=False)
    account_master.to_csv(DATA_DIR / "account_master.csv", index=False)
    monthly_transaction.to_csv(DATA_DIR / "monthly_transaction.csv", index=False)

    print("customer_master:", customer_master.shape)
    print("account_master:", account_master.shape)
    print("monthly_transaction:", monthly_transaction.shape)
    print("\n[persona tier distribution]")
    print(account_master["persona_tier"].value_counts())
    print("\n[payment_status distribution]")
    print(monthly_transaction["payment_status"].value_counts())
