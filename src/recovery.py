"""오뚝이(Ottugi) — 회복 게이지 (recovery_score) 계산.

게임 포인트가 아니라 **기존 모델 출력값과 연결된** 회복 상태 지표다. 임의 수치를 쓰지
않는다.

------------------------------------------------------------------------
공식 선택: 방식 (b) — risk_indicator 4단계 선형 매핑 + 단계 내 carryover_share 보간
------------------------------------------------------------------------
스펙이 제시한 두 후보:
  (a) recovery_score = (1 - transition_probability_3m) x 100
  (b) risk_indicator 4단계를 0~100 구간에 선형 매핑 + 단계 내 carryover_share 상대 위치 보간

**(b)를 선택한 이유:**
  1. (a)의 transition_probability_3m 은 hazard 모델의 이벤트("경고/심화로 전환")를 아직
     겪지 않은 계좌(관찰/주의)에만 정의된다. 이미 경고/심화인 사용자에게는 값이 없어
     게이지가 전 구간(관찰~심화)에서 연속적으로 동작하지 못한다.
  2. (b)는 이미 화면에 노출 중인 값(risk_indicator, carryover_share)만 재사용하므로
     "게이지가 왜 이 숫자인지"를 사용자가 다른 화면과 바로 대조할 수 있다.
  3. What-if 시뮬레이션으로 risk_indicator/carryover_share 가 바뀌면 게이지도 즉시
     같은 방향으로 움직인다(단계 경계를 넘으면 큰 폭, 같은 단계 내면 완만).
  4. 마스코트 상태(state_mapping)도 risk_indicator 로 구동되므로, 게이지와 캐릭터가
     같은 신호를 공유해 화면 일관성이 유지된다.

hazard 모델의 transition_probability_3m 은 게이지 값 자체가 아니라 게이지 **아래
힌트 문구**와 "회복 궤적" 보조 설명에서 활용한다.

------------------------------------------------------------------------
계산
------------------------------------------------------------------------
각 단계에 25점 폭의 구간을 배정한다 (높을수록 회복에 가까움):
    심화 [0, 25) · 경고 [25, 50) · 주의 [50, 75) · 관찰 [75, 100]
구간 중앙(midpoint)을 기준점으로 하고, 그 단계에서 관측되는 carryover_share 범위 안에서
현재 값의 상대 위치에 따라 ±12.5 를 가감한다 (carryover_share 가 낮을수록 = 회복에
가까울수록 상단). config.py 의 실제 임계값(CARRYOVER_SHARE_OBSERVE_CUTOFF=0.25,
RISK_LEVEL_THRESHOLD_DEFAULT=0.40)을 구간 경계로 사용한다.
"""

from __future__ import annotations

from config import CARRYOVER_SHARE_OBSERVE_CUTOFF, RISK_LEVEL_THRESHOLD_DEFAULT

# 단계별 [하한, 상한] (0~100). 높을수록 회복에 가깝다.
LEVEL_BANDS = {
    "심화": (0.0, 25.0),
    "경고": (25.0, 50.0),
    "주의": (50.0, 75.0),
    "관찰": (75.0, 100.0),
}

# 단계 내 carryover_share 보간에 쓸 (낮음=상단, 높음=하단) 참조 구간.
# 경계값은 config.py 실제 임계값 기준. 심화/경고 상단은 관측 최대값 근사(0.85)를 사용.
_CS_REF = {
    "관찰": (0.0, CARRYOVER_SHARE_OBSERVE_CUTOFF),                 # 0.00 ~ 0.25
    "주의": (CARRYOVER_SHARE_OBSERVE_CUTOFF, RISK_LEVEL_THRESHOLD_DEFAULT),  # 0.25 ~ 0.40
    "경고": (RISK_LEVEL_THRESHOLD_DEFAULT, 0.70),                  # 0.40 ~ 0.70
    "심화": (RISK_LEVEL_THRESHOLD_DEFAULT, 0.90),                  # 0.40 ~ 0.90
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def recovery_score(risk_indicator: str, carryover_share: float | None) -> float:
    """0~100. 높을수록 회복에 가까운 상태. (공식은 모듈 docstring 참고)"""
    lo, hi = LEVEL_BANDS.get(risk_indicator, LEVEL_BANDS["관찰"])
    mid = (lo + hi) / 2.0
    if carryover_share is None:
        return round(mid, 1)

    ref_lo, ref_hi = _CS_REF.get(risk_indicator, (0.0, 1.0))
    span = ref_hi - ref_lo if ref_hi > ref_lo else 1.0
    pos = _clip((float(carryover_share) - ref_lo) / span, 0.0, 1.0)  # 0=낮음(좋음), 1=높음(나쁨)
    # carryover_share 낮으면 +12.5, 높으면 -12.5
    score = mid + (0.5 - pos) * (hi - lo)
    return round(_clip(score, 0.0, 100.0), 1)


def recovery_hint(
    risk_indicator: str,
    *,
    transition_probability_3m: float | None = None,
    min_intervention_amount: float | None = None,
    simulation_delta_score: float | None = None,
) -> str:
    """게이지 아래 1문장. 단정적 지시가 아니라 시뮬레이션 결과 기반 안내로 표현한다."""
    if simulation_delta_score is not None:
        if simulation_delta_score > 1:
            return f"입력한 시나리오를 적용하면 이 수치가 약 {simulation_delta_score:.0f}점 높아지는 것으로 계산돼요."
        if simulation_delta_score < -1:
            return f"입력한 시나리오에서는 이 수치가 약 {abs(simulation_delta_score):.0f}점 낮아지는 것으로 계산돼요."
        return "입력한 시나리오에서는 이 수치가 크게 달라지지 않는 것으로 계산돼요."

    if min_intervention_amount and min_intervention_amount > 0:
        return (
            f"시뮬레이션상 월 약 {min_intervention_amount:,.0f}원을 추가로 상환하는 시나리오에서 "
            "이 수치의 하락 폭이 완만해지는 것으로 계산돼요."
        )
    if risk_indicator == "관찰":
        return "지금 결제 패턴을 유지하면 이 수치도 비슷하게 유지될 것으로 계산돼요."
    if transition_probability_3m is not None and transition_probability_3m >= 0.3:
        return "결제 비율을 조금 높이는 시나리오에서 회복 궤적이 완만해질 수 있는 것으로 계산돼요."
    return "결제 비율을 조금 높이거나 추가 상환을 가정하면 이 수치가 개선되는지 시뮬레이션에서 확인할 수 있어요."


# 상태별 절제된 안내 문구 (마스코트 이미지 옆에 표시). 겁주거나 죄책감 주는 표현 금지.
STATE_MESSAGES = {
    "관찰": "아직 안정적인 상태예요. 지금처럼만 유지하시면 됩니다.",
    "주의": "조금씩 리볼빙 의존도가 올라오고 있어요. 지금 결제 비율을 살펴볼 좋은 시점이에요.",
    "경고": "결제 여유가 얼마 남지 않았어요. 지금 상환 방식을 바꾸면 효과가 큰 구간이에요.",
    "심화": "지금은 어려운 상태지만, 아직 방향을 바꿀 수 있어요. 작은 변화부터 함께 살펴봐요.",
}
