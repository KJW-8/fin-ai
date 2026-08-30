"""오뚝이(Ottugi) 6단계 — 생성형 AI 코칭 메시지 (근거 태깅 JSON).

Claude Haiku 4.5(claude-haiku-4-5-20251001)를 호출해 위험도/SHAP/시뮬레이션 결과를
문장 단위로 근거(source)가 태깅된 JSON으로 코칭 메시지를 만든다.

    {"segments": [{"text": str, "source": "raw_data" | "shap" | "simulation"}]}

지금 단계에서는 API 키 없이 UI/UX를 먼저 완성하기 위해 mock 구현을 기본으로 쓰고,
환경변수 USE_MOCK_COACHING(기본값 "true")으로 mock/real을 전환한다. 검증 로직
(JSON Schema + source 일치성)은 mock/real 어느 쪽을 쓰든 동일하게 적용된다.
"""

from __future__ import annotations

import json
import os
from typing import Any

import jsonschema

MODEL_NAME = "claude-haiku-4-5-20251001"

VALID_SOURCES = ("raw_data", "shap", "simulation")

COACHING_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "enum": list(VALID_SOURCES)},
                },
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

# 모델 피처명 -> 코칭 메시지·SHAP 차트에 쓸 한국어 표현. 최대한 쉬운 일상어로 풀었다
# (예: committed_payment_ratio -> "이번 달 상환 비율", payment_ratio_gap -> "최소
# 상환액과의 여유분", carryover_share_slope_3m -> "최근 3개월 악화 속도").
FEATURE_LABELS = {
    "billing_amount": "이번 달 카드 사용액",
    "committed_payment_ratio": "이번 달 상환 비율",
    "carryover_share": "리볼빙 의존도",
    "carryover_share_delta_3m": "최근 3개월간 리볼빙 의존도 변화",
    "carryover_share_slope_3m": "최근 3개월 악화 속도",
    "committed_ratio_delta_3m": "최근 3개월간 상환 비율 변화",
    "payment_ratio_gap": "최소 상환액과의 여유분",
    "minimum_payment_streak": "연속 최소결제 이용 개월수",
    "revolving_streak_months": "연속 리볼빙 이용 개월수",
    "delinquency_count_6m": "최근 6개월 연체 횟수",
    "limit_utilization_ratio": "카드 한도 대비 사용률",
    "card_limit": "카드 한도",
    "minimum_payment_ratio": "최소결제비율",
    "revolving_principal_before_payment": "이월원금과 당월 사용액의 합",
    "ending_carryover_principal": "이월원금",
    "scheduled_principal_payment": "약정원금 상환액",
    "total_payment_amount": "총 결제액",
    "revolving_fee": "리볼빙 수수료",
    "actual_principal_paid": "실제 상환원금",
    "month_index": "가입 후 경과 개월수",
    "revolving_active": "리볼빙 이용 여부",
}

# 각 피처가 "왜" 위험도 예측에 영향을 주는지 한 줄 설명. SHAP 막대그래프 옆에 표시해
# 전문용어(피처명·SHAP 등) 없이도 요인의 의미를 이해할 수 있게 한다.
FEATURE_EXPLANATIONS = {
    "billing_amount": "이번 달 쓴 금액이 많을수록 다음 달 갚아야 할 금액도 커져요.",
    "committed_payment_ratio": "이번 달 상환 비율이 낮을수록 이월되는 금액이 늘어나요.",
    "carryover_share": "지금 이미 이월된 금액이 많으면, 다음 달에도 그 흐름이 이어지기 쉬워요.",
    "carryover_share_delta_3m": "최근 몇 달간 빠르게 나빠지고 있다면, 앞으로도 그 흐름이 이어질 가능성이 커요.",
    "carryover_share_slope_3m": "최근 나빠지는 속도가 빠를수록, 앞으로도 그 속도가 이어질 가능성이 커요.",
    "committed_ratio_delta_3m": "상환 비율을 스스로 낮춰온 흐름은 앞으로도 이어질 수 있는 행동 패턴이에요.",
    "payment_ratio_gap": "최소 상환액과 지금 상환액의 차이가 적을수록, 여유 없이 빠듯하게 갚고 있다는 뜻이에요.",
    "minimum_payment_streak": "최소한만 갚는 달이 길게 이어질수록 이월 금액이 계속 쌓여요.",
    "revolving_streak_months": "리볼빙을 오래 쓸수록 그 패턴에서 벗어나기 어려워질 수 있어요.",
    "delinquency_count_6m": "최근 연체가 잦았다면 앞으로도 결제 부담이 클 가능성이 있어요.",
    "limit_utilization_ratio": "한도를 많이 쓸수록 다음 달 갚아야 할 금액도 커질 수 있어요.",
    "card_limit": "한도 자체가 클수록 청구액 규모도 함께 커질 수 있어요.",
    "minimum_payment_ratio": "카드사가 정한 최소 상환 비율이 낮을수록, 덜 갚고 넘어가기 쉬운 구조예요.",
    "revolving_principal_before_payment": "이번 달 갚아야 할 원금 총액이 클수록 부담이 커져요.",
    "ending_carryover_principal": "이월된 원금 자체가 클수록 다음 달에도 그 영향이 이어져요.",
    "scheduled_principal_payment": "약정한 만큼 갚는 원금이 적을수록 이월액이 늘어나요.",
    "total_payment_amount": "이번 달 총 결제액이 적을수록 이월되는 금액이 늘어날 수 있어요.",
    "revolving_fee": "이월잔액에 붙는 수수료가 클수록 갚아야 할 금액이 더 늘어나요.",
    "actual_principal_paid": "실제로 갚은 원금이 적을수록 이월 금액이 늘어나요.",
    "month_index": "리볼빙을 이용해온 기간이 길수록 지금 패턴이 굳어질 가능성이 있어요.",
    "revolving_active": "리볼빙이 계속 활성화돼 있으면 이 패턴이 이어질 가능성이 커요.",
}


class CoachingValidationError(Exception):
    """코칭 메시지가 JSON Schema 또는 source 일치성 검증을 통과하지 못했을 때 발생."""


# ---------------------------------------------------------------------------
# 설정 로딩 — 로컬(os.environ)과 Streamlit Cloud(st.secrets) 양쪽 호환
# ---------------------------------------------------------------------------
# Streamlit Community Cloud에서는 API 키/플래그를 보통 st.secrets로 주입한다.
# os.environ에 이미 값이 있으면 그대로 두고(로컬 .env·쉘 환경 우선), 없을 때만
# st.secrets에서 읽어와 os.environ에 채워 넣는다. streamlit이 없거나 secrets가
# 설정되지 않은 실행 맥락(예: `python src/coaching.py`)에서도 조용히 무시된다.
_SECRET_KEYS = ("ANTHROPIC_API_KEY", "USE_MOCK_COACHING")


def _hydrate_env_from_st_secrets() -> None:
    missing = [k for k in _SECRET_KEYS if not os.environ.get(k)]
    if not missing:
        return
    try:
        import streamlit as st

        for key in missing:
            try:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
            except Exception:
                # st.secrets 접근 자체가 실패하는 맥락(secrets 파일 없음 등)
                pass
    except Exception:
        # streamlit 미설치 등 — 로컬 스크립트 실행 시 정상적인 경로
        pass


# ---------------------------------------------------------------------------
# 검증 레이어 (mock/real 공통)
# ---------------------------------------------------------------------------
def available_sources_for_context(context: dict) -> set[str]:
    """이번 호출에서 실제로 근거로 삼을 수 있는 source 집합.

    simulation 근거는 사용자가 상환 시뮬레이터를 실행했을 때만 존재하므로, context에
    simulation 결과가 없으면 "simulation" 태그를 쓰는 것 자체가 근거 없는 태깅이다.
    """
    sources = {"raw_data", "shap"}
    if context.get("simulation"):
        sources.add("simulation")
    return sources


def validate_coaching_message(message: Any, available_sources: set[str]) -> None:
    """JSON Schema 검증 + source 필드 일치성 검증. 실패 시 CoachingValidationError."""
    try:
        jsonschema.validate(instance=message, schema=COACHING_MESSAGE_SCHEMA)
    except jsonschema.ValidationError as e:
        raise CoachingValidationError(f"JSON Schema 검증 실패: {e.message}") from e

    used_sources = {seg["source"] for seg in message["segments"]}
    invalid = used_sources - available_sources
    if invalid:
        raise CoachingValidationError(
            f"허용되지 않은 source 태그 사용: {sorted(invalid)} "
            f"(이번 호출에서 사용 가능한 source: {sorted(available_sources)})"
        )


# ---------------------------------------------------------------------------
# mock 구현 — API 호출 없이, 실제 응답과 동일한 스키마의 예시 메시지를 생성
# ---------------------------------------------------------------------------
def mock_generate_coaching_message(context: dict) -> dict:
    """risk_indicator(관찰/주의/경고/심화)별로 그럴듯한 문장을 만들되, 실제 context에 담긴
    숫자(raw_data)와 SHAP 상위 피처를 그대로 문장에 반영해 페르소나별로 결과가 달라지게 한다.

    persona_tier로 분기하지 않는 이유: 실제 서비스에서는 페르소나 라벨이 없는 사용자
    직접입력값도 들어오므로, risk_indicator + 실측값 기반으로 생성해야 목업이 real
    구현과 동일한 입력 인터페이스로 모든 케이스(샘플 페르소나/직접 입력 모두)에서 동작한다.
    """
    risk = context.get("risk_indicator", "관찰")
    carryover_share = context.get("carryover_share")
    carryover_share_delta_3m = context.get("carryover_share_delta_3m")
    payment_ratio_gap = context.get("payment_ratio_gap")
    top_shap_features = context.get("top_shap_features") or []
    simulation = context.get("simulation")
    current_carryover_share = context.get("current_carryover_share")
    outlook = context.get("outlook") or []

    segments = []

    if current_carryover_share is not None and carryover_share is not None:
        segments.append(
            {
                "text": (
                    f"이번 달 리볼빙 의존도는 {current_carryover_share * 100:.1f}%이고, "
                    f"다음 달에는 {carryover_share * 100:.1f}%가 될 것으로 예측돼요."
                ),
                "source": "raw_data",
            }
        )
    elif carryover_share is not None:
        segments.append(
            {"text": f"다음 달 리볼빙 의존도는 {carryover_share * 100:.1f}%가 될 것으로 예측돼요.", "source": "raw_data"}
        )

    # SHAP 상위 요인은 최대 2개까지 근거 문장으로 반영 (실제 API 호출 시에는 이보다
    # 더 풍부하게 여러 개를 반영할 수 있음 — mock은 데모용 최소 예시)
    for top in top_shap_features[:2]:
        label = FEATURE_LABELS.get(top["feature"], top["feature"])
        direction = "높이는" if top.get("contribution", 0) > 0 else "낮추는"
        segments.append(
            {
                "text": f"최근 행동 패턴 중 '{label}' 항목이 다음 달 전망에 {direction} 방향으로 가장 크게 작용하고 있어요.",
                "source": "shap",
            }
        )

    if len(outlook) >= 3:
        last = outlook[-1]
        segments.append(
            {
                "text": (
                    f"이 흐름이 계속되면 {last['month_offset']}개월 후에는 위험 단계가 "
                    f"'{last['level']}'(리볼빙 의존도 약 {last['carryover_share'] * 100:.1f}%)까지 갈 수 있는 것으로 예측돼요."
                ),
                "source": "raw_data",
            }
        )

    level_templates = {
        "관찰": "현재는 결제 패턴이 안정적으로 유지되고 있어요. 지금처럼만 유지하시면 좋습니다.",
        "주의": (
            f"리볼빙 의존도가 최근 3개월간 {carryover_share_delta_3m * 100:+.1f}%p 상승하는 추세예요. "
            "이 흐름이 이어지면 부담이 커질 수 있으니 결제 비율을 조금 더 높여보는 걸 권해드려요."
        )
        if carryover_share_delta_3m is not None
        else "리볼빙 의존도가 상승하는 추세예요. 결제 비율을 조금 더 높여보는 걸 권해드려요.",
        "경고": (
            f"결제여유(약정결제비율-최소결제비율)가 {payment_ratio_gap * 100:.1f}%p까지 좁혀졌고 "
            "상승 추세도 계속되고 있어요. 지금 상환액을 늘리면 효과가 큰 구간입니다."
        )
        if payment_ratio_gap is not None
        else "결제여유가 좁아지고 상승 추세가 계속되고 있어요. 지금 상환액을 늘리면 효과가 큰 구간입니다.",
        "심화": "최근 3개월 이상 최소결제 수준만 반복하고 계세요. 이 패턴이 오래 지속되면 이월잔액이 계속 불어날 수 있어 지금 개입이 중요합니다.",
    }
    segments.append({"text": level_templates.get(risk, level_templates["관찰"]), "source": "raw_data"})

    if simulation:
        extra_payment = simulation.get("extra_payment", 0)
        new_share = simulation.get("new_predicted_carryover_share")
        new_risk = simulation.get("new_risk_indicator")
        if new_share is not None and new_risk is not None:
            segments.append(
                {
                    "text": (
                        f"매달 {extra_payment:,.0f}원을 추가로 상환하면, 다음 달 예상 리볼빙 의존도가 "
                        f"{new_share * 100:.1f}%로 낮아지고 경고 단계는 '{new_risk}'로 바뀔 것으로 예상돼요."
                    ),
                    "source": "simulation",
                }
            )

    return {"segments": segments}


# ---------------------------------------------------------------------------
# real 구현 — Claude Haiku 4.5 호출 (API 키 필요, UI 완성 후 전환 예정)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """당신은 '오뚝이'라는 리볼빙(일부결제금액이월약정) 조기경보 서비스의 코칭 메시지 작성자입니다.
사용자에게 제공된 실측_데이터(현재_상태 포함), 예측에_영향을_준_요인(모델이 다음 달을 예측할 때
반영한 요인들 — 보통 여러 개가 함께 옵니다), 향후_3개월_전망(월별 예측 궤적),
상환_시뮬레이션_결과(추가 상환 시 예상 변화, 있을 때만) 근거를 바탕으로 공감적이고 실행 가능한
코칭 문장을 작성하세요.

핵심 지시 — 종합적이지만 정돈된 개인 맞춤 코칭:
이 사용자만을 위한 코칭이 되도록, 받은 근거 항목들을 따로따로 나열하지 말고 하나의 이야기처럼
엮어서 설명하세요. 화면에는 segment 하나가 카드 하나로 표시되므로, 관련된 내용(예: 현재 상태와
그 이유가 되는 요인 여러 개)은 여러 segment로 잘게 쪼개지 말고 하나의 segment 안에 자연스러운
문단으로 함께 녹여서 설명하세요.
전체 흐름은 대략 다음 3~5개 segment로 구성하는 것을 기준으로 삼으세요 (근거가 부족하면 더 적어도
됩니다):
  1) raw_data: 현재 상태 + 다음 달 예측을 함께 묶어 설명
  2) shap: 예측에_영향을_준_요인들을 원인으로 묶어 설명 (요인이 여러 개면 한 문단 안에서 함께 언급)
  3) raw_data: 향후_3개월_전망을 반영해 이대로면 어떻게 되는지 설명
  4) simulation: 상환_시뮬레이션_결과가 있을 때만, 그걸 반영한 행동 제안
segment를 과도하게 쪼개 카드가 여러 개로 나열되면 오히려 읽기 피곤해지므로, 위 기준보다 많은
segment를 만들지 마세요. 각 segment는 반드시 아래 source 규칙에서 벗어나지 않는 범위 안에서만
작성하세요 (근거 없는 내용을 지어내지 마세요).

문체 지침 (중요 — 이걸 지키지 않으면 형식만 맞고 품질은 실패한 것입니다):
- 입력값은 이미 사람이 읽을 수 있는 한국어 라벨과 형식(%, 원 단위 등)으로 정리되어 있습니다.
  이 표현을 그대로 문장에 녹여 쓰세요. 영문 변수명이나 코드 용어는 절대 노출하지 마세요.
- 숫자를 단순히 재진술하지 말고, "그게 무슨 의미인지"·"왜 중요한지"까지 한 문장 안에서 풀어
  설명하세요.
  나쁜 예(숫자 재진술만 함): "리볼빙 의존도가 42%입니다."
  좋은 예(의미까지 풀어씀): "이번 달 카드값의 42%를 다 못 갚고 다음 달로 넘기고 계세요."
- 금융을 잘 모르는 사람도 한 번에 이해할 수 있는 쉬운 말로 쓰세요. 전문용어를 쓸 땐 바로 뒤에
  괄호 없이 자연스럽게 풀어 설명을 덧붙이세요.
- 딱딱한 보고서·안내문 톤이 아니라, 옆에서 챙겨주는 사람처럼 친근하고 공감하는 대화체로
  쓰세요 (예: "~하고 계세요", "~해보는 건 어떨까요", "~일 수 있어요").
- 각 segment는 1~2문장으로, 사실 전달과 "그래서 어떻다는 건지"를 함께 담은 하나의 완결된
  생각 단위로 쓰세요. 너무 짧게 끊어 단답형으로 만들지 마세요.

반드시 지켜야 할 규칙:
1. 출력은 아래 JSON 스키마를 따르는 JSON 객체 하나만 반환하세요. 다른 설명, 마크다운 코드펜스 없이 순수 JSON만 출력합니다.
   {"segments": [{"text": "문장", "source": "raw_data|shap|simulation"}]}
2. 각 segment의 source는 그 문장이 실제로 어떤 근거에서 나온 내용인지와 정확히 일치해야 합니다.
   실측_데이터·현재_상태·향후_3개월_전망을 인용·설명한 문장은 raw_data, 예측에_영향을_준_요인을
   설명한 문장은 shap, 상환_시뮬레이션_결과를 설명한 문장은 simulation으로 태깅하세요.
3. 입력에 상환_시뮬레이션_결과가 없으면 simulation을 근거로 하는 문장을 만들지 마세요.
4. "신용점수", "신용등급", "위험도 O점" 같은 표현은 쓰지 말고, 관찰/주의/경고/심화 4단계 용어만 사용하세요.
5. 특정 대출·카드 상품을 추천하지 마세요. 상환 행동에 대한 코칭에만 집중하세요.

스타일 참고용 예시 (실제 출력에 이 예시 자체를 포함하지 마세요):
입력 예: {"위험_단계": "경고", "실측_데이터": {"결제여유(약정결제비율이 최소결제비율보다 얼마나 여유있는지)": "3.0%p"}}
좋은 출력 문장 예: "지금 약정한 결제 비율이 카드사가 정한 최소 기준에 거의 다다랐어요 — 여유가 3%p밖에 안 남았거든요. 이 상태가 이어지면 매달 최소한만 갚는 패턴으로 굳어질 수 있어요."
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()
    return json.loads(text)


def real_generate_coaching_message(context: dict) -> dict:
    """Claude Haiku 4.5를 호출해 근거 태깅 JSON 코칭 메시지를 생성한다.

    ANTHROPIC_API_KEY 환경변수가 필요하다. UI/UX 완성 전까지는 사용하지 않는다
    (generate_coaching_message()가 USE_MOCK_COACHING=true일 때는 이 함수를 호출하지 않음).
    """
    import anthropic

    _hydrate_env_from_st_secrets()
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용 (없으면 st.secrets에서 주입됨)

    available_sources = available_sources_for_context(context)

    # Claude에게 넘기는 입력 자체를 사람이 읽을 수 있는 한국어 라벨·형식(%, 원)으로
    # 미리 번역/포맷팅해서 보낸다. 이전에는 payment_ratio_gap 같은 영문 변수명을
    # 그대로 넘겼는데, 그러면 모델이 그 용어를 그대로 문장에 노출하거나(혹은 대충
    # 번역만 하고) 단순 재진술에 머무는 경향이 있었다. 입력부터 "무엇에 대한
    # 숫자인지"가 분명해야 출력도 그만큼 자연스러워진다.
    def _pct(x, signed=False):
        if x is None:
            return None
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if x != x:  # NaN
            return None
        return f"{x*100:+.1f}%p" if signed else f"{x*100:.1f}%"

    raw_data_kr = {
        "현재_상태": {
            "현재_위험_단계": context.get("current_risk_indicator"),
            "현재_리볼빙_의존도": _pct(context.get("current_carryover_share")),
            "현재_연속_최소결제_개월수": context.get("current_streak"),
        },
        "다음_달_예측": {
            "예측_리볼빙_의존도(카드값 중 다음 달로 이월될 것으로 예측되는 비율)": _pct(context.get("carryover_share")),
            "최근_3개월간_리볼빙_의존도_변화": _pct(context.get("carryover_share_delta_3m"), signed=True),
            "결제여유(약정결제비율이 최소결제비율보다 얼마나 여유있는지)": _pct(context.get("payment_ratio_gap")),
        },
    }

    outlook_kr = [
        {"몇_개월_후": o.get("month_offset"), "예측_위험_단계": o.get("level"), "예측_리볼빙_의존도": _pct(o.get("carryover_share"))}
        for o in (context.get("outlook") or [])
        if o.get("month_offset", 0) > 0
    ]

    shap_kr = []
    for feat in context.get("top_shap_features") or []:
        label = FEATURE_LABELS.get(feat.get("feature"), feat.get("feature"))
        direction = "다음 달 위험을 높이는" if (feat.get("contribution") or 0) > 0 else "다음 달 위험을 낮추는"
        shap_kr.append({"요인": label, "영향_방향": direction})

    simulation_kr = None
    sim = context.get("simulation")
    if sim:
        simulation_kr = {
            "추가_상환액": f"{sim.get('extra_payment', 0):,.0f}원",
            "적용_후_예상_리볼빙_의존도": _pct(sim.get("new_predicted_carryover_share")),
            "적용_후_위험_단계": sim.get("new_risk_indicator"),
        }

    user_payload = {
        "위험_단계": context.get("risk_indicator"),
        "실측_데이터": raw_data_kr,
        "향후_3개월_전망": outlook_kr,
        "예측에_영향을_준_요인": shap_kr,
        "상환_시뮬레이션_결과": simulation_kr,
        "allowed_sources": sorted(available_sources),
    }

    response = client.messages.create(
        model=MODEL_NAME,
        # 1024였을 때 고위험 페르소나처럼 근거가 많은(요인 여러 개 + 3개월 전망 + 심화 서술)
        # 케이스에서 응답이 중간에 잘려 JSON 파싱이 실패하는 문제가 있었다. 컨텍스트가 더
        # 풍부해진 만큼 여유를 두되, 프롬프트에서도 segment를 과도하게 늘리지 말라고 함께
        # 지시해 두었다(위 SYSTEM_PROMPT의 "핵심 지시" 참고).
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    )
    stop_reason = getattr(response, "stop_reason", None)
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        return _extract_json(text)
    except json.JSONDecodeError as e:
        hint = " (max_tokens 한도에 도달해 응답이 중간에 잘렸을 가능성이 높습니다)" if stop_reason == "max_tokens" else ""
        raise CoachingValidationError(f"Claude 응답을 JSON으로 파싱하지 못했습니다{hint}: {e}") from e


# ---------------------------------------------------------------------------
# 진입점 — 환경변수 USE_MOCK_COACHING으로 mock/real 전환, 검증은 공통 적용
# ---------------------------------------------------------------------------
def generate_coaching_message(context: dict, use_mock: bool | None = None) -> dict:
    if use_mock is None:
        _hydrate_env_from_st_secrets()
        use_mock = os.environ.get("USE_MOCK_COACHING", "true").strip().lower() != "false"

    message = mock_generate_coaching_message(context) if use_mock else real_generate_coaching_message(context)

    available_sources = available_sources_for_context(context)
    validate_coaching_message(message, available_sources)
    return message


if __name__ == "__main__":
    sample_contexts = [
        {
            "risk_indicator": "관찰",
            "carryover_share": 0.091,
            "carryover_share_delta_3m": 0.002,
            "payment_ratio_gap": 0.62,
            "top_shap_features": [{"feature": "committed_payment_ratio", "contribution": 0.25}],
            "simulation": None,
        },
        {
            "risk_indicator": "주의",
            "carryover_share": 0.288,
            "carryover_share_delta_3m": 0.147,
            "payment_ratio_gap": 0.35,
            "top_shap_features": [{"feature": "carryover_share_slope_3m", "contribution": -0.09}],
            "simulation": None,
        },
        {
            "risk_indicator": "경고",
            "carryover_share": 0.435,
            "carryover_share_delta_3m": 0.209,
            "payment_ratio_gap": 0.045,
            "top_shap_features": [{"feature": "payment_ratio_gap", "contribution": -0.06}],
            "simulation": {
                "extra_payment": 150000,
                "new_predicted_carryover_share": 0.31,
                "new_risk_indicator": "주의",
            },
        },
        {
            "risk_indicator": "심화",
            "carryover_share": 0.769,
            "carryover_share_delta_3m": 0.05,
            "payment_ratio_gap": 0.0,
            "top_shap_features": [{"feature": "minimum_payment_streak", "contribution": 0.15}],
            "simulation": None,
        },
    ]

    print(f"USE_MOCK_COACHING={os.environ.get('USE_MOCK_COACHING', '(미설정, 기본 mock)')}")
    for ctx in sample_contexts:
        msg = generate_coaching_message(ctx)
        print(f"\n[risk_indicator={ctx['risk_indicator']}]")
        print(json.dumps(msg, ensure_ascii=False, indent=2))

    # 검증 실패 케이스 데모: simulation 근거가 없는데 simulation 태그를 붙인 경우
    print("\n[검증 실패 케이스 데모: simulation 근거 없이 simulation 태그 사용]")
    bad_message = {"segments": [{"text": "무단으로 시뮬레이션 근거를 주장하는 문장", "source": "simulation"}]}
    try:
        validate_coaching_message(bad_message, available_sources_for_context({"simulation": None}))
        print("검증 통과 (예상과 다름 - 버그)")
    except CoachingValidationError as e:
        print(f"예상대로 검증 실패: {e}")
