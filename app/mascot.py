"""오뚝이 마스코트 상태 시각화 컴포넌트.

assets/mascot/state_mapping.json 을 읽어 risk_indicator -> 캐릭터 이미지를 표시한다.
코드가 파일명의 의미를 추측하지 않는다 — 매핑은 전적으로 그 JSON 파일이 정한다.

risk_indicator 가 바뀌면(예: 시뮬레이션 실행 후) 이미지가 부드러운 페이드로 전환된다.
매핑 파일이나 이미지 파일이 없으면 이미지를 생략하고 안내 문구만 표시한다(앱은 안 깨진다).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st

import theme

BASE_DIR = Path(__file__).resolve().parent.parent
MASCOT_DIR = BASE_DIR / "assets" / "mascot"
MAPPING_PATH = MASCOT_DIR / "state_mapping.json"

# risk_indicator 는 4개(관찰/주의/경고/심화). '안정'은 매핑에만 있는 선택 항목.
# recovery_score 가 이 값 이상인 '관찰' 상태면 '안정' 이미지를 대신 쓴다.
STABLE_RECOVERY_CUTOFF = 90.0

STATE_MESSAGES = {
    "안정": "지금은 아주 안정적인 상태예요. 지금 흐름을 유지하시면 됩니다.",
    "관찰": "아직 안정적인 상태예요. 지금처럼만 유지하시면 됩니다.",
    "주의": "조금씩 리볼빙 의존도가 올라오고 있어요. 지금 결제 방식을 살펴볼 좋은 시점이에요.",
    "경고": "결제 여유가 얼마 남지 않았어요. 지금 상환 방식을 바꾸면 효과가 큰 구간이에요.",
    "심화": "지금은 어려운 상태지만, 아직 방향을 바꿀 수 있어요. 작은 변화부터 함께 살펴봐요.",
}

_FALLBACK_ORDER = ["안정", "관찰", "주의", "경고", "심화"]

# 마스코트 옆 제목: 기본 대비 1.5배, 마스코트 몸통색(딥 티얼)으로 통일.
ACCENT_TITLE_SIZE = "1.72rem"   # 기존 1.15rem x 1.5
ACCENT_BIG_TITLE_SIZE = "1.88rem"  # 기존 1.25rem x 1.5
ACCENT_TITLE_COLOR = theme.BRAND  # #0F4C4C — 오뚝이 몸통 딥 티얼


@st.cache_data
def _load_mapping() -> dict:
    if not MAPPING_PATH.exists():
        return {}
    try:
        return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data
def _img_data_uri(path_str: str) -> str | None:
    p = Path(path_str)
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def resolve_state(risk_indicator: str, recovery_score: float | None = None) -> str:
    """표시할 상태 키. '관찰'이면서 recovery_score 가 매우 높으면 '안정'으로 승격."""
    if risk_indicator == "관찰" and recovery_score is not None and recovery_score >= STABLE_RECOVERY_CUTOFF:
        return "안정"
    return risk_indicator if risk_indicator in _FALLBACK_ORDER else "관찰"


def image_uri_for_state(state: str) -> str | None:
    mp = _load_mapping()
    states = mp.get("states", {})
    base_dir = MASCOT_DIR / mp.get("base_dir", "all_elements")
    # 요청 상태 -> 없으면 fallback 순서상 가장 가까운(덜 심각한) 상태로
    order = _FALLBACK_ORDER
    start = order.index(state) if state in order else 1
    for k in [state] + order[start::-1] + order[start:]:
        fname = states.get(k)
        if fname:
            uri = _img_data_uri(str(base_dir / fname))
            if uri:
                return uri
    return None


def render(
    risk_indicator: str,
    *,
    recovery_score: float | None = None,
    caption: str | None = None,
    size_px: int = 150,
    key: str = "main",
    fill_height: bool = False,
    pad: str = "1.1rem 1.3rem",
) -> None:
    """마스코트 이미지 + 상태 문구. risk_indicator 변경 시 페이드 전환.

    fill_height=True: 카드가 부모(st.columns) 높이를 꽉 채우도록 -> 옆 카드와 높이 정렬.
    """
    state = resolve_state(risk_indicator, recovery_score)
    uri = image_uri_for_state(state)
    msg = caption or STATE_MESSAGES.get(state, STATE_MESSAGES["관찰"])
    color = theme.RISK_COLORS.get(risk_indicator, theme.RISK_COLORS["관찰"])

    if uri:
        img_html = (
            f'<img src="{uri}" alt="오뚝이 {state}" '
            f'style="width:{size_px}px;height:auto;display:block;'
            f'animation:ottugi-fade-{state}-{key} 0.5s ease;" />'
            f'<style>@keyframes ottugi-fade-{state}-{key}'
            f'{{from{{opacity:0;transform:translateY(6px);}}to{{opacity:1;transform:translateY(0);}}}}</style>'
        )
    else:
        img_html = (
            f'<div style="width:{size_px}px;height:{size_px}px;border-radius:16px;'
            f'background:{color["bg"]};border:2px solid {color["border"]};display:flex;'
            f'align-items:center;justify-content:center;font-size:2rem;">🪆</div>'
        )

    h = "height:100%;box-sizing:border-box;" if fill_height else ""
    st.markdown(
        theme.compact_html(f"""
        <div style="display:flex;gap:0.9rem;align-items:center;flex-wrap:wrap;{h}
                    background:{theme.SURFACE};border:1px solid {theme.LINE};
                    border-left:5px solid {color["main"]};border-radius:16px;padding:{pad};">
            <div style="flex-shrink:0;">{img_html}</div>
            <div style="flex:1;min-width:150px;">
                <div style="font-weight:900;font-size:1.1rem;color:{color["main"]};">{state}</div>
                <div style="color:{theme.INK};font-size:0.9rem;margin-top:3px;line-height:1.45;">{msg}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


@st.cache_data
def _accent_uri(key: str) -> str | None:
    mp = _load_mapping()
    fname = mp.get("accents", {}).get(key)
    if not fname:
        return None
    base_dir = MASCOT_DIR / mp.get("base_dir", "all_elements")
    return _img_data_uri(str(base_dir / fname))


def accent(key: str, *, size_px: int = 64) -> str:
    """제목 옆에 붙이는 작은 마스코트(표정/몸짓) HTML 문자열. 파일 없으면 빈 문자열.

    key: state_mapping.json 의 accents 키 (greeting/smile/cheer/focus/worry/surprise/
         shaky/steady/analyze/report/strategy/applaud).
    """
    uri = _accent_uri(key)
    if not uri:
        return ""
    return (f'<img src="{uri}" alt="오뚝이 {key}" '
            f'style="width:{size_px}px;height:auto;display:block;flex-shrink:0;" />')


def render_accent(key: str, *, size_px: int = 64) -> None:
    html = accent(key, size_px=size_px)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def state_img(risk_indicator: str, *, size_px: int = 44, recovery_score: float | None = None) -> str:
    """risk_indicator 에 대응하는 상태 캐릭터(회전각 다른 row1 png) <img> HTML. 없으면 "".
    라벨과 값 사이에 끼워 넣는 용도."""
    uri = image_uri_for_state(resolve_state(risk_indicator, recovery_score))
    if not uri:
        return ""
    return (f'<img src="{uri}" alt="오뚝이 {risk_indicator}" '
            f'style="width:{size_px}px;height:auto;display:inline-block;vertical-align:middle;flex-shrink:0;" />')


def section_with_accent(title: str, desc: str = "", *, accent_key: str = "", size_px: int = 52) -> None:
    """섹션 제목 **왼쪽**에 작은 마스코트가 붙은 헤더 (png -> 제목 순)."""
    acc = accent(accent_key, size_px=size_px) if accent_key else ""
    desc_html = f'<div style="color:{theme.SUBTLE};font-size:0.9rem;margin-top:2px;">{desc}</div>' if desc else ""
    st.markdown(
        theme.compact_html(f"""
        <div style="margin:1.3rem 0 0.5rem 0;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                {acc}<div style="font-weight:900;font-size:{ACCENT_TITLE_SIZE};color:{ACCENT_TITLE_COLOR};
                                 letter-spacing:-0.01em;line-height:1.15;">{title}</div>
            </div>{desc_html}
        </div>
        """),
        unsafe_allow_html=True,
    )


def big_title_with_accent(label: str, *, accent_key: str = "", accent_color: str | None = None, size_px: int = 52) -> None:
    """theme.big_section_title 배지 **왼쪽**에 마스코트 (png -> 배지 순). 텍스트는 몸통색으로 통일."""
    color = ACCENT_TITLE_COLOR  # accent_color 무시하고 마스코트 몸통색으로 통일
    acc = accent(accent_key, size_px=size_px) if accent_key else ""
    badge = (f'<div style="display:inline-block;background:{color}14;color:{color};font-weight:900;'
             f'font-size:{ACCENT_BIG_TITLE_SIZE};padding:0.45rem 1.1rem;border-radius:12px;border:1px solid {color}3a;'
             f'letter-spacing:-0.01em;line-height:1.15;">{label}</div>')
    st.markdown(
        theme.compact_html(f'<div style="display:flex;align-items:center;gap:0.6rem;margin:1.3rem 0 0.7rem 0;">{acc}{badge}</div>'),
        unsafe_allow_html=True,
    )


def sidebar_greeting(size_px: int = 118) -> None:
    """사이드바 워드마크 아래 큰 인사 캐릭터(말풍선 포함)."""
    uri = _accent_uri("greeting")
    if uri:
        st.markdown(
            f'<img src="{uri}" alt="오뚝이 인사" style="width:{size_px}px;height:auto;'
            f'display:block;margin:0.2rem auto 0.8rem auto;" />',
            unsafe_allow_html=True,
        )


def mapping_status() -> dict:
    """전문가 화면에서 매핑 상태를 보여주기 위한 헬퍼."""
    mp = _load_mapping()
    states = mp.get("states", {})
    base_dir = MASCOT_DIR / mp.get("base_dir", "all_elements")
    resolved = {}
    for k, fname in states.items():
        p = base_dir / fname if fname else None
        resolved[k] = {"file": fname, "exists": bool(p and p.exists())}
    accents = {}
    for k, fname in mp.get("accents", {}).items():
        p = base_dir / fname if fname else None
        accents[k] = {"file": fname, "exists": bool(p and p.exists()),
                      "추정": mp.get("accent_guess", {}).get(k, "")}
    return {
        "mapping_file_found": MAPPING_PATH.exists(),
        "needs_confirmation": mp.get("needs_confirmation", False),
        "confidence_note": mp.get("confidence_note", ""),
        "states": resolved,
        "accents": accents,
    }
