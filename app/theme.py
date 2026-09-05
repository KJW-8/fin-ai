"""오뚝이(Ottugi) 디자인 시스템 — CSS + 재사용 컴포넌트.

이 모듈은 순수하게 화면 표현(styling/rendering)만 다룬다. 예측·SHAP·시뮬레이션·
위험판정 등 핵심 비즈니스 로직은 전혀 포함하지 않으며, 오직 계산된 값을 어떻게
보여줄지만 결정한다.
"""

from __future__ import annotations

import re

import streamlit as st

# 오뚝이 워드마크 옆 요가(나무 자세) 아이콘 PNG (64x64, 흰색+투명배경) — base64 인라인.
# yoga_icon_svg()에서 mask-image로 색을 입혀 재사용한다.
YOGA_ICON_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAABINJREFUeJztm12IFlUYx5+jab7maquGfWBU20Ik2IcKfdBWF1JRZkJgJBREXVdUF0HZ14XkRdSN20UXfV1UQgVBBZUaqaDVjRqpW0vmElpqC0trtLv9upjZnH3ec953Zt6ZZ153/cHCzpn/Oed5/u95Z+acOa+IEcAFwDbgKs+5+4HPgZVW8ZgCLAEGgD2B8wuB34l4F5hrHWNpAEsTyd3ZQHc7MBbr+oAuyzhLAehOJP9ZCv2znOIIcI1FnKUAnAccjJMZAZakqOOADxMmDAJLLeItFKAG7E4k8lqGup3AoUTdQ8BFZcZbOMA76lNcmLH+jfGoGWcXcFZZ8RYK8CgTedqjqQG3AQ8CPcB0j+YV1c7zJgm0AtEV/+9E0APAbKVZHV/gkuwHrlW6OcDhhGYEuNo2owwAM4E9KrFHlOZmNbSTHAcuVvrHleYD26wyADymgj0B1JRmeyD5cXqVfi4wlDg/CnTbZpaCONDjKpnXlWYWpx50QhzwtP2p0qwvMvZpBbWzRkTmq7Lv1PGMFP2d7SkbUMfLM8TVlKIMWOQpG00eOOeGRKTuE1Z86ynTQ/6cDHE1pSgDfvaUPeAp29CgjVER2ZgsAC4XkR6l258tNAOAxYHv930e7Qse7TCwzqP9xNPmvTZZZQTY4gn2KHC+R3sl8AywCXgCuNCjWeNp7zdghk1GGQHWegIG2AHMzNhWN9FtVPNUWfG3DDCN+gehcd4GXMp2FgA/eNroB3x3ifYBuCNgAEQrPT1AFzCfaNaX/FsMrAT2Buqvqjq/VADvNzAhL29VnVdqgEXUT3ZaoR/orDqvTAC3Ej23t8pJYFnV+eQCeKkAAx6qOo/cANOBL1pIvrd5L20OcC4T1/fSsh2jJbCi5gJenHODIvJLjqoHnHOjzWWtU6oBMakegBSzm0uK4YwBBn3kMaDQOX8jLAzI00ce03LRriPAjDMGTJI+ctOuwa0AVlt01K4joENEPgJeJOVCStsCfN/CfADgPWBWWfFZjICRFuuvFZGtgO/dQ8tYGHCygDauE5GdwCUFtDUBCwOGG5zbICKbRGQwRTuXicjXnG6bp4DNDb7fy2NNDVhHtI+wGT+RccdJpQC9gUT+BeZ49DdQ/0ZYs4XT5e4APBdIQr/11fVWxSaFuKuI+CyuAUcD5YdDFYj2DPVI+DF6TETq9hW1JcA9gU9wc0B/BdHrtBB9wPXWeeQGWBZI5FWl6wA2Av8E9ENEO0nNFksKgWhh1Mf6hOZuoh1lPk4AL1PSg5AJwDFPYk8SLZ2/6Tk3BmwFHgZKXR2y2n3ZLyILVNlfItIpIl3x+QER+VFEvhGRr5xzRywCszKgT0RWqLJh59wxEbnJKAYvVusB+yrsuyFWQfh+KdJh1HdDrAzY6ymbOgY4534VkT9V8dQxIEZvgpxn2HcQSwN2quO22PVhacAOdaz3FleCpQG7ZOL+4allQLxZOjkKppYBMR8n/q8FVZMV4NLEhCfNQujkA/gyNuCPqmOpBOCW2ICDVcciYjcb/B/n3DbgDVG/KKmK/wDmVwLEy7ToaAAAAABJRU5ErkJggg=="
)

# ---------------------------------------------------------------------------
# 색상 토큰
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "관찰": {"main": "#1a7f5a", "bg": "#e6f4ee", "border": "#b7e0cd"},   # 안정적인 녹색
    "주의": {"main": "#a16a00", "bg": "#fff6e0", "border": "#f3d98b"},   # 노란색 계열
    "경고": {"main": "#c25a00", "bg": "#fff0e2", "border": "#f5c397"},   # 주황색 계열
    "심화": {"main": "#b3261e", "bg": "#fdeceb", "border": "#f2b8b5"},   # 빨간색 계열
}
RISK_ORDER = {"관찰": 0, "주의": 1, "경고": 2, "심화": 3}

SOURCE_META = {
    "raw_data": {"label": "실제 입력 데이터", "color": "#2563a8", "icon": "•"},
    "shap": {"label": "예측에 영향을 준 요인", "color": "#7a4fb5", "icon": "•"},
    "hazard": {"label": "위험 전환 전망", "color": "#b06a1a", "icon": "•"},
    # 예전엔 "#1a7f5a"(=RISK_COLORS["관찰"]과 동일한 값)를 썼는데, 브랜드 컬러 정비 과정에서
    # "상태 신호(위험도) 색상"과 "그 외 용도" 색상이 우연히 겹치던 것을 발견해 브랜드
    # 컬러로 바꿔 분리했다.
    "simulation": {"label": "상환 시뮬레이션 결과", "color": "#0F4C4C", "icon": "•"},
}

# AI 코칭 페이지에서 segment(source)별로 큰 섹션 제목을 붙일 때 쓰는 라벨.
# 같은 source가 여러 개면 호출부에서 뒤에 번호를 붙여 구분한다(예: "위험에 영향을 준 요인 2").
SOURCE_SECTION_TITLES = {
    "raw_data": "현재 상황",
    "shap": "위험에 영향을 준 요인",
    "simulation": "상환 시뮬레이션 결과",
}

BRAND = "#0F4C4C"        # 딥 티얼(deep teal) — "흔들려도 중심을 되찾는다"는 오뚝이의 회복탄력성
                          # 이미지에 신뢰감 있는 금융 톤을 더한 이 서비스만의 시그니처 컬러.
                          # RISK_COLORS(관찰=녹색 #1a7f5a 등 4단계 상태 신호)와는 다른 역할이므로
                          # 색상표에서 의도적으로 겹치지 않게 골랐다: 브랜드=차분한 딥 티얼(서비스
                          # 정체성), 위험도=신호등형 4색(관찰~심화, 상태 신호). 버튼·헤더·강조
                          # 텍스트 등 "브랜드"용으로만 쓰고, 위험 단계를 나타내는 곳에는 절대
                          # 쓰지 않는다.
BRAND_SOFT = "#e5efee"   # 브랜드 컬러의 옅은 톤 (배지·hover 배경 등에 사용)
INK = "#1c2226"
SUBTLE = "#6b7580"
LINE = "#e4e8eb"
SURFACE = "#ffffff"
PAGE_BG = "#f2f4f7"      # 라이트 그레이 (배경)

# 사이드바: 딥 티얼로 뒀더니 마스코트 몸통색(#0F4C4C)과 겹쳐 캐릭터가 배경에 묻혀서,
# 흰끼가 많이 도는 옅은 청록/연두 톤으로 바꿨다. 밝은 배경이라 사이드바 텍스트는
# 어두운 톤을 쓰고, 선택된 nav 탭만 딥 티얼로 강조한다.
SIDEBAR_BG = "#dcebe4"           # 옅은 세이지-청록 (washed teal-green)
SIDEBAR_TEXT = "#173a37"         # 어두운 티얼 (밝은 배경에서 가독)
SIDEBAR_TEXT_MUTED = "#5b7b75"
SIDEBAR_HOVER = "rgba(15,76,76,0.07)"
SIDEBAR_LINE = "rgba(15,76,76,0.13)"
SIDEBAR_SELECTED_BG = BRAND       # 선택 탭 = 딥 티얼
SIDEBAR_SELECTED_TEXT = "#ffffff"

# 브랜드 워드마크("오뚝이") 전용 서체. 마스코트 로고(assets/mascot/all_elements/label_08.png
# = "뚜기")의 두툼한 라운드 고딕 손글씨 느낌에 가장 근접한 무료 웹폰트로 Google Fonts의
# 'Black Han Sans'(검은고딕)를 쓴다. 손으로 다듬은 로고라 완전히 같지는 않지만 무게감·형태가
# 비슷하다. 폰트 로딩 실패 시 시스템 고딕으로 폴백한다. 단일 웨이트(400)이므로 font-weight는
# 지정하지 않는다.
LOGO_FONT = '"Black Han Sans", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif'


def compact_html(html: str) -> str:
    """Streamlit의 st.markdown(unsafe_allow_html=True)은 내부적으로 CommonMark 파서를
    거치는데, raw HTML 블록 안에 빈 줄(공백만 있는 줄)이 하나라도 있으면 그 지점에서
    HTML 인식이 끊기고 이후 내용이 그대로 텍스트로 노출된다(예: "</div>"가 화면에 그대로
    찍힘). f-string에 끼워 넣는 값이 빈 문자열이면 그 값만 있던 줄이 통째로 빈 줄이
    되어 버리므로, 모든 컴포넌트가 이 함수로 반환값을 감싸 빈 줄을 제거한다."""
    return "\n".join(line for line in html.split("\n") if line.strip() != "")


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&display=swap');
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
                "Malgun Gothic", "Segoe UI", sans-serif;
        }}
        /* 브랜드 워드마크("오뚝이") — 마스코트 로고 서체에 근접한 라운드 고딕 */
        .ottugi-wordmark {{
            font-family: {LOGO_FONT};
            letter-spacing: 0.03em;
        }}
        .stApp {{
            background: {PAGE_BG};
        }}
        /* 콘텐츠 최대 폭 + 여백 */
        .block-container {{
            max-width: 1080px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }}
        p, li, span, div {{
            line-height: 1.6;
        }}
        /* 기본 Streamlit 위젯 라벨 톤 다운 */
        [data-testid="stWidgetLabel"] p {{
            color: {SUBTLE};
            font-size: 0.86rem;
            font-weight: 600;
        }}
        /* 사이드바 — 진한 브랜드 컬러 배경으로 "안정감 있게 잡히는" 느낌을 준다 */
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG};
            border-right: 1px solid {SIDEBAR_LINE};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.6rem;
        }}
        /* 사이드바 전반 글씨 크기 확대 + 밝은 텍스트 색(어두운 배경 대비) */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {{
            font-size: 1.02rem !important;
            color: {SIDEBAR_TEXT};
        }}
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            font-size: 0.98rem !important;
            font-weight: 700;
            color: {SIDEBAR_TEXT} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            font-size: 0.92rem !important;
            color: {SIDEBAR_TEXT_MUTED} !important;
        }}
        /* 라디오 nav 메뉴: 동그라미 클릭형 대신 선으로 구분된 리스트 메뉴로.
           DOM 구조(실제 렌더링 확인함): label[data-testid="stRadioOption"] > div > div
           > [circle div (첫 자식), stMarkdownContainer]. circle div는 emotion 자동생성
           클래스라 이름이 불안정하므로 구조 선택자(:first-child)로 잡는다. */
        /* stElementContainer가 부모(259px) 대신 콘텐츠 크기(137px)로 줄어드는 문제가
           있어(실제 렌더링에서 getBoundingClientRect로 확인함), 사이드바 안에서는
           위젯 컨테이너를 전부 강제로 꽉 채운다. */
        section[data-testid="stSidebar"] [data-testid="stElementContainer"],
        section[data-testid="stSidebar"] [data-testid="stRadio"],
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{
            width: 100% !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            display: flex;
            flex-direction: column;
            gap: 0;
            width: 100%;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0;
            margin: 0;
            border-radius: 8px;
            border-bottom: 1px solid {SIDEBAR_LINE};
            transition: background 0.12s ease;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {{
            width: 100%;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:first-child {{
            border-top: 1px solid {SIDEBAR_LINE};
        }}
        /* 메인 네비게이션(6개 항목)의 5번째 옵션("전문 분석") 앞에 "서비스 검증" 구분
           라벨을 넣어, 고객용 4개 화면과 검증용 2개 화면을 시각적으로 분리한다.
           DEMO MODE 라디오는 옵션이 4개뿐이라 5번째가 없어 이 규칙의 영향을 받지 않는다. */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-testid="stRadioOption"]:nth-of-type(5) {{
            margin-top: 14px;
            position: relative;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-testid="stRadioOption"]:nth-of-type(5)::before {{
            content: "서비스 검증";
            display: block;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            color: {SIDEBAR_TEXT_MUTED};
            padding: 4px 0.9rem 6px;
        }}
        /* 원형 인디케이터 숨김 */
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div > div:first-child {{
            display: none;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {{
            padding: 0.8rem 0.9rem;
            width: 100%;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{
            font-size: 1.08rem !important;
            color: {SIDEBAR_TEXT} !important;
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{
            background: {SIDEBAR_HOVER};
        }}
        /* 선택된 항목: 딥 티얼 배경 + 흰 글씨 (청록색 강조 탭) */
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{
            background: {SIDEBAR_SELECTED_BG};
            border-color: {SIDEBAR_SELECTED_BG};
        }}
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {{
            font-weight: 800 !important;
            color: {SIDEBAR_SELECTED_TEXT} !important;
        }}
        /* 사이드바 안 expander("고객 정보 입력")는 흰색 카드로 띄워 어두운 배경과
           대비시키고, 그 안의 라벨/캡션은 원래(밝은 배경용) 색으로 되돌린다 */
        section[data-testid="stSidebar"] [data-testid="stExpander"] {{
            background: {SURFACE};
            border-radius: 12px;
            overflow: hidden;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] p,
        section[data-testid="stSidebar"] [data-testid="stExpander"] span,
        section[data-testid="stSidebar"] [data-testid="stExpander"] label {{
            color: {INK} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stWidgetLabel"] p {{
            color: {INK} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stCaptionContainer"] {{
            color: {SUBTLE} !important;
        }}
        /* 버튼 */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
            border: 1px solid {LINE};
        }}
        .stButton > button[kind="primary"] {{
            background: {BRAND};
            border-color: {BRAND};
        }}
        .stButton > button[kind="primary"]:hover {{
            background: #163c49;
            border-color: #163c49;
        }}
        /* 구분선 여백 정리 */
        hr {{
            margin: 1.6rem 0;
            border-color: {LINE};
        }}
        /* 사이드바(어두운 배경) 안의 구분선은 옅은 흰색 계열로 */
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.18);
        }}
        /* metric 위젯 폰트 강조 */
        [data-testid="stMetricValue"] {{
            font-size: 1.7rem;
            font-weight: 700;
            color: {INK};
        }}
        [data-testid="stMetricLabel"] {{
            color: {SUBTLE};
            font-weight: 600;
        }}
        /* expander */
        [data-testid="stExpander"] {{
            border: 1px solid {LINE};
            border-radius: 12px;
            background: {SURFACE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def yoga_icon_svg(color: str, size: int = 26) -> str:
    """오뚝이 워드마크 옆 아이콘 — 사용자가 실제로 보내준 나무 자세(브륵샤아사나) 요가
    PNG 아이콘(64x64, 흰색+투명배경)을 그대로 쓴다. base64로 인라인 임베드하고,
    CSS mask-image로 PNG의 실루엣(알파 채널)만 가져와 원하는 색(`color`)으로 칠한다 —
    이렇게 하면 원본이 흰색 PNG여도 이 함수 하나로 사이드바(흰색)든 밝은 헤더(브랜드색)든
    어떤 배경에도 맞는 색으로 재사용할 수 있다."""
    return (
        f'<span style="display:inline-block;vertical-align:middle;width:{size}px;height:{size}px;'
        f"background-color:{color};"
        f"-webkit-mask-image:url(data:image/png;base64,{YOGA_ICON_PNG_B64});"
        f"mask-image:url(data:image/png;base64,{YOGA_ICON_PNG_B64});"
        f"-webkit-mask-size:contain;mask-size:contain;"
        f"-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;"
        f'-webkit-mask-position:center;mask-position:center;"></span>'
    )


# ---------------------------------------------------------------------------
# 공통 컴포넌트 (HTML 문자열 반환 -> st.markdown(..., unsafe_allow_html=True))
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str, right_html: str = "") -> str:
    return compact_html(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.9rem 1.3rem;background:{SURFACE};border:1px solid {LINE};
                border-radius:16px;margin-bottom:1.4rem;">
        <div>
            <div class="ottugi-wordmark" style="font-size:1.6rem;color:{BRAND};">
                오뚝이
            </div>
            <div style="font-size:0.85rem;color:{SUBTLE};margin-top:2px;">{subtitle}</div>
        </div>
        <div>{right_html}</div>
    </div>
    """)


def demo_mode_badge(label: str) -> str:
    return compact_html(f"""
    <span style="background:{BRAND_SOFT};color:{BRAND};padding:5px 12px;border-radius:20px;
                 font-size:0.8rem;font-weight:700;border:1px solid {LINE};">
        Demo · {label}
    </span>
    """)


def section_header(title: str, desc: str = "") -> str:
    desc_html = f'<div style="color:{SUBTLE};font-size:0.92rem;margin-top:2px;">{desc}</div>' if desc else ""
    return compact_html(f"""
    <div style="margin:1.6rem 0 0.9rem 0;">
        <div style="font-size:1.15rem;font-weight:800;color:{INK};">{title}</div>
        {desc_html}
    </div>
    """)


def risk_badge_html(level: str, size: str = "md") -> str:
    c = RISK_COLORS.get(level, {"main": "#616161", "bg": "#eee", "border": "#ccc"})
    pad = "6px 18px" if size == "md" else "3px 12px"
    font = "1rem" if size == "md" else "0.78rem"
    return (
        f'<span style="background:{c["bg"]};color:{c["main"]};padding:{pad};border-radius:999px;'
        f'font-weight:800;font-size:{font};border:1px solid {c["border"]};display:inline-block;">'
        f"{level}</span>"
    )


RISK_LEVEL_ORDER = ["관찰", "주의", "경고", "심화"]

# risk_indicator 4단계가 각각 무엇을 의미하는지 — 스텝 인디케이터 옆에 고정 배치하는 설명.
# "숫자/차트만 단독 노출 금지" 원칙에 따라 항상 이 설명과 함께 보여준다.
RISK_LEVEL_EXPLANATIONS = {
    "관찰": "아직 안정적인 상태예요. 지금처럼만 유지하시면 됩니다.",
    "주의": "리볼빙 의존도가 조금씩 늘고 있어요. 결제 비율을 조금 높이면 안정을 되찾을 수 있어요.",
    "경고": "결제 여유가 얼마 남지 않았어요. 지금 추가로 상환하면 효과가 큰 시점이에요.",
    "심화": "최소한만 갚는 상태가 오래 이어지고 있어요. 지금부터 작은 상환 변화가 중요한 시점이에요.",
}


def risk_stepper_html(current_level: str) -> str:
    """관찰→주의→경고→심화 4단계를 신호등형 스텝 바로 표시. 현재 단계는 크고 진하게,
    나머지는 옅게 처리해 "지금 어디에 있는지"가 한눈에 들어오게 한다."""
    segments = []
    for lvl in RISK_LEVEL_ORDER:
        c = RISK_COLORS[lvl]
        if lvl == current_level:
            seg = (
                f'<div style="flex:1.3;text-align:center;background:{c["main"]};color:#ffffff;'
                f'font-weight:900;font-size:1.05rem;padding:1rem 0.5rem;border-radius:14px;'
                f'box-shadow:0 6px 16px {c["main"]}4d;">▲<br>{lvl}</div>'
            )
        else:
            seg = (
                f'<div style="flex:1;text-align:center;background:{c["bg"]};color:{c["main"]};'
                f'font-weight:700;font-size:0.92rem;opacity:0.6;padding:0.85rem 0.4rem;'
                f'border-radius:14px;">{lvl}</div>'
            )
        segments.append(seg)
    track = f'<div style="display:flex;gap:9px;align-items:center;">{"".join(segments)}</div>'
    explanation = (
        f'<div style="color:{INK};font-size:0.92rem;margin-top:0.9rem;padding:0.8rem 1rem;'
        f'background:{PAGE_BG};border-radius:10px;">'
        f'<b style="color:{RISK_COLORS[current_level]["main"]};">{current_level} 단계:</b> '
        f'{RISK_LEVEL_EXPLANATIONS.get(current_level, "")}</div>'
    )
    return compact_html(track + explanation)


def risk_hero_card(level: str, headline: str, sub_metrics_html: str, stepper_html: str = "") -> str:
    """현재 위험 단계 카드. stepper_html 을 넘기면 같은 카드 안에 이어 붙여, 같은 정보를
    두 개의 흰 카드로 반복해 보여주지 않는다."""
    c = RISK_COLORS.get(level, {"main": "#616161", "bg": "#eee", "border": "#ccc"})
    stepper_block = f'<div style="margin-top:1.2rem;">{stepper_html}</div>' if stepper_html else ""
    return compact_html(f"""
    <div style="background:{SURFACE};border:1px solid {LINE};border-radius:18px;padding:1.6rem 1.8rem;">
        <div style="color:{SUBTLE};font-size:0.85rem;font-weight:700;margin-bottom:0.5rem;">
            현재 리볼빙 위험도
        </div>
        <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
            <div style="font-size:2.4rem;font-weight:900;color:{c['main']};letter-spacing:-0.02em;">
                {level}
            </div>
            <div style="color:{SUBTLE};font-size:0.95rem;">{headline}</div>
        </div>
        <div style="margin-top:1.1rem;">{sub_metrics_html}</div>
        {stepper_block}
    </div>
    """)


def alert_card(icon: str, title: str, body: str, tone: str = "경고") -> str:
    c = RISK_COLORS.get(tone, {"main": "#616161", "bg": "#eee", "border": "#ccc"})
    return compact_html(f"""
    <div style="background:{c['bg']};border:1px solid {c['border']};border-radius:16px;
                padding:1.2rem 1.4rem;margin:1rem 0;">
        <div style="font-weight:800;color:{c['main']};font-size:1rem;margin-bottom:0.35rem;">
            {icon} {title}
        </div>
        <div style="color:{INK};font-size:0.95rem;">{highlight_text(body)}</div>
    </div>
    """)


def metric_tile(label: str, value: str, note: str = "", desc: str = "") -> str:
    """지표 타일. desc: 라벨과 수치 사이에 들어가는 작은 설명 문구(이 수치가 어떻게
    계산되는지). note: 수치 아래 보조 문구(기존 용도 유지)."""
    note_html = f'<div style="color:{SUBTLE};font-size:0.78rem;margin-top:4px;">{note}</div>' if note else ""
    desc_html = (
        f'<div style="color:{SUBTLE};font-size:0.72rem;line-height:1.45;margin-top:4px;">{desc}</div>'
        if desc else ""
    )
    return compact_html(f"""
    <div style="background:{SURFACE};border:1px solid {LINE};border-radius:14px;
                padding:1rem 1.2rem;flex:1;min-width:150px;">
        <div style="color:{INK};font-size:0.98rem;font-weight:700;letter-spacing:-0.01em;">{label}</div>
        {desc_html}
        <div style="color:{INK};font-size:1.5rem;font-weight:800;margin-top:5px;">{value}</div>
        {note_html}
    </div>
    """)


def metric_row(tiles_html: list[str]) -> str:
    return f'<div style="display:flex;gap:0.9rem;flex-wrap:wrap;margin:0.8rem 0;">{"".join(tiles_html)}</div>'


def recovery_gauge_html(score: float, *, hint: str = "", delta: float | None = None, fill_height: bool = False) -> str:
    """오뚝이 회복 게이지 (0~100 진행률 바). 숫자/막대 기반 — 캐릭터가 게이지를 대체하지 않는다.

    색은 점수 구간에 따라 위험색 4단계를 재사용한다(75+ 관찰, 50+ 주의, 25+ 경고, 그 이하 심화).
    fill_height=True: 옆 카드(마스코트)와 높이를 맞추도록 부모 높이를 채운다.
    """
    score = max(0.0, min(100.0, float(score)))
    tone = "관찰" if score >= 75 else "주의" if score >= 50 else "경고" if score >= 25 else "심화"
    c = RISK_COLORS[tone]["main"]
    delta_html = ""
    if delta is not None and abs(delta) >= 0.5:
        arrow = "▲" if delta > 0 else "▼"
        dcol = RISK_COLORS["관찰"]["main"] if delta > 0 else RISK_COLORS["경고"]["main"]
        delta_html = f'<span style="color:{dcol};font-size:0.9rem;font-weight:800;margin-left:8px;">{arrow} {abs(delta):.0f}</span>'
    hint_html = f'<div style="color:{SUBTLE};font-size:0.85rem;margin-top:0.5rem;line-height:1.45;">{hint}</div>' if hint else ""
    h = ("box-sizing:border-box;flex:1 1 300px;align-self:stretch;display:flex;flex-direction:column;"
         "justify-content:center;margin:0;") if fill_height else "margin:0.6rem 0;"
    # 신용점수 등 공식 지표로 오해되지 않도록, 게이지 아래 근거를 항상 짧게 밝혀둔다.
    disclaimer_html = (
        f'<div style="color:{SUBTLE};font-size:0.72rem;margin-top:0.5rem;opacity:0.85;">'
        "현재 금융 행동과 상환 시뮬레이션을 바탕으로 산출한 서비스 내 참고 지표예요.</div>"
    )
    return compact_html(f"""
    <div style="background:{SURFACE};border:1px solid {LINE};border-radius:16px;padding:0.95rem 1.2rem;{h}">
        <div style="display:flex;align-items:baseline;justify-content:space-between;">
            <div style="color:{SUBTLE};font-size:0.85rem;font-weight:700;">오뚝이 회복 게이지</div>
            <div><span style="color:{c};font-size:1.8rem;font-weight:900;">{score:.0f}</span>
                 <span style="color:{SUBTLE};font-size:0.9rem;">/ 100</span>{delta_html}</div>
        </div>
        <div style="background:{PAGE_BG};border-radius:999px;height:13px;margin-top:0.55rem;overflow:hidden;">
            <div style="width:{score:.1f}%;height:100%;background:{c};border-radius:999px;transition:width 0.5s ease;"></div>
        </div>
        {hint_html}
        {disclaimer_html}
    </div>
    """)


def mission_card_html(title: str, body_html: str, *, accent: str = BRAND) -> str:
    return compact_html(f"""
    <div style="background:{SURFACE};border:1px dashed {accent}80;border-left:4px solid {accent};
                border-radius:14px;padding:1.1rem 1.3rem;margin:0.6rem 0;">
        <div style="font-weight:900;color:{accent};font-size:1.02rem;margin-bottom:0.4rem;">{title}</div>
        <div style="color:{INK};font-size:0.93rem;line-height:1.55;">{body_html}</div>
    </div>
    """)


def card_open(padding: str = "1.4rem 1.6rem") -> str:
    return f'<div style="background:{SURFACE};border:1px solid {LINE};border-radius:16px;padding:{padding};margin-bottom:1rem;">'


def card_close() -> str:
    return "</div>"


def big_section_title(label: str, accent: str = BRAND, icon: str = "") -> str:
    """카드 본문과 시각적으로 분리된, 크고 눈에 띄는 섹션 제목(배지 스타일)."""
    return compact_html(f"""
    <div style="display:inline-block;background:{accent}17;color:{accent};font-weight:900;
                font-size:1.25rem;padding:0.5rem 1.1rem;border-radius:12px;
                margin:1.3rem 0 0.7rem 0;border:1px solid {accent}40;letter-spacing:-0.01em;">
        {icon} {label}
    </div>
    """)


def coaching_card(body_html: str, accent: str = BRAND) -> str:
    """coaching_section_title() 바로 아래에 오는 본문 박스 (제목은 별도 컴포넌트로 분리됨)."""
    return compact_html(f"""
    <div style="background:{SURFACE};border:1px solid {LINE};border-left:5px solid {accent};
                border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.4rem;">
        <div style="color:{INK};font-size:1.02rem;">{body_html}</div>
    </div>
    """)


def definitions_panel(items: list[tuple[str, str]]) -> str:
    """지표 이름 -> 설명 쌍을 담담한 카드로 보여준다 (수치가 어떤 데이터로 어떻게
    계산됐는지 사용자가 바로 아래에서 확인할 수 있도록)."""
    rows = []
    for label, desc in items:
        rows.append(
            f'<div style="padding:0.55rem 0;border-bottom:1px dashed {LINE};font-size:0.86rem;">'
            f'<span style="color:{INK};font-weight:700;">{label}</span>'
            f'<span style="color:{SUBTLE};"> — {desc}</span></div>'
        )
    body = "".join(rows)
    return compact_html(f"""
    <div style="background:{PAGE_BG};border:1px solid {LINE};border-radius:12px;padding:0.9rem 1.2rem;margin-top:0.6rem;">
        <div style="color:{SUBTLE};font-size:0.78rem;font-weight:800;margin-bottom:4px;letter-spacing:0.03em;">이 수치는 어떻게 계산되나요?</div>
        {body}
    </div>
    """)


def evidence_line(source: str, text: str) -> str:
    meta = SOURCE_META.get(source, {"label": source, "color": SUBTLE, "icon": "•"})
    return (
        f'<div style="padding:6px 0;border-bottom:1px dashed {LINE};font-size:0.88rem;">'
        f'<span style="color:{meta["color"]};font-weight:700;">{meta["icon"]} {meta["label"]}</span>'
        f'<div style="color:{INK};margin-top:2px;">{text}</div></div>'
    )


def cta_button_note(text: str) -> str:
    return f'<div style="color:{SUBTLE};font-size:0.8rem;margin-top:0.4rem;">{text}</div>'


def forecast_timeline_html(steps: list[dict]) -> str:
    """steps: [{"label": "현재", "level": "주의", "value": "34.2%", "icon": "<img.../>"(선택)}, ...]"""
    items = []
    for idx, s in enumerate(steps):
        c = RISK_COLORS.get(s["level"], {"main": "#616161", "bg": "#eee", "border": "#ccc"})
        arrow = "" if idx == 0 else f'<div style="color:{SUBTLE};font-size:2rem;font-weight:300;padding:0 14px;">→</div>'
        icon_html = (
            f'<div style="display:flex;justify-content:center;margin-bottom:6px;">{s["icon"]}</div>'
            if s.get("icon") else ""
        )
        items.append(
            f"""{arrow}
            <div style="text-align:center;flex:1;min-width:140px;">
                <div style="color:{SUBTLE};font-size:0.88rem;font-weight:700;margin-bottom:10px;">{s['label']}</div>
                {icon_html}
                <div style="background:{c['bg']};border:2px solid {c['border']};color:{c['main']};
                            border-radius:14px;padding:1rem 0.6rem;font-weight:900;font-size:1.3rem;">
                    {s['level']}
                </div>
                <div style="color:{SUBTLE};font-size:0.9rem;margin-top:8px;font-weight:700;">{s['value']}</div>
            </div>"""
        )
    return compact_html(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'gap:0;overflow-x:auto;padding:1rem 0.4rem;">{"".join(items)}</div>'
    )


# ---------------------------------------------------------------------------
# 텍스트 강조 — 숫자·%·원 금액·위험 단계 용어·핵심 지표명을 굵게/색상으로 강조
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*\s?%p?|\d[\d,]*\s?원|\d+\s?개월")
KEY_TERMS = ["리볼빙 의존도", "결제여유", "약정결제비율", "최소결제비율", "이월원금", "최소결제"]


def highlight_text(text: str, number_color: str = None) -> str:
    """본문 텍스트 안의 숫자(%, 원, 개월)와 관찰/주의/경고/심화, 핵심 지표명을
    굵게+색상으로 강조한다. 전체가 검정 텍스트라 중요한 값이 눈에 안 들어온다는
    피드백에 따른 것 — 페이지 전반의 서술형 텍스트에 공통 적용한다."""
    color = number_color or BRAND

    def _num_repl(m: re.Match) -> str:
        return f'<b style="color:{color};">{m.group(0)}</b>'

    out = _NUMBER_RE.sub(_num_repl, text)

    for level, c in RISK_COLORS.items():
        out = out.replace(level, f'<b style="color:{c["main"]};">{level}</b>')

    for term in KEY_TERMS:
        # 이미 <b>로 감싼 영역 안의 term은 건드리지 않도록, 태그가 섞이지 않은
        # 순수 term만 굵게 처리 (강조색 없이 굵기만 — 색을 너무 많이 쓰면 오히려
        # 안 읽히므로 숫자·위험단계 용어에만 색을 쓰고 지표명은 굵기만 준다).
        out = out.replace(term, f"<b>{term}</b>")

    return out


def footer_badge() -> str:
    return compact_html(f"""
    <div style="text-align:center;color:{SUBTLE};font-size:0.78rem;margin-top:2.5rem;padding-top:1rem;
                border-top:1px solid {LINE};">
        Demo Mode · 합성 금융데이터 기반 · 본 서비스의 지표는 공식 신용점수·신용등급이 아닌
        자가진단용 위험도 지표입니다.
    </div>
    """)
