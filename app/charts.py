"""오뚝이(Ottugi) 시각화 컴포넌트 (plotly).

이 모듈도 theme.py와 마찬가지로 순수 표현 계층이다. 여기서 그리는 모든 값은
model.py/risk.py/coaching.py가 이미 계산해 둔 값을 그대로 입력받아 차트로만
바꾼다 — 새로운 계산은 하지 않는다.

배경 밴드에 대한 설계 노트: risk_indicator는 carryover_share 하나만으로 정해지지
않고(수준+추세+결제여유+연속최소결제 결합, risk.py 참고) 시점마다 다르게 계산된다.
그래서 "carryover_share가 몇 % 이상이면 어떤 색"이라는 식으로 Y축 구간을 임의로
나누면 실제 판정 결과와 어긋날 수 있다(예: 45%인데도 추세가 꺾이면 '관찰'일 수 있음
— 실제로 이전 QA에서 이런 케이스가 발견됐다). 그래서 배경 밴드는 Y축 절대 수준이
아니라, 각 시점에 실제로 계산된 risk_indicator 색을 그 달의 X축 구간에 칠하는
방식으로 그린다 — 근사치가 아니라 실제 계산 결과를 그대로 시각화한 것이다.
"""

from __future__ import annotations

import plotly.graph_objects as go

import theme


def risk_trajectory_chart(outlook: list[dict]) -> go.Figure:
    """outlook: [{"month_offset": 0, "level": "주의", "carryover_share": 0.39}, ...]
    (model.recursive_forecast() + risk.classify_risk_indicator() 결과를 그대로 사용)
    """
    x = [o["month_offset"] for o in outlook]
    y = [o["carryover_share"] * 100 for o in outlook]
    levels = [o["level"] for o in outlook]

    fig = go.Figure()

    # 각 시점(달)의 실제 계산된 위험 단계 색을 그 구간의 배경으로 칠한다.
    for i, xi in enumerate(x):
        c = theme.RISK_COLORS[levels[i]]
        lo = xi - 0.5
        hi = xi + 0.5
        fig.add_shape(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1, fillcolor=c["bg"], opacity=1.0, line_width=0, layer="below")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text",
            line=dict(color=theme.BRAND, width=3),
            marker=dict(size=16, color=[theme.RISK_COLORS[l]["main"] for l in levels], line=dict(width=2, color="white")),
            text=[f"{v:.0f}%" for v in y],
            textposition="top center",
            textfont=dict(size=13, color=theme.INK),
            hovertext=[f"{'현재' if o['month_offset']==0 else str(o['month_offset'])+'개월 후'} · {lvl} · {v:.1f}%" for o, lvl, v in zip(outlook, levels, y)],
            hoverinfo="text",
            showlegend=False,
        )
    )

    tick_labels = ["현재" if o["month_offset"] == 0 else f"{o['month_offset']}개월 후" for o in outlook]
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickmode="array", tickvals=x, ticktext=tick_labels, showgrid=False),
        yaxis=dict(title="리볼빙 의존도(%)", range=[0, max(100, max(y) * 1.15)], showgrid=True, gridcolor=theme.LINE),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_bar_chart(shap_dict: dict, feature_labels: dict, k: int = 6) -> go.Figure:
    """SHAP 기여도를 한국어 라벨로 매핑한 가로 막대그래프. 양수(위험을 높이는 방향)는
    경고색, 음수(낮추는 방향)는 브랜드색으로 구분한다."""
    items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    items = list(reversed(items))  # plotly 가로 막대는 아래->위 순이라, 위에서부터 중요도 순으로 보이게 뒤집는다
    labels = [feature_labels.get(f, f) for f, _ in items]
    values = [v for _, v in items]
    colors = [theme.RISK_COLORS["경고"]["main"] if v > 0 else theme.BRAND for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[("위험 ↑" if v > 0 else "위험 ↓") for v in values],
            textposition="outside",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(
        height=90 + 46 * len(items),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True, zerolinecolor=theme.LINE),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def intervention_comparison_chart(scenarios: list[dict]) -> go.Figure:
    """scenarios: [{"extra_payment": 0, "level": "경고", "carryover_share": 0.48}, ...]
    (model.simulate_extra_payment()/simulate_intervention_trajectory() 결과를 그대로 사용)
    """
    x = [f"+{s['extra_payment']//10000}만원" if s["extra_payment"] > 0 else "현재 그대로" for s in scenarios]
    y = [s["carryover_share"] * 100 for s in scenarios]
    colors = [theme.RISK_COLORS[s["level"]]["main"] for s in scenarios]

    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker_color=colors,
            text=[s["level"] for s in scenarios],
            textposition="outside",
            textfont=dict(size=13, weight=700),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="예측 리볼빙 의존도(%)", range=[0, max(100, max(y) * 1.25)], showgrid=True, gridcolor=theme.LINE),
        xaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig
