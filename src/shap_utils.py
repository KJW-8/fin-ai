"""오뚝이(Ottugi) 4단계 — SHAP 연동.

f1(S 예측 모델), f2(r 예측 모델) 각각에 대해 독립적으로 SHAP TreeExplainer를 적용한다.
두 모델은 서로 다른 타깃(S_(t+1), r_(t+1))을 예측하므로 SHAP 기여도도 분리해서
산출·직렬화한다 (하나로 합치면 "무엇에 대한 기여도인지"가 불명확해지므로).

직렬화 스키마 (prediction_log.shap_contributions 확장):
    {
      "account_id": ...,
      "month_index": t,                # 이 시점까지의 데이터로 t+1을 설명
      "predicted_S": float,
      "predicted_r": float,
      "base_value_S": float,            # SHAP base value (전체 평균 예측)
      "base_value_r": float,
      "shap_S": {feature_name: contribution_value, ...},
      "shap_r": {feature_name: contribution_value, ...}
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from model import build_feature_table

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_models() -> tuple[XGBRegressor, XGBRegressor, list[str]]:
    model_S = XGBRegressor()
    model_S.load_model(MODELS_DIR / "model_S.json")
    model_r = XGBRegressor()
    model_r.load_model(MODELS_DIR / "model_r.json")
    with open(MODELS_DIR / "feature_cols.json", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model_S, model_r, feature_cols


def build_explainers(model_S: XGBRegressor, model_r: XGBRegressor) -> tuple[shap.TreeExplainer, shap.TreeExplainer]:
    explainer_S = shap.TreeExplainer(model_S)
    explainer_r = shap.TreeExplainer(model_r)
    return explainer_S, explainer_r


def explain_row(explainer: shap.TreeExplainer, X_row: pd.DataFrame, feature_cols: list[str]) -> dict:
    """단일 행(1 x n_features)에 대한 SHAP 기여도를 {feature_name: value} 딕셔너리로 반환."""
    explanation = explainer(X_row)
    values = explanation.values[0]
    base_value = float(np.ravel(explanation.base_values)[0])
    contributions = {feature_cols[j]: float(values[j]) for j in range(len(feature_cols))}
    return contributions, base_value


def explain_customer_latest(
    account_id: str,
    feature_table: pd.DataFrame,
    model_S: XGBRegressor,
    model_r: XGBRegressor,
    explainer_S: shap.TreeExplainer,
    explainer_r: shap.TreeExplainer,
    feature_cols: list[str],
) -> dict:
    """해당 계좌의 가장 최근 관측월(month_index 최댓값) 기준으로 다음 달 S/r 예측 + SHAP 설명.

    Streamlit 앱(6단계)에서 "현재 시점 기준 다음 달 예측 근거"를 보여줄 때 사용하는 함수다.
    """
    rows = feature_table[feature_table["account_id"] == account_id]
    if rows.empty:
        raise ValueError(f"unknown account_id: {account_id}")
    latest = rows.loc[rows["month_index"].idxmax()]
    X_row = pd.DataFrame([latest[feature_cols].values], columns=feature_cols)

    pred_S = float(model_S.predict(X_row)[0])
    pred_r = float(model_r.predict(X_row)[0])
    pred_r = float(np.clip(pred_r, latest["minimum_payment_ratio"], 1.0))

    shap_S, base_S = explain_row(explainer_S, X_row, feature_cols)
    shap_r, base_r = explain_row(explainer_r, X_row, feature_cols)

    return {
        "account_id": account_id,
        "month_index": int(latest["month_index"]),
        "predicted_S": pred_S,
        "predicted_r": pred_r,
        "base_value_S": base_S,
        "base_value_r": base_r,
        "shap_S": shap_S,
        "shap_r": shap_r,
    }


def global_feature_importance(explainer: shap.TreeExplainer, X: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """|SHAP| 평균 기준 전역 피처 중요도 순위 (문서 3-4장 "SHAP 기여도 1순위 후보" 가설 검증용)."""
    explanation = explainer(X)
    mean_abs = np.abs(explanation.values).mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_abs), key=lambda kv: -kv[1])
    return [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranking]


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)

    account_master = pd.read_csv(DATA_DIR / "account_master.csv")
    monthly_transaction = pd.read_csv(DATA_DIR / "monthly_transaction.csv")
    derived_features = pd.read_csv(DATA_DIR / "derived_features.csv")

    model_S, model_r, feature_cols = load_models()
    explainer_S, explainer_r = build_explainers(model_S, model_r)

    feature_table = build_feature_table(monthly_transaction, derived_features, account_master)

    # 페르소나(저/중/고위험) 대표 계좌 1개씩 SHAP 설명 산출 및 저장
    persona_accounts = account_master.groupby("persona_tier")["account_id"].first().to_dict()
    samples = []
    for tier, account_id in persona_accounts.items():
        result = explain_customer_latest(account_id, feature_table, model_S, model_r, explainer_S, explainer_r, feature_cols)
        result["persona_tier"] = tier
        samples.append(result)

    with open(OUTPUTS_DIR / "shap_contributions_sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("[페르소나별 SHAP 설명 샘플]")
    for s in samples:
        top_S = sorted(s["shap_S"].items(), key=lambda kv: -abs(kv[1]))[:5]
        top_r = sorted(s["shap_r"].items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"\n[{s['persona_tier']}] account={s['account_id'][:8]} month={s['month_index']}")
        print(f"  predicted_S={s['predicted_S']:.0f}, predicted_r={s['predicted_r']:.4f}")
        print(f"  top S 기여 피처: {top_S}")
        print(f"  top r 기여 피처: {top_r}")

    # 전역 피처 중요도 (test-like 표본: 전체 계좌의 최신월 기준)
    latest_rows = feature_table.loc[feature_table.groupby("account_id")["month_index"].idxmax()]
    X_latest = latest_rows[feature_cols]

    importance_S = global_feature_importance(explainer_S, X_latest, feature_cols)
    importance_r = global_feature_importance(explainer_r, X_latest, feature_cols)

    with open(OUTPUTS_DIR / "shap_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump({"model_S": importance_S, "model_r": importance_r}, f, ensure_ascii=False, indent=2)

    print("\n[모델 S 전역 피처 중요도 top5]")
    for item in importance_S[:5]:
        print(f"  {item['feature']}: {item['mean_abs_shap']:.4f}")
    print("\n[모델 r 전역 피처 중요도 top5]")
    for item in importance_r[:5]:
        print(f"  {item['feature']}: {item['mean_abs_shap']:.4f}")

    print("\nSaved outputs/shap_contributions_sample.json, outputs/shap_feature_importance.json")
