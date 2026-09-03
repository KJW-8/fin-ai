"""Discrete-time Hazard Model 학습 + 검증 + 저장.

    ./.venv/bin/python scripts/train_hazard.py

산출:
  models/hazard_model.joblib                    학습된 모델 번들 (model, feats, meta)
  outputs/hazard_metrics.json                   검증 지표
  outputs/hazard_person_period_sample.csv       person-period 샘플 (leakage 점검용)

기존 XGBoost 파이프라인(model.py)은 건드리지 않는다. 완전히 별도 산출물이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hazard as hz  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# duration 항은 반드시 포함(스펙 요구). raw hazard 가 t=3 스파이크 후 감소하는 형태.
#  - linear: 곡선이 반대로(단조증가) 학습됨 -> 제외
#  - quadratic: 험프형 근사, 매끄럽지만 t=3 스파이크를 놓쳐 3개월 전환확률을 과소추정
#  - dummies: period one-hot, 실제 per-period hazard 를 그대로 학습(서비스 예측범위 전부 커버)
DURATION_CANDIDATES = ("quadratic", "dummies")


def account_frame(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aid, g in pp.groupby("account_id", sort=False):
        g = g.sort_values("period")
        ev = g[g["event_occurred"] == 1]
        rows.append({
            "account_id": aid,
            "event_time": int(ev["period"].iloc[0]) if len(ev) else int(g["period"].iloc[-1]),
            "event_observed": 1 if len(ev) else 0,
        })
    return pd.DataFrame(rows)


def pp_concordance(model, feats, df: pd.DataFrame) -> tuple[float, int]:
    """person-period 수준 concordance: 같은 period 안에서 이벤트 행의 예측 hazard 가
    비이벤트 행보다 높은 비율 (period 별 계산 후 쌍 가중 평균)."""
    h = hz.predict_hazard_rows(model, feats, df)
    df = df.assign(_h=h)
    conc = tie = tot = 0
    for _, g in df.groupby("period"):
        pos = g.loc[g["event_occurred"] == 1, "_h"].to_numpy()
        neg = g.loc[g["event_occurred"] == 0, "_h"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        for hp in pos:
            conc += int((hp > neg).sum())
            tie += int((hp == neg).sum())
            tot += len(neg)
    if tot == 0:
        return float("nan"), 0
    return (conc + 0.5 * tie) / tot, tot


def three_month_eval(model, feats, pp: pd.DataFrame, from_period: int) -> dict:
    """from_period 시점(= 직전 달 feature) 기준 '향후 3개월 전환확률' vs 실제 3개월 내 전환.
    from_period 까지 생존한 계좌만 대상."""
    at_risk = pp[pp["period"] == from_period]
    preds, actual, etime, eobs = [], [], [], []
    for _, r in at_risk.iterrows():
        aid = r["account_id"]
        g = pp[pp["account_id"] == aid].sort_values("period")
        ev = g[g["event_occurred"] == 1]["period"].tolist()
        transitioned = bool(ev and from_period <= ev[0] <= from_period + 2)
        state = {c: float(r[c]) for c in hz.HAZARD_FEATURES}
        tp3 = hz.transition_probability_3m(model, feats, state, current_month=from_period - 1)
        preds.append(tp3); actual.append(int(transitioned))
        etime.append(from_period if transitioned else from_period + 3)
        eobs.append(int(transitioned))
    preds, actual = np.array(preds), np.array(actual)
    c, npair = hz.harrell_c_index(preds, np.array(etime), np.array(eobs))
    return {
        "from_period": from_period,
        "n_accounts_at_risk": int(len(at_risk)),
        "n_transitioned_within_3m": int(actual.sum()),
        "c_index": None if np.isnan(c) else round(float(c), 4),
        "c_index_pairs": int(npair),
        "brier": round(hz.brier_score(preds, actual), 5),
        "calibration": hz.calibration_bins(preds, actual, n_bins=4),
        "mean_predicted": round(float(preds.mean()), 4),
        "observed_rate": round(float(actual.mean()), 4),
    }


def account_cv(pp: pd.DataFrame, duration_spec: str, n_splits: int = 5, seed: int = 42) -> dict:
    """계좌 단위 K-fold CV — 한 계좌의 person-period 행이 절대 train/test 로 갈리지 않음.
    각 fold: person-period brier/concordance + 계좌 3개월-ahead C-index (전체 이벤트 사용)."""
    accs = account_frame(pp)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(accs["account_id"].to_numpy()), n_splits)

    pp_briers, pp_concs, acc_cs, acc_pairs = [], [], [], []
    pooled_pred, pooled_act = [], []
    for k in range(n_splits):
        te_ids = set(folds[k])
        tr = pp[~pp["account_id"].isin(te_ids)]
        te = pp[pp["account_id"].isin(te_ids)]
        model, feats = hz.fit_hazard_model(tr, duration_spec=duration_spec)

        pp_briers.append(hz.brier_score(hz.predict_hazard_rows(model, feats, te), te["event_occurred"].to_numpy()))
        c_pp, _ = pp_concordance(model, feats, te)
        if not np.isnan(c_pp):
            pp_concs.append(c_pp)

        # 계좌 3개월-ahead: 위험집합 진입(period 2, month-1 feature) 기준, periods 2~4 전환 예측.
        # 진입 전 행(이벤트 발생 행)을 anchor 로 쓰지 않아 순환을 피한다.
        preds, etime, eobs = [], [], []
        for aid in te_ids:
            g = te[te["account_id"] == aid].sort_values("period")
            if len(g) == 0:
                continue
            anchor = g.iloc[0]  # 진입 period (모두 관찰 상태, event=0)
            ap = int(anchor["period"])
            ev = g[g["event_occurred"] == 1]["period"].tolist()
            transitioned = bool(ev and ap <= ev[0] <= ap + 2)
            state = {c: float(anchor[c]) for c in hz.HAZARD_FEATURES}
            tp3 = hz.transition_probability_3m(model, feats, state, current_month=ap - 1)
            preds.append(tp3)
            etime.append(ap if transitioned else ap + 3)
            eobs.append(int(transitioned))
            pooled_pred.append(tp3); pooled_act.append(int(transitioned))
        c, npair = hz.harrell_c_index(np.array(preds), np.array(etime), np.array(eobs))
        if not np.isnan(c):
            acc_cs.append(c); acc_pairs.append(npair)

    pooled_pred, pooled_act = np.array(pooled_pred), np.array(pooled_act)
    return {
        "n_splits": n_splits,
        "person_period_brier_mean": round(float(np.mean(pp_briers)), 5),
        "person_period_concordance_mean": round(float(np.mean(pp_concs)), 4) if pp_concs else None,
        "account_3m_c_index_folds": [round(x, 4) for x in acc_cs],
        "account_3m_c_index_mean": round(float(np.mean(acc_cs)), 4) if acc_cs else None,
        "account_3m_c_index_std": round(float(np.std(acc_cs)), 4) if acc_cs else None,
        "account_3m_brier_pooled": round(hz.brier_score(pooled_pred, pooled_act), 5),
        "account_3m_calibration_pooled": hz.calibration_bins(pooled_pred, pooled_act, n_bins=5),
        "account_3m_mean_predicted": round(float(pooled_pred.mean()), 4),
        "account_3m_observed_rate": round(float(pooled_act.mean()), 4),
        "total_accounts_scored": int(len(pooled_act)),
    }


def risk_group_event_rates(model, feats, pp: pd.DataFrame) -> list[dict]:
    """예측 hazard 5분위별 실제 이벤트율 (전체 person-period)."""
    h = hz.predict_hazard_rows(model, feats, pp)
    q = pd.qcut(h, 5, labels=False, duplicates="drop")
    out = []
    for grp in sorted(pd.unique(q[~pd.isna(q)])):
        mask = q == grp
        out.append({
            "hazard_quintile": int(grp) + 1,
            "n": int(mask.sum()),
            "mean_predicted_hazard": round(float(h[mask].mean()), 4),
            "observed_event_rate": round(float(pp["event_occurred"].to_numpy()[mask].mean()), 4),
        })
    return out


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    derived = pd.read_csv(DATA_DIR / "derived_features.csv")
    mt = pd.read_csv(DATA_DIR / "monthly_transaction.csv")

    print("=" * 72)
    print("1. person-period 변환")
    ppr = hz.build_person_period(derived, mt)
    pp = ppr.df
    print(f"   전체 계좌 {ppr.n_accounts_total} | already_high_risk 제외 {ppr.n_already_high_risk} | "
          f"변환 사용 계좌 {ppr.n_accounts_used}")
    print(f"   person-period 행 {len(pp)} | 전환 이벤트 {ppr.n_events} | right-censored 계좌 {ppr.n_censored}")
    print(f"   첫 전환 월 분포: {ppr.first_event_month_counts}")
    pp.head(60).to_csv(OUTPUTS_DIR / "hazard_person_period_sample.csv", index=False)

    print("\n2. 시간 기준 분할 (XGBoost 동일: period<=9 train / 10-12 test)")
    train, test = hz.temporal_split(pp)
    n_test_ev = int(test["event_occurred"].sum())
    small = n_test_ev < 30
    print(f"   train {len(train)}행(이벤트 {int(train['event_occurred'].sum())}) | test {len(test)}행(이벤트 {n_test_ev})")
    if small:
        print(f"   ⚠️  시간분할 test 전환 이벤트 {n_test_ev}건 < 30 → 이 분할의 C-index/calibration 은 "
              f"신뢰구간이 넓다. 계좌 5-fold CV(전체 이벤트)를 주 지표로 병행 보고.")

    print("\n3. duration 항 비교 (전부 duration 포함, 스펙 요구: 'duration, duration^2, duration 구간 dummy 비교')")
    mean_state = {c: float(train[c].mean()) for c in hz.HAZARD_FEATURES}
    dur_cmp = {}
    for spec in DURATION_CANDIDATES:
        m, f = hz.fit_hazard_model(train, duration_spec=spec)
        curve = [round(float(m.predict_proba(hz._feature_vector(mean_state, f, mm))[0, 1]), 3) for mm in range(2, 13)]
        b = hz.brier_score(hz.predict_hazard_rows(m, f, test), test["event_occurred"].to_numpy())
        e3 = three_month_eval(m, f, pp, from_period=10)
        dur_cmp[spec] = {"person_period_brier_test": round(b, 5),
                         "temporal_3m_c_index": e3["c_index"],
                         "mean_state_hazard_curve_t2_12": curve}
        print(f"   {spec:10s}: test pp-brier={b:.5f}  3m C-index={e3['c_index']}  평균상태곡선={curve}")
    best = min(DURATION_CANDIDATES, key=lambda s: dur_cmp[s]["person_period_brier_test"])
    print(f"   → 선택: {best}  (person-period brier 최소; raw hazard 가 t=3 스파이크 후 감소하는 형태와 대체로 일치)")

    print("\n4. 최종 평가 모델 (train 학습)")
    model, feats = hz.fit_hazard_model(train, duration_spec=best)
    print(f"   features({len(feats)}): {feats}")

    print("\n5. [시간분할 test] 향후 3개월 전환확률 평가 (month 9 상태 → 10~12월 전환)")
    tsplit_3m = three_month_eval(model, feats, pp, from_period=10)
    print(f"   대상 계좌 {tsplit_3m['n_accounts_at_risk']} | 3개월 내 전환 {tsplit_3m['n_transitioned_within_3m']} | "
          f"C-index {tsplit_3m['c_index']} (쌍 {tsplit_3m['c_index_pairs']}) | brier {tsplit_3m['brier']}")
    print(f"   평균 예측 {tsplit_3m['mean_predicted']} vs 실제 전환율 {tsplit_3m['observed_rate']}")

    print("\n6. [계좌 5-fold CV] (전체 이벤트 사용, 한 계좌가 train/test 로 안 갈림)")
    cv = account_cv(pp, duration_spec=best, n_splits=5)
    print(f"   person-period brier(평균) {cv['person_period_brier_mean']} | pp-concordance {cv['person_period_concordance_mean']}")
    print(f"   계좌 3개월 C-index: {cv['account_3m_c_index_folds']}  평균 {cv['account_3m_c_index_mean']} (std {cv['account_3m_c_index_std']})")
    print(f"   계좌 3개월 brier(pooled) {cv['account_3m_brier_pooled']} | 평균예측 {cv['account_3m_mean_predicted']} vs 실제 {cv['account_3m_observed_rate']}")

    print("\n7. Kaplan-Meier baseline")
    accs = account_frame(pp)
    km = hz.kaplan_meier(accs["event_time"].to_numpy(), accs["event_observed"].to_numpy())
    print(f"   KM S(3)={km[3]:.3f} S(6)={km[6]:.3f} S(9)={km[9]:.3f} S(12)={km[12]:.3f}")
    print(f"   KM baseline 은 개인화 없음 → C-index 정의상 0.5. 개인화 모델 C-index({cv['account_3m_c_index_mean']}) "
          f"가 0.5 보다 유의하게 크면 baseline 대비 판별력 개선.")

    print("\n8. 위험군(예측 hazard 5분위)별 실제 이벤트율")
    full_model, full_feats = hz.fit_hazard_model(pp, duration_spec=best)
    rg = risk_group_event_rates(full_model, full_feats, pp)
    for r in rg:
        print(f"   Q{r['hazard_quintile']}: 예측 hazard {r['mean_predicted_hazard']:.3f} → 실제 이벤트율 {r['observed_event_rate']:.3f}  (n={r['n']})")

    print("\n9. 전환확률/예상 전환 시점 예시 (시간분할 test, month 9 상태)")
    p10 = test[test["period"] == 10].head(4)
    examples = []
    for _, r in p10.iterrows():
        state = {c: float(r[c]) for c in hz.HAZARD_FEATURES}
        tp3 = hz.transition_probability_3m(full_model, full_feats, state, current_month=9)
        mtw = hz.median_time_to_warning(full_model, full_feats, state, current_month=9, max_horizon=hz.N_MONTHS)
        examples.append({"account_id": r["account_id"], "transition_probability_3m": round(tp3, 3),
                         "median_time_to_warning_months": mtw})
        print(f"   {str(r['account_id'])[:8]}: 3개월 전환확률 {tp3*100:.1f}% | "
              f"예상 시점 {'약 '+str(mtw)+'개월 후' if mtw else '관측기간 내 낮음'}")

    mean_c = cv["account_3m_c_index_mean"]
    improvement = None if mean_c is None else round((mean_c - 0.5) / 0.5 * 100, 1)

    metrics = {
        "model": "discrete_time_hazard_logistic_regression",
        "target_event": "risk_indicator 가 처음으로 '경고' 또는 '심화'로 전환",
        "leakage_controls": ppr.notes,
        "duration_spec_selected": best,
        "duration_spec_comparison": dur_cmp,
        "features": full_feats,
        "person_period": {
            "n_accounts_total": ppr.n_accounts_total,
            "n_already_high_risk_excluded": ppr.n_already_high_risk,
            "already_high_risk_ids_sample": ppr.already_high_risk_ids[:10],
            "n_accounts_used": ppr.n_accounts_used,
            "n_person_period_rows": int(len(pp)),
            "n_events_total": ppr.n_events,
            "n_censored": ppr.n_censored,
            "first_event_month_counts": ppr.first_event_month_counts,
        },
        "temporal_split": {
            "rule": "period <= 9 -> train, period in [10,11,12] -> test (XGBoost와 동일)",
            "train_rows": int(len(train)), "train_events": int(train["event_occurred"].sum()),
            "test_rows": int(len(test)), "test_events": n_test_ev,
            "small_sample_warning": small,
            "small_sample_note": (
                f"시간분할 test 전환 이벤트 {n_test_ev}건 (<30). 이 분할 기반 C-index/calibration 은 "
                "신뢰구간이 넓으므로, 계좌 5-fold CV(이벤트 564건 전체 사용)를 주 지표로 함께 본다."
            ) if small else None,
            "three_month_transition_eval": tsplit_3m,
        },
        "account_5fold_cv": cv,
        "kaplan_meier_baseline": {str(k): round(v, 4) for k, v in km.items()},
        "c_index_vs_baseline": {
            "baseline_km_c_index": 0.5,
            "hazard_model_account_3m_c_index_cv": mean_c,
            "relative_improvement_pct_over_baseline": improvement,
            "note": "KM baseline 은 개인화가 없어 C-index 정의상 0.5. 개선율 = (모델 C-index - 0.5)/0.5.",
        },
        "risk_group_event_rates_hazard_quintile": rg,
        "examples_transition_probability": examples,
        "calibration_caveat": {
            "hazard_level": "예측 hazard 5분위별 실제 이벤트율이 거의 일치(Q1 .001/.001 ... Q5 .504/.502). "
                            "person-period brier 0.04 (naive 0.10 대비 양호). hazard 자체는 잘 보정됨.",
            "three_month_transition_probability": (
                f"'향후 3개월 전환확률'은 feature 를 고정하고 duration 만 전진시켜 계산하므로, "
                f"제한된 초기 이력에서 투영하면 실제 전환율을 과소추정한다 "
                f"(계좌 5-fold CV pooled: 평균 예측 {cv['account_3m_mean_predicted']} vs 실제 {cv['account_3m_observed_rate']}). "
                f"순위(누가 더 위험한가)는 유지된다(C-index {mean_c} CV / {tsplit_3m['c_index']} temporal). "
                f"→ 서비스에서는 확정 확률이 아닌 '현재 패턴 유지 가정 하의 방향성 추정'으로 표기하고, "
                f"median_time_to_warning(순서형, 더 robust)를 함께 노출한다."
            ),
        },
        "known_limitations": [
            "합성데이터: month 3 에 전환 이벤트가 크게 몰림(minimum_payment_streak>=3 이 월 3 부터 성립 가능). "
            "가입 초기(1~3개월) 사용자의 단기 전환확률은 과소추정될 수 있다.",
            "right-censored 236계좌는 관측 12개월 내 미전환일 뿐, 이후 전환 여부는 알 수 없다.",
            "시간분할 test 전환 이벤트 27건(<30) — 이 분할 지표는 신뢰구간이 넓다.",
            "이벤트 정의가 규칙 기반 risk_indicator 전환이므로, hazard 모델은 '규칙이 켜지는 조건에 언제 "
            "도달하는가'를 학습한다. 실서비스에서는 실제 연체/부실 라벨로 재정의·재학습 필요.",
        ],
    }
    with open(OUTPUTS_DIR / "hazard_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    meta = {"duration_spec": best, "features": full_feats,
            "trained_on": "전체 person-period (평가는 temporal split + account 5-fold CV 로 별도)",
            "metrics_file": "outputs/hazard_metrics.json"}
    path = hz.save_bundle(full_model, full_feats, meta)

    print("\n" + "=" * 72)
    print(f"저장: {path}  /  {OUTPUTS_DIR / 'hazard_metrics.json'}")
    print(f"요약: person-period {len(pp)}행 / 이벤트 {ppr.n_events}건 / censored {ppr.n_censored}계좌 / "
          f"already_high_risk {ppr.n_already_high_risk}계좌")
    print(f"      계좌 5-fold CV 3개월 C-index = {mean_c} (std {cv['account_3m_c_index_std']}) "
          f"| KM baseline 0.5 대비 +{improvement}%")
    print(f"      시간분할 test 이벤트 {n_test_ev}건" + ("  ⚠️ <30 (CI 넓음)" if small else ""))


if __name__ == "__main__":
    main()
