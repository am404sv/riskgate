# -*- coding: utf-8 -*-
"""Додаткові аналізи робочої моделі:

  1. збіжність u(T) за кількістю MC-прогонів і бутстреп-стабільність u_max
  2. сітка чутливості показників зонування до (t_w, t_b, u_max)
  3. абляція радіуса контекстного профілю r = 0 / 1 / 2          (навчання, ~10 хв)
  4. бейзлайни LogReg / RandomForest / HistGradientBoosting      (~5 хв)
  5. стабільність робочої моделі за 5 початковими значеннями ГВЧ (навчання, ~25 хв)
  6. групова атрибуція Integrated Gradients для однієї транзакції зони BLACK
  7. криві PR / ROC, діаграма надійності, вибірка (s, u)

    python src/analysis_extra.py          # повний прогін
    python src/analysis_extra.py --fast   # без пп. 3–5 (їхні попередні результати зберігаються)

Результат: results/results_analysis.json, results/sensitivity_grid.json
"""
import argparse
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common import (RESULTS, RESULTS_ANALYSIS, RESULTS_MAIN, SCORES_GTT, load_json, load_splits, require,
                    save_json, set_threads)
from engine import (BLACK, fit_temperature, load_scores, mc_predict, pick_zoning, threshold_metrics,
                    train_model, val_thresholds, zone_assign, zone_metrics)
from models import GROUP_NAMES, GROUPS, build_main, build_main_for_subset, load_main

SENS_GRID = os.path.join(RESULTS, "sensitivity_grid.json")

IDX_R0 = list(range(0, 165)) + [165]          # базові + часовий крок
IDX_R1 = IDX_R0 + list(range(166, 186))       # + степені, середні сусідів, той самий крок, розмах (r = 1)
IDX_R2 = list(range(0, 188))                  # + охоплення другого кола (r = 2)


def u_convergence(model, data, zcfg):
    sub = np.random.RandomState(0).choice(len(data["X_test"]), 4000, replace=False)
    _, u50, P50 = mc_predict(model, data["X_test"][sub], T=50, seed=777)
    s50 = P50.mean(0)
    z50 = zone_assign(s50, u50, zcfg["tw"], zcfg["tb"], zcfg["umax"])
    out = []
    for T in (5, 10, 20, 30, 40):
        uT, sT = P50[:T].std(0), P50[:T].mean(0)
        zT = zone_assign(sT, uT, zcfg["tw"], zcfg["tb"], zcfg["umax"])
        out.append({"T": T, "spearman_vs_T50": float(spearmanr(uT, u50).statistic),
                    "mean_abs_diff": float(np.mean(np.abs(uT - u50))),
                    "zone_agreement": float((zT == z50).mean())})
    return out


def umax_stability(u_val, policy_umax, n_boot=30):
    rng = np.random.RandomState(42)
    boot = [float(np.quantile(u_val[rng.randint(0, len(u_val), len(u_val))], 0.90)) for _ in range(n_boot)]
    return {"boot_mean": float(np.mean(boot)), "boot_std": float(np.std(boot)), "policy_umax": policy_umax}


def sensitivity_grid(c_test, u_test, y_test):
    grid = []
    for umax in np.round(np.quantile(u_test, [0.75, 0.85, 0.90, 0.95, 0.99]), 4):
        for tb in np.round(np.linspace(0.55, 0.98, 12), 3):
            for tw in np.round(np.linspace(0.05, 0.60, 12), 3):
                if tw >= tb:
                    continue
                m = zone_metrics(zone_assign(c_test, u_test, tw, tb, umax), y_test)
                grid.append({"tw": float(tw), "tb": float(tb), "umax": float(umax),
                             **{k: round(v, 4) for k, v in m.items()}})
    return grid


def fit_and_score(model, data, idxs=None, T=20, **train_kw):
    """Навчання + MC-оцінки + калібрування; повертає калібровані оцінки val/test і u."""
    sel = (lambda X: X[:, idxs]) if idxs is not None else (lambda X: X)
    model, _, _ = train_model(model, sel(data["X_train"]), data["y_train"], sel(data["X_val"]), data["y_val"],
                              lr=1e-3, **train_kw)
    s_v, u_v, _ = mc_predict(model, sel(data["X_val"]), T=T)
    s_t, u_t, _ = mc_predict(model, sel(data["X_test"]), T=T)
    _, cal = fit_temperature(s_v, data["y_val"])
    return cal(s_v), cal(s_t), u_v, u_t


def radius_ablation(data):
    out = []
    for name, idxs in (("r0_166", IDX_R0), ("r1_186", IDX_R1), ("r2_188", IDX_R2)):
        t0 = time.time()
        torch.manual_seed(42)
        cv, ct, _, _ = fit_and_score(build_main_for_subset(idxs), data, idxs, max_epochs=30, patience=5,
                                     tag=f"r{name[1]}")
        t_f1, _ = val_thresholds(cv, data["y_val"])
        out.append({"radius": name, "n_features": len(idxs),
                    "test_auprc": float(average_precision_score(data["y_test"], ct)),
                    "test_roc_auc": float(roc_auc_score(data["y_test"], ct)),
                    "test_f1_at_valopt": threshold_metrics(ct, data["y_test"], t_f1)["f1"],
                    "train_seconds": time.time() - t0})
        print("  ", out[-1], flush=True)
    return out


def baseline_row(name, p_val, p_test, data):
    t_f1, _ = val_thresholds(p_val, data["y_val"])
    return {"model": name, "roc_auc": float(roc_auc_score(data["y_test"], p_test)),
            "auprc": float(average_precision_score(data["y_test"], p_test)),
            **{f"opt_{k}": v for k, v in threshold_metrics(p_test, data["y_test"], t_f1).items()}}


def baselines(data):
    Xtr, ytr, Xva, Xte = data["X_train"], data["y_train"], data["X_val"], data["X_test"]
    rows = []
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    rows.append(baseline_row("LogReg", lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1], data))
    for seed in range(3):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample",
                                    random_state=seed).fit(Xtr, ytr)
        rows.append(baseline_row(f"RF_seed{seed}", rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1], data))
    sw = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, ytr, sample_weight=sw)
    rows.append(baseline_row("HistGB", hgb.predict_proba(Xva)[:, 1], hgb.predict_proba(Xte)[:, 1], data))
    for r in rows:
        print(f"   {r['model']:<10} ROC-AUC {r['roc_auc']:.4f}  AUPRC {r['auprc']:.4f}  F1 {r['opt_f1']:.4f}")
    return rows


def seed_stability(data, seeds=(42, 1, 2, 3, 4)):
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        cv, ct, u_v, u_t = fit_and_score(build_main(), data, max_epochs=40, patience=6, seed=seed,
                                         tag=f"seed{seed}")
        y_test = data["y_test"]
        t_f1, _ = val_thresholds(cv, data["y_val"])
        tm = threshold_metrics(ct, y_test, t_f1)
        zc, _ = pick_zoning(cv, u_v, data["y_val"])
        zt = zone_metrics(zone_assign(ct, u_t, zc["tw"], zc["tb"], zc["umax"]), y_test)
        runs.append({"seed": seed, "roc_auc": float(roc_auc_score(y_test, ct)),
                     "auprc": float(average_precision_score(y_test, ct)),
                     "f1": tm["f1"], "precision": tm["precision"], "recall": tm["recall"],
                     "esc_recall": zt["esc_recall"], "black_precision": zt["black_precision"],
                     "esc_load": zt["esc_load"], "avg_cost": zt["avg_cost"]})
        print("  ", runs[-1], flush=True)
    arr = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "seed"}
    return {"runs": runs,
            "mean_std": {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in arr.items()}}


def integrated_gradients(model, data, c_test, u_test, zcfg, steps=64):
    """Атрибуція за інтегрованими градієнтами (нульова базова точка) для першої
    правильно виявленої транзакції зони BLACK; частки трьох груп профілю."""
    z = zone_assign(c_test, u_test, zcfg["tw"], zcfg["tb"], zcfg["umax"])
    i = int(np.where((z == BLACK) & (data["y_test"] == 1))[0][0])
    xi = torch.tensor(data["X_test"][i:i + 1])
    base = torch.zeros_like(xi)
    path = (base + torch.linspace(0, 1, steps).view(-1, 1) * (xi - base)).requires_grad_(True)
    model.eval()
    grads = torch.autograd.grad(model(path).sum(), path)[0]
    ig = ((xi - base) * grads.mean(0, keepdim=True)).squeeze(0).detach().numpy()
    a = np.abs(ig)
    top = np.argsort(-a)[:12]
    return {"tx_index": i, "risk_score": float(c_test[i]),
            "group_share": {n: float(a[g].sum() / a.sum()) for n, g in zip(GROUP_NAMES, GROUPS)},
            "top_features": [{"name": data["profile_cols"][j], "attr": float(ig[j])} for j in top]}


def curves(c_test, u_test, y_test):
    out = {}
    prec, rec, _ = precision_recall_curve(y_test, c_test)
    st = max(1, len(prec) // 300)
    out["pr_curve"] = {"precision": prec[::st].tolist(), "recall": rec[::st].tolist()}
    fpr, tpr, _ = roc_curve(y_test, c_test)
    st = max(1, len(fpr) // 300)
    out["roc_curve"] = {"fpr": fpr[::st].tolist(), "tpr": tpr[::st].tolist()}
    bins = np.linspace(0, 1, 11)
    rel = []
    for i in range(10):
        m = (c_test >= bins[i]) & (c_test < bins[i + 1] + (1e-9 if i == 9 else 0))
        rel.append({"bin_mid": float((bins[i] + bins[i + 1]) / 2),
                    "conf": float(c_test[m].mean()) if m.sum() else None,
                    "acc": float(y_test[m].mean()) if m.sum() else None, "n": int(m.sum())})
    out["reliability"] = rel
    sub = np.random.RandomState(3).choice(len(c_test), 1500, replace=False)
    out["scatter"] = {"s": np.round(c_test[sub], 4).tolist(), "u": np.round(u_test[sub], 4).tolist(),
                      "y": y_test[sub].astype(int).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="пропустити абляцію, бейзлайни і 5 запусків")
    args = ap.parse_args()
    require(SCORES_GTT, "python src/evaluate.py")
    set_threads()
    data = load_splits()
    sc = load_scores(SCORES_GTT)
    zcfg = load_json(RESULTS_MAIN)["gtt"]["zoning"]["config"]
    model = load_main()
    res = load_json(RESULTS_ANALYSIS)

    print("1. збіжність u(T)...", flush=True)
    res["u_convergence"] = u_convergence(model, data, zcfg)
    res["umax_stability"] = umax_stability(sc["u_val"], zcfg["umax"])
    print("2. сітка чутливості...", flush=True)
    grid = sensitivity_grid(sc["c_test"], sc["u_test"], sc["y_test"])
    res["sensitivity_grid_size"] = len(grid)
    save_json(grid, SENS_GRID, compact=True)
    if not args.fast:
        print("3. абляція радіуса...", flush=True)
        res["radius_ablation"] = radius_ablation(data)
        print("4. бейзлайни...", flush=True)
        res["baselines"] = baselines(data)
        print("5. стабільність за 5 запусками...", flush=True)
        res["seeds"] = seed_stability(data)
    print("6. інтегровані градієнти...", flush=True)
    res["xai"] = integrated_gradients(model, data, sc["c_test"], sc["u_test"], zcfg)
    res.update(curves(sc["c_test"], sc["u_test"], sc["y_test"]))
    save_json(res, RESULTS_ANALYSIS)
    print("Збережено", RESULTS_ANALYSIS)


if __name__ == "__main__":
    main()
