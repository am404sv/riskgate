# -*- coding: utf-8 -*-
"""Адаптивна політика зонування П-А (штатна).

Пороги t_w, u_max і патерн-поріг t_p задаються квантилями (q_w, q_u, q_p) вікна
потоку — мітки не потрібні, тож політика застосовна в робочому шлюзі. Поріг
карантину t_b абсолютний: мінімальний, за якого precision BLACK на валідації >= 0.99
(запас на часовий дрейф). Квантилі добираються лише на валідації; на тесті пороги
перераховуються для кожного часового кроку за вікном цього кроку.

Результат: results_main.json -> gtt.policy_adaptive
"""
import numpy as np

from common import RESULTS_MAIN, RHO_NPZ, SCORES_GTT, load_json, load_raw, require, save_json
from engine import BLACK, GRAY, WHITE, load_scores, zone_metrics


def pick_tb(c, y, prec_target=0.99):
    for tb in np.unique(np.quantile(c, np.linspace(0.90, 0.9995, 200))):
        m = c >= tb
        if m.sum() and y[m].mean() >= prec_target:
            return float(tb)
    return float(np.quantile(c, 0.9995))


def assign_stream(c, u, rho, steps, qw, qu, qp, tb):
    """Зонування з порогами, перерахованими за вікном кожного часового кроку."""
    z = np.full(len(c), GRAY)
    for st in np.unique(steps):
        m = steps == st
        tw, umax, tp = np.quantile(c[m], qw), np.quantile(u[m], qu), np.quantile(rho[m], qp)
        zz = np.full(m.sum(), GRAY)
        zz[c[m] >= tb] = BLACK
        white = (c[m] < tw) & (u[m] < umax)
        zz[white] = WHITE
        zz[white & (rho[m] >= tp)] = GRAY          # патерн-алерт W -> G
        z[m] = zz
    return z


def main():
    require(SCORES_GTT, "python src/evaluate.py")
    require(RHO_NPZ, "python src/pattern_boost.py")
    sc = load_scores(SCORES_GTT)
    rho = np.load(RHO_NPZ)
    raw = load_raw()
    c_val, u_val, y_val, r_val, st_val = sc["c_val"], sc["u_val"], sc["y_val"], rho["rho_val"], raw["steps_val"]
    c_test, u_test, y_test, r_test, st_test = (sc["c_test"], sc["u_test"], sc["y_test"], rho["rho_test"],
                                               raw["steps_test"])

    tb = pick_tb(c_val, y_val)
    print(f"t_b (precision BLACK на валідації >= 0.99): {tb:.4f}")

    # мінімум середньої вартості за умов precision BLACK >= 0.95 і повноти ескалації >= 0.985
    best = None
    for qw in np.linspace(0.55, 0.95, 17):
        for qu in (0.80, 0.90, 0.95, 0.99):
            for qp in (0.55, 0.65, 0.75, 0.85, 0.95):
                m = zone_metrics(assign_stream(c_val, u_val, r_val, st_val, qw, qu, qp, tb), y_val)
                if m["black_precision"] < 0.95 or m["esc_recall"] < 0.985:
                    continue
                if best is None or -m["avg_cost"] > best[0]:
                    best = (-m["avg_cost"], {"qw": float(qw), "qu": float(qu), "qp": float(qp)}, m)
    cfg, mval = best[1], best[2]
    z_test = assign_stream(c_test, u_test, r_test, st_test, cfg["qw"], cfg["qu"], cfg["qp"], tb)
    mtest = zone_metrics(z_test, y_test)
    print("П-А квантилі:", cfg)
    print("  val :", {k: round(v, 4) for k, v in mval.items()})
    print("  test:", {k: round(v, 4) for k, v in mtest.items()})

    # стійкість до дрейфу: показники за часовими кроками тесту
    per_step = []
    for st in np.unique(st_test):
        m = st_test == st
        mm = zone_metrics(z_test[m], y_test[m])
        per_step.append({"step": int(st), "n": int(m.sum()),
                         "esc_recall": round(mm["esc_recall"], 4) if y_test[m].sum() else None,
                         "esc_load": round(mm["esc_load"], 4),
                         "black_precision": round(mm["black_precision"], 4) if (z_test[m] == BLACK).sum() else None})

    main_res = load_json(RESULTS_MAIN)
    main_res["gtt"]["policy_adaptive"] = {"config": {**cfg, "tb": tb}, "val": mval, "test": mtest,
                                          "per_step": per_step}
    save_json(main_res, RESULTS_MAIN)


if __name__ == "__main__":
    main()
