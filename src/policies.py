# -*- coding: utf-8 -*-
"""Дві політики зонування з фіксованими порогами (добір лише на валідації):

  П-Б (PB, збалансована): висока повнота ескалації із запасом на дрейф
  П-Е (PE, економна):     мінімальне навантаження за прийнятної повноти ескалації

Результат: results_main.json -> gtt.policies
"""
import numpy as np

from common import RESULTS_MAIN, SCORES_GTT, load_json, require, save_json
from engine import load_scores, zone_assign, zone_metrics

ESC_MIN_VAL = {"PB": 0.985, "PE": 0.95}


def search(c_val, u_val, y_val, esc_min_val, black_prec_min=0.95):
    """Мінімум навантаження G∪B за умов precision BLACK і повноти ескалації на валідації."""
    tb_grid = np.unique(np.quantile(c_val, np.linspace(0.85, 0.999, 80)))
    tw_grid = np.unique(np.quantile(c_val, np.linspace(0.02, 0.90, 120)))
    u_grid = np.quantile(u_val, [0.75, 0.85, 0.90, 0.95, 0.99])
    best = None
    for umax in u_grid:
        for tb in tb_grid:
            for tw in tw_grid:
                if tw >= tb:
                    continue
                m = zone_metrics(zone_assign(c_val, u_val, tw, tb, umax), y_val)
                if m["black_precision"] < black_prec_min or m["esc_recall"] < esc_min_val:
                    continue
                key = (-m["esc_load"], m["esc_recall"])
                if best is None or key > best[0]:
                    best = (key, {"tw": float(tw), "tb": float(tb), "umax": float(umax)}, m)
    return best[1], best[2]


def main():
    require(SCORES_GTT, "python src/evaluate.py")
    sc = load_scores(SCORES_GTT)
    res = {}
    for name, esc_min in ESC_MIN_VAL.items():
        cfg, mval = search(sc["c_val"], sc["u_val"], sc["y_val"], esc_min)
        z = zone_assign(sc["c_test"], sc["u_test"], cfg["tw"], cfg["tb"], cfg["umax"])
        mtest = zone_metrics(z, sc["y_test"])
        res[name] = {"config": cfg, "val": mval, "test": mtest, "esc_min_val": esc_min}
        print(name, "пороги:", {k: round(v, 4) for k, v in cfg.items()})
        print("  val :", {k: round(v, 4) for k, v in mval.items()})
        print("  test:", {k: round(v, 4) for k, v in mtest.items()})
    main_res = load_json(RESULTS_MAIN)
    main_res["gtt"]["policies"] = res
    save_json(main_res, RESULTS_MAIN)


if __name__ == "__main__":
    main()
