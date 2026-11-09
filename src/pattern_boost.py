# -*- coding: utf-8 -*-
"""Патерн-рівень керування: ризик оточення ρ може лише підвищити зону WHITE -> GRAY
(ніколи не послаблює контроль).

    ρ(tx) = max( s̄(tx), max_{v ∈ N1(tx)} s(v) )  — ризик 1-hop мотиву в графі переказів.

Поріг t_p добирається лише на валідації (цільова повнота ескалації >= 0.995).
Результат: results_main.json -> gtt.pattern_boost, results/pattern_rho.npz
"""
import os
import time

import numpy as np
import pandas as pd

from common import (DATA_RAW, RESULTS_MAIN, RHO_NPZ, SCORES_GTT, load_json, load_splits, require,
                    save_json, set_threads)
from engine import GRAY, WHITE, load_scores, single_predict, zone_assign, zone_metrics
from features import PROFILE_COLS, add_context_features, read_edges, read_features
from models import load_main


def neighbourhood_risk(data):
    """Оцінює всі 203 769 вершин графа одним детермінованим проходом і повертає
    максимальну оцінку серед сусідів першого кола для кожної вершини."""
    t0 = time.time()
    feats = read_features(os.path.join(DATA_RAW, "elliptic_txs_features.csv"))
    e = read_edges(os.path.join(DATA_RAW, "elliptic_txs_edgelist.csv"))
    df = add_context_features(feats, e)
    X_all = (df[PROFILE_COLS].fillna(0.0).values.astype(np.float32) - data["mu"]) / data["sd"]
    s_all = single_predict(load_main(), X_all.astype(np.float32), batch=8192)
    print(f"оцінено {len(X_all)} вершин за {time.time() - t0:.0f} с", flush=True)

    id2pos = pd.Series(np.arange(len(df)), index=df["txId"].values)
    sv, dv = e["src"].map(id2pos).values, e["dst"].map(id2pos).values
    nbr_max = np.zeros(len(df), dtype=np.float32)
    np.maximum.at(nbr_max, sv, s_all[dv])
    np.maximum.at(nbr_max, dv, s_all[sv])
    return id2pos, nbr_max


def boosted(c, u, rho, cfg, tp):
    z = zone_assign(c, u, cfg["tw"], cfg["tb"], cfg["umax"])
    z[(z == WHITE) & (rho >= tp)] = GRAY          # лише підвищення W -> G
    return z


def pick_tp(c, u, rho, y, cfg, esc_target=0.995):
    grid = np.quantile(rho, np.linspace(0.30, 0.98, 60))
    best = None
    for tp in grid:
        m = zone_metrics(boosted(c, u, rho, cfg, tp), y)
        if m["esc_recall"] >= esc_target and (best is None or -m["esc_load"] > best[0][0]):
            best = ((-m["esc_load"],), float(tp), m)
    if best is None:                               # ціль недосяжна: максимум повноти
        for tp in grid:
            m = zone_metrics(boosted(c, u, rho, cfg, tp), y)
            key = (m["esc_recall"], -m["esc_load"])
            if best is None or key > best[0]:
                best = (key, float(tp), m)
    return best[1], best[2]


def main():
    require(SCORES_GTT, "python src/evaluate.py")
    set_threads()
    data = load_splits()
    id2pos, nbr_max = neighbourhood_risk(data)

    sc = load_scores(SCORES_GTT)
    rho_val = np.maximum(sc["c_val"], nbr_max[pd.Series(data["txid_val"]).map(id2pos).values])
    rho_test = np.maximum(sc["c_test"], nbr_max[pd.Series(data["txid_test"]).map(id2pos).values])

    main_res = load_json(RESULTS_MAIN)
    if "policies" not in main_res.get("gtt", {}):
        raise SystemExit("Спершу виконайте: python src/policies.py")
    out = {}
    for pname, pol in main_res["gtt"]["policies"].items():
        cfg = pol["config"]
        tp, mval = pick_tp(sc["c_val"], sc["u_val"], rho_val, sc["y_val"], cfg)
        mtest = zone_metrics(boosted(sc["c_test"], sc["u_test"], rho_test, cfg, tp), sc["y_test"])
        out[pname] = {"t_p": tp, "val": mval, "test": mtest}
        print(pname, "t_p =", round(tp, 4))
        print("  val :", {k: round(v, 4) for k, v in mval.items()})
        print("  test:", {k: round(v, 4) for k, v in mtest.items()}, flush=True)

    main_res["gtt"]["pattern_boost"] = out
    save_json(main_res, RESULTS_MAIN)
    np.savez_compressed(RHO_NPZ, rho_val=rho_val.astype(np.float32), rho_test=rho_test.astype(np.float32))


if __name__ == "__main__":
    main()
