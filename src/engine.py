# -*- coding: utf-8 -*-
"""Навчання, MC-Dropout, калібрування, робочі пороги та зонування WHITE/GRAY/BLACK."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

WHITE, GRAY, BLACK = 0, 1, 2
# нормовані вартості режимів STD / ENH / QUAR (у.о. на транзакцію)
MODE_COSTS = (1.0, 6.702, 7.472)


# ------------------------------- навчання -------------------------------
def train_model(model, Xtr, ytr, Xva, yva, lr=1e-3, batch=512, max_epochs=60, patience=8,
                seed=42, tag=""):
    """AdamW + косинусний розклад, зважена BCE, рання зупинка за AUPRC на валідації."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva)
    n = len(Xtr_t)
    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).numpy()
        auprc, auc = average_precision_score(yva, pv), roc_auc_score(yva, pv)
        history.append({"epoch": ep, "loss": tot / n, "val_auprc": float(auprc), "val_auc": float(auc)})
        if ep % 5 == 0 or ep == 1:
            print(f"  [{tag}] епоха {ep:02d} loss {tot / n:.4f} val AUPRC {auprc:.4f} "
                  f"val AUC {auc:.4f} ({time.time() - t0:.0f} с)", flush=True)
        if auprc > best + 1e-4:
            best, best_state, bad = auprc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, history


# ------------------------------- інференс -------------------------------
@torch.no_grad()
def single_predict(model, X, batch=4096):
    model.eval()
    X_t = torch.tensor(X)
    return np.concatenate([torch.sigmoid(model(X_t[i:i + batch])).numpy()
                           for i in range(0, len(X_t), batch)])


@torch.no_grad()
def mc_predict(model, X, T=30, batch=4096, seed=123):
    """MC-Dropout: згладжена оцінка s̄ (середнє) і невизначеність u (СКВ) за T прогонів."""
    torch.manual_seed(seed)
    model.train()                                   # dropout увімкнено
    X_t = torch.tensor(X)
    P = np.stack([np.concatenate([torch.sigmoid(model(X_t[i:i + batch])).numpy()
                                  for i in range(0, len(X_t), batch)])
                  for _ in range(T)])               # [T, N]
    model.eval()
    return P.mean(0), P.std(0), P


# ------------------------------ калібрування ----------------------------
def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log(1 - p)


def fit_temperature(p_val, y_val):
    """Температурне масштабування логітів; повертає T і функцію калібрування."""
    z = torch.tensor(_logit(p_val), dtype=torch.float64)
    y = torch.tensor(y_val, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(z / torch.exp(logT), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(logT).item())
    return T, lambda p: 1.0 / (1.0 + np.exp(-_logit(p) / T))


def apply_temperature(p, T):
    return 1.0 / (1.0 + np.exp(-_logit(p) / T))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        if m.sum() > 0:
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


# ---------------------------- робочі пороги -----------------------------
def threshold_metrics(p, y, thr):
    pred = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def val_thresholds(p_val, y_val, prec_target=0.90):
    """Пороги, обрані лише на валідації: F1-оптимальний і мінімальний з precision >= prec_target."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    t_f1 = float(thr[np.argmax(f1s[:-1])])
    ok = np.where(prec[:-1] >= prec_target)[0]
    t_hp = float(thr[ok[0]]) if len(ok) else t_f1
    return t_f1, t_hp


def ranking_metrics(p, y):
    return {"roc_auc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


# ------------------------------- зонування ------------------------------
def zone_assign(p, u, tw, tb, umax):
    """BLACK: p >= t_b; WHITE: p < t_w і u < u_max; решта — GRAY."""
    z = np.full(len(p), GRAY)
    z[p >= tb] = BLACK
    z[(p < tw) & (u < umax)] = WHITE
    return z


def zone_metrics(z, y, costs=MODE_COSTS):
    black, gray, white = z == BLACK, z == GRAY, z == WHITE
    esc = gray | black
    n_pos = max(y.sum(), 1)
    return {
        "share_white": float(white.mean()), "share_gray": float(gray.mean()),
        "share_black": float(black.mean()),
        "black_precision": float(y[black].mean()) if black.sum() else 0.0,
        "black_recall": float(y[black].sum() / n_pos),
        "esc_recall": float(y[esc].sum() / n_pos),
        "esc_load": float(esc.mean()),
        "missed_in_white": float(y[white].sum() / n_pos),
        "avg_cost": float(white.mean() * costs[0] + gray.mean() * costs[1] + black.mean() * costs[2]),
    }


def pick_zoning(p_val, u_val, y_val, black_prec_min=0.95, esc_recall_min=0.90):
    """Сітковий добір (t_w, t_b, u_max) на валідації: максимум повноти ескалації з
    штрафом за навантаження за умови precision BLACK >= black_prec_min
    (умова поступово послаблюється, якщо недосяжна)."""
    tb_grid = np.unique(np.quantile(p_val, np.linspace(0.80, 0.999, 60)))
    tw_grid = np.unique(np.quantile(p_val, np.linspace(0.30, 0.95, 60)))
    u_grid = np.quantile(u_val, [0.80, 0.85, 0.90, 0.95, 0.98])
    for prec_min in (black_prec_min, 0.85, 0.7, 0.0):
        best = None
        for umax in u_grid:
            for tb in tb_grid:
                for tw in tw_grid:
                    if tw >= tb:
                        continue
                    m = zone_metrics(zone_assign(p_val, u_val, tw, tb, umax), y_val)
                    if m["black_precision"] < prec_min:
                        continue
                    key = (m["esc_recall"] >= esc_recall_min,
                           m["esc_recall"] - 0.6 * m["esc_load"], -m["esc_load"])
                    if best is None or key > best[0]:
                        best = (key, {"tw": float(tw), "tb": float(tb), "umax": float(umax)}, m)
        if best is not None:
            return best[1], best[2]
    raise RuntimeError("порожня сітка зонування")


# ------------------------- повне оцінювання моделі ----------------------
def evaluate(model, data, T=30):
    """Тест без MC і з MC-Dropout, калібрування, робочі точки, зонування.
    Повертає (метрики, масиви оцінок для подальших кроків)."""
    X_val, y_val, X_test, y_test = data["X_val"], data["y_val"], data["X_test"], data["y_test"]
    r = {"test_single": ranking_metrics(single_predict(model, X_test), y_test)}
    s_val, u_val, _ = mc_predict(model, X_val, T=T)
    s_test, u_test, _ = mc_predict(model, X_test, T=T)
    r["test_mc"] = ranking_metrics(s_test, y_test)
    Temp, cal = fit_temperature(s_val, y_val)
    c_val, c_test = cal(s_val), cal(s_test)
    r["calibration"] = {"temperature": Temp, "ece_before": ece(s_test, y_test),
                        "ece_after": ece(c_test, y_test)}
    t_f1, t_hp = val_thresholds(c_val, y_val)
    r["test_at_05"] = threshold_metrics(c_test, y_test, 0.5)
    r["test_at_f1opt"] = threshold_metrics(c_test, y_test, t_f1)
    r["test_at_p90"] = threshold_metrics(c_test, y_test, t_hp)
    r["op_thresholds"] = {"f1opt": t_f1, "p90": t_hp}
    zcfg, zval = pick_zoning(c_val, u_val, y_val)
    z_test = zone_assign(c_test, u_test, zcfg["tw"], zcfg["tb"], zcfg["umax"])
    r["zoning"] = {"config": zcfg, "val": zval, "test": zone_metrics(z_test, y_test)}
    scores = {"s_test": s_test, "u_test": u_test, "c_test": c_test, "y_test": y_test,
              "s_val": s_val, "u_val": u_val, "c_val": c_val, "y_val": y_val}
    return r, scores


def evaluate_single(model, data):
    """Оцінювання одним детермінованим проходом (для довідкових моделей)."""
    X_val, y_val, X_test, y_test = data["X_val"], data["y_val"], data["X_test"], data["y_test"]
    p_val, p_test = single_predict(model, X_val), single_predict(model, X_test)
    t_f1, t_hp = val_thresholds(p_val, y_val)
    m = ranking_metrics(p_test, y_test)
    return {"eval_mode": "single-pass", "test_single": m, "test_mc": m,
            "test_at_05": threshold_metrics(p_test, y_test, 0.5),
            "test_at_f1opt": threshold_metrics(p_test, y_test, t_f1),
            "test_at_p90": threshold_metrics(p_test, y_test, t_hp)}


def save_scores(path, scores):
    arrs = {k: (v.astype(np.int8) if k.startswith("y_") else v.astype(np.float32))
            for k, v in scores.items()}
    np.savez_compressed(path, **arrs)


def load_scores(path):
    sc = np.load(path)
    return {k: (sc[k].astype(int) if k.startswith("y_") else sc[k]) for k in sc.files}
