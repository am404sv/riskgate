# -*- coding: utf-8 -*-
"""Транзакційний профіль із 188 ознак.

Профіль = 165 базових ознак Elliptic (93 локальні + 72 агреговані)
        + часовий крок
        + 22 контекстні ознаки графа переказів (радіус оточення r <= 2).
Усі контекстні ознаки обчислюються за O(|E|).
"""
import numpy as np
import pandas as pd

N_BASE = 165
BASE_COLS = [f"f{i}" for i in range(1, N_BASE + 1)]
NB_FEATS = ["f1", "f2", "f3", "f4", "f5"]

CTX_COLS = [
    "time_step",
    "ctx_in_deg", "ctx_out_deg", "ctx_total_deg", "ctx_deg_ratio", "ctx_log_in", "ctx_log_out",
    "ctx_pred_mean_f1", "ctx_pred_mean_f2", "ctx_pred_mean_f3", "ctx_pred_mean_f4", "ctx_pred_mean_f5",
    "ctx_pred_deg_mean",
    "ctx_succ_mean_f1", "ctx_succ_mean_f2", "ctx_succ_mean_f3", "ctx_succ_mean_f4", "ctx_succ_mean_f5",
    "ctx_succ_deg_mean",
    "ctx_2hop_pred", "ctx_2hop_succ", "ctx_same_step_nbrs", "ctx_nbr_step_span",
]
PROFILE_COLS = BASE_COLS + CTX_COLS
assert len(CTX_COLS) == 23 and len(PROFILE_COLS) == 188


def read_features(path):
    """elliptic_txs_features.csv -> DataFrame зі стовпцями txId, time_step, f1..f165."""
    feats = pd.read_csv(path, header=None)
    feats = feats.rename(columns={0: "txId", 1: "time_step"})
    return feats.rename(columns={i + 2: BASE_COLS[i] for i in range(N_BASE)})


def read_edges(path):
    edges = pd.read_csv(path)
    return edges.rename(columns={edges.columns[0]: "src", edges.columns[1]: "dst"})


def add_context_features(df, e):
    """Додає до df контекстні ознаки графа e (стовпці src, dst). Повертає новий DataFrame."""
    df = df.copy()
    out_deg = e.groupby("src").size()
    in_deg = e.groupby("dst").size()

    # степені вершини (r = 1)
    df["ctx_in_deg"] = df["txId"].map(in_deg).fillna(0).astype(np.float32)
    df["ctx_out_deg"] = df["txId"].map(out_deg).fillna(0).astype(np.float32)
    df["ctx_total_deg"] = df["ctx_in_deg"] + df["ctx_out_deg"]
    df["ctx_deg_ratio"] = (df["ctx_out_deg"] + 1.0) / (df["ctx_in_deg"] + 1.0)
    df["ctx_log_in"] = np.log1p(df["ctx_in_deg"])
    df["ctx_log_out"] = np.log1p(df["ctx_out_deg"])

    # середні значення перших п'яти локальних ознак у попередників і наступників (r = 1)
    node_attr = df.set_index("txId")[NB_FEATS + ["ctx_total_deg", "time_step"]]
    pred = e.merge(node_attr, left_on="src", right_index=True, how="left")
    pred_agg = pred.groupby("dst")[NB_FEATS + ["ctx_total_deg"]].mean()
    pred_agg.columns = [f"ctx_pred_mean_{c}" for c in NB_FEATS] + ["ctx_pred_deg_mean"]
    succ = e.merge(node_attr, left_on="dst", right_index=True, how="left")
    succ_agg = succ.groupby("src")[NB_FEATS + ["ctx_total_deg"]].mean()
    succ_agg.columns = [f"ctx_succ_mean_{c}" for c in NB_FEATS] + ["ctx_succ_deg_mean"]
    df = df.merge(pred_agg, left_on="txId", right_index=True, how="left")
    df = df.merge(succ_agg, left_on="txId", right_index=True, how="left")

    # охоплення другого кола (r = 2)
    e2 = e.copy()
    e2["src_in_deg"] = e2["src"].map(in_deg).fillna(0)
    n2_pred = e2.groupby("dst")["src_in_deg"].sum()
    e2["dst_out_deg"] = e2["dst"].map(out_deg).fillna(0)
    n2_succ = e2.groupby("src")["dst_out_deg"].sum()
    df["ctx_2hop_pred"] = np.log1p(df["txId"].map(n2_pred).fillna(0)).astype(np.float32)
    df["ctx_2hop_succ"] = np.log1p(df["txId"].map(n2_succ).fillna(0)).astype(np.float32)

    # часовий контекст: сусіди в тому самому кроці та часовий розмах оточення
    nbr = pd.concat([
        e.assign(v=e["dst"], u=e["src"])[["v", "u"]],
        e.assign(v=e["src"], u=e["dst"])[["v", "u"]],
    ])
    nbr = nbr.merge(df[["txId", "time_step"]], left_on="u", right_on="txId", how="left")
    nbr = nbr.merge(df[["txId", "time_step"]], left_on="v", right_on="txId", how="left",
                    suffixes=("_u", "_v"))
    same_step = nbr[nbr["time_step_u"] == nbr["time_step_v"]].groupby("v").size()
    df["ctx_same_step_nbrs"] = np.log1p(df["txId"].map(same_step).fillna(0)).astype(np.float32)
    span = (nbr["time_step_u"] - nbr["time_step_v"]).abs().groupby(nbr["v"]).max()
    df["ctx_nbr_step_span"] = df["txId"].map(span).fillna(0).astype(np.float32)

    df[CTX_COLS] = df[CTX_COLS].fillna(0.0)
    return df
