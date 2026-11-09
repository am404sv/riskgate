# -*- coding: utf-8 -*-
"""Підготовка даних: Elliptic -> профіль із 188 ознак і суворе часове розбиття.

  train = часові кроки 1..30   (26 905 розмічених транзакцій)
  val   = часові кроки 31..34  ( 2 989)
  test  = часові кроки 35..49  (16 670)

Результат: data/prepared/elliptic_188.npz
"""
import os

import numpy as np
import pandas as pd

from common import DATA_PREP, DATA_RAW, PREPARED_NPZ, require
from features import PROFILE_COLS, add_context_features, read_edges, read_features

TRAIN_LAST, VAL_LAST = 30, 34
EXPECTED = (26905, 2989, 16670)


def main():
    require(os.path.join(DATA_RAW, "elliptic_txs_features.csv"), "python src/download_elliptic.py")
    print("Читання CSV...", flush=True)
    feats = read_features(os.path.join(DATA_RAW, "elliptic_txs_features.csv"))
    classes = pd.read_csv(os.path.join(DATA_RAW, "elliptic_txs_classes.csv"))
    edges = read_edges(os.path.join(DATA_RAW, "elliptic_txs_edgelist.csv"))
    print("ознаки:", feats.shape, "класи:", classes.shape, "ребра:", edges.shape, flush=True)

    classes = classes.rename(columns={"class": "label"})
    df = feats.merge(classes, on="txId", how="left")
    df = add_context_features(df, edges)

    # мітки: "1" — незаконна (illicit), "2" — законна, "unknown" — без мітки
    lab = df[df["label"].astype(str).isin(["1", "2"])].copy()
    lab["y"] = (lab["label"].astype(str) == "1").astype(np.int64)
    print("розмічено:", len(lab), "з них незаконних:", int(lab["y"].sum()), flush=True)

    parts = {
        "train": lab[lab["time_step"] <= TRAIN_LAST],
        "val": lab[(lab["time_step"] > TRAIN_LAST) & (lab["time_step"] <= VAL_LAST)],
        "test": lab[lab["time_step"] > VAL_LAST],
    }
    sizes = tuple(len(p) for p in parts.values())
    print("train/val/test:", sizes, flush=True)
    if sizes != EXPECTED:
        print("УВАГА: розміри вибірок відрізняються від очікуваних", EXPECTED, flush=True)
    y_tr = parts["train"]["y"].values
    print(f"частка незаконних у train: {y_tr.mean():.4f}, pos_weight: {(1 - y_tr.mean()) / y_tr.mean():.3f}",
          flush=True)

    arrs = {}
    for name, part in parts.items():
        arrs[f"X_{name}"] = part[PROFILE_COLS].values.astype(np.float32)
        arrs[f"y_{name}"] = part["y"].values.astype(np.int64)
        arrs[f"steps_{name}"] = part["time_step"].values.astype(np.int32)
        arrs[f"txid_{name}"] = part["txId"].values.astype(np.int64)
    os.makedirs(DATA_PREP, exist_ok=True)
    np.savez_compressed(PREPARED_NPZ, **arrs, profile_cols=np.array(PROFILE_COLS))
    print("Збережено:", PREPARED_NPZ, flush=True)


if __name__ == "__main__":
    main()
