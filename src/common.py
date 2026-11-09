# -*- coding: utf-8 -*-
"""Спільні шляхи, завантаження підготовлених даних і робота з файлами результатів."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "elliptic")
DATA_PREP = os.path.join(ROOT, "data", "prepared")
RESULTS = os.path.join(ROOT, "results")

PREPARED_NPZ = os.path.join(DATA_PREP, "elliptic_188.npz")
RESULTS_MAIN = os.path.join(RESULTS, "results_main.json")
RESULTS_SPEED = os.path.join(RESULTS, "results_speed.json")
RESULTS_ANALYSIS = os.path.join(RESULTS, "results_analysis.json")
SCORES_GTT = os.path.join(RESULTS, "scores_gtt.npz")
RHO_NPZ = os.path.join(RESULTS, "pattern_rho.npz")


def set_threads():
    """Кількість потоків PyTorch: усі ядра, крім двох (як в експериментах дисертації)."""
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 2))


def require(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"Не знайдено {os.path.relpath(path, ROOT)}. Спершу виконайте: {hint}")


def load_raw():
    """Повертає підготовлений архів (сирі ознаки, мітки, кроки, txId)."""
    require(PREPARED_NPZ, "python src/download_elliptic.py && python src/prep_data.py")
    return np.load(PREPARED_NPZ)


def load_splits():
    """Стандартизовані (за статистиками train) вибірки train/val/test."""
    d = load_raw()
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    return {
        "X_train": (X_train - mu) / sd, "y_train": d["y_train"],
        "X_val": (X_val - mu) / sd, "y_val": d["y_val"],
        "X_test": (X_test - mu) / sd, "y_test": d["y_test"],
        "steps_val": d["steps_val"], "steps_test": d["steps_test"],
        "txid_val": d["txid_val"], "txid_test": d["txid_test"],
        "profile_cols": [str(c) for c in d["profile_cols"]],
        "mu": mu, "sd": sd,
    }


def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, compact=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=1)
