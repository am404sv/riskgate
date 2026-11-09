# -*- coding: utf-8 -*-
"""Навчання моделей risk engine на профілі з 188 ознак.

  mlp — MLP 256-128 (довідкова модель), ~15 с
  ftt — трансформер із токеном на кожну ознаку; бюджет навчання свідомо обмежено
        трьома епохами: модель потрібна для оцінювання вартості пофічевої токенізації,
        а не для змагання за якість; ~12 хв
  gtt — груповий трансформер над PLE-кодуванням (робоча модель), ~5 хв

Ваги зберігаються в results/model_<tag>.pt (робоча модель перезаписує постачену
results/model_gtt.pt), після навчання запускається оцінювання (evaluate.py).
Через багатопотокові обчислення повторне навчання дає близькі, але не побітово
однакові результати.

    python src/train_models.py                 # усі три моделі
    python src/train_models.py --models gtt    # лише робоча
"""
import argparse
import time

import numpy as np
import torch

from common import RESULTS_MAIN, load_json, load_splits, save_json, set_threads
from engine import train_model
from evaluate import evaluate_models, weights_path
from models import GROUPS, PLE_EDGES, FeatureTokenTransformer, GTTPLE, MLPRisk, make_ple_edges

# тег -> гіперпараметри навчання
PLAN = {
    "mlp": {"lr": 2e-3, "max_epochs": 60, "patience": 8},
    "ftt": {"lr": 1e-3, "max_epochs": 3, "patience": 2},
    "gtt": {"lr": 1e-3, "max_epochs": 40, "patience": 7},
}


def make_model(tag, data):
    if tag == "mlp":
        return MLPRisk()
    if tag == "ftt":
        return FeatureTokenTransformer()
    edges = make_ple_edges(data["X_train"], n_bins=24)
    np.save(PLE_EDGES, edges.numpy())
    return GTTPLE(GROUPS, edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="mlp,ftt,gtt", help="перелік через кому: mlp,ftt,gtt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tags = [t.strip() for t in args.models.split(",") if t.strip()]

    set_threads()
    data = load_splits()
    print(f"train {data['X_train'].shape}  val {data['X_val'].shape}  test {data['X_test'].shape}", flush=True)

    main_res = load_json(RESULTS_MAIN)
    for tag in tags:
        torch.manual_seed(args.seed)
        model = make_model(tag, data)
        t0 = time.time()
        model, best, hist = train_model(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                                        seed=args.seed, tag=tag, **PLAN[tag])
        train_s = time.time() - t0
        torch.save(model.state_dict(), weights_path(tag))
        main_res[tag] = {"tag": tag, "train_seconds": train_s, "epochs": len(hist),
                         "sec_per_epoch": train_s / max(len(hist), 1), "val_best_auprc": float(best),
                         "history": hist}
        save_json(main_res, RESULTS_MAIN)
        print(f"[{tag}] навчено за {train_s:.0f} с, епох {len(hist)}, найкраща val AUPRC {best:.4f}", flush=True)

    evaluate_models(tags, data)


if __name__ == "__main__":
    main()
