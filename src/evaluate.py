# -*- coding: utf-8 -*-
"""Оцінювання моделей на тестовій вибірці (кроки 35–49).

Для кожної моделі, ваги якої є в results/, обчислює ROC-AUC/AUPRC, калібрування,
робочі точки (пороги лише з валідації) і зонування; оновлює results/results_main.json.
Для робочої моделі зберігає results/scores_gtt.npz — вхід для кроків політик.

    python src/evaluate.py                # усі доступні моделі
    python src/evaluate.py --models gtt   # лише робоча модель
"""
import argparse
import os

import torch

from common import RESULTS, RESULTS_MAIN, SCORES_GTT, load_json, load_splits, save_json, set_threads
from engine import evaluate, evaluate_single, save_scores
from features import PROFILE_COLS
from models import (IDX_AGG, IDX_CTX, IDX_LOCAL, FeatureTokenTransformer, MLPRisk, build_main,
                    load_state, n_params)

# тег -> (конструктор, режим оцінювання)
REGISTRY = {
    "mlp": (MLPRisk, "mc"),
    "gtt": (build_main, "mc"),
    "ftt": (FeatureTokenTransformer, "single"),
}
KINDS = {"mlp": "mlp", "gtt": "gtt_ple", "ftt": "ftt"}


def weights_path(tag):
    return os.path.join(RESULTS, f"model_{tag}.pt")


def meta_block(data):
    y = data["y_train"]
    return {"n_features": len(PROFILE_COLS),
            "profile_groups": {"local": len(IDX_LOCAL), "aggregated": len(IDX_AGG),
                               "context": len(IDX_CTX)},
            "train": int(len(y)), "val": int(len(data["y_val"])), "test": int(len(data["y_test"])),
            "pos_weight": float((y == 0).sum() / (y == 1).sum()),
            "torch_threads": torch.get_num_threads(), "model_kind": "gtt_ple"}


def evaluate_models(tags, data=None):
    data = data or load_splits()
    main = load_json(RESULTS_MAIN)
    main["meta"] = meta_block(data)
    for tag in tags:
        path = weights_path(tag)
        if not os.path.exists(path):
            print(f"[{tag}] ваг {os.path.basename(path)} немає — пропуск "
                  f"(створюються командою python src/train_models.py)")
            continue
        ctor, mode = REGISTRY[tag]
        model = load_state(ctor(), path)
        print(f"[{tag}] оцінювання ({'MC-Dropout, T=30' if mode == 'mc' else 'один прохід'})...", flush=True)
        entry = main.get(tag, {})
        entry.update({"tag": tag, "kind": KINDS[tag], "params": n_params(model)})
        if mode == "mc":
            r, scores = evaluate(model, data, T=30)
            if tag == "gtt":
                save_scores(SCORES_GTT, scores)
        else:
            r = evaluate_single(model, data)
        entry.update(r)
        main[tag] = entry
        print(f"  test ROC-AUC {r['test_mc']['roc_auc']:.4f}  AUPRC {r['test_mc']['auprc']:.4f}  "
              f"F1@opt {r['test_at_f1opt']['f1']:.4f}", flush=True)
    save_json(main, RESULTS_MAIN)
    print("Оновлено", RESULTS_MAIN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="mlp,gtt,ftt", help="перелік через кому: mlp,gtt,ftt")
    args = ap.parse_args()
    set_threads()
    evaluate_models([t.strip() for t in args.models.split(",") if t.strip()])


if __name__ == "__main__":
    main()
