# -*- coding: utf-8 -*-
"""Послідовний запуск усього конвеєра.

    python run_all.py              # дані -> оцінювання постаченої моделі -> політики -> аналіз (--fast)
                                   #   -> бенчмарки -> демонстрація шлюзу            (~30 хв на 4-ядерному CPU)
    python run_all.py --train      # те саме, але з навчанням моделей і повним аналізом (~2 год)
    python run_all.py --from policies   # продовжити з указаного кроку
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("download", ["download_elliptic.py"]),
    ("prep", ["prep_data.py"]),
    ("train", ["train_models.py"]),
    ("evaluate", ["evaluate.py"]),
    ("policies", ["policies.py"]),
    ("pattern", ["pattern_boost.py"]),
    ("adaptive", ["adaptive_policy.py"]),
    ("analysis", ["analysis_extra.py"]),
    ("bench", ["bench_speed.py"]),
    ("gateway", ["gateway_service.py", "--n", "5000"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="навчити моделі заново і виконати повний аналіз")
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS], help="почати з кроку")
    args = ap.parse_args()

    names = [s for s, _ in STEPS]
    first = names.index(args.start) if args.start else 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    trained = False
    for name, cmd in STEPS[first:]:
        if name == "train" and not args.train:
            continue
        if name == "evaluate" and trained:
            continue                                   # train_models.py уже оцінив моделі
        if name == "analysis" and not args.train:
            cmd = cmd + ["--fast"]
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", cmd[0])] + cmd[1:], cwd=ROOT, env=env)
        if rc != 0:
            sys.exit(f"Крок {name} завершився з кодом {rc}")
        trained = trained or name == "train"
        print(f"=== {name}: {time.time() - t0:.0f} с", flush=True)
    print("\nГотово. Результати в results/.")


if __name__ == "__main__":
    main()
