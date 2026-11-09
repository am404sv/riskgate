# -*- coding: utf-8 -*-
"""Завантаження набору Elliptic Bitcoin Dataset (три CSV-файли, ~150 МБ у zip).

Основне джерело — дзеркало PyTorch Geometric (data.pyg.org), резервне — Hugging Face.
Файли зберігаються в data/elliptic/.
"""
import os
import sys
import time
import urllib.request
import zipfile

from common import DATA_RAW

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

SOURCES = [
    ("pyg", "https://data.pyg.org/datasets/elliptic/{}.zip", True),
    ("hf", "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/{}", False),
]


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "riskgate-downloader"})
    total, t0 = 0, time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (25 << 20) < (1 << 20):
                speed = total / 1e6 / max(time.time() - t0, 0.01)
                print(f"  {os.path.basename(dest)}: {total / 1e6:.0f} МБ ({speed:.1f} МБ/с)", flush=True)
    return total


def unzip_csv(zip_path, final):
    with zipfile.ZipFile(zip_path) as z:
        member = [m for m in z.namelist() if m.endswith(".csv")][0]
        with z.open(member) as src, open(final, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    os.remove(zip_path)


def download(fname):
    final = os.path.join(DATA_RAW, fname)
    if os.path.exists(final) and os.path.getsize(final) > 1000:
        print(f"Пропуск {fname}: уже є ({os.path.getsize(final) / 1e6:.1f} МБ)")
        return True
    for name, tmpl, is_zip in SOURCES:
        url = tmpl.format(fname)
        dest = final + (".zip" if is_zip else "")
        try:
            print(f"GET [{name}] {url}", flush=True)
            n = fetch(url, dest)
            print(f"  отримано {n / 1e6:.1f} МБ", flush=True)
            if is_zip:
                unzip_csv(dest, final)
            return True
        except Exception as e:  # пробуємо наступне джерело
            print(f"  помилка [{name}]: {e}", flush=True)
            for p in (final, final + ".zip"):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    return False


def main():
    os.makedirs(DATA_RAW, exist_ok=True)
    for fname in FILES:
        if not download(fname):
            sys.exit(f"Не вдалося завантажити {fname}")
    for fname in FILES:
        p = os.path.join(DATA_RAW, fname)
        print(f"  {fname}: {os.path.getsize(p) / 1e6:.1f} МБ")
    print("Завантаження завершено.")


if __name__ == "__main__":
    main()
