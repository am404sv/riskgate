# -*- coding: utf-8 -*-
"""Мікробенчмарки швидкодії комплексу:

  1. затримка інференсу risk engine: пакети 1/64/256/1024, FP32 і INT8, 1 потік і N потоків
  2. журнал рішень Ed25519: підпис/перевірка, hash-chain, виявлення підміни (1000 записів)
  3. режими передконсенсусної обробки STD / ENH / QUAR (пакет 256)
  4. наскрізний шлюз: потік 20 000 підписаних транзакцій, середня/p95 затримка, TPS
     зі шлюзом і без нього (потоковий MC-Dropout з T = 5)

Довідкові моделі mlp і ftt беруться з results/, а якщо їх не навчено — з випадковими
вагами (на затримку ваги не впливають).
Результат: results/results_speed.json
"""
import hashlib
import io
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sklearn.metrics import average_precision_score, roc_auc_score

from common import RESULTS, RESULTS_MAIN, RESULTS_SPEED, load_json, load_splits, save_json, set_threads
from engine import WHITE, zone_assign
from models import FeatureTokenTransformer, MLPRisk, load_main, load_state, n_params, quantize_int8

B = 256              # пакет для режимів і шлюзу
N_STREAM = 20000     # довжина потоку транзакцій
MC_STREAM = 5        # MC-прогонів у потоковому режимі шлюзу
MC_ENH = 10          # MC-прогонів у режимі ENH


def load_reference(ctor, tag):
    path = os.path.join(RESULTS, f"model_{tag}.pt")
    if os.path.exists(path):
        return load_state(ctor(), path), "trained"
    torch.manual_seed(0)
    return ctor().eval(), "untrained"


def model_mb(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def bench_forward(model, X, batch, n_iter=30, warmup=5):
    xb = torch.tensor(X[:batch])
    with torch.no_grad():
        for _ in range(warmup):
            model(xb)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(xb)
    return (time.perf_counter() - t0) / n_iter * 1000.0       # мс на пакет


def mc_scores(model, xb, T):
    model.train()
    with torch.no_grad():
        P = torch.stack([torch.sigmoid(model(xb)) for _ in range(T)])
    model.eval()
    return P.mean(0).numpy(), P.std(0).numpy()


def latency_table(models, X):
    lat = {}
    n_threads = torch.get_num_threads()
    for threads in (1, n_threads):
        torch.set_num_threads(threads)
        for tag, m in models.items():
            for batch in (1, 64, 256, 1024):
                ms = bench_forward(m, X, batch)
                lat[f"{tag}_t{threads}_b{batch}"] = {"ms_batch": ms, "ms_per_tx": ms / batch,
                                                     "tps": batch / (ms / 1000.0)}
        for tag, m in models.items():
            mq = quantize_int8(m)
            for batch in (1, 256):
                ms = bench_forward(mq, X, batch)
                lat[f"{tag}_int8_t{threads}_b{batch}"] = {"ms_batch": ms, "ms_per_tx": ms / batch,
                                                          "tps": batch / (ms / 1000.0)}
    torch.set_num_threads(n_threads)
    return lat


def make_record(sk, i, prev_hash, risk, score_bps):
    rec = {"tx_id": f"tx{i:06d}", "risk": int(risk), "score_bps": int(score_bps),
           "model_v": 2, "policy_v": 2, "feat_commit": hashlib.sha256(f"f{i}".encode()).hexdigest(),
           "ts": 1757300000 + i, "prev": prev_hash}
    blob = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
    rec["record_hash"] = hashlib.sha256(blob).hexdigest()
    rec["sig"] = sk.sign(bytes.fromhex(rec["record_hash"])).hex()
    return rec


def verify_journal(pk, journal):
    prev, bad = None, 0
    for rec in journal:
        body = {k: v for k, v in rec.items() if k not in ("record_hash", "sig")}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ok = hashlib.sha256(blob).hexdigest() == rec["record_hash"] and rec["prev"] == prev
        if ok:
            try:
                pk.verify(bytes.fromhex(rec["sig"]), bytes.fromhex(rec["record_hash"]))
            except Exception:
                ok = False
        bad += (not ok)
        prev = rec["record_hash"]
    return bad


def journal_bench(sk, pk):
    msg = hashlib.sha256(b"decision-record").digest()
    n_sig = 2000
    t0 = time.perf_counter()
    sigs = [sk.sign(msg) for _ in range(n_sig)]
    t_sign = (time.perf_counter() - t0) / n_sig * 1000
    t0 = time.perf_counter()
    for s in sigs:
        pk.verify(s, msg)
    t_verify = (time.perf_counter() - t0) / n_sig * 1000

    rng = np.random.RandomState(0)
    t0 = time.perf_counter()
    journal, prev = [], None
    for i in range(1000):
        r = make_record(sk, i, prev, rng.randint(0, 3), rng.randint(0, 10000))
        journal.append(r)
        prev = r["record_hash"]
    t_build = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    bad_clean = verify_journal(pk, journal)
    t_check = (time.perf_counter() - t0) * 1000
    tampered = json.loads(json.dumps(journal))
    tampered[500]["score_bps"] = 9999                  # підміна одного поля в середині журналу
    return ({"sign_ms": t_sign, "verify_ms": t_verify},
            {"build_1000_ms": t_build, "verify_1000_ms": t_check, "clean_invalid": int(bad_clean),
             "tampered_invalid_detected": int(verify_journal(pk, tampered))})


def main():
    set_threads()
    data = load_splits()
    X_test = data["X_test"].astype(np.float32)
    zcfg = load_json(RESULTS_MAIN)["gtt"]["zoning"]["config"]
    res = {}

    gtt = load_main()
    mlp, mlp_state = load_reference(MLPRisk, "mlp")
    ftt, ftt_state = load_reference(FeatureTokenTransformer, "ftt")
    models = {"mlp": mlp, "gtt": gtt, "ftt": ftt}

    print("1. затримка інференсу...", flush=True)
    res["latency"] = latency_table(models, X_test)
    res["model_size"] = {tag: {"params": n_params(m), "file_mb": model_mb(m)} for tag, m in models.items()}
    res["model_size"]["mlp"]["weights"], res["model_size"]["ftt"]["weights"] = mlp_state, ftt_state
    with torch.no_grad():
        p_fp32 = torch.sigmoid(gtt(torch.tensor(X_test))).numpy()
        p_int8 = torch.sigmoid(quantize_int8(gtt)(torch.tensor(X_test))).numpy()
    y = data["y_test"]
    res["int8_quality"] = {
        "fp32": {"roc_auc": float(roc_auc_score(y, p_fp32)), "auprc": float(average_precision_score(y, p_fp32))},
        "int8": {"roc_auc": float(roc_auc_score(y, p_int8)), "auprc": float(average_precision_score(y, p_int8))},
        "max_abs_diff": float(np.abs(p_fp32 - p_int8).max())}

    print("2. журнал Ed25519...", flush=True)
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    res["ed25519"], res["journal"] = journal_bench(sk, pk)

    print("3. режими STD / ENH / QUAR...", flush=True)
    Xb = torch.tensor(X_test[:B])
    tx_hashes = [hashlib.sha256(f"tx{i}".encode()).digest() for i in range(B)]
    sig_b = [sk.sign(h) for h in tx_hashes]

    def mode_std():                                    # формат + підпис
        for i in range(B):
            pk.verify(sig_b[i], tx_hashes[i])
            _ = len(tx_hashes[i]) == 32

    def mode_enh():                                    # + ризик і невизначеність, журнал пакетами
        mode_std()
        s, u = mc_scores(gtt, Xb, T=MC_ENH)
        z = zone_assign(s, u, zcfg["tw"], zcfg["tb"], zcfg["umax"])
        for i in range(0, B, 32):
            sk.sign(hashlib.sha256(s[i:i + 32].tobytes() + u[i:i + 32].tobytes()).digest())
        return z

    def mode_quar():                                   # + контрольний прохід і запис на кожну транзакцію
        z = mode_enh()
        with torch.no_grad():
            gtt.eval()
            _ = torch.sigmoid(gtt(Xb)).numpy()
        for i in range(B):
            sk.sign(hashlib.sha256(tx_hashes[i] + bytes([int(z[i])])).digest())

    def bench_mode(fn, n_iter=8, warmup=2):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000 / B)
        return {"mean_ms_tx": float(np.mean(ts)), "p95_ms_tx": float(np.percentile(ts, 95))}

    m_std, m_enh, m_quar = bench_mode(mode_std), bench_mode(mode_enh), bench_mode(mode_quar)
    res["modes"] = {"STD": m_std, "ENH": m_enh, "QUAR": m_quar,
                    "normalized": {"STD": 1.0, "ENH": m_enh["mean_ms_tx"] / m_std["mean_ms_tx"],
                                   "QUAR": m_quar["mean_ms_tx"] / m_std["mean_ms_tx"]}}

    print("4. наскрізний шлюз, потік 20 000 транзакцій...", flush=True)
    stream_X = X_test[np.random.RandomState(1).randint(0, len(X_test), N_STREAM)]
    pool = ThreadPoolExecutor(max_workers=6)          # перевірка Ed25519 звільняє GIL

    def verify_batch(i0, i1):
        def one(i):
            pk.verify(sig_b[i % B], tx_hashes[i % B])
            return hashlib.sha256(tx_hashes[i % B]).digest()
        list(pool.map(one, range(i0, i1)))

    def run_stream(with_gateway):
        lat = []
        t_start = time.perf_counter()
        for s0 in range(0, N_STREAM, B):
            t0 = time.perf_counter()
            e0 = min(s0 + B, N_STREAM)
            verify_batch(s0, e0)
            if with_gateway:
                s, u = mc_scores(gtt, torch.tensor(stream_X[s0:e0]), T=MC_STREAM)
                z = zone_assign(s, u, zcfg["tw"], zcfg["tb"], zcfg["umax"])
                for j in np.where(z != WHITE)[0]:      # журнал лише для ескальованих
                    sk.sign(hashlib.sha256(stream_X[s0 + j].tobytes()[:64]).digest())
            lat.extend([(time.perf_counter() - t0) * 1000 / (e0 - s0)] * (e0 - s0))
        total = time.perf_counter() - t_start
        return {"mean_ms": float(np.mean(lat)), "p95_ms": float(np.percentile(lat, 95)), "tps": N_STREAM / total}

    run_stream(True)                                   # прогрів
    off, on = run_stream(False), run_stream(True)
    res["gateway"] = {"without": off, "with": on, "added_ms": on["mean_ms"] - off["mean_ms"],
                      "n_stream": N_STREAM, "batch": B, "mc_T": MC_STREAM}
    res["env"] = {"cpu_count": os.cpu_count(), "torch": torch.__version__,
                  "threads_used": torch.get_num_threads(),
                  "platform": f"{platform.system()} {platform.release()}, {platform.processor()}"}
    save_json(res, RESULTS_SPEED)
    print(json.dumps({k: res[k] for k in ("modes", "gateway", "int8_quality")}, indent=1))
    print("Збережено", RESULTS_SPEED)


if __name__ == "__main__":
    main()
