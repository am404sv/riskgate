# -*- coding: utf-8 -*-
"""RiskGate — передконсенсусний шлюз ризик-орієнтованого керування захищеними транзакціями
(виконуване ядро апаратно-програмного комплексу).

Тракт обробки пакета транзакцій:
  n_acc  приймання та перевірка формату
  n_val  перевірка підпису Ed25519
  m_prof транзакційний профіль (188 ознак)
  m_re   нейромережевий risk engine (GTT+PLE, вибіркове INT8)
  m_unc  невизначеність MC-Dropout (T прогонів) і температурне калібрування
  m_pol  зонування WHITE/GRAY/BLACK -> протокольна директива STD/ENH/QUAR
  n_pre  маршрутизація в черги mempool / inspection / quarantine
  m_log  криптографічно верифікований журнал рішень (hash-chain + Ed25519)

    python src/gateway_service.py --n 5000

Вихід: results/session_report.json і results/journal.jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import time

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from common import RESULTS, RESULTS_MAIN, load_json, load_raw, require, save_json, set_threads
from engine import apply_temperature, zone_assign
from models import load_main, quantize_int8


class DecisionJournal:
    """Журнал рішень лише на дописування: кожен запис містить hash попереднього
    і підписаний Ed25519 hash власного вмісту."""

    def __init__(self, path, signing_key=None):
        self.path = path
        self.sk = signing_key or Ed25519PrivateKey.generate()
        self.pk = self.sk.public_key()
        self.prev = None
        open(path, "w").close()

    @staticmethod
    def _digest(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def append(self, tx_id, zone, score_bps, u_bps, model_v, policy_v, feat_commit):
        rec = {"tx_id": tx_id, "zone": zone, "score_bps": int(score_bps), "u_bps": int(u_bps),
               "model_v": model_v, "policy_v": policy_v, "feat_commit": feat_commit,
               "ts": time.time_ns() // 1_000_000, "prev": self.prev}
        rec["record_hash"] = self._digest(rec)
        rec["sig"] = self.sk.sign(bytes.fromhex(rec["record_hash"])).hex()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.prev = rec["record_hash"]
        return rec

    def verify(self):
        prev, n, bad = None, 0, 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                body = {k: v for k, v in rec.items() if k not in ("record_hash", "sig")}
                ok = self._digest(body) == rec["record_hash"] and rec["prev"] == prev
                if ok:
                    try:
                        self.pk.verify(bytes.fromhex(rec["sig"]), bytes.fromhex(rec["record_hash"]))
                    except Exception:
                        ok = False
                bad += (not ok)
                prev = rec["record_hash"]
                n += 1
        return {"records": n, "invalid": bad}


class RiskGate:
    ZONES = ("WHITE", "GRAY", "BLACK")
    MODES = ("STD", "ENH", "QUAR")
    QUEUES = ("mempool", "inspection", "quarantine")

    def __init__(self, model, policy, scaler, mc_passes=5, int8=True, journal_path="journal.jsonl"):
        self.model = quantize_int8(model) if int8 else model
        self.tw, self.tb, self.umax = policy["tw"], policy["tb"], policy["umax"]
        self.temperature = policy.get("temperature", 1.0)
        self.mu, self.sd = scaler
        self.T = mc_passes
        self.journal = DecisionJournal(journal_path)
        self.queues = {q: [] for q in self.QUEUES}
        self.stats = collections.Counter()
        self.policy_v = policy.get("version", "zoning-1")
        self.model_v = policy.get("model_version", "gtt_ple-int8" if int8 else "gtt_ple-fp32")

    def _score(self, X):
        """Позаланцюговий контур: s̄ і u за MC-Dropout, калібрування температурою."""
        xb = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        self.model.train()                             # dropout увімкнено
        with torch.no_grad():
            P = torch.stack([torch.sigmoid(self.model(xb)) for _ in range(self.T)])
        self.model.eval()
        return apply_temperature(P.mean(0).numpy(), self.temperature), P.std(0).numpy()

    def process_batch(self, tx_ids, X, signatures, pubkeys):
        """Повний передконсенсусний прохід пакета; повертає директиви для кожної транзакції."""
        t0 = time.perf_counter()
        valid = []
        for i, txid in enumerate(tx_ids):              # n_acc + n_val
            try:
                pubkeys[i].verify(signatures[i], txid.encode())
                valid.append(i)
            except Exception:
                self.stats["rejected_sig"] += 1
        if not valid:
            return []
        s, u = self._score(X[valid])                   # m_re + m_unc
        z = zone_assign(s, u, self.tw, self.tb, self.umax)   # m_pol
        directives = []
        for j, i in enumerate(valid):
            k = int(z[j])
            self.queues[self.QUEUES[k]].append(tx_ids[i])   # n_pre
            self.stats[self.ZONES[k]] += 1
            if k > 0:                                  # m_log: журналюються ескальовані рішення
                self.journal.append(tx_ids[i], self.ZONES[k], round(s[j] * 10000), round(u[j] * 10000),
                                    self.model_v, self.policy_v,
                                    hashlib.sha256(X[i].tobytes()).hexdigest()[:16])
            directives.append({"tx_id": tx_ids[i], "zone": self.ZONES[k], "mode": self.MODES[k],
                               "prio": 2 - k, "queue": self.QUEUES[k],
                               "score": float(s[j]), "u": float(u[j])})
        self.stats["latency_ms_sum"] += (time.perf_counter() - t0) * 1000
        self.stats["batches"] += 1
        return directives


def main():
    ap = argparse.ArgumentParser(description="Демонстрація шлюзу RiskGate на потоці підписаних транзакцій")
    ap.add_argument("--n", type=int, default=5000, help="кількість транзакцій потоку")
    ap.add_argument("--mc", type=int, default=5, help="кількість MC-прогонів T")
    ap.add_argument("--batch", type=int, default=256, help="розмір пакета")
    ap.add_argument("--fp32", action="store_true", help="без INT8-квантування")
    args = ap.parse_args()

    require(RESULTS_MAIN, "python src/evaluate.py")
    set_threads()
    d = load_raw()
    mu, sd = d["X_train"].mean(0), d["X_train"].std(0) + 1e-6
    pool = d["X_test"]                                 # сирі профілі; стандартизує сам шлюз

    g = load_json(RESULTS_MAIN)["gtt"]
    policy = dict(g["zoning"]["config"], temperature=g["calibration"]["temperature"])
    gate = RiskGate(load_main(), policy, (mu, sd), mc_passes=args.mc, int8=not args.fp32,
                    journal_path=os.path.join(RESULTS, "journal.jsonl"))

    # потік мережі на дозволах: підписані транзакції з профілями з тестового пулу
    idx = np.random.RandomState(7).randint(0, len(pool), args.n)
    user_sk = Ed25519PrivateKey.generate()
    user_pk = user_sk.public_key()
    print(f"RiskGate: потік {args.n} тх, пакет {args.batch}, MC T={args.mc}, "
          f"INT8: {'ні' if args.fp32 else 'так'}", flush=True)
    t_start = time.perf_counter()
    for s0 in range(0, args.n, args.batch):
        e0 = min(s0 + args.batch, args.n)
        ids = [f"tx-{i:06d}" for i in range(s0, e0)]
        sigs = [user_sk.sign(t.encode()) for t in ids]
        gate.process_batch(ids, pool[idx[s0:e0]], sigs, [user_pk] * len(ids))
    wall = time.perf_counter() - t_start

    report = {
        "stream": args.n, "batch": args.batch, "mc_T": args.mc, "int8": not args.fp32,
        "zones": {z: gate.stats[z] for z in RiskGate.ZONES},
        "queues": {k: len(v) for k, v in gate.queues.items()},
        "wall_seconds": wall, "tps": args.n / wall,
        "gateway_ms_per_tx": gate.stats["latency_ms_sum"] / args.n,
        "journal": gate.journal.verify(), "policy": policy,
    }
    save_json(report, os.path.join(RESULTS, "session_report.json"))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
