# -*- coding: utf-8 -*-
"""Нейромережеві моделі risk engine.

  MLPRisk                 — багатошаровий перцептрон 256-128 (довідкова модель)
  GroupTokenTransformer   — груповий трансформер (GTT): самоувага над токенами груп ознак
  FeatureTokenTransformer — трансформер із токеном на кожну ознаку (довідка щодо складності)
  GTTPLE                  — GTT над кусково-лінійним кодуванням ознак (PLE); робоча модель

PLE: Gorishniy Y., Rubachev I., Babenko A. On Embeddings for Numerical Features
in Tabular Deep Learning. NeurIPS 2022.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from common import RESULTS

N_FEAT = 188
IDX_LOCAL = list(range(0, 93))      # f1..f93    структурні локальні
IDX_AGG = list(range(93, 165))      # f94..f165  структурні агреговані (r = 1)
IDX_CTX = list(range(165, 188))     # часовий крок + 22 контекстні
GROUPS = [IDX_LOCAL, IDX_AGG, IDX_CTX]
GROUP_NAMES = ["структурні локальні (f1-f93)", "структурні агреговані (f94-f165)",
               "часові+контекстні (23)"]

MODEL_GTT = os.path.join(RESULTS, "model_gtt.pt")
PLE_EDGES = os.path.join(RESULTS, "ple_edges.npy")


class MLPRisk(nn.Module):
    """Повнозв'язна мережа 256-128."""

    def __init__(self, d_in=N_FEAT, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _encoder(d_model, n_heads, n_layers, p):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
        dropout=p, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class GroupTokenTransformer(nn.Module):
    """Кожна смислова група ознак проєктується в k токенів; самоувага працює над
    3k+1 токенами замість 188 пофічевих: O((3k+1)^2 d) проти O(188^2 d) на транзакцію."""

    def __init__(self, groups=GROUPS, d_model=64, k_per_group=4, n_layers=3, n_heads=4, p=0.15,
                 dims_per_feature=1):
        super().__init__()
        self.groups = [torch.tensor(g, dtype=torch.long) for g in groups]
        self.projs = nn.ModuleList([nn.Linear(len(g) * dims_per_feature, d_model * k_per_group)
                                    for g in groups])
        self.k, self.d = k_per_group, d_model
        n_tokens = len(groups) * k_per_group + 1
        self.group_emb = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.encoder = _encoder(d_model, n_heads, n_layers, p)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(p), nn.Linear(d_model, 1))

    def tokens(self, x):
        toks = [proj(x[:, g]).view(x.size(0), self.k, self.d) for g, proj in zip(self.groups, self.projs)]
        t = torch.cat(toks, dim=1)
        return torch.cat([self.cls.expand(x.size(0), 1, self.d), t], dim=1) + self.group_emb.unsqueeze(0)

    def forward(self, x):
        return self.head(self.encoder(self.tokens(x))[:, 0]).squeeze(-1)


class FeatureTokenTransformer(nn.Module):
    """Окремий токен для кожної з 188 ознак."""

    def __init__(self, n_feat=N_FEAT, d_model=24, n_layers=2, n_heads=4, p=0.15):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_feat, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.encoder = _encoder(d_model, n_heads, n_layers, p)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(p), nn.Linear(d_model, 1))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)      # [B, F, d]
        cls = self.cls.expand(x.size(0), 1, t.size(-1))
        return self.head(self.encoder(torch.cat([cls, t], dim=1))[:, 0]).squeeze(-1)


def make_ple_edges(X_train, n_bins=24):
    """Межі квантильних бінів кожної ознаки (лише за навчальною вибіркою)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return torch.tensor(np.quantile(X_train, qs, axis=0).T.copy(), dtype=torch.float32)


class PLE(nn.Module):
    """Кусково-лінійне кодування: ознака -> вектор ступенів заповнення n_bins бінів."""

    def __init__(self, edges):
        super().__init__()
        self.register_buffer("edges", edges)               # [F, B-1]
        self.n_bins = edges.shape[1] + 1

    def forward(self, x):                                  # [N, F] -> [N, F, B]
        e = self.edges.unsqueeze(0)
        x = x.unsqueeze(-1)
        lo = torch.cat([torch.full_like(e[..., :1], -1e4), e], dim=-1)
        hi = torch.cat([e, torch.full_like(e[..., :1], 1e4)], dim=-1)
        return ((x - lo) / (hi - lo + 1e-9)).clamp(0, 1)


class GTTPLE(GroupTokenTransformer):
    """Груповий трансформер над PLE-кодованими ознаками (робоча модель risk engine)."""

    def __init__(self, groups, edges, d_model=64, k_per_group=4, n_layers=3, n_heads=4, p=0.15):
        super().__init__(groups, d_model, k_per_group, n_layers, n_heads, p,
                         dims_per_feature=edges.shape[1] + 1)
        self.ple = PLE(edges)

    def tokens(self, x):
        enc = self.ple(x)                                  # [N, F, B]
        toks = [proj(enc[:, g].flatten(1)).view(x.size(0), self.k, self.d)
                for g, proj in zip(self.groups, self.projs)]
        t = torch.cat(toks, dim=1)
        return torch.cat([self.cls.expand(x.size(0), 1, self.d), t], dim=1) + self.group_emb.unsqueeze(0)


def build_main(groups=GROUPS, edges=None):
    if edges is None:
        edges = torch.tensor(np.load(PLE_EDGES))
    return GTTPLE(groups, edges)


def build_main_for_subset(idxs):
    """Робоча модель для підмножини ознак (абляція радіуса оточення)."""
    groups = [[idxs.index(i) for i in g if i in idxs] for g in GROUPS]
    groups = [g for g in groups if g]
    edges = torch.tensor(np.load(PLE_EDGES)[idxs])
    return GTTPLE(groups, edges)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_main():
    """Робоча модель із results/model_gtt.pt."""
    return load_state(build_main(), MODEL_GTT)


def quantize_int8(m):
    """Динамічне INT8-квантування лінійних шарів. Для трансформерів квантуються лише
    вхідні проєкції та голова (~90 % ваг): енкодер лишається FP32, бо квантовані шари
    всередині nn.TransformerEncoder вимикають його оптимізований шлях виконання."""
    q = torch.ao.quantization.quantize_dynamic
    mq = copy.deepcopy(m)
    if hasattr(mq, "projs"):
        mq.projs = nn.ModuleList([q(p, {nn.Linear}, dtype=torch.qint8) for p in mq.projs])
        mq.head = q(mq.head, {nn.Linear}, dtype=torch.qint8)
        return mq
    if hasattr(mq, "encoder"):
        mq.head = q(mq.head, {nn.Linear}, dtype=torch.qint8)
        return mq
    return q(mq, {nn.Linear}, dtype=torch.qint8)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
