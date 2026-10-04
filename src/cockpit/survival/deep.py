"""Modèle de survie récurrent : GRU sur la séquence des relevés, sortie en temps discret.

Le réseau lit les `length` derniers relevés avant l'instant de prédiction (entrées par relevé
de `sequences.py`, plus un canal de masque pour les historiques courts), concatène l'état final
avec les spécifications du camion, et prédit un risque instantané h_k pour chaque tranche de
temps (0-1], (1-2], …, (48-52] : les mêmes horizons que la calibration et les décisions.
S(H_k) = prod_{i <= k} (1 - h_i).

Perte (vraisemblance de survie en temps discret, qui gère la censure) :
- réparé dans la tranche j : -[sum_{k<j} log(1 - h_k) + log h_j] ;
- censuré après la tranche j : -sum_{k<=j} log(1 - h_k).
Inspiré de DeepHit / Dynamic-DeepHit, en plus simple.
"""

from __future__ import annotations

import numpy as np

from cockpit.survival.calibration import HORIZONS
from cockpit.survival.sequences import SequenceStore

H = np.asarray(HORIZONS, dtype=float)


def discrete_targets(duration: np.ndarray, event: np.ndarray):
    """Masques de survie (n, K) et indicatrice de l'événement (n, K)."""
    K = len(H)
    j = np.searchsorted(H, duration, side="left")          # tranche de l'événement
    in_range = (event == 1) & (j < K)
    n_surv = np.where(in_range, j, np.searchsorted(H, duration, side="right"))
    k = np.arange(K)[None, :]
    surv_mask = (k < n_surv[:, None]).astype(np.float32)
    ev = ((k == j[:, None]) & in_range[:, None]).astype(np.float32)
    return surv_mask, ev


class GRUSurvival:
    name = "gru_seq"

    def __init__(self, length=32, hidden=64, dropout=0.2, lr=2e-3, batch=1024, epochs=12,
                 patience=2, seed=0, threads=4):
        self.length, self.hidden, self.dropout, self.lr = length, hidden, dropout, lr
        self.batch, self.epochs, self.patience, self.seed = batch, epochs, patience, seed
        self.threads = threads

    def _net(self, n_in, n_static):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Sequential(nn.Linear(n_in, self.hidden), nn.ReLU())
                s.gru = nn.GRU(self.hidden, self.hidden, batch_first=True)
                s.head = nn.Sequential(nn.Linear(self.hidden + n_static, self.hidden), nn.ReLU(),
                                       nn.Dropout(self.dropout), nn.Linear(self.hidden, len(H)))

            def forward(s, x, z):
                _, h = s.gru(s.inp(x))
                return torch.sigmoid(s.head(torch.cat([h[-1], z], dim=1))).clamp(1e-6, 1 - 1e-6)

        return Net()

    def _batches(self, store, vid, cut, static, idx):
        import torch

        for b in range(0, len(idx), self.batch):
            i = idx[b:b + self.batch]
            x = torch.from_numpy(store.window(vid[i], cut[i], self.length))
            yield i, x, torch.from_numpy(static[i])

    def _loss(self, h, surv, ev, w):
        import torch

        ll = (surv * torch.log(1 - h)).sum(1) + (ev * torch.log(h)).sum(1)
        return -(w * ll).sum() / w.sum()

    def fit(self, store: SequenceStore, vid, cut, static, duration, event, weight=None,
            groups=None):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        n = len(vid)
        w = np.ones(n, np.float32) if weight is None else weight.astype(np.float32)
        surv, ev = discrete_targets(duration, event)
        # Arrêt précoce sur 10 % des camions de l'ensemble d'apprentissage
        g = vid if groups is None else groups
        held_veh = rng.choice(np.unique(g), max(1, len(np.unique(g)) // 10), replace=False)
        held = np.isin(g, held_veh)
        tr_idx, va_idx = np.flatnonzero(~held), np.flatnonzero(held)

        self.net_ = self._net(store.X.shape[1] + 1, static.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        T = {k: torch.from_numpy(v) for k, v in (("surv", surv), ("ev", ev), ("w", w))}
        best, best_state, wait = np.inf, None, 0
        self.history_ = []
        for _epoch in range(self.epochs):
            self.net_.train()
            for i, x, z in self._batches(store, vid, cut, static, rng.permutation(tr_idx)):
                opt.zero_grad()
                loss = self._loss(self.net_(x, z), T["surv"][i], T["ev"][i], T["w"][i])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            val = self._eval_loss(store, vid, cut, static, va_idx, T)
            self.history_.append(val)
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def _eval_loss(self, store, vid, cut, static, idx, T):
        import torch

        self.net_.eval()
        tot, wsum = 0.0, 0.0
        with torch.no_grad():
            for i, x, z in self._batches(store, vid, cut, static, idx):
                h = self.net_(x, z)
                ll = (T["surv"][i] * torch.log(1 - h)).sum(1) + (T["ev"][i] * torch.log(h)).sum(1)
                tot += float(-(T["w"][i] * ll).sum())
                wsum += float(T["w"][i].sum())
        return tot / wsum

    def survival(self, store: SequenceStore, vid, cut, static) -> np.ndarray:
        """S aux HORIZONS, (n, K)."""
        import torch

        self.net_.eval()
        out = np.empty((len(vid), len(H)), np.float64)
        with torch.no_grad():
            for i, x, z in self._batches(store, vid, cut, static, np.arange(len(vid))):
                out[i] = torch.cumprod(1 - self.net_(x, z), dim=1).numpy()
        return out
