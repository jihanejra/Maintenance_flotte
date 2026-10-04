"""Calibration des courbes de survie, horizon par horizon (régression isotonique).

Pour chaque horizon h, la cible binaire « réparé dans les h pas de temps » est connue si le
camion a été réparé, ou s'il a été suivi plus de h pas de temps après la coupure. On ajuste une
régression isotonique de cette cible sur la probabilité prédite 1 - S(h), sur des prédictions
hors échantillon (validation croisée), puis on impose que la probabilité de panne cumulée
croisse avec l'horizon.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 52)


class SurvivalCalibrator:
    def __init__(self, horizons=HORIZONS):
        self.horizons = tuple(horizons)

    def fit(self, S: np.ndarray, duration: np.ndarray, event: np.ndarray, weight=None):
        """`S` : survies prédites hors échantillon aux `horizons` (n, H)."""
        self.iso_ = []
        for j, h in enumerate(self.horizons):
            known = (event == 1) | (duration > h)
            y = ((event == 1) & (duration <= h))[known].astype(float)
            w = None if weight is None else weight[known]
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(1 - S[known, j], y, sample_weight=w)
            self.iso_.append(iso)
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        F = np.column_stack([iso.predict(1 - S[:, j]) for j, iso in enumerate(self.iso_)])
        F = np.maximum.accumulate(F, axis=1)          # panne cumulée croissante en h
        return 1 - F

    def survival_at(self, S_cal: np.ndarray, times) -> np.ndarray:
        """Interpolation linéaire de la survie calibrée entre les horizons calibrés."""
        grid = np.concatenate([[0.0], self.horizons])
        vals = np.column_stack([np.ones(len(S_cal)), S_cal])
        t = np.clip(np.asarray(times, float), 0, grid[-1])
        j = np.clip(np.searchsorted(grid, t, "right") - 1, 0, len(grid) - 2)
        a = (t - grid[j]) / (grid[j + 1] - grid[j])
        return vals[:, j] * (1 - a) + vals[:, j + 1] * a
