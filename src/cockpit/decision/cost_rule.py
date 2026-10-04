"""Passage des probabilités à la décision : la classe qui minimise le coût attendu.

Pour un camion de probabilités p (classes 0..4) et la matrice de coûts C (ligne = vraie
classe, colonne = classe prédite), le coût attendu de la prédiction j vaut sum_k p_k C[k, j].
On choisit le j le moins cher, et non la classe la plus probable : avec une panne non détectée
50 fois plus chère qu'une fausse alerte, une probabilité de panne de quelques pour cent suffit
à justifier une alerte.
"""

from __future__ import annotations

import numpy as np

from cockpit.config import load_config


def cost_matrix() -> np.ndarray:
    return np.asarray(load_config("costs")["challenge_cost_matrix"], dtype=float)


def expected_costs(p: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
    """(n, 5) : coût attendu de chaque prédiction possible."""
    return p @ (cost_matrix() if C is None else C)


def decide(p: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
    return expected_costs(p, C).argmin(axis=1)


def most_likely(p: np.ndarray) -> np.ndarray:
    return p.argmax(axis=1)


def challenge_cost(y_true: np.ndarray, y_pred: np.ndarray, C: np.ndarray | None = None) -> float:
    C = cost_matrix() if C is None else C
    return float(C[y_true.astype(int), y_pred.astype(int)].sum())


def prior_shift(p: np.ndarray, prior_from: np.ndarray, prior_to: np.ndarray) -> np.ndarray:
    """Recalage des probabilités sur une autre répartition des classes (règle de Bayes).

    p'(k | x) ∝ p(k | x) · π_to(k) / π_from(k). Sert à passer de la répartition de nos
    échantillons d'entraînement à celle du protocole d'évaluation SCANIA, qui sur-représente
    les instants proches de la panne.
    """
    q = p * (np.asarray(prior_to) / np.asarray(prior_from))[None, :]
    return q / q.sum(axis=1, keepdims=True)


def class_prior(labels: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(labels)) if weight is None else weight
    counts = np.bincount(labels.astype(int), weights=w, minlength=5)
    return counts / counts.sum()
