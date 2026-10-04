"""Chargement des fichiers de configuration YAML et résolution des chemins du projet."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Racine du projet : déduite de l'emplacement du code en installation éditable (pip install -e),
# imposée par COCKPIT_ROOT quand le paquet est installé ailleurs (image Docker, Hugging Face).
PROJECT_ROOT = Path(os.environ.get("COCKPIT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache
def load_config(name: str) -> dict[str, Any]:
    """Charge `config/<name>.yaml`."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Chemin absolu d'un dossier déclaré dans `config/data.yaml` (raw, interim, processed...)."""
    p = PROJECT_ROOT / load_config("data")["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
