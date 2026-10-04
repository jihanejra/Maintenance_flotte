"""Téléchargement des fichiers bruts SCANIA Component X depuis le Swedish National Data Service.

Le téléchargement est idempotent : un fichier déjà présent avec la bonne taille n'est pas
retéléchargé. Usage : `python -m cockpit.ingest.download`
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from tqdm import tqdm

from cockpit.config import load_config, path

log = logging.getLogger(__name__)
CHUNK = 1 << 20


def _remote_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    size = r.headers.get("Content-Length")
    return int(size) if size else None


def download_file(url: str, dest: Path) -> Path:
    expected = _remote_size(url)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        log.info("déjà présent : %s", dest.name)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f, tqdm(total=expected, unit="B", unit_scale=True,
                                        desc=dest.name) as bar:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))
    if expected is not None and tmp.stat().st_size != expected:
        raise OSError(f"taille inattendue pour {dest.name}")
    tmp.replace(dest)
    return dest


def download_all() -> list[Path]:
    src = load_config("data")["source"]
    raw = path("raw")
    return [download_file(f"{src['base_url']}/{name}", raw / name) for name in src["files"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_all()
