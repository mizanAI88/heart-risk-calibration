"""Explicit, opt-in download of the processed Cleveland file.

The CLI prints the licence and terms first; nothing is fetched unless the
caller passes --accept-terms, and never when a CI environment is detected.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping
from urllib.request import urlopen

from heart_risk_calibration.config import DataConfig
from heart_risk_calibration.errors import ConfigError, TermsNotAcceptedError

CI_ENV_VARS: tuple[str, ...] = ("CI", "GITHUB_ACTIONS")
Fetcher = Callable[[str, Path], int]


def terms_message(cfg: DataConfig) -> str:
    return (
        f"Dataset: UCI Heart Disease ({cfg.dataset_url})\n"
        f"Licence: {cfg.licence} ({cfg.licence_url})\n"
        f"File that would be fetched: {cfg.download_url}\n"
        "Nothing has been downloaded. Re-run with --accept-terms after reading the "
        "licence to fetch the file into the data root."
    )


def fetch_with_urllib(url: str, dest: Path) -> int:
    """Stream `url` to `dest`; returns bytes written. Only called after acceptance."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with urlopen(url) as response, dest.open("wb") as handle:  # noqa: S310 (explicit opt-in)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    return total


def download_cleveland(data_root: Path, cfg: DataConfig, accept_terms: bool,
                       fetcher: Fetcher = fetch_with_urllib,
                       env: Mapping[str, str] | None = None) -> Path:
    """Fetch the file into `data_root`. Raises unless terms were accepted and not in CI."""
    environ = os.environ if env is None else env
    if any(environ.get(var) for var in CI_ENV_VARS):
        raise TermsNotAcceptedError("download is disabled in CI environments by design")
    if not accept_terms:
        raise TermsNotAcceptedError(terms_message(cfg))
    dest = Path(data_root) / cfg.file_name
    if dest.is_file():
        return dest
    written = fetcher(cfg.download_url, dest)
    if written <= 0 or not dest.is_file():
        raise ConfigError(f"download of {cfg.download_url} produced no file at {dest}")
    return dest
