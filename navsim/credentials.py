"""Credential storage.  Only the free Copernicus key is needed.

One file, one place, owner-readable only, outside the project tree so it
cannot be committed by accident.  Every provider is optional: the pipeline
degrades to whatever is configured rather than refusing to start.

Design rules, all of them load-bearing:

* **Never echo a secret.**  :func:`masked` is the only way a key reaches a log,
  a window title or a report, and it shows four characters.
* **Environment wins.**  A key in the environment overrides the stored one, so
  CI and shared machines never need the file.
* **Mode 0600, and the directory 0700.**  Checked on every write.
* **Keys are not validated on save.**  A typo should surface as a clear
  authentication failure from the service, not as a silent empty result.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "navsim"
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"


@dataclass(frozen=True)
class Provider:
    key: str                      # storage key
    label: str                    # shown in the GUI
    fields: tuple[str, ...]       # sub-fields, e.g. ("client_id", "secret")
    env: tuple[str, ...]          # environment overrides, same order
    help: str
    signup: str


#: Everything the pipeline can use.  Order is the order shown in the GUI.
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        "copernicus", "Copernicus Data Space (Sentinel-1 radar)",
        ("client_id", "client_secret"),
        ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"),
        "FREE. Sentinel-1 radar. Register, then create an OAuth client under "
        "Sentinel Hub > User settings.",
        "https://dataspace.copernicus.eu/",
    ),
)
PROVIDERS_BY_KEY = {p.key: p for p in PROVIDERS}


# --------------------------------------------------------------------------
def _load() -> dict:
    if not CREDENTIALS_PATH.exists():
        return {}
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save(data: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CONFIG_DIR, stat.S_IRWXU)
    CREDENTIALS_PATH.write_text(json.dumps(data, indent=2))
    os.chmod(CREDENTIALS_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return CREDENTIALS_PATH


def get(provider: str, field: str | None = None) -> str | None:
    """Fetch one credential field.  Environment wins over the stored file."""
    p = PROVIDERS_BY_KEY.get(provider)
    if p is None:
        return None
    field = field or p.fields[0]
    if field in p.fields:
        env_name = p.env[p.fields.index(field)]
        if os.environ.get(env_name):
            return os.environ[env_name]
    entry = _load().get(provider) or {}
    val = entry.get(field)
    return val or None


def get_all(provider: str) -> dict[str, str]:
    p = PROVIDERS_BY_KEY.get(provider)
    if p is None:
        return {}
    return {f: (get(provider, f) or "") for f in p.fields}


def set_credentials(provider: str, **fields: str) -> Path:
    """Store one provider's fields.  Blank values clear that field."""
    p = PROVIDERS_BY_KEY.get(provider)
    if p is None:
        raise KeyError(f"unknown provider {provider!r}")
    unknown = set(fields) - set(p.fields)
    if unknown:
        raise KeyError(f"{provider} has no field(s) {sorted(unknown)}")
    data = _load()
    entry = dict(data.get(provider) or {})
    for k, v in fields.items():
        v = (v or "").strip()
        if v:
            entry[k] = v
        else:
            entry.pop(k, None)
    if entry:
        data[provider] = entry
    else:
        data.pop(provider, None)
    return _save(data)


def clear(provider: str) -> None:
    data = _load()
    if data.pop(provider, None) is not None:
        _save(data)


def masked(provider: str, field: str | None = None) -> str:
    """A safe-to-display form: never more than the last four characters."""
    v = get(provider, field)
    if not v:
        return "(not set)"
    if len(v) <= 8:
        return "****"
    return f"{v[:4]}{'*' * 8}{v[-4:]}"


def source(provider: str, field: str | None = None) -> str:
    p = PROVIDERS_BY_KEY.get(provider)
    if p is None:
        return "unknown provider"
    field = field or p.fields[0]
    if field in p.fields and os.environ.get(p.env[p.fields.index(field)]):
        return f"environment ({p.env[p.fields.index(field)]})"
    if (_load().get(provider) or {}).get(field):
        return str(CREDENTIALS_PATH)
    return "not configured"


def configured(provider: str) -> bool:
    """True only when every field a provider needs is present."""
    p = PROVIDERS_BY_KEY.get(provider)
    return bool(p) and all(get(provider, f) for f in p.fields)


def status() -> list[dict]:
    return [{
        "key": p.key, "label": p.label, "configured": configured(p.key),
        "fields": {f: masked(p.key, f) for f in p.fields},
        "source": source(p.key), "help": p.help, "signup": p.signup,
    } for p in PROVIDERS]
