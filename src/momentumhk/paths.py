# src/momentumhk/paths.py
from pathlib import Path
import os

# repo root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]

SRC       = ROOT / "src" / "momentumhk"
DATA      = ROOT / "data"
RAW       = DATA / "raw"
INTERIM   = DATA / "interim"
CLEANED   = DATA / "cleaned"
OUTPUTS   = ROOT / "outputs"
FIGURES   = OUTPUTS / "figures"
REPORTS   = OUTPUTS / "reports"
CONFIGS   = ROOT / "configs"
LOGS      = ROOT / "logs"

# ensure dirs exist when imported (optional)
for d in (RAW, INTERIM, CLEANED, FIGURES, REPORTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

#def resolve_env(key: str, default: str = "") -> str:
#    return os.getenv(key, default)
