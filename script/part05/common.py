"""Paths and small JSON helpers; no model is imported into the Agent process."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets' / 'part05'
OUT = ROOT / 'outputs' / 'part05'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))
