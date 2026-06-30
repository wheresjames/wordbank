"""Shared helpers for Wordbank audio engine adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ENGINES_DIR = Path(__file__).resolve().parent
AUDIO_ROOT = ENGINES_DIR.parent


def engine_dir(engine: str) -> Path:
  return ENGINES_DIR / engine


def venv_dir(engine: str) -> Path:
  return engine_dir(engine) / ".venv"


def cache_dir(engine: str) -> Path:
  return engine_dir(engine) / "cache"


def venv_python(engine: str) -> Path:
  return venv_dir(engine) / "bin" / "python"


def venv_bin(engine: str, name: str) -> Path:
  return venv_dir(engine) / "bin" / name


def ensure_engine_dir(engine: str) -> None:
  engine_dir(engine).mkdir(parents=True, exist_ok=True)
  cache_dir(engine).mkdir(parents=True, exist_ok=True)


def run(command: list[str | Path], *, env: dict[str, str] | None = None) -> None:
  cmd = [str(part) for part in command]
  try:
    subprocess.check_call(cmd, env=env)
  except subprocess.CalledProcessError as exc:
    raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from None


def create_venv(engine: str, python_executable: str | Path | None = None) -> None:
  ensure_engine_dir(engine)
  if not venv_python(engine).exists():
    run([python_executable or sys.executable, "-m", "venv", venv_dir(engine)])
  run([venv_python(engine), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])


def pip_install(engine: str, packages: list[str], *, python_executable: str | Path | None = None) -> None:
  create_venv(engine, python_executable)
  run([venv_python(engine), "-m", "pip", "install", *packages])


def venv_modules_available(engine: str, modules: list[str]) -> bool:
  py = venv_python(engine)
  if not py.exists():
    return False
  code = (
    "import importlib.util, sys; "
    "mods = sys.argv[1:]; "
    "sys.exit(0 if all(importlib.util.find_spec(mod) for mod in mods) else 1)"
  )
  result = subprocess.run([str(py), "-c", code, *modules], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return result.returncode == 0


def require_venv_modules(engine: str, modules: list[str], install_command: str) -> None:
  py = venv_python(engine)
  if not py.exists():
    raise SystemExit(f"{engine} is not installed. Run: {install_command}")
  if not venv_modules_available(engine, modules):
    names = ", ".join(modules)
    raise SystemExit(
      f"{engine} runtime is incomplete; missing Python module(s): {names}. "
      f"Run: {install_command}"
    )


def load_phrase_items(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a YAML mapping/object")
  items = data.get("items") or []
  if not isinstance(items, list):
    raise SystemExit(f"{path} items must be a list")
  out: list[dict[str, Any]] = []
  for index, item in enumerate(items, start=1):
    if isinstance(item, str):
      item = {"text": item}
    if not isinstance(item, dict):
      raise SystemExit(f"Item #{index} must be a string or mapping/object")
    text = str(item.get("text") or "").strip()
    output_name = str(item.get("output_name") or "").strip()
    if not text or not output_name:
      raise SystemExit(f"Item #{index} must define text and output_name")
    out.append(item)
  return out


def status_payload(engine: str, *, executable: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
  venv = venv_dir(engine)
  py = venv_python(engine)
  payload: dict[str, Any] = {
    "engine": engine,
    "env_installed": py.exists(),
    "python": str(py),
    "cache_dir": str(cache_dir(engine)),
  }
  if executable:
    payload["executable"] = str(venv_bin(engine, executable))
    payload["executable_installed"] = venv_bin(engine, executable).exists() or shutil.which(executable) is not None
  if extra:
    payload.update(extra)
  return payload


def print_status(payload: dict[str, Any], json_output: bool) -> None:
  if json_output:
    print(json.dumps(payload, indent=2))
    return
  for key, value in payload.items():
    print(f"{key}: {value}")


def local_env(engine: str) -> dict[str, str]:
  env = dict(os.environ)
  env.setdefault("HF_HOME", str(cache_dir(engine) / "huggingface"))
  env.setdefault("TRANSFORMERS_CACHE", str(cache_dir(engine) / "huggingface"))
  env.setdefault("NLTK_DATA", str(cache_dir(engine) / "nltk_data"))
  return env
