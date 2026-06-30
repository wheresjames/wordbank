#!/usr/bin/env python3
"""MeloTTS adapter for Wordbank audio generation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from common import (
  cache_dir,
  local_env,
  pip_install,
  print_status,
  require_venv_modules,
  run,
  status_payload,
  venv_dir,
  venv_python,
)


ENGINE = "melotts"
INSTALL_COMMAND = "python tools/itemaudio/engines/melotts.py --install"
MELOTTS_GIT_REQUIREMENT = "git+https://github.com/myshell-ai/MeloTTS.git@209145371cff8fc3bd60d7be902ea69cbdb7965a"
SUPPORTED_PYTHON_MIN = (3, 9)
SUPPORTED_PYTHON_MAX = (3, 11)
NLTK_PACKAGES = ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]


def run_embedded(code: str, *args: str | Path) -> None:
  py = venv_python(ENGINE)
  try:
    subprocess.check_call([str(py), "-c", code, *(str(arg) for arg in args)], env=local_env(ENGINE))
  except subprocess.CalledProcessError as exc:
    raise SystemExit(f"MeloTTS command failed with exit code {exc.returncode}") from None


def python_version(py: str | Path) -> tuple[int, int]:
  result = subprocess.run(
    [str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
  )
  major, minor = result.stdout.strip().split(".", 1)
  return int(major), int(minor)


def assert_supported_python(py: str | Path, *, existing_venv: bool = False) -> None:
  version = python_version(py)
  if SUPPORTED_PYTHON_MIN <= version <= SUPPORTED_PYTHON_MAX:
    return
  source = f"existing MeloTTS venv at {venv_dir(ENGINE)} uses" if existing_venv else "selected Python is"
  raise SystemExit(
    f"MeloTTS requires Python 3.9-3.11; {source} Python {version[0]}.{version[1]}. "
    "Set MELOTTS_PYTHON to a supported interpreter before first install. "
    f"On this machine, try: MELOTTS_PYTHON=/home/xbob/.local/bin/python3.11 {INSTALL_COMMAND}"
  )


def install_python() -> str:
  env_python = os.environ.get("MELOTTS_PYTHON")
  candidates = [env_python] if env_python else []
  candidates.extend(shutil.which(name) for name in ("python3.11", "python3.10", "python3.9"))
  for candidate in candidates:
    if not candidate:
      continue
    try:
      assert_supported_python(candidate)
      return candidate
    except Exception:
      continue
  raise SystemExit(
    "MeloTTS requires Python 3.9-3.11 and no supported interpreter was found. "
    "Install Python 3.11 or set MELOTTS_PYTHON to a supported interpreter."
  )


def install() -> None:
  if venv_python(ENGINE).exists():
    assert_supported_python(venv_python(ENGINE), existing_venv=True)
    base_python = None
  else:
    base_python = install_python()
  pip_install(ENGINE, [MELOTTS_GIT_REQUIREMENT, "PyYAML"], python_executable=base_python)
  download_nltk_data()
  run([venv_python(ENGINE), "-m", "unidic", "download"])


def download_nltk_data() -> None:
  require_venv_modules(ENGINE, ["nltk"], INSTALL_COMMAND)
  target = cache_dir(ENGINE) / "nltk_data"
  target.mkdir(parents=True, exist_ok=True)
  code = """
import sys
import nltk

target = sys.argv[1]
for package in sys.argv[2:]:
    nltk.download(package, download_dir=target, quiet=True, raise_on_error=True)
    print(f"Downloaded NLTK data {package} to {target}")
"""
  run_embedded(code, target, *NLTK_PACKAGES)


def require_nltk_data() -> None:
  require_venv_modules(ENGINE, ["nltk"], INSTALL_COMMAND)
  code = """
import sys
import nltk.data

missing = []
for package in sys.argv[1:]:
    try:
        nltk.data.find(f"taggers/{package}/")
    except LookupError:
        missing.append(package)
if missing:
    raise SystemExit(
        "MeloTTS is missing NLTK data: "
        + ", ".join(missing)
        + ". Run: python tools/itemaudio/engines/melotts.py --download EN"
    )
"""
  run_embedded(code, *NLTK_PACKAGES)


def download(model: str) -> None:
  # MeloTTS downloads models lazily when TTS(language=...) is constructed.
  if not model:
    raise SystemExit("--download requires a MeloTTS language/model code, e.g. EN")
  require_venv_modules(ENGINE, ["melo", "yaml"], INSTALL_COMMAND)
  download_nltk_data()
  code = """
import sys
from melo.api import TTS
language = sys.argv[1]
TTS(language=language, device="cpu")
print(f"Downloaded/loaded MeloTTS language {language}")
"""
  run_embedded(code, model)


def process(phrases: Path, output_dir: Path) -> None:
  require_venv_modules(ENGINE, ["melo", "yaml"], INSTALL_COMMAND)
  require_nltk_data()
  worker = r"""
import sys
from pathlib import Path

import yaml
from melo.api import TTS

def hparams_to_dict(value):
    if isinstance(value, dict):
        return value
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return dict(items())
        except Exception:
            pass
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {key: item for key, item in attrs.items() if not key.startswith("_")}
    return {}

phrases_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
with phrases_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
items = data.get("items") or []
models = {}
for index, item in enumerate(items, start=1):
    if not isinstance(item, dict):
        continue
    text = str(item.get("text") or "").strip()
    language = str(item.get("model") or "").strip()
    speaker = str(item.get("voice") or item.get("speaker") or "").strip()
    output_name = str(item.get("output_name") or "").strip()
    if not text or not language or not output_name:
        raise SystemExit(f"Item #{index} must define text, model, and output_name")
    if language not in models:
        models[language] = TTS(language=language, device="cpu")
    tts = models[language]
    speaker_ids = hparams_to_dict(tts.hps.data.spk2id)
    if not speaker_ids:
        raise SystemExit(f"MeloTTS language {language} did not expose any speakers")
    speaker_id = speaker_ids.get(speaker) if speaker else next(iter(speaker_ids.values()))
    if speaker_id is None:
        available = ", ".join(sorted(str(key) for key in speaker_ids))
        raise SystemExit(f"Unknown MeloTTS speaker {speaker!r} for language {language}; available: {available}")
    out = output_dir / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text, speaker_id, str(out))
    print(f"Writing {out}: {text} ({language}/{speaker or speaker_id})")
"""
  output_dir.mkdir(parents=True, exist_ok=True)
  run_embedded(worker, phrases, output_dir)


def main() -> None:
  parser = argparse.ArgumentParser(description="Wordbank MeloTTS adapter")
  parser.add_argument("--process", help="Phrase YAML to synthesize")
  parser.add_argument("--output", default="./output_audio", help="Output directory")
  parser.add_argument("--install", action="store_true", help="Install MeloTTS into its venv")
  parser.add_argument("--download", nargs="?", const="", help="Download/load a MeloTTS language model")
  parser.add_argument("--status", action="store_true", help="Report adapter status")
  parser.add_argument("--json", action="store_true", help="Emit JSON for --status")
  args = parser.parse_args()

  if args.status:
    print_status(status_payload(ENGINE, extra={"cache_dir": str(cache_dir(ENGINE))}), args.json)
    return
  if args.install:
    install()
    return
  if args.download is not None:
    download(args.download)
    return
  if args.process:
    process(Path(args.process), Path(args.output))
    return
  parser.error("one of --process, --install, --download, or --status is required")


if __name__ == "__main__":
  main()
