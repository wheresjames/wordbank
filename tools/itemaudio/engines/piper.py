#!/usr/bin/env python3
"""Piper TTS adapter for Wordbank audio generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from common import cache_dir, load_phrase_items, pip_install, print_status, status_payload, venv_bin


ENGINE = "piper"
PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_url(voice: str) -> str | None:
  # Voice names follow <locale>-<name>-<quality>, e.g. de_DE-thorsten-high,
  # which map to <lang>/<locale>/<name>/<quality>/<voice>.onnx in the
  # rhasspy/piper-voices Hugging Face repo.
  parts = voice.split("-")
  if len(parts) != 3:
    return None
  locale, name, quality = parts
  lang = locale.split("_", 1)[0]
  return f"{PIPER_VOICES_BASE}/{lang}/{locale}/{name}/{quality}/{voice}.onnx"


def voice_path(voice: str) -> Path:
  raw = Path(voice)
  if raw.exists():
    return raw
  cache = cache_dir(ENGINE)
  if raw.suffix == ".onnx":
    return cache / raw.name
  return cache / f"{voice}.onnx"


def download_voice(target: str) -> None:
  if not target:
    raise SystemExit("--download requires a local .onnx path or URL")
  cache_dir(ENGINE).mkdir(parents=True, exist_ok=True)
  if target.startswith(("http://", "https://")):
    dest = cache_dir(ENGINE) / Path(target.split("?", 1)[0]).name
    urllib.request.urlretrieve(target, dest)
    print(f"Downloaded {target} -> {dest}")
    return
  path = Path(target)
  if path.exists():
    dest = cache_dir(ENGINE) / path.name
    dest.write_bytes(path.read_bytes())
    print(f"Copied {path} -> {dest}")
    return
  url = voice_url(target)
  if url:
    # Piper needs the model and its .onnx.json config side by side.
    for source in (url, f"{url}.json"):
      dest = cache_dir(ENGINE) / Path(source).name
      urllib.request.urlretrieve(source, dest)
      print(f"Downloaded {source} -> {dest}")
    return
  print(f"No download performed for {target!r}; provide a URL or existing .onnx file.")


def process(phrases: Path, output_dir: Path) -> None:
  exe = venv_bin(ENGINE, "piper")
  if not exe.exists():
    raise SystemExit("Piper is not installed. Run: python engines/piper.py --install")
  output_dir.mkdir(parents=True, exist_ok=True)
  for item in load_phrase_items(phrases):
    text = str(item["text"])
    voice = str(item.get("voice") or item.get("model") or "").strip()
    if not voice:
      raise SystemExit("Piper item missing voice/model")
    model = voice_path(voice)
    if not model.exists():
      raise SystemExit(f"Piper voice not found: {model}")
    out = output_dir / str(item["output_name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
      [str(exe), "--model", str(model), "--output_file", str(out)],
      input=text,
      text=True,
      check=True,
    )
    print(f"Writing {out}: {text} ({model.name})")


def main() -> None:
  parser = argparse.ArgumentParser(description="Wordbank Piper TTS adapter")
  parser.add_argument("--process", help="Phrase YAML to synthesize")
  parser.add_argument("--output", default="./output_audio", help="Output directory")
  parser.add_argument("--install", action="store_true", help="Install Piper into its venv")
  parser.add_argument("--download", nargs="?", const="", help="Download/copy a Piper .onnx voice")
  parser.add_argument("--status", action="store_true", help="Report adapter status")
  parser.add_argument("--json", action="store_true", help="Emit JSON for --status")
  args = parser.parse_args()

  if args.status:
    voices = sorted(p.name for p in cache_dir(ENGINE).glob("*.onnx")) if cache_dir(ENGINE).exists() else []
    print_status(status_payload(ENGINE, executable="piper", extra={"voices": voices}), args.json)
    return
  if args.install:
    pip_install(ENGINE, ["piper-tts"])
    return
  if args.download is not None:
    download_voice(args.download)
    return
  if args.process:
    process(Path(args.process), Path(args.output))
    return
  parser.error("one of --process, --install, --download, or --status is required")


if __name__ == "__main__":
  main()
