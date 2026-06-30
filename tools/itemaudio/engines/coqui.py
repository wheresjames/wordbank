#!/usr/bin/env python3
"""Coqui engine adapter for Wordbank audio generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COQUI_SCRIPT = ROOT / "coqui_local_tts.py"
COQUI_LOCAL = ROOT / "coqui_local"


def run_coqui(args: list[str]) -> None:
  subprocess.check_call([str(COQUI_SCRIPT), *args])


def status(json_output: bool = False) -> None:
  venv = COQUI_LOCAL / ".venv"
  payload = {
    "engine": "coqui",
    "env_installed": venv.exists(),
    "tts_installed": (venv / "bin" / "tts").exists(),
    "cache_dir": str(COQUI_LOCAL / "cache"),
  }
  if json_output:
    print(json.dumps(payload, indent=2))
    return
  print(f"engine: {payload['engine']}")
  print(f"env_installed: {payload['env_installed']}")
  print(f"tts_installed: {payload['tts_installed']}")
  print(f"cache_dir: {payload['cache_dir']}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Wordbank Coqui TTS adapter")
  parser.add_argument("--process", help="Phrase YAML to synthesize")
  parser.add_argument("--output", default="./output_audio", help="Output directory")
  parser.add_argument("--install", action="store_true", help="Create/repair the Coqui environment")
  parser.add_argument("--download", nargs="?", const="", help="Download configured Coqui models")
  parser.add_argument("--status", action="store_true", help="Report adapter status")
  parser.add_argument("--json", action="store_true", help="Emit JSON for --status")
  args = parser.parse_args()

  if args.status:
    status(args.json)
    return
  if args.install or args.download is not None:
    run_coqui(["--download"])
    return
  if args.process:
    run_coqui(["--process", args.process, "--output", args.output])
    return
  parser.error("one of --process, --install, --download, or --status is required")


if __name__ == "__main__":
  main()
