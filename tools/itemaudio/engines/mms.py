#!/usr/bin/env python3
"""Meta MMS-TTS adapter for Wordbank audio generation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import cache_dir, local_env, pip_install, print_status, require_venv_modules, status_payload, venv_python


ENGINE = "mms"
INSTALL_COMMAND = "python tools/itemaudio/engines/mms.py --install"


def run_embedded(code: str, *args: str | Path) -> None:
  py = venv_python(ENGINE)
  try:
    subprocess.check_call([str(py), "-c", code, *(str(arg) for arg in args)], env=local_env(ENGINE))
  except subprocess.CalledProcessError as exc:
    raise SystemExit(f"MMS command failed with exit code {exc.returncode}") from None


def install() -> None:
  pip_install(ENGINE, ["torch", "transformers", "scipy", "PyYAML"])


def download(model: str) -> None:
  if not model:
    raise SystemExit("--download requires a Hugging Face model id, e.g. facebook/mms-tts-deu")
  require_venv_modules(ENGINE, ["transformers", "torch", "scipy", "yaml"], INSTALL_COMMAND)
  code = """
import sys
from transformers import VitsModel, AutoTokenizer
model_id = sys.argv[1]
try:
    AutoTokenizer.from_pretrained(model_id)
    VitsModel.from_pretrained(model_id)
except OSError as exc:
    text = str(exc)
    if (
        "valid model identifier" in text
        or "Repository Not Found" in text
        or "401 Client Error" in text
        or "Can't load the configuration" in text
    ):
        raise SystemExit(
            f"Could not download {model_id}: Hugging Face does not list this as a public model, "
            "or it requires authentication. Check the model id or run `hf auth login` for private/gated repos."
        ) from None
    raise
print(f"Downloaded {model_id}")
"""
  run_embedded(code, model)


def process(phrases: Path, output_dir: Path) -> None:
  require_venv_modules(ENGINE, ["transformers", "torch", "scipy", "yaml"], INSTALL_COMMAND)
  worker = r"""
import sys
from pathlib import Path

import scipy.io.wavfile
import torch
import yaml
from transformers import VitsModel, AutoTokenizer

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
    model_id = str(item.get("model") or item.get("voice") or "").strip()
    output_name = str(item.get("output_name") or "").strip()
    if not text or not model_id or not output_name:
        raise SystemExit(f"Item #{index} must define text, model, and output_name")
    if model_id not in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id)
        except OSError as exc:
            text = str(exc)
            if (
                "valid model identifier" in text
                or "Repository Not Found" in text
                or "401 Client Error" in text
                or "Can't load the configuration" in text
            ):
                raise SystemExit(
                    f"Could not load {model_id} for item #{index}: Hugging Face does not list this as a public model, "
                    "or it requires authentication. Check the model id or run `hf auth login` for private/gated repos."
                ) from None
            raise
        models[model_id] = (tokenizer, model)
    tokenizer, model = models[model_id]
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    waveform = output.squeeze().cpu().numpy()
    sample_rate = int(model.config.sampling_rate)
    out = output_dir / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(out, sample_rate, waveform)
    print(f"Writing {out}: {text} ({model_id})")
"""
  output_dir.mkdir(parents=True, exist_ok=True)
  run_embedded(worker, phrases, output_dir)


def main() -> None:
  parser = argparse.ArgumentParser(description="Wordbank MMS-TTS adapter")
  parser.add_argument("--process", help="Phrase YAML to synthesize")
  parser.add_argument("--output", default="./output_audio", help="Output directory")
  parser.add_argument("--install", action="store_true", help="Install MMS dependencies into its venv")
  parser.add_argument("--download", nargs="?", const="", help="Download a Hugging Face MMS model")
  parser.add_argument("--status", action="store_true", help="Report adapter status")
  parser.add_argument("--json", action="store_true", help="Emit JSON for --status")
  args = parser.parse_args()

  if args.status:
    print_status(status_payload(ENGINE, extra={"huggingface_cache": str(cache_dir(ENGINE) / "huggingface")}), args.json)
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
