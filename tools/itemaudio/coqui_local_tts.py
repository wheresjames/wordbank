#!/usr/bin/env python3
"""
coqui_local_tts.py

Local Coqui TTS test runner for Debian-based Linux.

Commands:
  ./coqui_local_tts.py --setup
  ./coqui_local_tts.py --download
  ./coqui_local_tts.py --process phrases.yml --output ./audio

The --download command creates a local virtualenv in ./coqui_local/.venv and
downloads Python dependencies plus the configured Coqui model into ./coqui_local/cache.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "coqui_local"
VENV = LOCAL / ".venv"
CACHE = LOCAL / "cache"
CONFIG = ROOT / "config.yml"
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)
PYTHON_CANDIDATES = ("python3.11", "python3.10", "python3.9")
SETUPTOOLS_SPEC = "setuptools<82"
TRANSFORMERS_SPEC = "transformers>=4.33,<5"
DEFAULT_LANGUAGE_MODELS = {
    "en": "tts_models/en/ljspeech/vits",
    "de": "tts_models/de/thorsten/vits",
    "fr": "tts_models/fr/css10/vits",
    "es": "tts_models/es/css10/vits",
    "it": "tts_models/it/mai_female/vits",
    "ja": "tts_models/ja/kokoro/tacotron2-DDC",
    "zh-cn": "tts_models/zh-CN/baker/tacotron2-DDC-GST",
}

def run(cmd, env=None):
    display = []
    summarize_next = False
    for token in (str(c) for c in cmd):
        if summarize_next:
            # Avoid dumping the entire inline worker script into the log.
            display.append("'<inline script: %d lines>'" % (token.count("\n") + 1))
            summarize_next = False
            continue
        display.append(shlex.quote(token))
        if token == "-c":
            summarize_next = True
    print("+", " ".join(display))
    try:
        subprocess.check_call([str(c) for c in cmd], env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None

def python_version(python):
    code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    try:
        out = subprocess.check_output([str(python), "-c", code], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    major, minor = out.split(".", 1)
    return int(major), int(minor)

def is_supported_python(version):
    return version is not None and MIN_PYTHON <= version < MAX_PYTHON

def format_python_range():
    return f">={MIN_PYTHON[0]}.{MIN_PYTHON[1]}, <{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"

def python_hint():
    lines = [
        "Install a compatible interpreter, then rerun with COQUI_PYTHON.",
    ]
    if which("uv"):
        lines.extend(
            [
                "With uv:",
                "  uv python install 3.11",
                "  rm -rf ./coqui_local/.venv",
                "  COQUI_PYTHON=\"$(uv python find 3.11)\" ./coqui_local_tts.py --download",
            ]
        )
    lines.extend(
        [
            "With apt on distributions that package Python 3.11:",
            "  sudo apt install python3.11 python3.11-venv",
            "  rm -rf ./coqui_local/.venv",
            "  COQUI_PYTHON=python3.11 ./coqui_local_tts.py --download",
        ]
    )
    return "\n".join(lines)

def coqui_python():
    requested = os.environ.get("COQUI_PYTHON")
    candidates = [requested] if requested else [sys.executable]
    candidates.extend(which(name) for name in PYTHON_CANDIDATES)

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)

        version = python_version(candidate)
        if is_supported_python(version):
            return candidate

    current = python_version(sys.executable)
    current_text = "unknown" if current is None else ".".join(str(part) for part in current)
    requested_text = f" COQUI_PYTHON={requested!r} was not compatible." if requested else ""
    raise SystemExit(
        "Coqui TTS requires Python "
        f"{format_python_range()}, but this script is running with Python {current_text}."
        f"{requested_text}\n"
        f"{python_hint()}"
    )

def venv_python():
    return VENV / "bin" / "python"

def patch_tts_transformers_import():
    pattern = "lib/python*/site-packages/TTS/tts/layers/xtts/stream_generator.py"
    matches = list(VENV.glob(pattern))
    if not matches:
        return

    path = matches[0]
    text = path.read_text(encoding="utf-8")
    old = """from transformers import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
"""
    new = """from transformers import (
    ConstrainedBeamSearchScorer,
"""
    if old not in text:
        return

    text = text.replace(old, new, 1)
    if "from transformers.generation.beam_search import BeamSearchScorer\n" not in text:
        text = text.replace(
            ")\nfrom transformers.generation.utils import GenerateOutput, SampleOutput, logger\n",
            ")\nfrom transformers.generation.beam_search import BeamSearchScorer\nfrom transformers.generation.utils import GenerateOutput, SampleOutput, logger\n",
            1,
        )
    path.write_text(text, encoding="utf-8")

def patch_tts_torch_load():
    pattern = "lib/python*/site-packages/TTS/utils/io.py"
    matches = list(VENV.glob(pattern))
    if not matches:
        return

    path = matches[0]
    text = path.read_text(encoding="utf-8")
    marker = '    is_local = os.path.isdir(path) or os.path.isfile(path)\n'
    inject = """    if "weights_only" not in kwargs:\n        kwargs["weights_only"] = False\n\n"""
    if inject.strip() in text:
        return
    if marker not in text:
        return

    text = text.replace(marker, inject + marker, 1)
    path.write_text(text, encoding="utf-8")

def ensure_venv_python_supported():
    py = venv_python()
    if not py.exists():
        return

    version = python_version(py)
    if is_supported_python(version):
        return

    version_text = "unknown" if version is None else ".".join(str(part) for part in version)
    raise SystemExit(
        f"Existing virtualenv uses unsupported Python {version_text}; Coqui TTS requires "
        f"Python {format_python_range()}.\n"
        f"{python_hint()}"
    )

def local_env():
    e = os.environ.copy()
    e["TTS_HOME"] = str(CACHE / "tts")
    e["HF_HOME"] = str(CACHE / "huggingface")
    e["TRANSFORMERS_CACHE"] = str(CACHE / "transformers")
    return e

def setup():
    print("Setting up Coqui TTS...")

def read_simple_config():
    if not CONFIG.exists():
        raise SystemExit("Missing config.yml. Run: ./coqui_local_tts.py --setup")

    cfg = {}
    for raw in CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            cfg[key] = None
        elif value.lower() in ("true", "false"):
            cfg[key] = value.lower() == "true"
        else:
            cfg[key] = value.strip("\"'")
    return cfg

def download():
    LOCAL.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    ensure_venv_python_supported()

    if not VENV.exists():
        run([coqui_python(), "-m", "venv", VENV])

    py = venv_python()

    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", SETUPTOOLS_SPEC], env=local_env())
    run([py, "-m", "pip", "install", "TTS[ja]", "PyYAML", TRANSFORMERS_SPEC], env=local_env())
    patch_tts_transformers_import()
    patch_tts_torch_load()

    code = f"""
from pathlib import Path

import yaml
from TTS.api import TTS

config_path = Path({str(CONFIG)!r})
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {{}}

model_names = []
legacy = config.get("language_models") or {{}}
if isinstance(legacy, dict):
    for model in legacy.values():
        if model and model not in model_names:
            model_names.append(model)

languages = config.get("languages") or {{}}
if isinstance(languages, dict):
    for lang_spec in languages.values():
        if not isinstance(lang_spec, dict):
            continue
        voice_sets = lang_spec.get("voice_sets") or {{}}
        if not isinstance(voice_sets, dict):
            continue
        for route in voice_sets.values():
            if not isinstance(route, dict):
                continue
            if str(route.get("engine") or "").lower() != "coqui":
                continue
            model = route.get("model")
            if model and model not in model_names:
                model_names.append(model)

fallback_model = config.get("model_name")
if not model_names and fallback_model:
    model_names.append(fallback_model)
if not model_names:
    raise SystemExit("No models configured. Set 'model_name' or 'language_models' in config.yml.")

for model in model_names:
    print("Downloading/loading model:", model)
    TTS(model)

print("Models ready.")
"""
    run([py, "-c", code], env=local_env())

def process(yml_file, output_dir):
    py = venv_python()
    if not py.exists():
        raise SystemExit("Local environment missing. Run: ./coqui_local_tts.py --download")

    if not Path(yml_file).exists():
        raise SystemExit(f"Input YAML file not found: {yml_file}")

    if not CONFIG.exists():
        raise SystemExit("Missing config.yml. Run: ./coqui_local_tts.py --setup")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = """
import hashlib
import re
import sys
from pathlib import Path

import yaml
from TTS.api import TTS

config_path = Path(sys.argv[1])
phrases_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping/object")
    return data

def safe_name(text, index, language):
    short = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())[:48].strip("_")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    lang = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(language or "und"))
    return f"{index:04d}_{lang}_{short or 'item'}_{digest}"

config = load_yaml(config_path)
phrases = load_yaml(phrases_path)

model_name = config.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
language_models = {}
legacy_language_models = config.get("language_models") or {}
if isinstance(legacy_language_models, dict):
    language_models.update({str(k).lower(): v for k, v in legacy_language_models.items() if v})
profile_languages = config.get("languages") or {}
if isinstance(profile_languages, dict):
    for raw_lang, lang_spec in profile_languages.items():
        if not isinstance(lang_spec, dict):
            continue
        selected = str(lang_spec.get("selected") or "").strip()
        voice_sets = lang_spec.get("voice_sets") or {}
        route = voice_sets.get(selected) if isinstance(voice_sets, dict) else None
        if isinstance(route, dict) and str(route.get("engine") or "").lower() == "coqui" and route.get("model"):
            language_models[str(raw_lang).lower()] = route.get("model")
output_format = config.get("output_format", "wav")
skip_unsupported_languages = bool(config.get("skip_unsupported_languages", False))

global_defaults = {
    "language": config.get("default_language", "en"),
    "speaker": config.get("default_speaker"),
    "speaker_wav": config.get("default_speaker_wav"),
    "split_sentences": config.get("split_sentences", False),
}

phrase_defaults = phrases.get("defaults") or {}
if not isinstance(phrase_defaults, dict):
    raise SystemExit("phrases.yml 'defaults' must be a mapping/object")
phrase_defaults = {k: v for k, v in phrase_defaults.items() if v not in (None, "")}

defaults = {**global_defaults, **phrase_defaults}

items = phrases.get("items") or []
if not isinstance(items, list):
    raise SystemExit("phrases.yml 'items' must be a list")

missing_languages = []
for index, item in enumerate(items, start=1):
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, dict):
        continue
    if item.get("model"):
        continue
    language = item.get("language") or defaults.get("language")
    language_key = str(language or "").lower()
    if language_models and language_key not in language_models and language_key not in missing_languages:
        missing_languages.append(language_key or "<unset>")

if missing_languages and not skip_unsupported_languages:
    raise SystemExit(
        "No Coqui single-language model configured for: "
        + ", ".join(missing_languages)
        + ". Add entries under 'language_models' in config.yml or remove those items."
    )

tts_by_model = {}

def tts_for_item(item, language):
    language_key = str(language or "").lower()
    selected_model = item.get("model") or language_models.get(language_key) or model_name
    if selected_model not in tts_by_model:
        print(f"Loading model: {selected_model}")
        tts_by_model[selected_model] = TTS(selected_model)
    return tts_by_model[selected_model], selected_model

count = 0
for index, item in enumerate(items, start=1):
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, dict):
        raise SystemExit(f"Item #{index} must be either a string or mapping/object")

    text = str(item.get("text", "")).strip()
    if not text:
        print(f"Skipping item #{index}: empty text")
        continue

    language = item.get("language") or defaults.get("language")
    speaker = item.get("speaker") or defaults.get("speaker")
    speaker_wav = item.get("speaker_wav") or defaults.get("speaker_wav")
    split_sentences = item.get("split_sentences", defaults.get("split_sentences", False))
    output_name = item.get("output_name")
    if not item.get("model") and language_models and str(language or "").lower() not in language_models:
        print(f"Skipping item #{index}: no model configured for language {language!r}")
        continue
    tts, selected_model = tts_for_item(item, language)

    if speaker_wav and tts.is_multi_speaker:
        speaker_wav = str(Path(speaker_wav))
        if not Path(speaker_wav).exists():
            raise SystemExit(f"Speaker WAV not found for item #{index}: {speaker_wav}")

    if tts.is_multi_speaker and not speaker and not speaker_wav:
        raise SystemExit(
            "Loaded model requires either `speaker` or `speaker_wav` for each item. "
            "Set `default_speaker_wav` in config.yml, add `speaker_wav` to phrases.yml items, "
            "or use `default_speaker` / `speaker` when the model has named speakers."
        )

    if output_name:
        out = output_dir / output_name
        if out.suffix == "":
            out = out.with_suffix("." + output_format)
    else:
        out = output_dir / f"{safe_name(text, index, language)}.{output_format}"

    out.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "text": text,
        "file_path": str(out),
        "split_sentences": bool(split_sentences),
    }

    if language and tts.is_multi_lingual:
        kwargs["language"] = language
    if speaker and tts.is_multi_speaker:
        kwargs["speaker"] = str(speaker)
    if speaker_wav and tts.is_multi_speaker:
        kwargs["speaker_wav"] = speaker_wav

    print(f"Writing {out}: [{language}] {text} ({selected_model})")
    tts.tts_to_file(**kwargs)
    count += 1

print(f"Done. Wrote {count} audio files to {output_dir}")
"""

    run([py, "-c", worker, CONFIG, yml_file, output_dir], env=local_env())

def process_translations(yml_file, output_dir):
    py = venv_python()
    if not py.exists():
        raise SystemExit("Local environment missing. Run: ./coqui_local_tts.py --download")

    if not Path(yml_file).exists():
        raise SystemExit(f"Input YAML file not found: {yml_file}")

    if not CONFIG.exists():
        raise SystemExit("Missing config.yml. Run: ./coqui_local_tts.py --setup")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = """
import re
import sys
from pathlib import Path

import yaml
from TTS.api import TTS

config_path = Path(sys.argv[1])
translations_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

LANGUAGE_ALIASES = {
    "jp": "ja",
    "cn": "zh-cn",
    "zh": "zh-cn",
    "ua": "uk",
}
IGNORED_LANGUAGES = {"cn", "zh", "zh-cn"}

def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping/object")
    return data

def file_stem(text):
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip().lower()).strip("_")
    return stem or "item"

def normalize_language(language):
    key = str(language or "").lower()
    return LANGUAGE_ALIASES.get(key, key)

def unique_path(path):
    if not path.exists():
        return path
    base = path.with_suffix("")
    suffix = path.suffix
    index = 2
    while True:
        candidate = Path(f"{base}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1

config = load_yaml(config_path)
data = load_yaml(translations_path)

model_name = config.get("model_name", "tts_models/en/ljspeech/vits")
language_models = config.get("language_models") or {}
if not isinstance(language_models, dict):
    raise SystemExit("config.yml 'language_models' must be a mapping/object")
language_models = {str(k).lower(): v for k, v in language_models.items() if v}
output_format = config.get("output_format", "wav")
split_sentences = bool(config.get("split_sentences", False))

categories = data.get("categories")
translations_legacy = data.get("translations")

if categories is not None:
    if not isinstance(categories, dict):
        raise SystemExit("translations.yml 'categories' must be a mapping/object")
    all_translations = {
        item: trans
        for cat_items in categories.values()
        if isinstance(cat_items, dict)
        for item, trans in cat_items.items()
        if isinstance(trans, dict)
    }
elif translations_legacy is not None:
    if not isinstance(translations_legacy, dict):
        raise SystemExit("translations.yml 'translations' must be a mapping/object")
    all_translations = translations_legacy
else:
    raise SystemExit("translations.yml must contain a 'categories' or 'translations' mapping/object")

tts_by_model = {}

def tts_for_language(language):
    selected_model = language_models.get(language) or model_name
    if selected_model not in tts_by_model:
        print(f"Loading model: {selected_model}")
        tts_by_model[selected_model] = TTS(selected_model)
    return tts_by_model[selected_model], selected_model

count = 0
skipped = 0
for english, translated in all_translations.items():
    if not isinstance(translated, dict):
        print(f"Skipping {english!r}: translations must be a mapping/object")
        skipped += 1
        continue

    base_name = file_stem(english)
    values = {"en": english, **translated}
    for raw_language, text in values.items():
        raw_key = str(raw_language or "").lower()
        language = normalize_language(raw_key)
        text = str(text or "").strip()

        if not text:
            skipped += 1
            continue
        if raw_key in IGNORED_LANGUAGES or language in IGNORED_LANGUAGES:
            skipped += 1
            continue
        if language_models and language not in language_models:
            print(f"Skipping {english!r} [{raw_language}]: no model configured for language {language!r}")
            skipped += 1
            continue

        lang_dir = output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        out = unique_path(lang_dir / f"{base_name}.{output_format}")

        tts, selected_model = tts_for_language(language)
        kwargs = {
            "text": text,
            "file_path": str(out),
            "split_sentences": split_sentences,
        }
        if tts.is_multi_lingual:
            kwargs["language"] = language

        print(f"Writing {out}: [{language}] {text} ({selected_model})")
        tts.tts_to_file(**kwargs)
        count += 1

print(f"Done. Wrote {count} audio files to {output_dir}; skipped {skipped}.")
"""

    run([py, "-c", worker, CONFIG, yml_file, output_dir], env=local_env())

def main():
    parser = argparse.ArgumentParser(description="Local Coqui TTS YAML test runner")
    parser.add_argument("--setup", action="store_true", help="Create ./config.yml and ./phrases.yml")
    parser.add_argument("--download", action="store_true", help="Create local venv and download Coqui/model")
    parser.add_argument("--process", metavar="PHRASES_YML", help="YAML file containing words/phrases and language tags")
    parser.add_argument("--translations", metavar="TRANSLATIONS_YML", help="YAML file mapping English items to translations")
    parser.add_argument("--output", default="./output_audio", help="Output folder")

    args = parser.parse_args()

    if args.setup:
        setup()
    if args.download:
        download()
    if args.process:
        process(args.process, args.output)
    if args.translations:
        process_translations(args.translations, args.output)

    if not any([args.setup, args.download, args.process, args.translations]):
        parser.print_help()

if __name__ == "__main__":
    main()
