#!/usr/bin/env python3
"""Prepare and optionally synthesize audio candidates from Wordbank words.yml."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from wordbank_common import (  # noqa: E402
  compose_text,
  load_audio_profile,
  load_corpus,
  normalize_lang,
  parse_wordlist,
  resolve_voice_set,
)


def build_entries(
    wordbank_dir: Path,
    keys: list[str],
    lang: str,
    voice_set: str,
    route: dict,
    takes: int,
    output_format: str,
    corpus=None,
):
  _, flat = load_corpus(wordbank_dir, corpus)
  missing = [key for key in keys if key not in flat]
  if missing:
    raise SystemExit("Missing Wordbank keys: " + ", ".join(missing))

  engine = str(route.get("engine") or "")
  model = route.get("model")
  voice = route.get("voice")
  entries = []
  for key in keys:
    item = flat[key]
    text = compose_text(key, item, lang)
    if not text:
      print(f"WARN: {key}: no text for language {lang}")
      continue
    for take in range(1, takes + 1):
      filename = f"{key}_{take:03d}.{output_format}"
      entry = {
        "key": key,
        "language": lang,
        "voice_set": voice_set,
        "engine": engine,
        "model": model,
        "voice": voice,
        "text": text,
        "take": take,
        "filename": filename,
        "status": "pending",
      }
      if route.get("speaker"):
        entry["speaker"] = route.get("speaker")
      if route.get("speaker_wav"):
        entry["speaker_wav"] = route.get("speaker_wav")
      entries.append(entry)
  return entries


def load_existing_entries(output_dir: Path) -> list[dict]:
  path = output_dir / "manifest.json"
  if not path.exists():
    return []
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return []
  if isinstance(data, list):
    return data
  if isinstance(data, dict) and isinstance(data.get("entries"), list):
    return data["entries"]
  return []


def audio_present_keys(output_dir: Path) -> set[str]:
  """Keys that already have at least one candidate audio file in this voice-set dir.

  Trusts the manifest first (a key counts only if its clip exists on disk), then
  sweeps for stray ``<key>_<take>.<ext>`` files not recorded in the manifest."""
  present: set[str] = set()
  for entry in load_existing_entries(output_dir):
    filename = str(entry.get("filename") or "")
    key = str(entry.get("key") or "")
    if key and filename and (output_dir / filename).exists():
      present.add(key)
  if output_dir.exists():
    for path in output_dir.iterdir():
      if not path.is_file():
        continue
      base, sep, take = path.stem.rpartition("_")
      if sep and base and take.isdigit():
        present.add(base)
  return present


def renumber_takes(existing: list[dict], new_entries: list[dict], output_format: str) -> list[dict]:
  """Shift new entries' take numbers/filenames past the highest existing take
  for the same key in this isolated voice-set dir, so merge never collides."""
  max_take: dict[str, int] = {}
  for entry in existing:
    key = str(entry.get("key") or "")
    take = entry.get("take")
    if isinstance(take, int):
      max_take[key] = max(max_take.get(key, 0), take)
  counter: dict[str, int] = {}
  for entry in new_entries:
    key = str(entry.get("key") or "")
    counter[key] = counter.get(key, max_take.get(key, 0)) + 1
    take = counter[key]
    entry["take"] = take
    entry["filename"] = f"{key}_{take:03d}.{output_format}"
  return new_entries


def write_manifest(output_dir: Path, entries: list[dict]) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
    json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)


def write_phrases(output_dir: Path, entries: list[dict]) -> Path:
  phrases = {
    "items": [
      {
        "text": entry["text"],
        "language": entry["language"],
        "model": entry.get("model"),
        "voice": entry.get("voice"),
        "speaker": entry.get("speaker"),
        "speaker_wav": entry.get("speaker_wav"),
        "output_name": entry["filename"],
      }
      for entry in entries
    ]
  }
  path = output_dir / "phrases.generated.yml"
  with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(phrases, f, sort_keys=False, allow_unicode=True, width=1000)
  return path


def normalize_outputs(output_dir: Path, entries: list[dict]) -> None:
  ffmpeg = shutil.which("ffmpeg")
  if not ffmpeg:
    raise SystemExit("ffmpeg is required to normalize synthesized audio to 22.05 kHz mono WAV")
  for entry in entries:
    path = output_dir / str(entry.get("filename") or "")
    if not path.exists() or path.suffix.lower() != ".wav":
      continue
    tmp = path.with_suffix(".normalized.wav")
    subprocess.check_call([
      ffmpeg,
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      str(path),
      "-ac",
      "1",
      "-ar",
      "22050",
      "-c:a",
      "pcm_s16le",
      str(tmp),
    ])
    tmp.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate audio candidate manifest from Wordbank")
  parser.add_argument("--wordbank", default=ROOT, help="Wordbank directory")
  parser.add_argument("--corpus", help="Corpus name or path under wordlists/ (default: children-001)")
  parser.add_argument("--wordlist", help="Optional app wordlist.yml; defaults to every corpus key")
  parser.add_argument("--keys", help="Optional comma-separated key filter")
  parser.add_argument("--lang", help="Single language to generate")
  parser.add_argument("--langs", help="Compatibility alias for --lang; must name exactly one language")
  parser.add_argument("--voice-set", help="Voice set name for the selected language (default: configured selected voice set)")
  parser.add_argument("--takes", type=int, default=1, help="Candidate takes per key/language")
  parser.add_argument("--output", default="output/audio", help="Audio output root; candidates are written under <output>/<lang>/<voice-set>")
  parser.add_argument("--format", default="wav", help="Audio file extension")
  parser.add_argument("--synthesize", action="store_true", help="Run the selected engine adapter after writing manifest")
  parser.add_argument("--merge", action="store_true", help="Append new takes to the existing manifest instead of replacing it")
  parser.add_argument("--skip-existing", action="store_true", help="Only generate keys that have no candidate audio yet (implies --merge)")
  args = parser.parse_args()

  wordbank_dir = Path(args.wordbank).resolve()
  audio_root = Path(args.output).resolve()
  langs = [normalize_lang(args.lang)] if args.lang else []
  if args.langs:
    langs.extend(normalize_lang(lang) for lang in args.langs.split(",") if lang.strip())
  langs = [lang for i, lang in enumerate(langs) if lang and lang not in langs[:i]]
  if len(langs) != 1:
    raise SystemExit("Phase 1 audio generation requires exactly one language via --lang or --langs")
  lang = langs[0]

  if args.wordlist:
    keys, _ = parse_wordlist(Path(args.wordlist).resolve())
  else:
    _, flat = load_corpus(wordbank_dir, args.corpus)
    keys = list(flat.keys())
  if args.keys:
    requested = {key.strip() for key in args.keys.split(",") if key.strip()}
    keys = [key for key in keys if key in requested]

  profile = load_audio_profile(wordbank_dir)
  voice_set, route = resolve_voice_set(profile, lang, args.voice_set)
  engine = str(route.get("engine") or "")
  output_dir = audio_root / lang / voice_set

  if args.skip_existing:
    present = audio_present_keys(output_dir)
    kept = [key for key in keys if key not in present]
    print(f"--skip-existing: {len(keys) - len(kept)} key(s) already have audio; generating {len(kept)} missing")
    keys = kept
    args.merge = True  # preserve existing entries; only append the missing keys

  entries = build_entries(wordbank_dir, keys, lang, voice_set, route, args.takes, args.format, args.corpus)

  if args.merge:
    existing = load_existing_entries(output_dir)
    entries = renumber_takes(existing, entries, args.format)
    write_manifest(output_dir, existing + entries)
  else:
    write_manifest(output_dir, entries)
  # Only the new entries need synthesis; existing clips keep their audio files.
  phrases_path = write_phrases(output_dir, entries)

  print(f"Wrote {len(entries)} candidate manifest entries to {output_dir / 'manifest.json'}")
  print(f"Wrote {engine} phrase file to {phrases_path}")

  if args.synthesize and entries:
    script = Path(__file__).resolve().parent / "engines" / f"{engine}.py"
    if not script.exists():
      raise SystemExit(f"Engine adapter not implemented: {engine}")
    result = subprocess.run([sys.executable, str(script), "--process", str(phrases_path), "--output", str(output_dir)])
    if result.returncode:
      raise SystemExit(f"{engine} synthesis failed with exit code {result.returncode}")
    normalize_outputs(output_dir, entries)


if __name__ == "__main__":
  main()
