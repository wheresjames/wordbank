#!/usr/bin/env python3
"""One-time migration from the old flat audio layout to the per-voice-set layout.

Old layout (pre WB-AUDIO refactor):

    output/audio/manifest.json          # single manifest, no voice_set field
    output/audio/<lang>/<key>_<take>.wav

New layout (see WB-AUDIO.md section 5):

    output/audio/<lang>/<voice_set>/manifest.json
    output/audio/<lang>/<voice_set>/<key>_<take>.wav

Each old entry is mapped to a voice set by matching its (language, model) against
the audio config. Files are moved into the nested directories, the manifest gains
`voice_set`/`engine`/`voice` provenance, and a per-(language, voice_set)
manifest.json is written. Approvals and human-review flags are preserved.

Usage:
    python tools/itemaudio/migrate_audio_layout.py --output output/audio [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = Path(__file__).resolve().parent / "config.yml"


def build_voice_set_map(config: dict) -> dict[tuple[str, str], dict]:
  """Map (language, model) -> route metadata (voice_set, engine, voice).

  The old manifest only records `model`, so model is the join key within a
  language. The config has no duplicate models within a language, so this is
  unambiguous.
  """
  out: dict[tuple[str, str], dict] = {}
  for lang, spec in (config.get("languages") or {}).items():
    for voice_set, route in (spec.get("voice_sets") or {}).items():
      model = str(route.get("model") or "").strip()
      if not model:
        continue
      out[(lang, model)] = {
        "voice_set": voice_set,
        "engine": route.get("engine"),
        "voice": route.get("voice"),
      }
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description="Migrate flat audio layout to per-voice-set layout")
  parser.add_argument("--output", default="output/audio", help="Audio output root")
  parser.add_argument("--dry-run", action="store_true", help="Report actions without moving files")
  args = parser.parse_args()

  audio_root = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
  manifest_path = audio_root / "manifest.json"
  if not manifest_path.exists():
    raise SystemExit(f"No flat manifest at {manifest_path}; nothing to migrate.")

  config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
  vs_map = build_voice_set_map(config)

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  entries = manifest.get("entries") if isinstance(manifest, dict) else manifest
  if not isinstance(entries, list):
    raise SystemExit("Unexpected manifest format: expected a list of entries.")

  # Group migrated entries by (language, voice_set).
  buckets: dict[tuple[str, str], list[dict]] = {}
  skipped_missing = 0
  skipped_unmapped: set[tuple[str, str]] = set()
  moved = 0

  for entry in entries:
    lang = str(entry.get("language") or entry.get("lang") or "").strip()
    model = str(entry.get("model") or "").strip()
    filename = str(entry.get("filename") or "")
    if not lang or not filename:
      continue
    src = audio_root / filename
    if not src.exists():
      skipped_missing += 1
      continue
    route = vs_map.get((lang, model))
    if not route:
      skipped_unmapped.add((lang, model))
      continue

    voice_set = route["voice_set"]
    new_name = Path(filename).name  # bare <key>_<take>.wav inside the voice-set dir
    dest_dir = audio_root / lang / voice_set
    dest = dest_dir / new_name

    new_entry = dict(entry)
    new_entry["filename"] = new_name
    new_entry["voice_set"] = voice_set
    if route.get("engine") and not new_entry.get("engine"):
      new_entry["engine"] = route["engine"]
    if route.get("voice") and not new_entry.get("voice"):
      new_entry["voice"] = route["voice"]
    buckets.setdefault((lang, voice_set), []).append(new_entry)

    if args.dry_run:
      moved += 1
      continue
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    moved += 1

  # Write per-voice-set manifests.
  for (lang, voice_set), bucket in sorted(buckets.items()):
    target = audio_root / lang / voice_set / "manifest.json"
    print(f"  {lang}/{voice_set}: {len(bucket)} entries -> {target}")
    if args.dry_run:
      continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entries": bucket}, indent=2, ensure_ascii=False), encoding="utf-8")

  if not args.dry_run:
    backup = manifest_path.with_suffix(".json.flat.bak")
    shutil.move(str(manifest_path), str(backup))
    print(f"Backed up old flat manifest -> {backup}")

  print(f"\nMoved {moved} clips into {len(buckets)} voice-set directories.")
  if skipped_missing:
    print(f"Skipped {skipped_missing} entries with missing files.")
  if skipped_unmapped:
    print(f"Skipped unmapped (language, model) pairs (no matching voice_set in config):")
    for lang, model in sorted(skipped_unmapped):
      print(f"  {lang}: {model!r}")
  if args.dry_run:
    print("\n(dry run — no files moved)")


if __name__ == "__main__":
  main()
