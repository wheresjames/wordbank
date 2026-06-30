#!/usr/bin/env python3
"""Generate a slim app asset bundle from the Wordbank corpus."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from wordbank_common import (
  FlowList,
  FlowMap,
  compose_text,
  default_output_root,
  load_audio_profile,
  load_audio_selections,
  load_corpus,
  load_image_selections,
  merged_item,
  normalize_lang,
  parse_wordlist,
  resolve_profile,
  selected_voice_sets,
  validate_requested_media,
  write_yaml,
)


def flow_inline_fields(item):
  for field in ("tr", "art", "cx"):
    if isinstance(item.get(field), dict):
      item[field] = FlowMap(item[field])
  if isinstance(item.get("pos"), list):
    item["pos"] = FlowList(item["pos"])
  return item


def selected_by_topic(keys, flat, tags_by_key):
  out = {}
  for key in keys:
    item = flat[key]
    topic = item["topic"]
    out.setdefault(topic, {})[key] = flow_inline_fields(merged_item(key, item, tags_by_key.get(key, {})))
  return out


def output_items_yml(keys, flat):
  categories = {}
  for key in keys:
    categories.setdefault(flat[key]["topic"], {"items": []})["items"].append(key)
  return {"categories": categories}


def output_translations_yml(keys, flat, langs):
  categories = {}
  for key in keys:
    item = flat[key]
    trans = {}
    for lang in langs:
      if lang == "en":
        continue
      text = compose_text(key, item, lang)
      if text is not None:
        trans[lang] = text
    categories.setdefault(item["topic"], {})[key] = FlowMap(trans)
  return {"categories": categories}


def image_metadata(keys: list[str], selections: dict[str, dict], profile: str) -> dict:
  """Per-image provenance carried in the package for lossless round-trip import."""
  out = {}
  for key in keys:
    entry = selections.get(key, {})
    out[key] = FlowMap({
      "filename": f"{key}.png",
      "prompt": entry.get("prompt"),
      "seed": entry.get("seed"),
      "profile": profile,
    })
  return out


def write_audio(src: Path, dst: Path, audio_format: str) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  if src.suffix == f".{audio_format}":
    shutil.copy2(src, dst)
    return
  ffmpeg = shutil.which("ffmpeg")
  if not ffmpeg:
    raise RuntimeError(f"ffmpeg is required to convert {src.suffix} audio to {audio_format}")
  command = [
    ffmpeg,
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    str(src),
  ]
  if audio_format == "ogg":
    command.extend([
    "-c:a",
    "libopus",
    "-b:a",
    "48k",
    "-vbr",
    "on",
    ])
  elif audio_format == "wav":
    command.extend(["-c:a", "pcm_s16le"])
  else:
    raise RuntimeError(f"Unsupported audio format: {audio_format}")
  command.append(str(dst))
  subprocess.run(command, check=True)


def copy_media(
    wordbank_dir: Path,
    out_dir: Path,
    keys: list[str],
    langs: list[str],
    audio_format: str,
    selections: dict[str, dict],
    audio_selections: dict[tuple[str, str], Path],
) -> None:
  for key in keys:
    src = Path(selections[key]["path"])
    dst = out_dir / "images" / f"{key}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    for lang in langs:
      # Copy the approved (or human-approved) clip selected in output/audio.
      src_audio = Path(audio_selections[(lang, key)])
      dst_audio = out_dir / "audio" / lang / f"{key}.{audio_format}"
      write_audio(src_audio, dst_audio, audio_format)


def generate_assets(args) -> int:
  wordbank_dir = Path(args.wordbank).resolve()
  wordlist_path = Path(args.wordlist).resolve() if args.wordlist else None
  out_dir = Path(args.out).resolve()
  langs = [normalize_lang(lang) for lang in args.langs.split(",") if lang.strip()]
  if not langs:
    raise SystemExit("--langs must name at least one language")

  profile = resolve_profile(wordbank_dir, getattr(args, "profile", None))
  output_root_arg = getattr(args, "output_root", None)
  output_root = Path(output_root_arg).resolve() if output_root_arg else default_output_root(wordbank_dir)
  selections = load_image_selections(output_root, profile)
  # Audio is built from the candidates manifest in the output root (output/audio),
  # not the retired ./audio corpus. 'approved' uses the approved status; 'human'
  # additionally requires the human-reviewed flag.
  audio_approval = getattr(args, "audio_approval", "approved")
  audio_selections = load_audio_selections(output_root, audio_approval, wordbank_dir=wordbank_dir)
  audio_voice_sets = selected_voice_sets(load_audio_profile(wordbank_dir))

  _, flat = load_corpus(wordbank_dir, getattr(args, "corpus", None))
  if wordlist_path:
    keys, tags_by_key = parse_wordlist(wordlist_path)
  else:
    keys, tags_by_key = sorted(flat), {}

  missing_keys = [key for key in keys if key not in flat]
  if missing_keys:
    for key in missing_keys:
      print(f"ERROR: {key}: key is not in Wordbank corpus")
    return 2

  media_errors = validate_requested_media(wordbank_dir, keys, langs, selections, audio_selections)
  ready_keys = []
  dropped = set()
  if media_errors:
    for error in media_errors:
      dropped.add(error.split(":", 1)[0])
      if not args.quiet_drops:
        print(f"WARN: {error}")
  for key in keys:
    if key not in dropped:
      ready_keys.append(key)

  if args.fail_on_drop and dropped:
    return 3

  if args.format != "assets":
    raise SystemExit("Only --format assets is implemented in the current asset-bundle generator")

  if args.clean and out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  write_yaml(out_dir / "words.yml", selected_by_topic(ready_keys, flat, tags_by_key))
  write_yaml(out_dir / "items.yml", output_items_yml(ready_keys, flat))
  write_yaml(out_dir / "translations.yml", output_translations_yml(ready_keys, flat, langs))
  copy_media(wordbank_dir, out_dir, ready_keys, langs, args.audio_format, selections, audio_selections)

  manifest = {
    "source_wordlist": str(wordlist_path) if wordlist_path else "(whole corpus)",
    "source_wordbank": str(wordbank_dir),
    "format": "assets",
    "profile": profile,
    "audio_format": args.audio_format,
    "audio_approval": audio_approval,
    "audio_voice_sets": FlowMap({lang: audio_voice_sets.get(lang) for lang in langs}),
    "languages": langs,
    "requested": len(keys),
    "emitted": len(ready_keys),
    "dropped": sorted(dropped),
    "items": image_metadata(ready_keys, selections, profile),
  }
  write_yaml(out_dir / "manifest.yml", manifest)

  print(f"Generated {len(ready_keys)} words into {out_dir}")
  if dropped:
    print(f"Dropped {len(dropped)} incomplete or unapproved words")
  return 0


def main() -> None:
  parser = argparse.ArgumentParser(description="Slice Wordbank to one app word list")
  parser.add_argument("--wordbank", default=Path(__file__).resolve().parent, help="Wordbank directory")
  parser.add_argument("--corpus", help="Corpus name or path under wordlists/ (default: children-001)")
  parser.add_argument("--profile", help="Image profile to build from (default: juggernaut-3d, else first available)")
  parser.add_argument("--output-root", help="Output/staging root holding per-profile candidates (default: <wordbank>/output)")
  parser.add_argument("--wordlist", help="Optional subset word-list.yml; defaults to the whole corpus")
  parser.add_argument("--langs", required=True, help="Comma-separated language list, e.g. en,de,fr,es")
  parser.add_argument("--out", required=True, help="Output asset directory")
  parser.add_argument("--format", default="assets", choices=["assets"], help="Output format")
  parser.add_argument("--audio-format", default="wav", choices=["wav", "ogg"], help="Audio file format to write into the package")
  parser.add_argument("--audio-approval", default="approved", choices=["approved", "human"], help="Audio approval level: 'approved' (standard) or 'human' (only human-reviewed clips, from the audio candidates manifest)")
  parser.add_argument("--clean", action="store_true", help="Remove output directory before writing")
  parser.add_argument("--fail-on-drop", action="store_true", help="Exit non-zero if any requested word is dropped")
  parser.add_argument("--quiet-drops", action="store_true", help="Only print the drop summary, not every dropped item")
  args = parser.parse_args()
  raise SystemExit(generate_assets(args))


if __name__ == "__main__":
  main()
