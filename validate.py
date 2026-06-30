#!/usr/bin/env python3
"""Validate Wordbank schema, word lists, and requested media readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from wordbank_common import (
  default_output_root,
  load_audio_selections,
  load_corpus,
  load_image_selections,
  normalize_lang,
  parse_wordlist,
  resolve_profile,
  validate_requested_media,
)


def main() -> None:
  parser = argparse.ArgumentParser(description="Validate Wordbank corpus and media")
  parser.add_argument("--wordbank", default=Path(__file__).resolve().parent, help="Wordbank directory")
  parser.add_argument("--corpus", help="Corpus name or path under wordlists/ (default: children-001)")
  parser.add_argument("--profile", help="Image profile for selection readiness (default: juggernaut-3d, else first)")
  parser.add_argument("--output-root", help="Output/staging root holding per-profile candidates (default: <wordbank>/output)")
  parser.add_argument("--wordlist", help="Optional app wordlist.yml to validate")
  parser.add_argument("--langs", help="Optional comma-separated language list for media validation")
  args = parser.parse_args()

  wordbank_dir = Path(args.wordbank).resolve()
  _, flat = load_corpus(wordbank_dir, args.corpus)
  print(f"OK: corpus schema valid ({len(flat)} keys)")

  keys = sorted(flat)
  if args.wordlist:
    wordlist_path = Path(args.wordlist).resolve()
    keys, _ = parse_wordlist(wordlist_path)
    missing = [key for key in keys if key not in flat]
    if missing:
      for key in missing:
        print(f"ERROR: {key}: key is not in Wordbank corpus")
      raise SystemExit(2)
    print(f"OK: {wordlist_path} references {len(keys)} known keys")

  if args.langs:
    langs = [normalize_lang(lang) for lang in args.langs.split(",") if lang.strip()]
    profile = resolve_profile(wordbank_dir, args.profile)
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(wordbank_dir)
    selections = load_image_selections(output_root, profile)
    audio_selections = load_audio_selections(output_root, "approved", wordbank_dir=wordbank_dir)
    errors = validate_requested_media(wordbank_dir, keys, langs, selections, audio_selections)
    if errors:
      for error in errors:
        print(f"ERROR: {error}")
      raise SystemExit(3)
    print(f"OK: approved media present for {len(keys)} keys across {', '.join(langs)}")


if __name__ == "__main__":
  main()
