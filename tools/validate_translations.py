#!/usr/bin/env python3
"""Validate and optionally repair corpus translations with deep-translator.

Examples:
  python tools/validate_translations.py wordlists/children-002.yml
  python tools/validate_translations.py wordlists/children-002.yml --topics adjective,verb
  python tools/validate_translations.py wordlists/children-002.yml --item-limit 10
  python tools/validate_translations.py wordlists/children-002.yml --apply
  python tools/validate_translations.py wordlists/children-002.yml --apply --kinds missing,empty,different --threshold 0.55

The check compares each stored translation with a fresh machine translation of
the English key. Low-similarity findings are useful review candidates, not proof
that the corpus is wrong; short words and synonyms can legitimately differ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGS = ["de", "fr", "es", "jp", "cn", "it", "ua"]
TRANSLATOR_LANGS = {
  "de": "de",
  "fr": "fr",
  "es": "es",
  "it": "it",
  "jp": "ja",
  "ja": "ja",
  "cn": "zh-CN",
  "zh-cn": "zh-CN",
  "ua": "uk",
  "uk": "uk",
}
ARTICLES = {
  "de": {"der", "die", "das", "den", "dem", "ein", "eine", "einen", "einem"},
  "fr": {"le", "la", "les", "l", "un", "une", "des"},
  "es": {"el", "la", "los", "las", "un", "una", "unos", "unas"},
  "it": {"il", "lo", "la", "i", "gli", "le", "l", "un", "una", "uno"},
}


@dataclass
class Finding:
  kind: str
  topic: str
  key: str
  lang: str
  current: str
  suggested: str
  score: float | None


@dataclass
class ValidationResult:
  findings: list[Finding]
  checked_items: int
  checked_translations: int


def parse_csv(raw: str | None, default: list[str]) -> list[str]:
  if not raw:
    return default
  return [part.strip() for part in raw.split(",") if part.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a YAML mapping/object")
  return data


def cache_key(source: str, lang: str) -> str:
  return f"{lang}\t{source}"


def load_cache(path: Path | None) -> dict[str, str]:
  if not path or not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a JSON object")
  return {str(k): str(v) for k, v in data.items()}


def save_cache(path: Path | None, cache: dict[str, str]) -> None:
  if not path:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")


def require_translator():
  try:
    from deep_translator import GoogleTranslator
  except ImportError as exc:
    raise SystemExit(
      "Missing dependency: deep-translator. Install it with: "
      "python -m pip install deep-translator"
    ) from exc
  return GoogleTranslator


def source_text(key: str, item: dict[str, Any]) -> str:
  text = key.replace("_", " ")
  pos = item.get("pos") or []
  if isinstance(pos, str):
    pos = [pos]
  pos = {str(value).lower() for value in pos if value is not None}
  if "v" in pos or "verb" in pos:
    return f"to {text}"
  return text


def normalize(text: str, lang: str) -> str:
  text = unicodedata.normalize("NFKC", text).casefold()
  text = text.replace("'", " ").replace("’", " ")
  text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
  words = [word for word in re.split(r"\s+", text.strip()) if word]
  articles = ARTICLES.get(lang.lower(), set())
  words = [word for word in words if word not in articles]
  return " ".join(words)


def similarity(current: str, suggested: str, lang: str) -> float:
  left = normalize(current, lang)
  right = normalize(suggested, lang)
  if not left and not right:
    return 1.0
  if not left or not right:
    return 0.0
  if left == right:
    return 1.0
  if left in right or right in left:
    return 0.9
  return SequenceMatcher(a=left, b=right).ratio()


def translate(
    source: str,
    lang: str,
    *,
    cache: dict[str, str],
    cache_path: Path | None,
    retries: int,
    delay: float,
) -> str:
  translator_lang = TRANSLATOR_LANGS.get(lang.lower())
  if not translator_lang:
    raise SystemExit(f"Unsupported language code {lang!r}; add it to TRANSLATOR_LANGS in this tool")
  key = cache_key(source, lang)
  if key in cache:
    return cache[key]
  GoogleTranslator = require_translator()
  translator = GoogleTranslator(source="en", target=translator_lang)
  for attempt in range(retries + 1):
    try:
      result = str(translator.translate(source)).strip()
      cache[key] = result
      save_cache(cache_path, cache)
      return result
    except Exception:
      if attempt >= retries:
        raise
      time.sleep(delay * (attempt + 1))
  raise AssertionError("unreachable")


def iter_entries(data: dict[str, Any], topics: set[str] | None, keys: set[str] | None):
  for topic, entries in data.items():
    if topics is not None and str(topic) not in topics:
      continue
    if not isinstance(entries, dict):
      raise SystemExit(f"topic {topic!r} must contain a mapping/object")
    for key, item in entries.items():
      key = str(key)
      if keys is not None and key not in keys:
        continue
      if not isinstance(item, dict):
        raise SystemExit(f"{topic}.{key} must be a mapping/object")
      yield str(topic), key, item


def validate(
    data: dict[str, Any],
    *,
    langs: list[str],
    topics: set[str] | None,
    keys: set[str] | None,
    threshold: float,
    cache: dict[str, str],
    cache_path: Path | None,
    retries: int,
    delay: float,
    limit: int | None,
    item_limit: int | None,
) -> ValidationResult:
  findings: list[Finding] = []
  checked = 0
  checked_items = 0
  for topic, key, item in iter_entries(data, topics, keys):
    if item_limit is not None and checked_items >= item_limit:
      return ValidationResult(findings, checked_items, checked)
    tr = item.setdefault("tr", {})
    if not isinstance(tr, dict):
      findings.append(Finding("missing", topic, key, "*", "", "", None))
      checked_items += 1
      continue
    source = source_text(key, item)
    for lang in langs:
      current = str(tr.get(lang) or "").strip()
      if not current:
        suggested = translate(
          source, lang, cache=cache, cache_path=cache_path, retries=retries, delay=delay
        )
        findings.append(Finding("empty" if lang in tr else "missing", topic, key, lang, current, suggested, None))
      else:
        suggested = translate(
          source, lang, cache=cache, cache_path=cache_path, retries=retries, delay=delay
        )
        score = similarity(current, suggested, lang)
        if score < threshold:
          findings.append(Finding("different", topic, key, lang, current, suggested, score))
      checked += 1
      if limit is not None and checked >= limit:
        checked_items += 1
        return ValidationResult(findings, checked_items, checked)
    checked_items += 1
  return ValidationResult(findings, checked_items, checked)


def apply_findings(data: dict[str, Any], findings: list[Finding], kinds: set[str]) -> int:
  changed = 0
  for finding in findings:
    if finding.kind not in kinds or finding.lang == "*":
      continue
    item = data[finding.topic][finding.key]
    tr = item.setdefault("tr", {})
    if tr.get(finding.lang) != finding.suggested:
      tr[finding.lang] = finding.suggested
      changed += 1
  return changed


def print_findings(findings: list[Finding]) -> None:
  for finding in findings:
    score = "" if finding.score is None else f" score={finding.score:.2f}"
    print(
      f"{finding.kind}: {finding.topic}.{finding.key} [{finding.lang}]{score}\n"
      f"  current:   {finding.current or '<missing>'}\n"
      f"  suggested: {finding.suggested or '<none>'}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description="Validate corpus translations with deep-translator")
  parser.add_argument("wordlist", type=Path, help="Corpus YAML file under wordlists/")
  parser.add_argument("--langs", default=",".join(DEFAULT_LANGS), help="Comma-separated corpus language codes")
  parser.add_argument("--topics", help="Comma-separated topics to check, e.g. adjective,verb")
  parser.add_argument("--keys", help="Comma-separated keys to check")
  parser.add_argument("--threshold", type=float, default=0.62, help="Similarity threshold for 'different' findings")
  parser.add_argument("--item-limit", type=int, help="Maximum corpus items/words to check, useful for test runs")
  parser.add_argument("--limit", type=int, help="Maximum individual language translation checks to run")
  parser.add_argument("--cache", type=Path, default=REPO_ROOT / ".translation-cache.json", help="JSON cache path")
  parser.add_argument("--no-cache", action="store_true", help="Do not read or write a translation cache")
  parser.add_argument("--retries", type=int, default=2, help="Retries per translation request")
  parser.add_argument("--delay", type=float, default=1.0, help="Base retry delay in seconds")
  parser.add_argument("--apply", action="store_true", help="Rewrite the YAML with suggested translations")
  parser.add_argument(
    "--kinds",
    default="missing,empty",
    help="Finding kinds to apply when --apply is set. Include 'different' to rewrite low-similarity translations.",
  )
  args = parser.parse_args()

  data = load_yaml(args.wordlist)
  langs = parse_csv(args.langs, DEFAULT_LANGS)
  topics = set(parse_csv(args.topics, [])) if args.topics else None
  keys = set(parse_csv(args.keys, [])) if args.keys else None
  cache_path = None if args.no_cache else args.cache
  cache = {} if args.no_cache else load_cache(cache_path)

  result = validate(
    data,
    langs=langs,
    topics=topics,
    keys=keys,
    threshold=args.threshold,
    cache=cache,
    cache_path=cache_path,
    retries=args.retries,
    delay=args.delay,
    limit=args.limit,
    item_limit=args.item_limit,
  )
  findings = result.findings
  print_findings(findings)
  print(
    f"Checked {result.checked_items} item(s), "
    f"{result.checked_translations} translation(s); findings: {len(findings)}"
  )

  if args.apply:
    kinds = set(parse_csv(args.kinds, []))
    changed = apply_findings(data, findings, kinds)
    if changed:
      with args.wordlist.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=1000)
      print(f"Applied {changed} translation correction(s) to {args.wordlist}")
    else:
      print("No matching findings to apply")

  raise SystemExit(1 if findings and not args.apply else 0)


if __name__ == "__main__":
  main()
