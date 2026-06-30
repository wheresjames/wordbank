#!/usr/bin/env python3
"""Shared helpers for Wordbank generation and validation tools."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

LANGUAGE_ALIASES = {
  "jp": "ja",
  "ja": "ja",
  "cn": "zh-cn",
  "zh": "zh-cn",
  "zh-cn": "zh-cn",
  "ua": "uk",
  "uk": "uk",
}

ARTICLE_LANGS = {"de", "fr", "es", "it"}
AUDIO_EXTENSIONS = ("wav", "ogg")


class FlowMap(dict):
  pass


class FlowList(list):
  pass


class FlowDumper(yaml.SafeDumper):
  pass


def _flow_map_representer(dumper, value):
  return dumper.represent_mapping("tag:yaml.org,2002:map", value.items(), flow_style=True)


def _flow_list_representer(dumper, value):
  return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


FlowDumper.add_representer(FlowMap, _flow_map_representer)
FlowDumper.add_representer(FlowList, _flow_list_representer)


def slugify(text: str) -> str:
  return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip().lower()).strip("_")


def normalize_lang(lang: str) -> str:
  key = str(lang or "").strip().lower()
  return LANGUAGE_ALIASES.get(key, key)


def load_yaml(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    yaml.dump(data, f, Dumper=FlowDumper, sort_keys=False, allow_unicode=True, width=1000)


WORDLISTS_DIRNAME = "wordlists"
DEFAULT_CORPUS_NAME = "children-001"
PROFILES_DIRNAME = "profiles"
OUTPUT_DIRNAME = "output"
DEFAULT_PROFILE_NAME = "juggernaut-3d"
AUDIO_CONFIG_RELATIVE = Path("tools") / "itemaudio" / "config.yml"


def audio_config_path(wordbank_dir: Path) -> Path:
  return wordbank_dir / AUDIO_CONFIG_RELATIVE


def load_audio_profile(wordbank_dir: Path) -> dict[str, Any]:
  path = audio_config_path(wordbank_dir)
  data = load_yaml(path) if path.exists() else {}
  if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a mapping/object")
  languages = data.get("languages")
  if not isinstance(languages, dict):
    raise SystemExit(f"{path} must define languages")
  return data


def selected_voice_sets(profile: dict[str, Any]) -> dict[str, str]:
  out: dict[str, str] = {}
  languages = profile.get("languages") or {}
  if not isinstance(languages, dict):
    return out
  for raw_lang, spec in languages.items():
    if not isinstance(spec, dict):
      continue
    selected = str(spec.get("selected") or "").strip()
    if selected:
      out[normalize_lang(str(raw_lang))] = selected
  return out


def resolve_voice_set(profile: dict[str, Any], language: str, voice_set: str | None = None) -> tuple[str, dict[str, Any]]:
  lang = normalize_lang(language)
  languages = profile.get("languages") or {}
  lang_spec = languages.get(lang) if isinstance(languages, dict) else None
  if not isinstance(lang_spec, dict):
    raise SystemExit(f"No audio voice sets configured for language {lang!r}")
  voice_sets = lang_spec.get("voice_sets") or {}
  if not isinstance(voice_sets, dict):
    raise SystemExit(f"Audio language {lang!r} must define voice_sets")
  name = str(voice_set or lang_spec.get("selected") or "").strip()
  if not name:
    raise SystemExit(f"No selected voice set configured for language {lang!r}")
  route = voice_sets.get(name)
  if not isinstance(route, dict):
    raise SystemExit(f"Voice set {name!r} is not configured for language {lang!r}")
  engine = str(route.get("engine") or "").strip()
  if not engine:
    raise SystemExit(f"Voice set {lang}/{name} must define an engine")
  return name, dict(route)


def audio_voice_set_dir(output_root: Path, language: str, voice_set: str) -> Path:
  return output_root / "audio" / normalize_lang(language) / voice_set


def list_corpora(wordbank_dir: Path) -> list[Path]:
  """All corpus files under <wordbank>/wordlists, sorted by name."""
  directory = wordbank_dir / WORDLISTS_DIRNAME
  if not directory.is_dir():
    return []
  files = list(directory.glob("*.yml")) + list(directory.glob("*.yaml"))
  return sorted(files, key=lambda p: p.name)


def resolve_corpus_path(wordbank_dir: Path, corpus: str | Path | None = None) -> Path:
  """Resolve a corpus selector to a file path.

  `corpus` may be an absolute/relative path, a bare name like "children-001"
  (with or without extension) found under <wordbank>/wordlists, or None to use
  the default (children-001, else the first wordlist).
  """
  directory = wordbank_dir / WORDLISTS_DIRNAME
  if corpus:
    candidate = Path(corpus)
    if candidate.is_absolute():
      return candidate
    if (directory / candidate).exists():
      return directory / candidate
    for ext in (".yml", ".yaml"):
      named = directory / f"{corpus}{ext}"
      if named.exists():
        return named
    return (wordbank_dir / candidate).resolve()
  default = directory / f"{DEFAULT_CORPUS_NAME}.yml"
  if default.exists():
    return default
  corpora = list_corpora(wordbank_dir)
  if corpora:
    return corpora[0]
  raise SystemExit(f"no corpus found under {directory}")


def load_corpus(
    wordbank_dir: Path,
    corpus: str | Path | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
  words_path = resolve_corpus_path(wordbank_dir, corpus)
  data = load_yaml(words_path)
  if not isinstance(data, dict):
    raise SystemExit(f"{words_path} must contain a mapping/object")

  flat: dict[str, dict[str, Any]] = {}
  for topic, entries in data.items():
    if not isinstance(entries, dict):
      raise SystemExit(f"{words_path} topic {topic!r} must contain a mapping/object")
    for key, item in entries.items():
      if key in flat:
        raise SystemExit(f"Duplicate key in {words_path}: {key}")
      if not isinstance(item, dict):
        raise SystemExit(f"{words_path} item {topic}.{key} must be a mapping/object")
      tr = item.get("tr")
      if not isinstance(tr, dict):
        raise SystemExit(f"{words_path} item {topic}.{key} missing tr mapping")
      copied = copy.deepcopy(item)
      copied["topic"] = str(topic)
      flat[str(key)] = copied
  return data, flat


# ── Profiles & per-profile image selections ──────────────────────────────────

def default_output_root(wordbank_dir: Path) -> Path:
  """The disposable output/staging root (gitignored)."""
  return wordbank_dir / OUTPUT_DIRNAME


def list_profiles(wordbank_dir: Path) -> list[str]:
  """All image profiles (directories holding a profile.yml), sorted by name."""
  directory = wordbank_dir / PROFILES_DIRNAME
  if not directory.is_dir():
    return []
  return sorted(p.name for p in directory.iterdir() if (p / "profile.yml").exists())


def resolve_profile(wordbank_dir: Path, profile: str | None = None) -> str:
  """Resolve a profile selector to a concrete profile name.

  Falls back to the canonical default profile, then the first available one.
  """
  names = list_profiles(wordbank_dir)
  if profile and profile in names:
    return profile
  if profile:
    raise SystemExit(f"profile {profile!r} not found under {wordbank_dir / PROFILES_DIRNAME}")
  if DEFAULT_PROFILE_NAME in names:
    return DEFAULT_PROFILE_NAME
  if names:
    return names[0]
  raise SystemExit(f"no profiles found under {wordbank_dir / PROFILES_DIRNAME}")


def profile_candidates_dir(output_root: Path, profile: str) -> Path:
  return output_root / "images" / profile / "candidates"


def _manifest_entries(data: Any) -> list[dict[str, Any]]:
  if isinstance(data, list):
    return [e for e in data if isinstance(e, dict)]
  if isinstance(data, dict) and isinstance(data.get("entries"), list):
    return [e for e in data["entries"] if isinstance(e, dict)]
  return []


def load_image_selections(output_root: Path, profile: str) -> dict[str, dict[str, Any]]:
  """Map word key -> the selected candidate for a profile.

  Selection (recorded in the profile's candidates/manifest.json) is the source of
  truth for images: each returned entry is the original manifest entry plus a
  resolved absolute ``path`` to the image file.
  """
  candidates = profile_candidates_dir(output_root, profile)
  manifest_path = candidates / "manifest.json"
  if not manifest_path.exists():
    return {}
  try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {}
  selections: dict[str, dict[str, Any]] = {}
  for entry in _manifest_entries(data):
    if not entry.get("selected"):
      continue
    key = slugify(entry.get("item") or entry.get("key") or "")
    filename = str(entry.get("filename") or "").strip()
    if not key or not filename:
      continue
    selections[key] = {**entry, "path": candidates / filename}
  return selections


def _approved_audio_from_manifest(
    manifest_path: Path,
    audio_dir: Path,
    level: str,
) -> dict[tuple[str, str], Path]:
  try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {}
  out: dict[tuple[str, str], Path] = {}
  for entry in _manifest_entries(data):
    if not (entry.get("status") == "approved" or entry.get("approved")):
      continue
    if level == "human" and not entry.get("human_approved"):
      continue
    key = slugify(entry.get("key") or entry.get("item") or "")
    lang = normalize_lang(str(entry.get("language") or entry.get("lang") or ""))
    filename = str(entry.get("filename") or "").strip()
    if not key or not lang or not filename:
      continue
    out[(lang, key)] = audio_dir / filename
  return out


def load_audio_selections(
    output_root: Path,
    level: str = "human",
    *,
    wordbank_dir: Path | None = None,
) -> dict[tuple[str, str], Path]:
  """Map (lang, key) -> approved clip path from selected voice-set manifests.

  Phase 1 audio output is isolated under
  ``output/audio/<language>/<voice_set>/manifest.json``. The selected voice set
  per language comes from ``tools/itemaudio/config.yml``. ``level='human'``
  requires ``human_approved``; any other value uses the plain approved status.
  """
  if wordbank_dir is not None:
    try:
      profile = load_audio_profile(wordbank_dir)
    except SystemExit:
      profile = {}
    selections: dict[tuple[str, str], Path] = {}
    for lang, voice_set in selected_voice_sets(profile).items():
      voice_dir = audio_voice_set_dir(output_root, lang, voice_set)
      manifest_path = voice_dir / "manifest.json"
      if manifest_path.exists():
        selections.update(_approved_audio_from_manifest(manifest_path, voice_dir, level))
    return selections

  # Fallback for callers that have not yet been passed the Wordbank root.
  manifest_path = output_root / "audio" / "manifest.json"
  if not manifest_path.exists():
    return {}
  return _approved_audio_from_manifest(manifest_path, output_root / "audio", level)


def parse_wordlist(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
  data = load_yaml(path)
  keys: list[str] = []
  tags: dict[str, dict[str, Any]] = {}

  def add_key(raw_key: Any, raw_tags: Any = None) -> None:
    key = slugify(str(raw_key))
    if not key:
      raise SystemExit(f"Invalid empty word key in {path}")
    if key not in tags:
      keys.append(key)
    if raw_tags is None:
      tags.setdefault(key, {})
    elif isinstance(raw_tags, dict):
      tags[key] = copy.deepcopy(raw_tags)
    else:
      raise SystemExit(f"Tags for {key!r} in {path} must be a mapping/object")

  if isinstance(data, list):
    for entry in data:
      if isinstance(entry, str):
        add_key(entry)
      elif isinstance(entry, dict):
        if len(entry) != 1:
          raise SystemExit(f"List mapping entries in {path} must contain exactly one key")
        raw_key, raw_tags = next(iter(entry.items()))
        add_key(raw_key, raw_tags)
      else:
        raise SystemExit(f"Word list entries in {path} must be strings or single-key mappings")
  elif isinstance(data, dict):
    for raw_key, raw_tags in data.items():
      add_key(raw_key, raw_tags or {})
  else:
    raise SystemExit(f"{path} must contain either a YAML list or mapping/object")

  return keys, tags


def compose_text(key: str, item: dict[str, Any], lang: str) -> str | None:
  lang = normalize_lang(lang)
  if lang == "en":
    return key.replace("_", " ")

  # The corpus stores translations under raw language codes (e.g. jp, cn) while
  # callers pass normalized codes (ja, zh-cn); normalize the keys so lookups for
  # aliased languages succeed.
  tr = {normalize_lang(k): v for k, v in (item.get("tr") or {}).items()}
  art = {normalize_lang(k): v for k, v in (item.get("art") or {}).items()}
  text = tr.get(lang)
  if text is None:
    return None
  text = str(text).strip()
  article = art.get(lang)
  if not article:
    return text
  article = str(article).strip()
  if article.endswith("'") or article.endswith("\u2019"):
    return f"{article}{text}"
  return f"{article} {text}"


def merged_item(key: str, item: dict[str, Any], tags: dict[str, Any]) -> dict[str, Any]:
  out = copy.deepcopy(item)
  topic = out.pop("topic", None)
  if topic is not None:
    out["topic"] = topic
  for tag_key, tag_value in tags.items():
    if tag_key in {"tr", "art", "pos", "topic"}:
      raise SystemExit(f"App tag {tag_key!r} for {key!r} conflicts with Wordbank field")
    out[tag_key] = copy.deepcopy(tag_value)
  return out


def validate_requested_media(
    wordbank_dir: Path,
    keys: list[str],
    langs: list[str],
    selections: dict[str, dict[str, Any]] | None = None,
    audio_selections: dict[tuple[str, str], Path] | None = None,
) -> list[str]:
  """Check each requested word has a selected image (per profile) and approved audio.

  ``selections`` is the per-profile image selection map (``load_image_selections``)
  and ``audio_selections`` the (lang, key) -> clip-path map for the requested
  approval level (``load_audio_selections``). Both image and audio are gated on the
  ``output/`` review state — there is no tracked-corpus fallback.
  """
  selections = selections or {}
  audio_selections = audio_selections or {}
  errors: list[str] = []
  for key in keys:
    selected = selections.get(key)
    if not selected or not Path(selected.get("path", "")).exists():
      errors.append(f"{key}: no selected image in profile")
    for lang in langs:
      clip = audio_selections.get((lang, key))
      if not clip or not Path(clip).exists():
        errors.append(f"{key}: no approved audio for {lang} in output")
  return errors


# ── Round-trip: import a built package back into a profile ────────────────────

def _read_package_image_metadata(package_dir: Path) -> dict[str, dict[str, Any]]:
  """Per-key image metadata from a built package manifest, keyed by word key.

  Looks for the ``items`` section written by generate_db.py (filename, prompt,
  seed, profile). Returns {} for packages built before per-image metadata.
  """
  manifest_path = package_dir / "manifest.yml"
  if not manifest_path.exists():
    return {}
  data = load_yaml(manifest_path)
  items = data.get("items") if isinstance(data, dict) else None
  if not isinstance(items, dict):
    return {}
  meta: dict[str, dict[str, Any]] = {}
  for raw_key, info in items.items():
    if isinstance(info, dict):
      meta[slugify(raw_key)] = dict(info)
  return meta


def _package_audio_level(package_dir: Path) -> str:
  """Whether the package's audio was a standard or human-approved build."""
  manifest_path = package_dir / "manifest.yml"
  if not manifest_path.exists():
    return "approved"
  data = load_yaml(manifest_path)
  level = data.get("audio_approval") if isinstance(data, dict) else None
  return "human" if level == "human" else "approved"


def _read_package_translations(package_dir: Path) -> dict[tuple[str, str], str]:
  """(lang, key) -> display text from a package's translations.yml (best effort)."""
  path = package_dir / "translations.yml"
  if not path.exists():
    return {}
  data = load_yaml(path)
  categories = data.get("categories") if isinstance(data, dict) else None
  if not isinstance(categories, dict):
    return {}
  out: dict[tuple[str, str], str] = {}
  for _topic, keys in categories.items():
    if not isinstance(keys, dict):
      continue
    for raw_key, langs in keys.items():
      if not isinstance(langs, dict):
        continue
      key = slugify(raw_key)
      for lang, text in langs.items():
        out[(normalize_lang(lang), key)] = str(text)
  return out


def import_package_audio(
    package_dir: Path,
    output_root: Path,
    *,
    human: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
  """Restore a built package's audio into ``output/audio`` as approved candidates.

  Each ``audio/<lang>/<key>.<ext>`` clip is copied to
  ``output/audio/<lang>/<key>_001.<ext>`` and recorded in
  ``output/audio/manifest.json`` as approved (and human-approved when ``human``),
  so a fresh checkout + archived package becomes a rebuildable working state.
  """
  package_dir = Path(package_dir)
  audio_src_root = package_dir / "audio"
  if not audio_src_root.is_dir():
    return {"audio_imported": 0, "audio_dir": str(output_root / "audio")}

  text_map = _read_package_translations(package_dir)
  audio_dir = output_root / "audio"
  manifest_path = audio_dir / "manifest.json"
  existing = _manifest_entries(json.loads(manifest_path.read_text(encoding="utf-8"))) \
    if manifest_path.exists() else []

  imported_pairs: set[tuple[str, str]] = set()
  new_entries: list[dict[str, Any]] = []
  imported: list[str] = []
  for lang_dir in sorted(p for p in audio_src_root.iterdir() if p.is_dir()):
    lang = normalize_lang(lang_dir.name)
    for src in sorted(lang_dir.iterdir()):
      ext = src.suffix.lower()
      if ext.lstrip(".") not in AUDIO_EXTENSIONS:
        continue
      key = slugify(src.stem)
      if not key or (lang, key) in imported_pairs:
        continue
      rel = f"{lang}/{key}_001{ext}"
      dst = audio_dir / rel
      if dst.exists() and not overwrite:
        continue
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, dst)
      imported_pairs.add((lang, key))
      imported.append(rel)
      entry: dict[str, Any] = {
        "key": key,
        "language": lang,
        "filename": rel,
        "take": 1,
        "status": "approved",
        "approved": True,
        "imported": True,
      }
      if human:
        entry["human_approved"] = True
      text = text_map.get((lang, key)) or (key.replace("_", " ") if lang == "en" else "")
      if text:
        entry["text"] = text
      new_entries.append(entry)

  def _pair(e: dict[str, Any]) -> tuple[str, str]:
    return (
      normalize_lang(str(e.get("language") or e.get("lang") or "")),
      slugify(e.get("key") or e.get("item") or ""),
    )

  kept = [e for e in existing if _pair(e) not in imported_pairs]
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.write_text(
    json.dumps(kept + new_entries, indent=2, ensure_ascii=False),
    encoding="utf-8",
  )
  return {"audio_imported": len(imported), "audio_dir": str(audio_dir), "human": human}


def import_package(
    package_dir: Path,
    output_root: Path,
    profile: str,
    *,
    prompt_template: str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
  """Populate a profile's candidates from a built package (or flat <key>.png folder).

  Each imported image is written into ``output/images/<profile>/candidates`` and
  marked as the *selected* candidate for its word, so a fresh checkout (profile
  recipe in git + archived package) becomes a fully reproducible working state.

  Only the selected image per word is recovered (packages carry no rejected
  candidates); existing entries for an imported key are replaced.
  """
  package_dir = Path(package_dir)
  images_dir = package_dir / "images"
  if not images_dir.is_dir():
    # Fall back: the package_dir itself is a flat folder of <key>.png files.
    images_dir = package_dir
  pngs = sorted(images_dir.glob("*.png"))
  if not pngs:
    raise SystemExit(f"no images found to import under {images_dir}")

  metadata = _read_package_image_metadata(package_dir)
  candidates = profile_candidates_dir(output_root, profile)
  candidates.mkdir(parents=True, exist_ok=True)
  manifest_path = candidates / "manifest.json"

  existing = _manifest_entries(json.loads(manifest_path.read_text(encoding="utf-8"))) \
    if manifest_path.exists() else []

  imported_keys: set[str] = set()
  new_entries: list[dict[str, Any]] = []
  imported: list[str] = []
  for src in pngs:
    key = slugify(src.stem)
    if not key or key in imported_keys:
      continue
    dst = candidates / f"{key}.png"
    if dst.exists() and not overwrite:
      continue
    shutil.copy2(src, dst)
    imported_keys.add(key)
    imported.append(key)
    info = metadata.get(key, {})
    entry: dict[str, Any] = {
      "item": key,
      "filename": dst.name,
      "selected": True,
      "imported": True,
    }
    prompt = info.get("prompt") or prompt_template
    if prompt:
      entry["prompt"] = prompt
    if info.get("seed") is not None:
      entry["seed"] = info.get("seed")
    if info.get("profile"):
      entry["source_profile"] = info.get("profile")
    new_entries.append(entry)

  # Drop any prior entries for the keys we just imported, then append the fresh ones.
  kept = [e for e in existing if slugify(e.get("item") or e.get("key") or "") not in imported_keys]
  manifest_path.write_text(
    json.dumps(kept + new_entries, indent=2, ensure_ascii=False),
    encoding="utf-8",
  )

  # Also restore the package's audio into output/audio (profile-independent), at
  # the approval level the package was built with, so the round-trip recovers
  # both images and audio.
  audio_result = import_package_audio(
    package_dir,
    output_root,
    human=_package_audio_level(package_dir) == "human",
    overwrite=overwrite,
  )
  return {
    "imported": imported,
    "count": len(imported),
    "candidates": str(candidates),
    **audio_result,
  }
