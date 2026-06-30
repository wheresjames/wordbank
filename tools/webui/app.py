#!/usr/bin/env python3
"""Local Wordbank web UI for review, extraction, validation, and packaging."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import yaml

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from wordbank_common import (  # noqa: E402
  _manifest_entries,
  compose_text,
  import_package,
  list_corpora,
  load_audio_profile,
  load_audio_selections,
  load_corpus,
  load_image_selections,
  normalize_lang,
  parse_wordlist,
  resolve_corpus_path,
  resolve_voice_set,
  slugify,
  validate_requested_media,
)

OUTPUT_ROOT_DEFAULT = ROOT / "output"
PROFILES_DIR = ROOT / "profiles"
DEFAULT_PROFILE = "juggernaut-3d"
DEFAULT_IMAGE_OUTPUT = OUTPUT_ROOT_DEFAULT / "images" / DEFAULT_PROFILE / "candidates"
DEFAULT_AUDIO_OUTPUT = OUTPUT_ROOT_DEFAULT / "audio"
# Extract is a throwaway convenience export (not part of the build workflow), so it
# defaults to a scratch /tmp folder rather than any in-repo directory.
DEFAULT_IMAGE_EXTRACT = Path("/tmp/wordbank-images")
DEFAULT_AUDIO_EXTRACT = Path("/tmp/wordbank-audio")
DEFAULT_IMAGE_WORKFLOW = ROOT / "tools" / "itemimages" / "workflow_api.json"
DEFAULT_COMFYUI_DIR = ROOT / "tools" / "itemimages" / "ComfyUI"
IMAGE_TOOLS_DIR = ROOT / "tools" / "itemimages"
AUDIO_TOOLS_DIR = ROOT / "tools" / "itemaudio"
SETUP_SCRIPT = IMAGE_TOOLS_DIR / "setup.py"
CIVITAI_KEY_FILE = IMAGE_TOOLS_DIR / ".civitai_key"
COQUI_SCRIPT = AUDIO_TOOLS_DIR / "coqui_local_tts.py"
COQUI_LOCAL_DIR = AUDIO_TOOLS_DIR / "coqui_local"
AUDIO_CONFIG = AUDIO_TOOLS_DIR / "config.yml"
# MeloTTS language codes map to these Hugging Face repos (see melo/download_utils.py).
MELOTTS_LANG_TO_HF_REPO = {
  "EN": "myshell-ai/MeloTTS-English",
  "EN_V2": "myshell-ai/MeloTTS-English-v2",
  "EN_NEWEST": "myshell-ai/MeloTTS-English-v3",
  "FR": "myshell-ai/MeloTTS-French",
  "JP": "myshell-ai/MeloTTS-Japanese",
  "ES": "myshell-ai/MeloTTS-Spanish",
  "ZH": "myshell-ai/MeloTTS-Chinese",
  "KR": "myshell-ai/MeloTTS-Korean",
}
# Pipeline Python packages: (pip name, importable module name).
PIPELINE_PACKAGES = [
  ("flask", "flask"),
  ("pyyaml", "yaml"),
  ("pillow", "PIL"),
  ("websocket-client", "websocket"),
  ("rembg", "rembg"),
  ("onnxruntime", "onnxruntime"),
]
STATE_PATH = Path(__file__).resolve().parent / "state.json"

app = Flask(__name__, static_folder="static", static_url_path="/static")


def _json_error(message: str, status: int = 400, **extra: Any):
  payload = {"ok": False, "error": message}
  payload.update(extra)
  return jsonify(payload), status


def _resolve_path(raw: str | None, default: Path | None = None) -> Path:
  if not raw:
    if default is None:
      raise ValueError("path is required")
    return default.resolve()
  path = Path(raw).expanduser()
  if not path.is_absolute():
    path = ROOT / path
  return path.resolve()


def _langs(raw: str | None) -> list[str]:
  if not raw:
    return []
  return [normalize_lang(lang) for lang in raw.split(",") if lang.strip()]


def _split_csv(raw: str | None) -> list[str]:
  if not raw:
    return []
  return [part.strip() for part in raw.split(",") if part.strip()]


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _write_json(path: Path, data: Any) -> None:
  # Write to a temp file then atomically replace, so an interrupt mid-write can't
  # leave a truncated/corrupt manifest (which would break the profile's review).
  tmp = path.with_name(path.name + ".tmp")
  with tmp.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
  os.replace(tmp, path)


def _shell_join(command: list[str]) -> str:
  return subprocess.list2cmdline(command)


def load_state() -> dict[str, Any]:
  if not STATE_PATH.exists():
    return {}
  data = _load_json(STATE_PATH)
  return data if isinstance(data, dict) else {}


def save_state(data: dict[str, Any]) -> dict[str, Any]:
  allowed = {
    "corpus",
    "wordlist",
    "langs",
    "output_root",
    "active_profile",
    "image_extract",
    "audio_extract",
    "image_workflow",
    "image_host",
    "image_comfyui",
    "package_out",
    "package_audio_format",
    "package_audio_approval",
    "recent",
  }
  clean = {key: value for key, value in data.items() if key in allowed}
  STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
  _write_json(STATE_PATH, clean)
  return clean


# ── Profiles & output root ───────────────────────────────────────────────────

def _profile_slug(name: str) -> str:
  slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
  return slug or "profile"


def get_output_root(state: dict[str, Any] | None = None) -> Path:
  state = load_state() if state is None else state
  raw = state.get("output_root")
  if raw:
    return Path(raw).expanduser().resolve()
  return OUTPUT_ROOT_DEFAULT.resolve()


def list_profile_names() -> list[str]:
  if not PROFILES_DIR.is_dir():
    return []
  return sorted(
    p.name for p in PROFILES_DIR.iterdir()
    if (p / "profile.yml").exists()
  )


def get_active_profile(state: dict[str, Any] | None = None) -> str:
  state = load_state() if state is None else state
  names = list_profile_names()
  name = state.get("active_profile")
  if name in names:
    return name
  if DEFAULT_PROFILE in names:
    return DEFAULT_PROFILE
  return names[0] if names else DEFAULT_PROFILE


def image_candidates_dir(profile: str | None = None, root: Path | None = None) -> Path:
  root = root or get_output_root()
  profile = profile or get_active_profile()
  return root / "images" / profile / "candidates"


def audio_output_dir(root: Path | None = None) -> Path:
  return (root or get_output_root()) / "audio"


def audio_approvals(root: Path | None = None) -> dict[tuple[str, str], Path]:
  """(lang, key) -> approved clip path from output/audio — the audio readiness and
  serving source (replaces the retired ./audio corpus)."""
  return load_audio_selections(root or get_output_root(), "approved", wordbank_dir=ROOT)


def load_audio_profile_for_ui() -> dict[str, Any]:
  try:
    return load_audio_profile(ROOT)
  except SystemExit:
    return {"schema_version": 1, "name": "default", "type": "audio", "languages": {}, "downloads": []}


def save_audio_profile_for_ui(profile: dict[str, Any]) -> None:
  AUDIO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
  with AUDIO_CONFIG.open("w", encoding="utf-8") as f:
    yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True, width=1000)


def audio_profile_summary(root: Path | None = None) -> dict[str, Any]:
  root = root or audio_output_dir()
  profile = load_audio_profile_for_ui()
  languages = profile.get("languages") if isinstance(profile.get("languages"), dict) else {}
  out_langs: dict[str, Any] = {}
  for raw_lang, lang_spec in languages.items():
    if not isinstance(lang_spec, dict):
      continue
    lang = normalize_lang(raw_lang)
    selected = str(lang_spec.get("selected") or "").strip()
    voice_sets = lang_spec.get("voice_sets") if isinstance(lang_spec.get("voice_sets"), dict) else {}
    out_sets: dict[str, Any] = {}
    for name, route in voice_sets.items():
      if not isinstance(route, dict):
        continue
      voice_dir = root / lang / name
      manifest_path = voice_dir / "manifest.json"
      count = 0
      approved = 0
      human = 0
      if manifest_path.exists():
        try:
          entries = _manifest_entries(_load_json(manifest_path))
          count = len(entries)
          approved = sum(1 for e in entries if e.get("status") == "approved" or e.get("approved"))
          human = sum(1 for e in entries if e.get("human_approved"))
        except Exception:
          pass
      out_sets[name] = {
        "name": name,
        "selected": name == selected,
        "engine": route.get("engine"),
        "model": route.get("model"),
        "voice": route.get("voice"),
        "speaker": route.get("speaker"),
        "speaker_wav": route.get("speaker_wav"),
        "output": str(voice_dir),
        "manifest": str(manifest_path),
        "entries": count,
        "approved": approved,
        "human": human,
      }
    out_langs[lang] = {"selected": selected, "voice_sets": out_sets}
  return {
    "path": str(AUDIO_CONFIG),
    "output_root": str(root.resolve()),
    "profile": profile,
    "languages": out_langs,
    "downloads": profile.get("downloads") or [],
  }


def audio_request_dir(data: dict[str, Any] | None = None, *, args=None) -> Path:
  data = data or {}
  root = _resolve_path((args.get("output") if args else data.get("output")), DEFAULT_AUDIO_OUTPUT)
  language = str((args.get("language") if args else data.get("language")) or "").strip()
  voice_set = str((args.get("voice_set") if args else data.get("voice_set")) or "").strip()
  if language and voice_set:
    return root / normalize_lang(language) / voice_set
  return root


def load_profile(name: str) -> dict[str, Any]:
  path = PROFILES_DIR / name / "profile.yml"
  if not path.exists():
    raise FileNotFoundError(f"profile {name!r} not found")
  data = yaml.safe_load(path.read_text(encoding="utf-8"))
  return data if isinstance(data, dict) else {}


def validate_profile(data: dict[str, Any]) -> None:
  if not isinstance(data, dict):
    raise ValueError("profile must be a mapping")
  if not str(data.get("name") or "").strip():
    raise ValueError("profile name is required")
  model = data.get("model")
  if not isinstance(model, dict) or not str(model.get("checkpoint") or "").strip():
    raise ValueError("model.checkpoint is required")
  prompt = data.get("prompt")
  template = str((prompt or {}).get("template") or "").strip() if isinstance(prompt, dict) else ""
  if not template:
    raise ValueError("prompt.template is required")
  if "{item}" not in template:
    raise ValueError("prompt.template must contain the {item} placeholder")


def save_profile(name: str, data: dict[str, Any]) -> str:
  slug = _profile_slug(name)
  data = dict(data)
  data["name"] = slug
  data.setdefault("schema_version", 1)
  data.setdefault("type", "images")
  validate_profile(data)
  directory = PROFILES_DIR / slug
  directory.mkdir(parents=True, exist_ok=True)
  with (directory / "profile.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=10000)
  return slug


def _yaml_block(value: Any) -> str:
  if not value:
    return ""
  return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=10000).strip()


def profiles_summary(state: dict[str, Any] | None = None) -> dict[str, Any]:
  state = load_state() if state is None else state
  active = get_active_profile(state)
  items = []
  for name in list_profile_names():
    try:
      p = load_profile(name)
    except Exception:
      p = {}
    items.append({
      "name": name,
      "description": p.get("description", ""),
      "checkpoint": (p.get("model") or {}).get("checkpoint"),
      "active": name == active,
    })
  return {"profiles": items, "active_profile": active}


def _media_response(base: Path, filename: str):
  if not base.exists():
    return _json_error(f"{base} does not exist", 404)
  return send_from_directory(base, filename)


def _python_script_command(script: Path) -> list[str]:
  return [sys.executable, "-u", str(script)]


def _path_status(path: Path) -> dict[str, Any]:
  return {
    "path": str(path),
    "exists": path.exists(),
    "is_file": path.is_file(),
    "is_dir": path.is_dir(),
  }


def _strip_image_filter_args(command: list[str]) -> list[str]:
  stripped: list[str] = []
  skip_next = False
  filter_flags_with_values = {"--process-items", "--process-categories"}
  filter_flags = {"--needs-selection", "--resume"}
  for part in command:
    if skip_next:
      skip_next = False
      continue
    if part in filter_flags_with_values:
      skip_next = True
      continue
    if part in filter_flags:
      continue
    stripped.append(part)
  return stripped


def health_summary(host: str) -> dict[str, Any]:
  comfy_ok = False
  comfy_error = None
  try:
    urllib.request.urlopen(f"http://{host}/", timeout=1.5).close()
    comfy_ok = True
  except Exception as exc:
    comfy_error = str(exc)
  return {
    "image_script": _path_status(ROOT / "tools" / "itemimages" / "generate_images.py"),
    "audio_script": _path_status(ROOT / "tools" / "itemaudio" / "generate_candidates.py"),
    "image_workflow": _path_status(DEFAULT_IMAGE_WORKFLOW),
    "comfyui_dir": _path_status(DEFAULT_COMFYUI_DIR),
    "comfyui": {
      "host": host,
      "reachable": comfy_ok,
      "error": comfy_error,
    },
  }


def _module_available(module: str) -> bool:
  try:
    return importlib.util.find_spec(module) is not None
  except (ImportError, ValueError):
    return False


def _venv_modules_available(python_path: Path, modules: list[str]) -> bool:
  if not python_path.exists():
    return False
  code = (
    "import importlib.util, sys; "
    "mods = sys.argv[1:]; "
    "sys.exit(0 if all(importlib.util.find_spec(mod) for mod in mods) else 1)"
  )
  try:
    result = subprocess.run(
      [str(python_path), "-c", code, *modules],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      timeout=5,
      check=False,
    )
    return result.returncode == 0
  except Exception:
    return False


def _audio_engine_runtime_status(engine: str) -> dict[str, Any]:
  if engine == "coqui":
    cli = COQUI_LOCAL_DIR / ".venv" / "bin" / "tts"
    return {"runtime_installed": cli.exists(), "runtime_detail": str(cli)}
  engine_dir = AUDIO_TOOLS_DIR / "engines" / engine
  venv = engine_dir / ".venv"
  py = venv / "bin" / "python"
  if engine == "piper":
    exe = venv / "bin" / "piper"
    return {"runtime_installed": exe.exists() or shutil.which("piper") is not None, "runtime_detail": str(exe)}
  if engine == "mms":
    modules = ["transformers", "torch", "scipy", "yaml"]
    return {"runtime_installed": _venv_modules_available(py, modules), "runtime_detail": ", ".join(modules)}
  if engine == "melotts":
    modules = ["melo", "yaml"]
    return {"runtime_installed": _venv_modules_available(py, modules), "runtime_detail": ", ".join(modules)}
  return {"runtime_installed": py.exists(), "runtime_detail": str(py)}


def _comfyui_reachable(host: str) -> bool:
  try:
    urllib.request.urlopen(f"http://{host}/", timeout=1.0).close()
    return True
  except Exception:
    return False


def _read_yaml(path: Path) -> Any:
  if not path.exists():
    return None
  try:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
  except Exception:
    return None


def _dir_has_files(path: Path) -> bool:
  return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def _audio_voice_cache_status(engine: str, route: dict[str, Any]) -> dict[str, Any]:
  model = str(route.get("model") or "").strip()
  voice = str(route.get("voice") or "").strip()
  if engine == "coqui" and model:
    path = COQUI_LOCAL_DIR / "cache" / "tts" / "tts" / model.replace("/", "--")
    return {"cached": _dir_has_files(path), "cache_path": str(path)}
  if engine == "piper" and voice:
    raw = Path(voice)
    path = raw if raw.exists() else AUDIO_TOOLS_DIR / "engines" / "piper" / "cache" / (raw.name if raw.suffix == ".onnx" else f"{voice}.onnx")
    return {"cached": path.is_file(), "cache_path": str(path)}
  if engine in {"mms", "melotts"}:
    identifier = model or voice
    if not identifier:
      return {"cached": False, "cache_path": ""}
    if engine == "melotts":
      # MeloTTS routes use a language code (EN, FR, ...) but the model is cached
      # under its Hugging Face repo name (myshell-ai/MeloTTS-English, ...).
      identifier = MELOTTS_LANG_TO_HF_REPO.get(identifier.upper(), identifier)
    cache = AUDIO_TOOLS_DIR / "engines" / engine / "cache"
    hf_model_path = cache / "huggingface" / "hub" / f"models--{identifier.replace('/', '--')}"
    loose_matches = list(cache.rglob(f"*{Path(identifier).name}*")) if cache.exists() else []
    cached = _dir_has_files(hf_model_path) or any(p.is_file() or _dir_has_files(p) for p in loose_matches)
    return {"cached": cached, "cache_path": str(hf_model_path if hf_model_path.exists() else cache)}
  return {"cached": False, "cache_path": ""}


def _read_text(path: str) -> str:
  try:
    return Path(path).read_text(encoding="utf-8", errors="replace")
  except Exception:
    return ""


def _cpu_info() -> dict[str, Any]:
  model = ""
  for line in _read_text("/proc/cpuinfo").splitlines():
    if line.lower().startswith("model name"):
      model = line.split(":", 1)[1].strip()
      break
  return {
    "model": model or platform.processor() or platform.machine(),
    "arch": platform.machine(),
    "cores_logical": os.cpu_count(),
  }


def _memory_info() -> dict[str, Any]:
  totals: dict[str, int] = {}
  for line in _read_text("/proc/meminfo").splitlines():
    if line.startswith(("MemTotal:", "MemAvailable:")):
      name, value, *_ = line.replace(":", "").split()
      totals[name] = int(value)  # kB
  gb = lambda kb: round(kb / 1024 / 1024, 1) if kb else None
  return {
    "total_gb": gb(totals.get("MemTotal")),
    "available_gb": gb(totals.get("MemAvailable")),
  }


def _gpu_info() -> dict[str, Any]:
  gpus: list[dict[str, Any]] = []
  cuda_version = None

  smi = shutil.which("nvidia-smi")
  if smi:
    try:
      out = subprocess.run(
        [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
      )
      if out.returncode == 0:
        for line in out.stdout.strip().splitlines():
          parts = [p.strip() for p in line.split(",")]
          if not parts or not parts[0]:
            continue
          mem = parts[1] if len(parts) > 1 else ""
          gpus.append({
            "vendor": "NVIDIA",
            "name": parts[0],
            "memory_mb": int(mem) if mem.isdigit() else None,
            "driver": parts[2] if len(parts) > 2 else None,
          })
      plain = subprocess.run([smi], capture_output=True, text=True, timeout=5)
      match = re.search(r"CUDA Version:\s*([\d.]+)", plain.stdout)
      if match:
        cuda_version = match.group(1)
    except Exception:
      pass

  # Supplement with any display controllers lspci can see (e.g. an integrated
  # GPU alongside the discrete NVIDIA one) that nvidia-smi does not report.
  lspci = shutil.which("lspci")
  if lspci:
    try:
      out = subprocess.run([lspci], capture_output=True, text=True, timeout=5)
      for line in out.stdout.splitlines():
        if not re.search(r"VGA compatible controller|3D controller|Display controller", line):
          continue
        # lspci line: "<pci-slot> <class>: <device>"; the slot itself contains
        # colons, so drop it by splitting off the leading whitespace-delimited
        # field before separating class from device on the first ": ".
        rest = line.split(None, 1)[1] if " " in line else line
        desc = rest.split(": ", 1)[1].strip() if ": " in rest else rest.strip()
        if "NVIDIA" in desc and any(g["vendor"] == "NVIDIA" for g in gpus):
          continue  # already covered with richer detail by nvidia-smi
        vendor = desc.split(" ", 1)[0]
        gpus.append({"vendor": vendor, "name": desc, "memory_mb": None, "driver": None})
    except Exception:
      pass

  return {"gpus": gpus, "cuda_version": cuda_version, "detected": bool(gpus)}


def system_info() -> dict[str, Any]:
  return {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu": _cpu_info(),
    "memory": _memory_info(),
    "gpu": _gpu_info(),
  }


def tools_status(host: str) -> dict[str, Any]:
  comfy = DEFAULT_COMFYUI_DIR

  packages = [
    {"name": name, "module": module, "installed": _module_available(module)}
    for name, module in PIPELINE_PACKAGES
  ]

  # The shared model files each profile needs come from the profile's own
  # downloads: list — profiles are the single source of truth (no model_sets.yml).
  sets: list[dict[str, Any]] = []
  for name in list_profile_names():
    try:
      spec = load_profile(name)
    except Exception:
      continue
    downloads = []
    for dl in spec.get("downloads", []) or []:
      dest = comfy / str(dl.get("dest") or "")
      exists = dest.exists()
      downloads.append({
        "dest": dl.get("dest"),
        "size_hint": dl.get("size_hint", ""),
        "exists": exists,
        "size_mb": round(dest.stat().st_size / 1_000_000, 1) if exists else None,
      })
    sets.append({
      "name": name,
      "description": spec.get("description", ""),
      "downloads": downloads,
      "complete": bool(downloads) and all(d["exists"] for d in downloads),
    })

  audio_config = _read_yaml(AUDIO_CONFIG) or {}
  audio_engines: dict[str, dict[str, Any]] = {
    name: {"engine": name, "voice_sets": []}
    for name in ("coqui", "piper", "mms", "melotts")
  }
  coqui_models = []
  languages = audio_config.get("languages") if isinstance(audio_config, dict) else None
  if isinstance(languages, dict):
    for lang, lang_spec in languages.items():
      if not isinstance(lang_spec, dict):
        continue
      selected = str(lang_spec.get("selected") or "").strip()
      voice_sets = lang_spec.get("voice_sets") or {}
      for voice_set, route in (voice_sets.items() if isinstance(voice_sets, dict) else []):
        if not isinstance(route, dict):
          continue
        engine = str(route.get("engine") or "").strip()
        if engine not in audio_engines:
          audio_engines[engine] = {"engine": engine, "voice_sets": []}
        cache_status = _audio_voice_cache_status(engine, route)
        audio_engines[engine]["voice_sets"].append({
          "lang": lang,
          "voice_set": voice_set,
          "selected": voice_set == selected,
          "model": route.get("model"),
          "voice": route.get("voice"),
          **cache_status,
        })
        if engine != "coqui":
          continue
        model = route.get("model")
        if not model:
          continue
        cache_root = COQUI_LOCAL_DIR / "cache" / "tts" / "tts"
        model_dir = cache_root / str(model).replace("/", "--")
        installed = model_dir.is_dir() and any(model_dir.iterdir())
        coqui_models.append({
          "lang": lang,
          "voice_set": voice_set,
          "selected": voice_set == selected,
          "model": model,
          "installed": installed,
          "cached": cache_status["cached"],
          "cache_path": cache_status["cache_path"],
        })
  for engine, info in audio_engines.items():
    engine_dir = AUDIO_TOOLS_DIR / "engines" / engine
    venv = engine_dir / ".venv" if engine != "coqui" else COQUI_LOCAL_DIR / ".venv"
    info["venv_installed"] = venv.exists()
    info.update(_audio_engine_runtime_status(engine))
    info["configured"] = bool(info["voice_sets"])

  return {
    "system": system_info(),
    "python": packages,
    "comfyui": {
      "installed": (comfy / ".git").exists(),
      "node_installed": (comfy / "custom_nodes" / "websocket_image_save.py").exists(),
      "reachable": _comfyui_reachable(host),
      "host": host,
      "path": str(comfy),
    },
    "image_models": {
      "civitai_key": CIVITAI_KEY_FILE.exists(),
      "sets": sets,
    },
    "coqui": {
      "venv_installed": (COQUI_LOCAL_DIR / ".venv").exists(),
      "tts_installed": (COQUI_LOCAL_DIR / ".venv" / "bin" / "tts").exists(),
      "config": str(AUDIO_CONFIG),
      "models": coqui_models,
    },
    "audio_engines": audio_engines,
  }


def _summarize_exception(exc: BaseException) -> dict[str, Any]:
  return {
    "ok": False,
    "error": str(exc),
    "traceback": traceback.format_exc().splitlines()[-8:],
  }


def _exception_response(exc: BaseException):
  status = 500
  if isinstance(exc, FileNotFoundError):
    status = 404
  elif isinstance(exc, (KeyError, ValueError)):
    status = 400
  return jsonify(_summarize_exception(exc)), status


class Job:
  def __init__(
      self,
      kind: str,
      command: list[str],
      cwd: Path = ROOT,
      artifacts: dict[str, str] | None = None,
  ):
    self.id = uuid.uuid4().hex[:12]
    self.kind = kind
    self.command = command
    self.cwd = cwd
    self.artifacts = artifacts or {}
    self.status = "queued"
    self.exit_code: int | None = None
    self.started_at: float | None = None
    self.finished_at: float | None = None
    self.lines: list[str] = []
    self._total_lines = 0  # lines ever appended; survives buffer truncation
    self.error: str | None = None
    self.progress: dict[str, Any] = {}
    self._cancel_requested = False
    self._proc: subprocess.Popen[str] | None = None
    self._lock = threading.Lock()

  def append(self, line: str) -> None:
    with self._lock:
      cleaned = line.rstrip("\n")
      self.lines.append(cleaned)
      self._total_lines += 1
      self._update_progress(cleaned)
      if len(self.lines) > 2000:
        self.lines = self.lines[-2000:]

  def lines_since(self, cursor: int) -> tuple[list[str], int]:
    """Return (new_lines, next_cursor) for an absolute line cursor.

    The cursor counts every line ever produced, so it stays valid even after the
    in-memory buffer drops old lines — preventing the live log stream from
    stalling once a job emits more than the retained tail."""
    with self._lock:
      base = self._total_lines - len(self.lines)
      start = max(cursor, base) - base
      return list(self.lines[start:]), self._total_lines

  def _update_progress(self, line: str) -> None:
    if self.kind == "image-generate":
      if line.startswith("[") and "/" in line and "]" in line:
        head, _, label = line.partition("]")
        nums = head.strip("[").split("/", 1)
        if len(nums) == 2 and nums[0].isdigit() and nums[1].isdigit():
          self.progress.update({
            "current": int(nums[0]),
            "total": int(nums[1]),
            "label": label.strip(),
          })
      elif line.startswith("Done!"):
        self.progress["summary"] = line
      elif "FAILED" in line or "Attempt" in line:
        self.progress["last_warning"] = line
    elif self.kind == "audio-generate":
      if line.startswith("Writing "):
        self.progress["current"] = self.progress.get("current", 0) + 1
        self.progress["label"] = line
      elif "candidate manifest entries" in line:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
          self.progress["total"] = int(parts[1])
        self.progress["summary"] = line
      elif line.startswith("Done.") or line.startswith("Wrote "):
        self.progress["summary"] = line
      elif line.startswith("WARN:"):
        self.progress["last_warning"] = line
      elif "Processing" in line or "->" in line:
        self.progress["label"] = line
    elif self.kind == "package":
      if line.startswith("Generated ") or line.startswith("Dropped "):
        self.progress["summary"] = line

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return {
        "id": self.id,
        "kind": self.kind,
        "command": self.command,
        "cwd": str(self.cwd),
        "status": self.status,
        "exit_code": self.exit_code,
        "started_at": self.started_at,
        "finished_at": self.finished_at,
        "error": self.error,
        "progress": dict(self.progress),
        "artifacts": dict(self.artifacts),
        "lines": list(self.lines),
      }

  def run(self) -> None:
    self.status = "running"
    self.started_at = time.time()
    self.append("$ " + " ".join(self.command))
    try:
      env = dict(os.environ, PYTHONUNBUFFERED="1")
      self._proc = subprocess.Popen(
        self.command,
        cwd=self.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
      )
      assert self._proc.stdout is not None
      for line in self._proc.stdout:
        self.append(line)
      self.exit_code = self._proc.wait()
      if self._cancel_requested:
        self.status = "canceled"
      else:
        self.status = "succeeded" if self.exit_code == 0 else "failed"
      if self.exit_code and self.status != "canceled":
        self.error = f"Command exited with {self.exit_code}"
    except Exception as exc:
      self.status = "failed"
      self.error = str(exc)
      self.append(traceback.format_exc())
    finally:
      self.finished_at = time.time()

  def _signal_group(self, sig: int) -> None:
    if self._proc is None:
      return
    try:
      os.killpg(os.getpgid(self._proc.pid), sig)
    except (ProcessLookupError, PermissionError):
      # Process group already gone, or no permission; fall back to the child.
      try:
        self._proc.send_signal(sig)
      except ProcessLookupError:
        pass

  def cancel(self) -> None:
    if self._proc is None or self.status != "running":
      return
    self._cancel_requested = True
    self.append("Cancel requested.")
    self._signal_group(signal.SIGTERM)
    try:
      self._proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
      self.append("Terminate timed out; killing process.")
      self._signal_group(signal.SIGKILL)


class JobStore:
  def __init__(self):
    self._jobs: dict[str, Job] = {}
    self._lock = threading.Lock()

  def start(
      self,
      kind: str,
      command: list[str],
      cwd: Path = ROOT,
      artifacts: dict[str, str] | None = None,
  ) -> Job:
    job = Job(kind, command, cwd, artifacts=artifacts)
    with self._lock:
      self._jobs[job.id] = job
    thread = threading.Thread(target=job.run, daemon=True)
    thread.start()
    return job

  def list(self) -> list[dict[str, Any]]:
    with self._lock:
      jobs = list(self._jobs.values())
    return [job.snapshot() for job in sorted(jobs, key=lambda j: j.started_at or 0, reverse=True)]

  def get(self, job_id: str) -> Job | None:
    with self._lock:
      return self._jobs.get(job_id)


JOBS = JobStore()


def validation_summary(wordlist: Path | None, langs: list[str], corpus: str | None = None) -> dict[str, Any]:
  _, flat = load_corpus(ROOT, corpus)
  if wordlist:
    keys, tags = parse_wordlist(wordlist)
  else:
    keys, tags = sorted(flat), {}

  profile = get_active_profile()
  output_root = get_output_root()
  selections = load_image_selections(output_root, profile)
  audio_sel = audio_approvals(output_root)
  human_sel = load_audio_selections(output_root, "human", wordbank_dir=ROOT)

  missing_keys = [key for key in keys if key not in flat]
  known_keys = [key for key in keys if key in flat]
  translation_missing: dict[str, list[str]] = {}
  image_missing: list[str] = []
  audio_missing: dict[str, list[str]] = {lang: [] for lang in langs}
  audio_human: dict[str, int] = {lang: 0 for lang in langs}
  ready_keys: list[str] = []

  for key in known_keys:
    item = flat[key]
    selected = selections.get(key)
    has_image = bool(selected and Path(selected.get("path", "")).exists())
    if not has_image:
      image_missing.append(key)
    key_audio_ok = True
    for lang in langs:
      if compose_text(key, item, lang) is None:
        translation_missing.setdefault(lang, []).append(key)
      clip = audio_sel.get((lang, key))
      if not clip or not Path(clip).exists():
        audio_missing.setdefault(lang, []).append(key)
        key_audio_ok = False
      if (lang, key) in human_sel:
        audio_human[lang] += 1
    if has_image and key_audio_ok:
      ready_keys.append(key)

  media_errors = validate_requested_media(ROOT, known_keys, langs, selections, audio_sel) if langs else []
  return {
    "corpus_keys": len(flat),
    "selected_keys": len(keys),
    "known_keys": len(known_keys),
    "ready": len(ready_keys),
    "wordlist": str(wordlist) if wordlist else None,
    "languages": langs,
    "profile": profile,
    "missing_keys": missing_keys,
    "translation_missing": translation_missing,
    "media_errors": media_errors,
    "image": {
      "profile": profile,
      "selected": len(known_keys) - len(image_missing),
      "approved": len(known_keys) - len(image_missing),
      "missing_or_unapproved": image_missing,
    },
    "audio": {
      lang: {
        "approved": len(known_keys) - len(missing),
        "human": audio_human[lang],
        "missing_or_unapproved": missing,
      }
      for lang, missing in audio_missing.items()
    },
    "tags_by_key": tags,
  }


def corpus_detail(key: str, langs: list[str], corpus: str | None = None) -> dict[str, Any]:
  _, flat = load_corpus(ROOT, corpus)
  key = slugify(key)
  if key not in flat:
    raise FileNotFoundError(f"{key} not found in Wordbank corpus")
  item = flat[key]
  profile = get_active_profile()
  output_root = get_output_root()
  selected = load_image_selections(output_root, profile).get(key)
  image_path = Path(selected["path"]) if selected else None
  audio_sel = audio_approvals(output_root)
  audio_paths = {lang: audio_sel.get((lang, key)) for lang in langs}
  return {
    "key": key,
    "item": item,
    "topic": item.get("topic"),
    "text": key.replace("_", " "),
    "translations": {lang: compose_text(key, item, lang) for lang in langs},
    "image": {
      "profile": profile,
      "path": str(image_path) if image_path else None,
      "exists": bool(image_path and image_path.exists()),
      "selected": bool(image_path and image_path.exists()),
      "approved": bool(image_path and image_path.exists()),
    },
    "audio": {
      lang: {
        "paths": [str(path)] if path else [],
        "exists": bool(path and Path(path).exists()),
        "approved": bool(path and Path(path).exists()),
      }
      for lang, path in audio_paths.items()
    },
  }


def maintenance_report(wordlist: Path | None, langs: list[str], image_output: Path, audio_output: Path, corpus: str | None = None) -> str:
  summary = validation_summary(wordlist, langs, corpus)
  lines = [
    "# Wordbank Maintenance Report",
    "",
    f"- Wordbank: `{ROOT}`",
    f"- Corpus: `{resolve_corpus_path(ROOT, corpus)}`",
    f"- Word-list: `{wordlist or 'all corpus keys'}`",
    f"- Languages: `{','.join(langs)}`",
    f"- Corpus keys: {summary['corpus_keys']}",
    f"- Selected keys: {summary['selected_keys']}",
    f"- Known keys: {summary['known_keys']}",
    f"- Images ready: {summary['image']['approved']}/{summary['known_keys']}",
  ]
  for lang, info in summary["audio"].items():
    lines.append(f"- Audio ready {lang}: {info['approved']}/{summary['known_keys']}")
  lines.extend([
    "",
    "## Staging",
    "",
    f"- Image output: `{image_output}`",
    f"- Image manifest: `{image_output / 'manifest.json'}`",
    f"- Audio output: `{audio_output}`",
    f"- Audio manifest: `{audio_output / 'manifest.json'}`",
    "",
    "## Issues",
    "",
  ])
  if summary["missing_keys"]:
    lines.append(f"- Missing keys: {', '.join(summary['missing_keys'])}")
  if summary["image"]["missing_or_unapproved"]:
    lines.append(f"- Missing or unapproved images ({len(summary['image']['missing_or_unapproved'])}): {', '.join(summary['image']['missing_or_unapproved'][:120])}")
  for lang, missing in summary["translation_missing"].items():
    lines.append(f"- Missing translations {lang} ({len(missing)}): {', '.join(missing[:120])}")
  for lang, info in summary["audio"].items():
    missing = info["missing_or_unapproved"]
    if missing:
      lines.append(f"- Missing or unapproved audio {lang} ({len(missing)}): {', '.join(missing[:120])}")
  if lines[-1] == "":
    lines.append("No issues for selected inputs.")
  lines.append("")
  return "\n".join(lines)


def build_image_groups(output_dir: Path, corpus: str | None = None) -> list[dict[str, Any]]:
  groups: dict[str, dict[str, Any]] = {}
  order: list[str] = []

  def ensure(item: str, category: str | None) -> dict[str, Any]:
    key = slugify(item)
    group = groups.get(key)
    if group is None:
      group = {
        "item": item,
        "key": key,
        "category": category or None,
        "images": [],
        "selected": None,
      }
      groups[key] = group
      order.append(key)
    elif category and not group["category"]:
      group["category"] = category
    return group

  # Seed from the corpus so every word appears in the review list, even those
  # with zero candidates in this profile — otherwise un-generated words go
  # unnoticed. The category is the corpus topic.
  try:
    _, flat = load_corpus(ROOT, corpus)
  except SystemExit:
    flat = {}
  for word_key, item in flat.items():
    ensure(word_key, str(item.get("topic") or "").strip() or None)

  # Overlay the profile's generated candidates from its manifest (if any).
  manifest_path = output_dir / "manifest.json"
  if manifest_path.exists():
    for entry in _manifest_entries(_load_json(manifest_path)):
      item = str(entry.get("item") or entry.get("key") or "").strip()
      filename = str(entry.get("filename") or "")
      if not item or not filename or not (output_dir / filename).exists():
        continue
      category = str(entry.get("category") or "").strip() or None
      group = ensure(item, category)
      # A generator slot collision can leave two manifest entries for one file;
      # the later entry reflects what's on disk, so drop any earlier duplicate
      # rather than showing two cards that select/delete together.
      group["images"] = [im for im in group["images"] if im["filename"] != filename]
      selected = bool(entry.get("selected"))
      group["images"].append({
        "filename": filename,
        "selected": selected,
        "prompt": entry.get("prompt"),
        "negative": entry.get("negative"),
        "seed": entry.get("seed"),
        "category": entry.get("category"),
        "width": entry.get("width"),
        "height": entry.get("height"),
        "generated_at": entry.get("generated_at"),
      })
      if selected:
        group["selected"] = filename

  # Attach the profile-derived default prompt (category template with {item}
  # filled in) so words with zero candidates still show a usable prompt to edit
  # and generate from. The profile is named by the output dir (…/<profile>/candidates).
  try:
    prompt_cfg = load_profile(output_dir.parent.name).get("prompt") or {}
  except Exception:
    prompt_cfg = {}
  default_template = str(prompt_cfg.get("template") or "")
  cat_templates = prompt_cfg.get("categories") or {}
  for group in groups.values():
    template = str(cat_templates.get(group["category"]) or default_template)
    if template:
      corpus_item = flat.get(group["key"]) or {}
      desc = str(corpus_item.get("desc") or "").strip() or group["key"].replace("_", " ")
      group["default_prompt"] = template.replace("{item}", group["key"]).replace("{desc}", desc)
    else:
      group["default_prompt"] = ""

  # Group by category (uncategorized last), then alphabetically by item.
  return sorted(
    (groups[key] for key in order),
    key=lambda g: ((1, "") if not g["category"] else (0, g["category"].lower()), g["item"].lower()),
  )


def set_image_selection(output_dir: Path, item: str, filename: str | None) -> None:
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  # Match by slug: the manifest's item field is unnormalized (multi-word words
  # may be stored with a space, e.g. "jet ski", while the UI sends the slug
  # "jet_ski"), so exact-string matching would miss those entries.
  target = slugify(item)
  for entry in _manifest_entries(manifest):
    if slugify(entry.get("item") or entry.get("key") or "") != target:
      continue
    if filename is not None and entry.get("filename") == filename:
      entry["selected"] = True
    else:
      entry.pop("selected", None)
  _write_json(manifest_path, manifest)


def _profile_items_data(
    profile: dict[str, Any],
    corpus: str | None,
    single: tuple[str, str | None] | None = None,
) -> dict[str, Any]:
  """Build an items.yml-equivalent dict from a profile + corpus.

  The profile supplies the prompt templates / negatives / rembg model (the
  style); the corpus supplies which words exist and their topic (category). This
  replaces the old setup.py --switch-generated items.yml.
  """
  prompt = profile.get("prompt") or {}
  default_template = str(prompt.get("template") or "")
  default_negative = str(prompt.get("negative") or "")
  cat_templates = prompt.get("categories") or {}
  cat_negatives = prompt.get("negative_categories") or {}
  data: dict[str, Any] = {
    "default_template": default_template,
    "default_negative_prompt": default_negative,
    "rembg_model": profile.get("rembg_model") or "u2net",
    "categories": {},
    # item key -> corpus scene description, injected into templates via {desc}.
    "descriptions": {},
  }

  if single is not None:
    item, custom = single
    if custom:
      # A one-off custom prompt: escape braces so template.format() emits it
      # verbatim instead of substituting {item}/{desc}.
      template = custom.replace("{", "{{").replace("}", "}}")
      negative = default_negative
    else:
      _, flat = load_corpus(ROOT, corpus)
      corpus_item = flat.get(item) or {}
      topic = str(corpus_item.get("topic") or "").strip()
      template = str(cat_templates.get(topic) or default_template)
      negative = str(cat_negatives.get(topic) or default_negative)
      desc = str(corpus_item.get("desc") or "").strip()
      if desc:
        data["descriptions"][item] = desc
    data["categories"]["regen"] = {
      "template": template,
      "negative_prompt": negative,
      "items": [item],
    }
    return data

  # Full / filtered batch: group corpus words by topic and apply the matching
  # profile template. generate_images.py applies --process-items/-categories.
  _, flat = load_corpus(ROOT, corpus)
  by_topic: dict[str, list[str]] = {}
  for key, item in flat.items():
    topic = str(item.get("topic") or "default")
    by_topic.setdefault(topic, []).append(key)
    desc = str(item.get("desc") or "").strip()
    if desc:
      data["descriptions"][key] = desc
  for topic, keys in by_topic.items():
    data["categories"][topic] = {
      "template": str(cat_templates.get(topic) or default_template),
      "negative_prompt": str(cat_negatives.get(topic) or default_negative),
      "items": keys,
    }
  return data


def _patch_workflow_for_profile(profile: dict[str, Any]) -> dict[str, Any]:
  """Load the base ComfyUI workflow and patch model + generation settings from
  the profile, in memory (the global workflow_api.json is never rewritten)."""
  workflow = json.loads(DEFAULT_IMAGE_WORKFLOW.read_text(encoding="utf-8"))
  model = profile.get("model") or {}
  gen = profile.get("generation") or {}
  for node in workflow.values():
    if not isinstance(node, dict):
      continue
    ct = node.get("class_type")
    inputs = node.setdefault("inputs", {})
    if ct == "CheckpointLoaderSimple" and model.get("checkpoint"):
      inputs["ckpt_name"] = model["checkpoint"]
    elif ct == "LoraLoader":
      if model.get("lora"):
        strength = model.get("lora_strength")
        strength = 1.0 if strength is None else strength
        inputs["lora_name"] = model["lora"]
        inputs["strength_model"] = strength
        inputs["strength_clip"] = strength
      else:
        inputs["strength_model"] = 0
        inputs["strength_clip"] = 0
    elif ct == "KSampler":
      if gen.get("sampler"):
        inputs["sampler_name"] = gen["sampler"]
      if gen.get("scheduler"):
        inputs["scheduler"] = gen["scheduler"]
      if gen.get("steps") is not None:
        inputs["steps"] = gen["steps"]
      if gen.get("cfg") is not None:
        inputs["cfg"] = gen["cfg"]
    elif ct == "EmptyLatentImage":
      if gen.get("width"):
        inputs["width"] = gen["width"]
      if gen.get("height"):
        inputs["height"] = gen["height"]
  return workflow


def prepare_profile_generation(
    profile_name: str,
    output_dir: Path,
    corpus: str | None = None,
    single: tuple[str, str | None] | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
  """Derive the per-job items + patched-workflow file paths from a profile and
  return them plus images_per_item. Files live in the profile's image dir (the
  parent of candidates/). With ``write=False`` the paths are computed but no
  files are written — used by the read-only "Copy Command" path."""
  profile = load_profile(profile_name)
  profile_dir = output_dir.parent
  items_path = profile_dir / "_gen_items.json"
  workflow_path = profile_dir / "_gen_workflow.json"
  if write:
    profile_dir.mkdir(parents=True, exist_ok=True)
    _write_json(items_path, _profile_items_data(profile, corpus, single))
    _write_json(workflow_path, _patch_workflow_for_profile(profile))
  images_per_item = (profile.get("generation") or {}).get("images_per_item") or 1
  return {
    "items": items_path,
    "workflow": workflow_path,
    "images_per_item": int(images_per_item),
    "profile": profile,
  }


def delete_image(output_dir: Path, item: str, filename: str) -> dict[str, Any]:
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  entries = _manifest_entries(manifest)
  kept: list[dict[str, Any]] = []
  matched = False
  target = slugify(item)
  for entry in entries:
    if slugify(entry.get("item") or entry.get("key") or "") == target and str(entry.get("filename") or "") == filename:
      matched = True
      continue
    kept.append(entry)
  if not matched:
    raise FileNotFoundError(f"{filename} not found in manifest for {item}")
  path = output_dir / filename
  if path.exists():
    path.unlink()
  if isinstance(manifest, dict) and isinstance(manifest.get("entries"), list):
    manifest["entries"] = kept
    _write_json(manifest_path, manifest)
  else:
    _write_json(manifest_path, kept)
  return {"ok": True, "filename": filename}


def purge_unselected_images(output_dir: Path) -> dict[str, Any]:
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  entries = _manifest_entries(manifest)
  # Group by slug so a word's spaced and underscored manifest entries (e.g.
  # "jet ski" vs "jet_ski") are treated as the same word.
  items_with_selection = {
    slugify(entry.get("item") or entry.get("key") or "")
    for entry in entries
    if entry.get("selected") and slugify(entry.get("item") or entry.get("key") or "")
  }
  removed: list[str] = []
  errors: list[str] = []
  kept: list[dict[str, Any]] = []
  for entry in entries:
    item = slugify(entry.get("item") or entry.get("key") or "")
    filename = str(entry.get("filename") or "")
    if item in items_with_selection and not entry.get("selected"):
      path = output_dir / filename
      try:
        if path.exists():
          path.unlink()
        removed.append(filename)
        continue
      except OSError as exc:
        errors.append(f"{filename}: {exc}")
    kept.append(entry)
  if isinstance(manifest, dict) and isinstance(manifest.get("entries"), list):
    manifest["entries"] = kept
    _write_json(manifest_path, manifest)
  else:
    _write_json(manifest_path, kept)
  return {"ok": not errors, "removed": removed, "errors": errors}


def _rel_to_root(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def extract_images(output_dir: Path, overwrite: bool, dest_dir: Path | None = None) -> dict[str, Any]:
  dest = (dest_dir or DEFAULT_IMAGE_EXTRACT).resolve()
  manifest = _load_json(output_dir / "manifest.json")
  operations: list[tuple[Path, Path]] = []
  skipped: list[str] = []
  errors: list[str] = []
  seen: set[str] = set()
  for entry in _manifest_entries(manifest):
    item = str(entry.get("item") or entry.get("key") or "").strip()
    key = slugify(item)
    if not key or key in seen:
      continue
    group = [
      candidate for candidate in _manifest_entries(manifest)
      if slugify(str(candidate.get("item") or candidate.get("key") or "")) == key
      and candidate.get("selected")
    ]
    seen.add(key)
    if not group:
      skipped.append(f"{key}: no selection")
      continue
    if len(group) > 1:
      errors.append(f"{key}: multiple selected candidates")
      continue
    selected = group[0]
    src = output_dir / str(selected.get("filename") or "")
    if not src.exists():
      errors.append(f"{key}: missing source {selected.get('filename')}")
      continue
    dst = dest / f"{key}{src.suffix or '.png'}"
    if dst.exists() and not overwrite:
      skipped.append(f"{key}: target exists ({_rel_to_root(dst)}); enable overwrite to replace it")
      continue
    operations.append((src, dst))
  if errors:
    return {"ok": False, "copied": [], "skipped": skipped, "errors": errors}
  copied: list[str] = []
  for src, dst in operations:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(_rel_to_root(dst))
  return {"ok": not errors, "copied": copied, "skipped": skipped, "errors": errors, "dest": str(dest)}


def build_audio_groups(
  output_dir: Path,
  language: str | None = None,
  voice_set: str | None = None,
  corpus: str | None = None,
) -> list[dict[str, Any]]:
  groups: dict[str, dict[str, Any]] = {}
  order: list[str] = []

  def ensure(key: str, lang: str, text: str) -> dict[str, Any]:
    group_id = f"{lang}:{key}"
    group = groups.get(group_id)
    if group is None:
      group = {
        "id": group_id,
        "key": key,
        "language": lang,
        "voice_set": voice_set,
        "engine": None,
        "model": None,
        "voice": None,
        "text": text or "",
        "clips": [],
        "approved": None,
        "human_approved": None,
      }
      groups[group_id] = group
      order.append(group_id)
    elif text and not group["text"]:
      group["text"] = text
    return group

  # Seed from the corpus so every word appears in the review list for the
  # selected (language, voice_set), even those with zero clips — this mirrors
  # the image review list and lets words be generated one at a time. Words with
  # no translation for the language are skipped (they can't be synthesized).
  if language and voice_set:
    lang = normalize_lang(language)
    try:
      _, route = resolve_voice_set(load_audio_profile_for_ui(), lang, voice_set)
    except Exception:
      route = {}
    try:
      _, flat = load_corpus(ROOT, corpus)
    except SystemExit:
      flat = {}
    for key, item in flat.items():
      text = compose_text(key, item, lang)
      if text is None:
        continue
      group = ensure(key, lang, text)
      group["engine"] = route.get("engine")
      group["model"] = route.get("model")
      group["voice"] = route.get("voice")

  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path) if manifest_path.exists() else []
  for entry in _manifest_entries(manifest):
    key = str(entry.get("key") or entry.get("item") or "").strip()
    lang = normalize_lang(str(entry.get("language") or entry.get("lang") or ""))
    filename = str(entry.get("filename") or "")
    if not key or not lang or not filename or not (output_dir / filename).exists():
      continue
    group = ensure(key, lang, str(entry.get("text") or ""))
    # Provenance from the manifest reflects what was actually generated; prefer
    # it over the route defaults seeded above.
    group["voice_set"] = entry.get("voice_set", group["voice_set"])
    group["engine"] = entry.get("engine") or group["engine"]
    group["model"] = entry.get("model") or group["model"]
    group["voice"] = entry.get("voice") or group["voice"]
    approved = entry.get("status") == "approved" or bool(entry.get("approved"))
    human_approved = bool(entry.get("human_approved"))
    group["clips"].append({
      "filename": filename,
      "take": entry.get("take"),
      "status": entry.get("status", "pending"),
      "approved": approved,
      "human_approved": human_approved,
      "voice_set": entry.get("voice_set"),
      "engine": entry.get("engine"),
      "model": entry.get("model"),
      "voice": entry.get("voice"),
      "speaker": entry.get("speaker"),
    })
    if approved:
      group["approved"] = filename
    if human_approved:
      group["human_approved"] = filename
  # Group by language, then alphabetically by key.
  return sorted(
    (groups[group_id] for group_id in order),
    key=lambda g: (g["language"], g["key"]),
  )


def set_audio_status(output_dir: Path, key: str, language: str, filename: str, status: str) -> None:
  if status not in {"pending", "approved", "rejected", "flagged"}:
    raise ValueError("invalid status")
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  lang = normalize_lang(language)
  for entry in _manifest_entries(manifest):
    same_group = entry.get("key") == key and normalize_lang(str(entry.get("language") or "")) == lang
    if same_group and status == "approved":
      entry["status"] = "pending"
      entry.pop("approved", None)
    if same_group and entry.get("filename") == filename:
      entry["status"] = status
      if status == "approved":
        entry["approved"] = True
      else:
        entry.pop("approved", None)
  _write_json(manifest_path, manifest)


def approve_audio_language(output_dir: Path, language: str) -> dict[str, Any]:
  """Approve one clip per word for a whole language (the first take), so every
  word that lacks an approved clip gets one in a single action. Words that
  already have an approved clip are left untouched."""
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  lang = normalize_lang(language)
  groups: dict[str, list[dict[str, Any]]] = {}
  order: list[str] = []
  for entry in _manifest_entries(manifest):
    if normalize_lang(str(entry.get("language") or entry.get("lang") or "")) != lang:
      continue
    key = str(entry.get("key") or entry.get("item") or "").strip()
    filename = str(entry.get("filename") or "")
    if not key or not filename or not (output_dir / filename).exists():
      continue
    if key not in groups:
      groups[key] = []
      order.append(key)
    groups[key].append(entry)
  approved = 0
  already = 0
  for key in order:
    clips = groups[key]
    if any(c.get("status") == "approved" or c.get("approved") for c in clips):
      already += 1
      continue
    first = clips[0]
    first["status"] = "approved"
    first["approved"] = True
    approved += 1
  if approved:
    _write_json(manifest_path, manifest)
  return {
    "ok": True,
    "language": lang,
    "approved": approved,
    "already_approved": already,
    "groups": len(order),
  }


def disapprove_audio_language(output_dir: Path, language: str) -> dict[str, Any]:
  """Clear approval for a whole language — the inverse of approve_audio_language.

  Every clip for that language is reset to pending and loses both the ``approved``
  and ``human_approved`` flags (human review implies approval, so it goes too)."""
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  lang = normalize_lang(language)
  cleared_words: set[str] = set()
  changed = False
  for entry in _manifest_entries(manifest):
    if normalize_lang(str(entry.get("language") or entry.get("lang") or "")) != lang:
      continue
    if entry.get("status") == "approved" or entry.get("approved") or entry.get("human_approved"):
      key = str(entry.get("key") or entry.get("item") or "").strip()
      if key:
        cleared_words.add(key)
      if entry.get("status") == "approved":
        entry["status"] = "pending"
      entry.pop("approved", None)
      entry.pop("human_approved", None)
      changed = True
  if changed:
    _write_json(manifest_path, manifest)
  return {"ok": True, "language": lang, "cleared": len(cleared_words)}


def set_audio_human_approved(output_dir: Path, key: str, language: str, human: bool) -> dict[str, Any]:
  """Toggle the per-word *human-reviewed* flag (a second approval level above the
  bulk ``approved`` status). Marking a word human-approved also approves its clip
  (the approved take, else the first), so a single click both approves and human-
  approves. Unmarking only clears the human flag and leaves ``approved`` as-is."""
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  lang = normalize_lang(language)
  entries = [
    e for e in _manifest_entries(manifest)
    if e.get("key") == key and normalize_lang(str(e.get("language") or e.get("lang") or "")) == lang
  ]
  if not entries:
    raise FileNotFoundError(f"no audio clips for {lang}/{key}")
  if human:
    target = next((e for e in entries if e.get("status") == "approved" or e.get("approved")), entries[0])
    for e in entries:
      if e is target:
        e["status"] = "approved"
        e["approved"] = True
        e["human_approved"] = True
      else:
        # Keep a single approved take per word.
        if e.get("status") == "approved":
          e["status"] = "pending"
        e.pop("approved", None)
        e.pop("human_approved", None)
  else:
    for e in entries:
      e.pop("human_approved", None)
  _write_json(manifest_path, manifest)
  return {"ok": True, "key": key, "language": lang, "human_approved": human}


def delete_audio_clip(output_dir: Path, key: str, language: str, filename: str) -> dict[str, Any]:
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  lang = normalize_lang(language)
  kept: list[dict[str, Any]] = []
  matched = False
  for entry in _manifest_entries(manifest):
    same = (
      entry.get("key") == key
      and normalize_lang(str(entry.get("language") or "")) == lang
      and str(entry.get("filename") or "") == filename
    )
    if same:
      matched = True
      continue
    kept.append(entry)
  if not matched:
    raise FileNotFoundError(f"{filename} not found in manifest for {lang}/{key}")
  path = output_dir / filename
  if path.exists():
    path.unlink()
  if isinstance(manifest, dict) and isinstance(manifest.get("entries"), list):
    manifest["entries"] = kept
    _write_json(manifest_path, manifest)
  else:
    _write_json(manifest_path, kept)
  return {"ok": True, "filename": filename}


def purge_unapproved_audio(output_dir: Path) -> dict[str, Any]:
  manifest_path = output_dir / "manifest.json"
  manifest = _load_json(manifest_path)
  entries = _manifest_entries(manifest)
  approved_groups = {
    (normalize_lang(str(e.get("language") or "")), str(e.get("key") or ""))
    for e in entries
    if (e.get("status") == "approved" or e.get("approved"))
    and e.get("language") and e.get("key")
  }
  removed: list[str] = []
  errors: list[str] = []
  kept: list[dict[str, Any]] = []
  for entry in entries:
    group = (normalize_lang(str(entry.get("language") or "")), str(entry.get("key") or ""))
    is_approved = entry.get("status") == "approved" or entry.get("approved")
    filename = str(entry.get("filename") or "")
    if group in approved_groups and not is_approved:
      path = output_dir / filename
      try:
        if path.exists():
          path.unlink()
        removed.append(filename)
        continue
      except OSError as exc:
        errors.append(f"{filename}: {exc}")
    kept.append(entry)
  if isinstance(manifest, dict) and isinstance(manifest.get("entries"), list):
    manifest["entries"] = kept
    _write_json(manifest_path, manifest)
  else:
    _write_json(manifest_path, kept)
  return {"ok": not errors, "removed": removed, "errors": errors}


def extract_audio(output_dir: Path, overwrite: bool, dest_dir: Path | None = None, human_only: bool = False) -> dict[str, Any]:
  dest = (dest_dir or DEFAULT_AUDIO_EXTRACT).resolve()
  manifest = _load_json(output_dir / "manifest.json")
  operations: list[tuple[Path, Path]] = []
  skipped: list[str] = []
  errors: list[str] = []
  group_ids = sorted({
    (normalize_lang(str(entry.get("language") or "")), str(entry.get("key") or ""))
    for entry in _manifest_entries(manifest)
    if entry.get("language") and entry.get("key")
  })
  for lang, key in group_ids:
    approved = [
      entry for entry in _manifest_entries(manifest)
      if normalize_lang(str(entry.get("language") or "")) == lang
      and entry.get("key") == key
      and (entry.get("status") == "approved" or entry.get("approved"))
      and (not human_only or entry.get("human_approved"))
    ]
    if not approved:
      skipped.append(f"{lang}/{key}: no {'human-approved' if human_only else 'approved'} clip")
      continue
    if len(approved) > 1:
      errors.append(f"{lang}/{key}: multiple approved clips")
      continue
    selected = approved[0]
    src = output_dir / str(selected.get("filename") or "")
    if not src.exists():
      errors.append(f"{lang}/{key}: missing source {selected.get('filename')}")
      continue
    ext = src.suffix.lower() if src.suffix.lower() in {".wav", ".ogg"} else ".wav"
    dst = dest / lang / f"{key}{ext}"
    if dst.exists() and not overwrite:
      skipped.append(f"{lang}/{key}: target exists ({_rel_to_root(dst)}); enable overwrite to replace it")
      continue
    operations.append((src, dst))
  if errors:
    return {"ok": False, "copied": [], "skipped": skipped, "errors": errors}
  copied: list[str] = []
  for src, dst in operations:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(_rel_to_root(dst))
  return {"ok": not errors, "copied": copied, "skipped": skipped, "errors": errors, "dest": str(dest)}


@app.route("/")
def index():
  return send_from_directory(app.static_folder, "index.html")


def _corpora_payload() -> dict[str, Any]:
  corpora = list_corpora(ROOT)
  default = resolve_corpus_path(ROOT, None)
  return {
    "corpora": [
      {"name": path.stem, "filename": path.name, "path": str(path)}
      for path in corpora
    ],
    "default": default.stem if default.exists() else None,
  }


@app.route("/api/config")
def api_config():
  state = load_state()
  root = get_output_root(state)
  profile = get_active_profile(state)
  return jsonify({
    "root": str(ROOT),
    "output_root": str(root),
    "active_profile": profile,
    "default_image_output": str(image_candidates_dir(profile, root)),
    "default_image_extract": str(DEFAULT_IMAGE_EXTRACT),
    "default_audio_output": str(audio_output_dir(root)),
    "default_audio_extract": str(DEFAULT_AUDIO_EXTRACT),
    "default_image_workflow": str(DEFAULT_IMAGE_WORKFLOW),
    "default_comfyui_dir": str(DEFAULT_COMFYUI_DIR),
    "default_comfyui_host": "127.0.0.1:8188",
    **_corpora_payload(),
    **profiles_summary(state),
  })


@app.route("/api/corpora")
def api_corpora():
  return jsonify({"ok": True, **_corpora_payload()})


@app.route("/api/profiles")
def api_profiles():
  try:
    return jsonify({"ok": True, "output_root": str(get_output_root()), **profiles_summary()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/profiles/active", methods=["POST"])
def api_profiles_active():
  try:
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    if name not in list_profile_names():
      return _json_error(f"profile {name!r} not found", 404)
    state = load_state()
    state["active_profile"] = name
    save_state(state)
    return jsonify({"ok": True, "active_profile": name})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/profiles/<name>/import", methods=["POST"])
def api_profile_import(name: str):
  """Round-trip: import a built package's selected images into a profile (§5.1)."""
  try:
    if name not in list_profile_names():
      return _json_error(f"profile {name!r} not found", 404)
    data = request.get_json(force=True)
    package_dir = _resolve_path(data.get("package"))
    if not package_dir.is_dir():
      return _json_error(f"{package_dir} is not a directory", 404)
    try:
      template = (load_profile(name).get("prompt") or {}).get("template")
    except Exception:
      template = None
    result = import_package(
      package_dir,
      get_output_root(),
      name,
      prompt_template=template,
      overwrite=bool(data.get("overwrite", True)),
    )
    return jsonify({"ok": True, **result})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/profiles/<name>", methods=["GET", "POST", "DELETE"])
def api_profile(name: str):
  try:
    if request.method == "GET":
      profile = load_profile(name)
      prompt = profile.get("prompt") or {}
      edit = {
        "categories_yaml": _yaml_block(prompt.get("categories")),
        "negative_categories_yaml": _yaml_block(prompt.get("negative_categories")),
        "downloads_yaml": _yaml_block(profile.get("downloads")),
      }
      return jsonify({"ok": True, "name": name, "profile": profile, "edit": edit})

    if request.method == "DELETE":
      directory = PROFILES_DIR / name
      if not (directory / "profile.yml").exists():
        return _json_error(f"profile {name!r} not found", 404)
      if len(list_profile_names()) <= 1:
        return _json_error("cannot delete the last profile")
      shutil.rmtree(directory)
      state = load_state()
      if state.get("active_profile") == name:
        state.pop("active_profile", None)
        save_state(state)
      return jsonify({"ok": True})

    # POST — create or update from the editor payload.
    data = request.get_json(force=True)
    try:
      categories = yaml.safe_load(data.get("categories_yaml") or "") or {}
      negative_categories = yaml.safe_load(data.get("negative_categories_yaml") or "") or {}
      downloads = yaml.safe_load(data.get("downloads_yaml") or "") or []
    except yaml.YAMLError as exc:
      return _json_error(f"invalid YAML in advanced fields: {exc}")
    if not isinstance(categories, dict) or not isinstance(negative_categories, dict):
      return _json_error("category overrides must be YAML mappings")
    if not isinstance(downloads, list):
      return _json_error("downloads must be a YAML list")
    profile = {
      "schema_version": 1,
      "name": data.get("name") or name,
      "description": data.get("description", ""),
      "type": "images",
      "model": {
        "checkpoint": (data.get("model") or {}).get("checkpoint"),
        "lora": (data.get("model") or {}).get("lora") or None,
        "lora_strength": (data.get("model") or {}).get("lora_strength"),
      },
      "rembg_model": data.get("rembg_model") or "u2net",
      "prompt": {
        "template": (data.get("prompt") or {}).get("template", ""),
        "categories": categories,
        "negative": (data.get("prompt") or {}).get("negative", ""),
        "negative_categories": negative_categories,
      },
      "generation": data.get("generation") or {},
      "downloads": downloads,
    }
    slug = save_profile(profile["name"], profile)
    return jsonify({"ok": True, "name": slug})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/state", methods=["GET", "POST"])
def api_state():
  try:
    if request.method == "GET":
      return jsonify({"ok": True, "state": load_state(), "path": str(STATE_PATH)})
    data = request.get_json(force=True)
    return jsonify({"ok": True, "state": save_state(data), "path": str(STATE_PATH)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/health")
def api_health():
  try:
    host = request.args.get("host") or "127.0.0.1:8188"
    return jsonify({"ok": True, "health": health_summary(host)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/tools/status")
def api_tools_status():
  try:
    host = request.args.get("host") or "127.0.0.1:8188"
    return jsonify({"ok": True, "tools": tools_status(host)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/tools/civitai-key", methods=["POST"])
def api_tools_civitai_key():
  try:
    data = request.get_json(force=True)
    token = str(data.get("token") or "").strip()
    if not token:
      return _json_error("token is required")
    CIVITAI_KEY_FILE.write_text(token, encoding="utf-8")
    return jsonify({"ok": True})
  except Exception as exc:
    return _exception_response(exc)


def _known_model_dests() -> set[str]:
  """All model-file dests referenced by any profile's downloads: list."""
  dests: set[str] = set()
  for name in list_profile_names():
    try:
      for dl in load_profile(name).get("downloads") or []:
        dest = str(dl.get("dest") or "").strip()
        if dest:
          dests.add(dest)
    except Exception:
      continue
  return dests


@app.route("/api/tools/delete-model", methods=["POST"])
def api_tools_delete_model():
  """Delete a downloaded model file from the shared ComfyUI models dir. Only
  files some profile references (and only under ComfyUI/models) can be removed."""
  try:
    data = request.get_json(force=True)
    dest = str(data.get("dest") or "").strip()
    if not dest:
      return _json_error("dest is required")
    if dest not in _known_model_dests():
      return _json_error(f"{dest!r} is not a known profile model file", 400)
    models_root = (DEFAULT_COMFYUI_DIR / "models").resolve()
    path = (DEFAULT_COMFYUI_DIR / dest).resolve()
    if models_root != path.parent and models_root not in path.parents:
      return _json_error("refusing to delete outside the ComfyUI models dir", 400)
    if not path.exists():
      return jsonify({"ok": True, "deleted": False, "dest": dest})
    freed = path.stat().st_size
    path.unlink()
    return jsonify({"ok": True, "deleted": True, "dest": dest, "freed_mb": round(freed / 1_000_000, 1)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs/tool", methods=["POST"])
def api_tool_job():
  try:
    data = request.get_json(force=True)
    action = str(data.get("action") or "")
    set_name = str(data.get("set") or "").strip()
    if action == "comfyui-install":
      command = _python_script_command(SETUP_SCRIPT) + ["--install"]
    elif action == "image-download":
      if not set_name:
        return _json_error("set is required")
      command = _python_script_command(SETUP_SCRIPT) + ["--download", set_name]
    elif action == "coqui-download":
      command = _python_script_command(COQUI_SCRIPT) + ["--download"]
    elif action in {"piper-install", "mms-install", "melotts-install"}:
      engine = action.removesuffix("-install")
      command = _python_script_command(AUDIO_TOOLS_DIR / "engines" / f"{engine}.py") + ["--install"]
    elif action in {"piper-download", "mms-download", "melotts-download"}:
      engine = action.removesuffix("-download")
      if not set_name:
        return _json_error("set is required")
      command = _python_script_command(AUDIO_TOOLS_DIR / "engines" / f"{engine}.py") + ["--download", set_name]
    else:
      return _json_error(f"unknown tool action: {action}")
    job = JOBS.start("tool", command)
    return jsonify({"ok": True, "job": job.snapshot()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/validate")
def api_validate():
  try:
    wordlist_raw = request.args.get("wordlist") or None
    wordlist = _resolve_path(wordlist_raw) if wordlist_raw else None
    if wordlist is not None and not wordlist.exists():
      return _json_error(f"{wordlist} not found", 404)
    corpus = request.args.get("corpus") or None
    return jsonify({"ok": True, "summary": validation_summary(wordlist, _langs(request.args.get("langs")), corpus)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/corpus")
def api_corpus():
  try:
    wordlist_raw = request.args.get("wordlist") or None
    wordlist = _resolve_path(wordlist_raw) if wordlist_raw else None
    langs = _langs(request.args.get("langs"))
    corpus = request.args.get("corpus") or None
    _, flat = load_corpus(ROOT, corpus)
    output_root = get_output_root()
    selections = load_image_selections(output_root, get_active_profile())
    audio_sel = audio_approvals(output_root)
    keys, tags = parse_wordlist(wordlist) if wordlist else (sorted(flat), {})
    items = []
    for key in keys:
      item = flat.get(key)
      if not item:
        items.append({"key": key, "missing": True})
        continue
      selected = selections.get(key)
      items.append({
        "key": key,
        "topic": item.get("topic"),
        "text": key.replace("_", " "),
        "translations": {lang: compose_text(key, item, lang) for lang in langs},
        "image_approved": bool(selected and Path(selected.get("path", "")).exists()),
        "audio_approved": {lang: (lang, key) in audio_sel for lang in langs},
        "tags": tags.get(key, {}),
      })
    return jsonify({"ok": True, "items": items})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/corpus/<key>")
def api_corpus_detail(key: str):
  try:
    corpus = request.args.get("corpus") or None
    return jsonify({"ok": True, "detail": corpus_detail(key, _langs(request.args.get("langs")), corpus)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/images/groups")
def api_image_groups():
  try:
    output_dir = _resolve_path(request.args.get("output"), DEFAULT_IMAGE_OUTPUT)
    corpus = request.args.get("corpus") or None
    return jsonify({"ok": True, "output": str(output_dir), "groups": build_image_groups(output_dir, corpus)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/images/select", methods=["POST"])
def api_image_select():
  try:
    data = request.get_json(force=True)
    output_dir = _resolve_path(data.get("output"), DEFAULT_IMAGE_OUTPUT)
    set_image_selection(output_dir, data["item"], data.get("filename"))
    return jsonify({"ok": True})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/images/extract", methods=["POST"])
def api_image_extract():
  try:
    data = request.get_json(force=True)
    profile = str(data.get("profile") or "").strip() or get_active_profile()
    output_dir = _resolve_path(data.get("output"), image_candidates_dir(profile))
    # Export Selected is an optional convenience dump, decoupled from packaging (§4).
    dest_dir = _resolve_path(data.get("dest"), DEFAULT_IMAGE_EXTRACT)
    result = extract_images(output_dir, bool(data.get("overwrite")), dest_dir)
    return jsonify(result), 200 if result["ok"] else 409
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/images/delete", methods=["POST"])
def api_image_delete():
  try:
    data = request.get_json(force=True)
    output_dir = _resolve_path(data.get("output"), DEFAULT_IMAGE_OUTPUT)
    item = str(data.get("item") or "").strip()
    filename = str(data.get("filename") or "").strip()
    if not item or not filename:
      return _json_error("item and filename are required")
    return jsonify(delete_image(output_dir, item, filename))
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/images/purge", methods=["POST"])
def api_image_purge():
  try:
    data = request.get_json(force=True)
    output_dir = _resolve_path(data.get("output"), DEFAULT_IMAGE_OUTPUT)
    result = purge_unselected_images(output_dir)
    return jsonify(result), 200 if result["ok"] else 409
  except Exception as exc:
    return _exception_response(exc)


@app.route("/media/images/<path:filename>")
def media_image(filename: str):
  output_dir = _resolve_path(request.args.get("output"), DEFAULT_IMAGE_OUTPUT)
  return _media_response(output_dir, filename)


@app.route("/approved/images/<key>.png")
def approved_image(key: str):
  # The selected candidate in the active profile is the deliverable for a word.
  selected = load_image_selections(get_output_root(), get_active_profile()).get(slugify(key))
  if selected:
    path = Path(selected["path"])
    if path.exists():
      return send_from_directory(path.parent, path.name)
  return _json_error(f"no selected image for {key}", 404)


@app.route("/api/audio/profile")
def api_audio_profile():
  try:
    audio_root = _resolve_path(request.args.get("output"), DEFAULT_AUDIO_OUTPUT)
    return jsonify({"ok": True, **audio_profile_summary(audio_root)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/profile/select", methods=["POST"])
def api_audio_profile_select():
  try:
    data = request.get_json(force=True)
    language = normalize_lang(str(data.get("language") or ""))
    voice_set = str(data.get("voice_set") or "").strip()
    if not language or not voice_set:
      return _json_error("language and voice_set are required")
    profile = load_audio_profile_for_ui()
    languages = profile.setdefault("languages", {})
    lang_spec = languages.get(language)
    if not isinstance(lang_spec, dict):
      return _json_error(f"language {language!r} is not configured", 404)
    voice_sets = lang_spec.get("voice_sets")
    if not isinstance(voice_sets, dict) or voice_set not in voice_sets:
      return _json_error(f"voice set {voice_set!r} is not configured for {language}", 404)
    lang_spec["selected"] = voice_set
    save_audio_profile_for_ui(profile)
    return jsonify({"ok": True, "language": language, "selected": voice_set})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/profile/voice-set", methods=["POST"])
def api_audio_profile_voice_set():
  try:
    data = request.get_json(force=True)
    language = normalize_lang(str(data.get("language") or ""))
    name = slugify(data.get("name") or data.get("voice_set") or "")
    engine = str(data.get("engine") or "").strip()
    if not language or not name or not engine:
      return _json_error("language, name and engine are required")
    route: dict[str, Any] = {"engine": engine}
    for field in ("model", "voice", "speaker", "speaker_wav"):
      value = str(data.get(field) or "").strip()
      if value:
        route[field] = value
    profile = load_audio_profile_for_ui()
    languages = profile.setdefault("languages", {})
    lang_spec = languages.setdefault(language, {"selected": name, "voice_sets": {}})
    if not isinstance(lang_spec, dict):
      return _json_error(f"language {language!r} has invalid configuration")
    voice_sets = lang_spec.setdefault("voice_sets", {})
    if not isinstance(voice_sets, dict):
      return _json_error(f"language {language!r} has invalid voice_sets")
    voice_sets[name] = route
    if not str(lang_spec.get("selected") or "").strip() or data.get("select"):
      lang_spec["selected"] = name
    downloads = profile.setdefault("downloads", [])
    if isinstance(downloads, list):
      wanted = {"engine": engine}
      if route.get("model"):
        wanted["model"] = route["model"]
      if route.get("voice"):
        wanted["voice"] = route["voice"]
      if wanted not in downloads:
        downloads.append(wanted)
    save_audio_profile_for_ui(profile)
    return jsonify({"ok": True, "language": language, "voice_set": name, "route": route})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/groups")
def api_audio_groups():
  try:
    output_dir = audio_request_dir(args=request.args)
    groups = build_audio_groups(
      output_dir,
      language=request.args.get("language") or None,
      voice_set=request.args.get("voice_set") or None,
      corpus=request.args.get("corpus") or None,
    )
    return jsonify({"ok": True, "output": str(output_dir), "groups": groups})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/status", methods=["POST"])
def api_audio_status():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    set_audio_status(output_dir, data["key"], data["language"], data["filename"], data["status"])
    return jsonify({"ok": True})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/approve-language", methods=["POST"])
def api_audio_approve_language():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    language = str(data.get("language") or "").strip()
    if not language:
      return _json_error("language is required")
    return jsonify(approve_audio_language(output_dir, language))
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/disapprove-language", methods=["POST"])
def api_audio_disapprove_language():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    language = str(data.get("language") or "").strip()
    if not language:
      return _json_error("language is required")
    return jsonify(disapprove_audio_language(output_dir, language))
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/human-approve", methods=["POST"])
def api_audio_human_approve():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    key = str(data.get("key") or "").strip()
    language = str(data.get("language") or "").strip()
    if not key or not language:
      return _json_error("key and language are required")
    return jsonify(set_audio_human_approved(output_dir, key, language, bool(data.get("human"))))
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/delete", methods=["POST"])
def api_audio_delete():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    key = str(data.get("key") or "").strip()
    language = str(data.get("language") or "").strip()
    filename = str(data.get("filename") or "").strip()
    if not key or not language or not filename:
      return _json_error("key, language and filename are required")
    return jsonify(delete_audio_clip(output_dir, key, language, filename))
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/purge", methods=["POST"])
def api_audio_purge():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    result = purge_unapproved_audio(output_dir)
    return jsonify(result), 200 if result["ok"] else 409
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/audio/extract", methods=["POST"])
def api_audio_extract():
  try:
    data = request.get_json(force=True)
    output_dir = audio_request_dir(data)
    dest_dir = _resolve_path(data.get("dest"), DEFAULT_AUDIO_EXTRACT)
    result = extract_audio(output_dir, bool(data.get("overwrite")), dest_dir, bool(data.get("human_only")))
    return jsonify(result), 200 if result["ok"] else 409
  except Exception as exc:
    return _exception_response(exc)


@app.route("/media/audio/<path:filename>")
def media_audio(filename: str):
  output_dir = audio_request_dir(args=request.args)
  return _media_response(output_dir, filename)


def _serve_approved_audio(lang: str, key: str):
  # The approved clip in output/audio is the deliverable for a word+language.
  clip = audio_approvals().get((normalize_lang(lang), slugify(key)))
  if clip and Path(clip).exists():
    return send_from_directory(Path(clip).parent, Path(clip).name)
  return _json_error(f"no approved audio for {lang}/{key}", 404)


@app.route("/approved/audio/<lang>/<key>.wav")
def approved_audio(lang: str, key: str):
  return _serve_approved_audio(lang, key)


@app.route("/approved/audio/<lang>/<key>.ogg")
def approved_audio_ogg(lang: str, key: str):
  return _serve_approved_audio(lang, key)


@app.route("/api/report")
def api_report():
  try:
    wordlist_raw = request.args.get("wordlist") or None
    wordlist = _resolve_path(wordlist_raw) if wordlist_raw else None
    if wordlist is not None and not wordlist.exists():
      return _json_error(f"{wordlist} not found", 404)
    text = maintenance_report(
      wordlist,
      _langs(request.args.get("langs")),
      _resolve_path(request.args.get("image_output"), DEFAULT_IMAGE_OUTPUT),
      _resolve_path(request.args.get("audio_output"), DEFAULT_AUDIO_OUTPUT),
      request.args.get("corpus") or None,
    )
    return Response(text, mimetype="text/markdown")
  except Exception as exc:
    return _exception_response(exc)


# ── Command builders ─────────────────────────────────────────────────────────
# One builder per job kind, shared by the run endpoints and the read-only "Copy
# Command" endpoint, so the copied command always matches what Build runs.

def build_package_command(data: dict[str, Any]) -> list[str]:
  wordlist_raw = data.get("wordlist")
  wordlist = _resolve_path(wordlist_raw) if wordlist_raw else None
  out_dir = _resolve_path(data.get("out"))
  langs = _langs(data.get("langs"))
  command = [
    sys.executable, str(ROOT / "generate_db.py"),
    "--langs", ",".join(langs),
    "--out", str(out_dir),
    "--format", "assets",
  ]
  if wordlist is not None:
    command += ["--wordlist", str(wordlist)]
  if data.get("corpus"):
    command += ["--corpus", str(data.get("corpus"))]
  command += ["--profile", str(data.get("profile") or get_active_profile())]
  command += ["--output-root", str(get_output_root())]
  command += ["--audio-format", data.get("audio_format") or "wav"]
  command += ["--audio-approval", "human" if data.get("audio_approval") == "human" else "approved"]
  if data.get("clean"):
    command.append("--clean")
  if data.get("fail_on_drop"):
    command.append("--fail-on-drop")
  if data.get("quiet_drops"):
    command.append("--quiet-drops")
  return command


def build_image_generate_command(
    data: dict[str, Any], *, write: bool, single_item: str | None = None,
) -> tuple[list[str], str, Path]:
  """Returns (command, profile, output). ``single_item`` switches to the
  per-word regenerate form; ``write`` controls whether the per-job items/workflow
  files are actually written (False for the read-only Copy Command path)."""
  profile = str(data.get("profile") or "").strip() or get_active_profile()
  output = _resolve_path(data.get("output"), image_candidates_dir(profile))
  prompt = data.get("prompt")
  single = (single_item, prompt if prompt else None) if single_item else None
  prepared = prepare_profile_generation(profile, output, data.get("corpus") or None, single=single, write=write)
  command = _python_script_command(ROOT / "tools" / "itemimages" / "generate_images.py") + [
    "--items", str(prepared["items"]),
    "--workflow", str(prepared["workflow"]),
    "--output", str(output),
    "--host", str(data.get("host") or "127.0.0.1:8188"),
    "--images-per-item", str(prepared["images_per_item"]),
  ]
  if single_item:
    command += ["--process-items", single_item]
  if data.get("keep_originals"):
    command.append("--keep-originals")
  if data.get("no_rmbg"):
    command.append("--no-rmbg")
  if data.get("no_comfyui"):
    command.append("--no-comfyui")
  elif data.get("comfyui"):
    command += ["--comfyui", str(_resolve_path(data.get("comfyui")))]
  if not single_item:
    process_items = ",".join(_split_csv(data.get("process_items")))
    process_categories = ",".join(_split_csv(data.get("process_categories")))
    if process_items:
      command += ["--process-items", process_items]
    if process_categories:
      command += ["--process-categories", process_categories]
    if data.get("needs_selection"):
      command.append("--needs-selection")
    if data.get("resume"):
      command.append("--resume")
  return command, profile, output


def build_audio_generate_command(data: dict[str, Any]) -> tuple[list[str], Path]:
  output = _resolve_path(data.get("output"), DEFAULT_AUDIO_OUTPUT)
  langs = _langs(data.get("langs"))
  if len(langs) != 1:
    raise ValueError("Phase 1 audio generation requires exactly one language")
  lang = langs[0]
  profile = load_audio_profile(ROOT)
  voice_set, _route = resolve_voice_set(profile, lang, data.get("voice_set"))
  voice_output = output / lang / voice_set
  command = _python_script_command(ROOT / "tools" / "itemaudio" / "generate_candidates.py") + [
    "--lang", lang,
    "--voice-set", voice_set,
    "--takes", str(int(data.get("takes") or 1)),
    "--output", str(output),
    "--format", data.get("format") or "wav",
  ]
  if data.get("corpus"):
    command += ["--corpus", str(data.get("corpus"))]
  if data.get("wordlist"):
    command += ["--wordlist", str(_resolve_path(data.get("wordlist")))]
  keys = ",".join(_split_csv(data.get("keys")))
  if keys:
    command += ["--keys", keys]
  if data.get("synthesize"):
    command.append("--synthesize")
  if data.get("merge"):
    command.append("--merge")
  if data.get("skip_existing"):
    command.append("--skip-existing")
  return command, voice_output


@app.route("/api/commands", methods=["POST"])
def api_commands():
  """Read-only: returns the command a job would run, without running it or
  writing any files (see prepare_profile_generation write=False)."""
  try:
    data = request.get_json(force=True)
    kind = data.get("kind")
    if kind == "package":
      command = build_package_command(data)
    elif kind == "image-generate":
      command, _, _ = build_image_generate_command(data, write=False)
    elif kind == "audio-generate":
      command, _ = build_audio_generate_command(data)
    else:
      return _json_error("unknown command kind")
    return jsonify({"ok": True, "command": command, "shell": _shell_join(command)})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs/package", methods=["POST"])
def api_package_job():
  try:
    data = request.get_json(force=True)
    wordlist_raw = data.get("wordlist")
    wordlist = _resolve_path(wordlist_raw) if wordlist_raw else None
    out_dir = _resolve_path(data.get("out"))
    langs = _langs(data.get("langs"))
    if wordlist is not None and not wordlist.exists():
      return _json_error(f"{wordlist} not found", 404)
    if not langs:
      return _json_error("At least one language is required")
    command = build_package_command(data)
    job = JOBS.start("package", command, artifacts={"output": str(out_dir), "manifest": str(out_dir / "manifest.yml")})
    return jsonify({"ok": True, "job": job.snapshot()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs/image-generate", methods=["POST"])
def api_image_generate_job():
  try:
    data = request.get_json(force=True)
    profile = str(data.get("profile") or "").strip() or get_active_profile()
    if profile not in list_profile_names():
      return _json_error(f"profile {profile!r} not found", 404)
    if not DEFAULT_IMAGE_WORKFLOW.exists():
      return _json_error(f"{DEFAULT_IMAGE_WORKFLOW} not found", 404)
    command, _, output = build_image_generate_command(data, write=True)
    job = JOBS.start(
      "image-generate",
      command,
      artifacts={
        "output": str(output),
        "manifest": str(output / "manifest.json"),
        "failed": str(output / "failed.txt"),
      },
    )
    return jsonify({"ok": True, "job": job.snapshot()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs/image-regenerate", methods=["POST"])
def api_image_regenerate_job():
  try:
    data = request.get_json(force=True)
    profile = str(data.get("profile") or "").strip() or get_active_profile()
    if profile not in list_profile_names():
      return _json_error(f"profile {profile!r} not found", 404)
    item = str(data.get("item") or "").strip()
    if not item:
      return _json_error("item is required")
    if not DEFAULT_IMAGE_WORKFLOW.exists():
      return _json_error(f"{DEFAULT_IMAGE_WORKFLOW} not found", 404)
    command, _, output = build_image_generate_command(data, write=True, single_item=item)
    job = JOBS.start(
      "image-generate",
      command,
      artifacts={
        "output": str(output),
        "manifest": str(output / "manifest.json"),
        "failed": str(output / "failed.txt"),
      },
    )
    return jsonify({"ok": True, "job": job.snapshot()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs/audio-generate", methods=["POST"])
def api_audio_generate_job():
  try:
    data = request.get_json(force=True)
    langs = _langs(data.get("langs"))
    if len(langs) != 1:
      return _json_error("Phase 1 audio generation requires exactly one language")
    wordlist_raw = data.get("wordlist")
    if wordlist_raw and not _resolve_path(wordlist_raw).exists():
      return _json_error(f"{_resolve_path(wordlist_raw)} not found", 404)
    command, output = build_audio_generate_command(data)
    job = JOBS.start(
      "audio-generate",
      command,
      artifacts={
        "output": str(output),
        "manifest": str(output / "manifest.json"),
        "phrases": str(output / "phrases.generated.yml"),
      },
    )
    return jsonify({"ok": True, "job": job.snapshot()})
  except Exception as exc:
    return _exception_response(exc)


@app.route("/api/jobs")
def api_jobs():
  return jsonify({"ok": True, "jobs": JOBS.list()})


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return _json_error("job not found", 404)
  return jsonify({"ok": True, "job": job.snapshot()})


@app.route("/api/jobs/<job_id>/events")
def api_job_events(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return _json_error("job not found", 404)

  def stream():
    sent = 0
    while True:
      new_lines, sent = job.lines_since(sent)
      for line in new_lines:
        yield f"event: log\ndata: {json.dumps(line)}\n\n"
      snapshot = job.snapshot()
      yield f"event: state\ndata: {json.dumps({k: snapshot[k] for k in ('status', 'exit_code', 'error', 'progress', 'artifacts')})}\n\n"
      if snapshot["status"] in {"succeeded", "failed", "canceled"}:
        # Flush any lines appended after the last read but before the job ended.
        final_lines, sent = job.lines_since(sent)
        for line in final_lines:
          yield f"event: log\ndata: {json.dumps(line)}\n\n"
        break
      time.sleep(0.5)

  return Response(stream(), mimetype="text/event-stream")


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_job_cancel(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return _json_error("job not found", 404)
  job.cancel()
  return jsonify({"ok": True, "job": job.snapshot()})


@app.route("/api/jobs/<job_id>/failed-items")
def api_job_failed_items(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return _json_error("job not found", 404)
  failed = job.artifacts.get("failed")
  if not failed:
    return _json_error("job has no failed-items artifact", 404)
  path = Path(failed)
  if not path.exists():
    return jsonify({"ok": True, "items": [], "path": str(path)})
  items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
  return jsonify({"ok": True, "items": items, "path": str(path)})


@app.route("/api/jobs/<job_id>/retry-failed", methods=["POST"])
def api_job_retry_failed(job_id: str):
  job = JOBS.get(job_id)
  if not job:
    return _json_error("job not found", 404)
  if job.kind != "image-generate":
    return _json_error("retry-failed is only available for image generation jobs")
  failed = job.artifacts.get("failed")
  if not failed:
    return _json_error("job has no failed-items artifact", 404)
  path = Path(failed)
  if not path.exists():
    return _json_error(f"{path} not found", 404)
  items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
  if not items:
    return _json_error("failed-items file is empty")
  command = _strip_image_filter_args(job.command)
  command.extend(["--process-items", ",".join(items)])
  retry = JOBS.start("image-generate", command, artifacts=dict(job.artifacts))
  return jsonify({"ok": True, "items": items, "job": retry.snapshot()})


def main() -> None:
  parser = argparse.ArgumentParser(description="Wordbank local web UI")
  parser.add_argument("--port", type=int, default=5050)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--no-browser", action="store_true")
  args = parser.parse_args()

  url = f"http://{args.host}:{args.port}"
  print(f"Wordbank UI -> {url}")
  if not args.no_browser:
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
  app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
  main()
