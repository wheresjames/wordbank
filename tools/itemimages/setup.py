#!/usr/bin/env python3
"""
ComfyUI environment and model manager.

Usage:
  python setup.py --list                         list profiles and the model files they need
  python setup.py --install                      clone/update ComfyUI, install deps
  python setup.py --download  sdxl-storybook     download the models a profile needs
  python setup.py --civitai-key TOKEN            save CivitAI API token for gated downloads
  python setup.py --status                       show what model files are installed

The model files come from each profile's `downloads:` list
(profiles/<name>/profile.yml); style activation is per profile, not here.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.resolve()
COMFYUI_DIR  = ROOT / "ComfyUI"
KEY_FILE         = ROOT / ".civitai_key"
PROFILES_DIR     = ROOT.parent.parent / "profiles"   # <repo>/profiles
COMFYUI_REPO   = "https://github.com/comfyanonymous/ComfyUI.git"

# Custom node needed to stream images back over the websocket
WEBSOCKET_NODE_SRC = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/custom_nodes/websocket_image_save.py"


def list_profiles() -> list[str]:
    """Profile names (directories holding a profile.yml), sorted."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.name for p in PROFILES_DIR.iterdir() if (p / "profile.yml").exists())


def profile_spec(name: str) -> dict:
    """A profile.yml as a dict (its `downloads:` list is the model-file source)."""
    import yaml
    path = PROFILES_DIR / name / "profile.yml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def redact(text: str) -> str:
    """Mask the CivitAI token so it never appears in printed URLs / logs."""
    return re.sub(r"(token=)[^&\s]+", r"\1***", str(text))


def civitai_key() -> str:
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    return os.environ.get("CIVITAI_API_KEY", "")


def download_file(url: str, dest: Path, size_hint: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = f"{dest.name}  ({size_hint})" if size_hint else dest.name
    print(f"  Downloading {label}")
    print(f"    {redact(url)}")

    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "setup.py/1.0"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
            total    = int(resp.headers.get("Content-Length", 0))
            received = 0
            chunk    = 1 << 20   # 1 MB
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                received += len(block)
                if total:
                    pct = received * 100 // total
                    bar = "█" * (pct // 4) + "░" * (25 - pct // 4)
                    mb  = received / 1_048_576
                    print(f"\r    [{bar}] {pct:3d}%  {mb:.0f} MB", end="", flush=True)
            print()
        return True
    except urllib.error.HTTPError as e:
        print(f"\n    ERROR {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n    ERROR: {redact(e)}")
        if dest.exists():
            dest.unlink()
        return False


def remote_size(url: str) -> int:
    """Return Content-Length for url via HEAD, or 0 on failure."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "setup.py/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0


def fetch_model(spec: dict) -> bool:
    dest      = COMFYUI_DIR / spec["dest"]
    size_hint = spec.get("size_hint", "")

    # Build the download URL up front so we can HEAD-check it if needed
    if "url" in spec:
        url = spec["url"]
    elif "civitai" in spec:
        key = civitai_key()
        base = f"https://civitai.com/api/download/models/{spec['civitai']}"
        url  = f"{base}?token={key}" if key else base
    else:
        url = None

    if dest.exists():
        if url:
            expected = remote_size(url)
            actual   = dest.stat().st_size
            if expected and actual < expected:
                print(f"  Incomplete ({actual/1e6:.0f} MB of {expected/1e6:.0f} MB) — re-downloading: {spec['dest']}")
                dest.unlink()
            else:
                print(f"  Already exists: {spec['dest']}")
                return True
        else:
            print(f"  Already exists: {spec['dest']}")
            return True

    if url is None:
        print(f"  No URL for {spec['dest']} — add manually.")
        return False

    if "url" in spec:
        return download_file(url, dest, size_hint)

    # CivitAI version download
    if "civitai" in spec:
        ok = download_file(url, dest, size_hint)
        if not ok and not civitai_key():
            print("    Tip: CivitAI may require an API key for this file.")
            print("    Run:  python setup.py --civitai-key YOUR_TOKEN")
        return ok

    print(f"  No URL for {spec['dest']} — add manually.")
    return False


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list():
    print("\nProfiles and the model files they need\n")
    for name in list_profiles():
        spec = profile_spec(name)
        print(f"  {name}")
        if spec.get("description"):
            print(f"    {spec['description']}")
        for dl in spec.get("downloads") or []:
            dest   = COMFYUI_DIR / dl["dest"]
            status = "✓" if dest.exists() else "✗ missing"
            print(f"    [{status}]  {dl['dest']}  {dl.get('size_hint','')}")
        print()


def cmd_status():
    cmd_list()


def cmd_civitai_key(token: str):
    KEY_FILE.write_text(token.strip())
    print(f"Saved CivitAI API key → {KEY_FILE}")


def cmd_install():
    print("── Install / update ComfyUI ─────────────────────────────────────")

    # Clone or update
    if not (COMFYUI_DIR / ".git").exists():
        print("Cloning ComfyUI …")
        subprocess.run(["git", "clone", COMFYUI_REPO, str(COMFYUI_DIR)], check=True)
    else:
        print("Updating ComfyUI …")
        subprocess.run(["git", "-C", str(COMFYUI_DIR), "pull", "--ff-only"], check=True)

    # Install ComfyUI deps
    req_file = COMFYUI_DIR / "requirements.txt"
    if req_file.exists():
        print("\nInstalling ComfyUI requirements …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)

    # Install our own pipeline deps
    print("\nInstalling pipeline requirements …")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "websocket-client", "rembg", "pillow", "onnxruntime",
        "pyyaml", "flask",
    ], check=True)

    # Custom node
    node_path = COMFYUI_DIR / "custom_nodes" / "websocket_image_save.py"
    if not node_path.exists():
        print("\nFetching websocket_image_save custom node …")
        download_file(WEBSOCKET_NODE_SRC, node_path)

    # Ensure model subdirs exist
    for sub in ("checkpoints", "loras", "vae", "controlnet", "embeddings"):
        (COMFYUI_DIR / "models" / sub).mkdir(parents=True, exist_ok=True)

    print("\nInstall complete.")


def cmd_download(profile_name: str):
    if profile_name not in list_profiles():
        print(f"Unknown profile '{profile_name}'.  Run --list to see options.")
        sys.exit(1)

    downloads = profile_spec(profile_name).get("downloads") or []
    print(f"── Downloading models for profile: {profile_name} ──────────────────")
    all_ok = True
    for spec in downloads:
        ok = fetch_model(spec)
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\nAll models ready. Select the '{profile_name}' profile to use this style.")
    else:
        print("\nSome downloads failed. See messages above.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI model manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",        action="store_true",   help="List profiles and the model files they need")
    group.add_argument("--status",      action="store_true",   help="Show install status (alias for --list)")
    group.add_argument("--install",     action="store_true",   help="Clone/update ComfyUI and install deps")
    group.add_argument("--download",    metavar="PROFILE",     help="Download the models a profile needs")
    group.add_argument("--civitai-key", metavar="TOKEN",       help="Save your CivitAI API token")
    args = parser.parse_args()

    if args.list:              cmd_list()
    elif args.status:          cmd_status()
    elif args.install:         cmd_install()
    elif args.download:        cmd_download(args.download)
    elif args.civitai_key:     cmd_civitai_key(args.civitai_key)


if __name__ == "__main__":
    main()
