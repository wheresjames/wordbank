#!/usr/bin/env python3
"""
Batch image generator: ComfyUI → rembg (U2Net) → transparent PNG

Usage:
    python generate_images.py --items items.yml --workflow workflow_api.json

Outputs:
    output/
        cat_001.png              ← transparent PNG
        cat_001_original.png     ← original with white bg (optional, --keep-originals)
        manifest.json

Install deps:
    pip install websocket-client rembg pillow onnxruntime pyyaml
    # GPU acceleration (optional): pip install onnxruntime-gpu
"""

import argparse
import io as _io
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
import websocket  # pip install websocket-client
import yaml       # pip install pyyaml

from PIL import Image
from rembg import remove as rembg_remove, new_session


# ── Config ────────────────────────────────────────────────────────────────────

COMFYUI_HOST = "127.0.0.1:8188"

WIDTH     = 512   # SDXL needs 1024 — 512 produces poor results
HEIGHT    = 512
STEPS     = 30     # more steps helps prompt adherence
CFG       = 6.5    # slightly higher CFG helps follow "single subject" instruction
SAMPLER   = "dpmpp_2m"   # better prompt adherence than euler
SCHEDULER = "karras"     # smoother results with dpmpp
DENOISE   = 1.0

IMAGES_PER_ITEM    = 1
MAX_RETRIES        = 3
RETRY_DELAY        = 5
GENERATION_TIMEOUT = 300


# ── Items / category loader ───────────────────────────────────────────────────

def load_items_yaml(path: str):
    """
    Load items.yml and return (items, item_category, get_prompt, get_negative_prompt, rembg_model).

    items                — flat list of item strings in declaration order
    item_category        — dict mapping lowercase item → category name
    get_prompt           — callable(item) → positive prompt string
    get_negative_prompt  — callable(item) → negative prompt string
    rembg_model          — rembg session name to use for background removal
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    default_template  = data.get("default_template", "")
    default_negative  = data.get("default_negative_prompt", "")
    rembg_model       = data.get("rembg_model", "u2net")
    categories        = data.get("categories", {})
    # Optional per-item scene descriptions from the corpus; templates may inject
    # them via the {desc} placeholder. Falls back to the humanized item name.
    descriptions      = data.get("descriptions", {}) or {}

    items             = []
    item_category     = {}
    cat_templates     = {}
    cat_negatives     = {}

    for cat_name, cat_data in categories.items():
        cat_templates[cat_name] = cat_data.get("template", default_template)
        cat_negatives[cat_name] = cat_data.get("negative_prompt", default_negative)
        for item in cat_data.get("items", []):
            items.append(item)
            item_category[item.strip().lower()] = cat_name

    def get_prompt(item: str) -> str:
        cat      = item_category.get(item.strip().lower())
        template = cat_templates.get(cat, default_template) if cat else default_template
        desc     = str(descriptions.get(item) or "").strip() or item.replace("_", " ")
        return template.format(item=item, desc=desc)

    def get_negative_prompt(item: str) -> str:
        cat = item_category.get(item.strip().lower())
        return cat_negatives.get(cat, default_negative) if cat else default_negative

    return items, item_category, get_prompt, get_negative_prompt, rembg_model


# ── rembg background removal ──────────────────────────────────────────────────
#
# Supported model names (set via rembg_model in the items file / profile):
#   u2net            ~170 MB  general purpose, good for flat illustrations
#   birefnet-general ~400 MB  best for 3D renders and complex objects; handles
#                             fine edges, enclosed areas, and white-on-white well
#   isnet-general-use ~170 MB intermediate quality, faster than birefnet
#
# Models download automatically from HuggingFace on first use.

def load_rmbg_model(model_name: str = "u2net"):
    print(f"Loading rembg / {model_name} background removal model …")
    session = new_session(model_name)
    print(f"  rembg ready ({model_name})")
    return session, None   # device not needed — rembg handles it internally


def remove_background(session, _device, img: Image.Image) -> Image.Image:
    """Return img as RGBA with background pixels made transparent."""
    return rembg_remove(img, session=session)


# ── ComfyUI helpers ───────────────────────────────────────────────────────────

def load_workflow(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_node_by_class(workflow: dict, class_type: str):
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    return None, None


def build_prompt(workflow: dict, positive: str, negative: str, seed: int) -> dict:
    wf = json.loads(json.dumps(workflow))  # deep copy

    _, ksampler = find_node_by_class(wf, "KSampler")
    if ksampler is None:
        _, ksampler = find_node_by_class(wf, "KSamplerAdvanced")

    if ksampler:
        ksampler["inputs"]["seed"]         = seed
        ksampler["inputs"]["steps"]        = STEPS
        ksampler["inputs"]["cfg"]          = CFG
        ksampler["inputs"]["sampler_name"] = SAMPLER
        ksampler["inputs"]["scheduler"]    = SCHEDULER
        ksampler["inputs"]["denoise"]      = DENOISE

        pos_node_id = ksampler["inputs"]["positive"][0]
        neg_node_id = ksampler["inputs"]["negative"][0]
        wf[pos_node_id]["inputs"]["text"]  = positive
        wf[neg_node_id]["inputs"]["text"]  = negative

    _, latent = find_node_by_class(wf, "EmptyLatentImage")
    if latent:
        latent["inputs"]["width"]      = WIDTH
        latent["inputs"]["height"]     = HEIGHT
        latent["inputs"]["batch_size"] = 1

    return wf


def queue_prompt(prompt: dict, client_id: str) -> str:
    payload = {"prompt": prompt, "client_id": client_id}
    data    = json.dumps(payload).encode("utf-8")
    req     = urllib.request.Request(
        f"http://{COMFYUI_HOST}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
            # Surface the most useful part of ComfyUI's validation response
            if "node_errors" in detail:
                msgs = []
                for node_id, node_err in detail["node_errors"].items():
                    for err in node_err.get("errors", []):
                        msgs.append(f"  node {node_id}: {err.get('details', err.get('message', ''))}")
                if msgs:
                    raise RuntimeError(f"ComfyUI rejected prompt (HTTP {e.code}):\n" + "\n".join(msgs)) from None
            if "error" in detail:
                raise RuntimeError(f"ComfyUI rejected prompt (HTTP {e.code}): {detail['error'].get('message', body)}") from None
        except (json.JSONDecodeError, KeyError):
            pass
        raise RuntimeError(f"ComfyUI rejected prompt (HTTP {e.code}): {body}") from None


def get_images_for_prompt(ws_conn, prompt_id: str) -> list[bytes]:
    deadline = time.time() + GENERATION_TIMEOUT
    progress_active = False
    while time.time() < deadline:
        raw = ws_conn.recv()
        if isinstance(raw, str):
            msg = json.loads(raw)
            t   = msg.get("type")
            d   = msg.get("data", {})
            if t == "progress" and d.get("prompt_id") == prompt_id:
                v, m = d["value"], d["max"]
                filled = v * 25 // m
                bar = "█" * filled + "░" * (25 - filled)
                print(f"\r  [{bar}] {v}/{m}", end="", flush=True)
                progress_active = True
            elif t == "executing" and d.get("node") is None and d.get("prompt_id") == prompt_id:
                if progress_active:
                    print()  # newline after the progress bar
                break

    history = json.loads(
        urllib.request.urlopen(f"http://{COMFYUI_HOST}/history/{prompt_id}").read()
    )
    entry = history.get(prompt_id, {})
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for msg_type, msg_data in status.get("messages", []):
            if msg_type == "execution_error":
                raise RuntimeError(
                    f"ComfyUI execution error in node {msg_data.get('node_id')} "
                    f"({msg_data.get('node_type')}): {msg_data.get('exception_message', '').strip()}"
                )
        raise RuntimeError("ComfyUI reported an execution error (no details available)")
    images = []
    for node_output in entry.get("outputs", {}).values():
        for img_info in node_output.get("images", []):
            url = (
                f"http://{COMFYUI_HOST}/view"
                f"?filename={img_info['filename']}"
                f"&subfolder={img_info.get('subfolder', '')}"
                f"&type={img_info.get('type', 'output')}"
            )
            images.append(urllib.request.urlopen(url).read())
    return images


# Deliberately duplicated from wordbank_common.slugify: this script runs
# standalone (no wordbank_common import) so it can be invoked outside the repo.
def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("/", "_")


def item_has_generated_image(item: str, output_dir: str, manifest: list[dict]) -> bool:
    """Return True when output_dir already has at least one generated image for item."""
    manifest_filenames = [
        entry.get("filename")
        for entry in manifest
        if entry.get("item") == item and entry.get("filename")
    ]
    if any(os.path.exists(os.path.join(output_dir, filename)) for filename in manifest_filenames):
        return True

    slug = re.escape(slugify(item))
    generated_image_pattern = re.compile(rf"^{slug}_\d{{3}}\.png$")
    return any(
        generated_image_pattern.match(filename)
        for filename in os.listdir(output_dir)
    )


# ── ComfyUI lifecycle ─────────────────────────────────────────────────────────

def _default_comfyui_dir() -> "str | None":
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI")
    return candidate if os.path.isfile(os.path.join(candidate, "main.py")) else None


def start_comfyui(comfyui_dir: str, host: str) -> subprocess.Popen:
    parts = host.rsplit(":", 1)
    listen_addr = parts[0] if len(parts) == 2 else "127.0.0.1"
    port        = parts[1] if len(parts) == 2 else "8188"

    print(f"Starting ComfyUI from {comfyui_dir} …")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", listen_addr, "--port", port],
        cwd=comfyui_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url      = f"http://{host}/"
    deadline = time.time() + 90
    print("Waiting for ComfyUI to be ready ", end="", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early (return code {proc.returncode})")
        try:
            urllib.request.urlopen(url, timeout=2)
            print(" ready.")
            return proc
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)

    proc.terminate()
    raise RuntimeError("ComfyUI did not become ready within 90 seconds")


def stop_comfyui(proc: "subprocess.Popen | None") -> None:
    if proc is None or proc.poll() is not None:
        return
    print("\nStopping ComfyUI …", end=" ", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global COMFYUI_HOST  # must be declared before any use

    parser = argparse.ArgumentParser(
        description="Kid-art batch generator with transparent backgrounds"
    )
    parser.add_argument("--items",          required=True,        help="YAML file with categories and items (items.yml)")
    parser.add_argument("--workflow",       required=True,        help="ComfyUI API-format workflow JSON")
    parser.add_argument("--output",         default="output",     help="Output directory (default: ./output)")
    parser.add_argument("--host",           default=COMFYUI_HOST, help="ComfyUI host:port")
    parser.add_argument("--keep-originals", action="store_true",  help="Also save white-bg PNGs")
    parser.add_argument("--no-rmbg",        action="store_true",  help="Skip background removal")
    parser.add_argument("--comfyui",             default=None,         help="Path to ComfyUI directory to auto-start (default: ./ComfyUI if present)")
    parser.add_argument("--no-comfyui",          action="store_true",  help="Skip auto-starting ComfyUI (assume it is already running)")
    parser.add_argument("--process-items",       default=None,         help="Comma-separated items to (re)generate, e.g. dog,cat,car")
    parser.add_argument("--process-categories",  default=None,         help="Comma-separated categories to (re)generate, e.g. food,animal")
    parser.add_argument("--needs-selection",      action="store_true",  help="Only process items with no selected image in the manifest")
    parser.add_argument("--resume",               action="store_true",  help="Only process items that have no generated images in the output directory")
    parser.add_argument("--images-per-item",      type=int, default=IMAGES_PER_ITEM, help="How many candidate images to generate per item")
    args = parser.parse_args()

    COMFYUI_HOST = args.host

    comfyui_dir = args.comfyui
    if comfyui_dir is None and not args.no_comfyui:
        comfyui_dir = _default_comfyui_dir()

    comfyui_proc = None
    try:
        if comfyui_dir and not args.no_comfyui:
            comfyui_proc = start_comfyui(comfyui_dir, COMFYUI_HOST)

        _run(args)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_comfyui(comfyui_proc)


def _run(args):
    items, item_category, get_prompt, get_negative_prompt, rembg_model_name = load_items_yaml(args.items)

    filter_items = {i.strip().lower() for i in args.process_items.split(",")} if args.process_items else None
    filter_cats  = {c.strip().lower() for c in args.process_categories.split(",")} if args.process_categories else None

    if filter_items or filter_cats:
        items = [
            item for item in items
            if (filter_items and item.strip().lower() in filter_items)
            or (filter_cats  and item_category.get(item.strip().lower()) in filter_cats)
        ]

    if not items:
        print("ERROR: No items found in input file.")
        sys.exit(1)

    workflow = load_workflow(args.workflow)
    os.makedirs(args.output, exist_ok=True)

    manifest_path = os.path.join(args.output, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Resuming — {len(manifest)} entries already done.")
    else:
        manifest = []

    if args.resume:
        before = len(items)
        items = [
            item for item in items
            if not item_has_generated_image(item, args.output, manifest)
        ]
        print(f"--resume: {len(items)} of {before} items have no generated images yet.")

    if args.needs_selection:
        selected_items = {e["item"] for e in manifest if e.get("selected")}
        items = [item for item in items if item not in selected_items]
        print(f"--needs-selection: {len(items)} items have no selection yet.")

    if not items:
        print("No items need image generation.")
        return

    already_done = {e["item"] for e in manifest}
    force_new    = bool(filter_items or filter_cats or args.needs_selection or args.resume)

    # Load RMBG once — stays in GPU memory for the whole batch
    rmbg_model, rmbg_device = None, None
    if not args.no_rmbg:
        rmbg_model, rmbg_device = load_rmbg_model(rembg_model_name)

    client_id = str(uuid.uuid4())
    ws_conn   = websocket.WebSocket()
    ws_conn.connect(f"ws://{COMFYUI_HOST}/ws?clientId={client_id}")
    print(f"Connected to ComfyUI at {COMFYUI_HOST}\n")

    total, completed, failed = len(items), 0, []

    for idx, item in enumerate(items, 1):
        if item in already_done and not force_new:
            print(f"[{idx}/{total}] Skipping '{item}' (already done)")
            completed += 1
            continue

        positive = get_prompt(item)
        negative = get_negative_prompt(item)
        slug     = slugify(item)
        print(f"[{idx}/{total}] {item}")

        # Reserve filenames against BOTH the manifest and the files on disk, so a
        # regenerate never reuses a slot whose file was overwritten or removed.
        # Reusing a slot produced two manifest entries pointing at one file, which
        # then selected and deleted together.
        used = {e.get("filename") for e in manifest}

        def next_free_slot():
            n = 1
            while n <= 999:
                fn = f"{slug}_{n:03d}.png"
                if fn not in used and not os.path.exists(os.path.join(args.output, fn)):
                    return n, fn
                n += 1
            return None, None

        images_per_item = getattr(args, "images_per_item", None) or IMAGES_PER_ITEM
        made = 0
        while made < images_per_item:
            made += 1
            variation, filename = next_free_slot()
            if variation is None:
                print(f"  ✗ No free filename slot up to {slug}_999.png, skipping")
                break
            used.add(filename)  # reserve now so the next variation picks a fresh slot
            seed     = random.randint(1, 2**32 - 1)
            filepath = os.path.join(args.output, filename)

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Step 1 — Generate
                    patched   = build_prompt(workflow, positive, negative, seed)
                    prompt_id = queue_prompt(patched, client_id)
                    raw_imgs  = get_images_for_prompt(ws_conn, prompt_id)

                    if not raw_imgs:
                        raise RuntimeError("No images returned from ComfyUI")

                    pil_img = Image.open(_io.BytesIO(raw_imgs[0]))

                    # Step 2 — Remove background
                    if rmbg_model is not None:
                        if args.keep_originals:
                            orig_path = os.path.join(
                                args.output, f"{slug}_{variation:03d}_original.png"
                            )
                            pil_img.save(orig_path)
                            print(f"  Original saved → {os.path.basename(orig_path)}")

                        print("  Removing background …", end=" ", flush=True)
                        pil_img = remove_background(rmbg_model, rmbg_device, pil_img)
                        print("done")

                    # Write atomically so an interrupt can't leave a truncated PNG.
                    tmp_img = filepath + ".tmp"
                    pil_img.save(tmp_img, format="PNG")
                    os.replace(tmp_img, filepath)
                    print(f"  ✓ {filename}  (seed={seed})")
                    success = True
                    break

                except Exception as e:
                    print(f"  ✗ Attempt {attempt}/{MAX_RETRIES}: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        try:
                            ws_conn.ping()
                        except Exception:
                            ws_conn = websocket.WebSocket()
                            ws_conn.connect(f"ws://{COMFYUI_HOST}/ws?clientId={client_id}")

            if success:
                manifest.append({
                    "item":         item,
                    "prompt":       positive,
                    "category":     item_category.get(item.strip().lower(), "default"),
                    "negative":     negative,
                    "seed":         seed,
                    "filename":     filename,
                    "transparent":  rmbg_model is not None,
                    "width":        WIDTH,
                    "height":       HEIGHT,
                    "steps":        STEPS,
                    "cfg":          CFG,
                    "sampler":      SAMPLER,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                # Atomic manifest write: a hard kill mid-write would otherwise
                # truncate manifest.json and break the whole profile's review.
                tmp_manifest = manifest_path + ".tmp"
                with open(tmp_manifest, "w") as mf:
                    json.dump(manifest, mf, indent=2)
                os.replace(tmp_manifest, manifest_path)
                completed += 1
            else:
                print(f"  ✗ FAILED — skipping '{item}'")
                failed.append(item)

    ws_conn.close()

    print("\n" + "─" * 50)
    print(f"Done!  {completed}/{total} images generated.")
    print(f"Manifest: {manifest_path}")
    if failed:
        failed_path = os.path.join(args.output, "failed.txt")
        with open(failed_path, "w") as ff:
            ff.write("\n".join(failed) + "\n")
        print(f"Failed ({len(failed)}): {failed_path}  →  retry with --items failed.txt")


if __name__ == "__main__":
    main()
