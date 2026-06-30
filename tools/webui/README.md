# Wordbank Web UI

Local browser UI for basic Wordbank maintenance.

Run:

```bash
python tools/webui/app.py --port 5050
```

Then open:

```text
http://127.0.0.1:5050
```

## Phase 1 Coverage

Implemented:

- corpus selection from `wordlists/*.yml` (default `children-001`)
- dashboard validation for the selected corpus, optional app `wordlist.yml`, and selected languages
- corpus readiness table
- image review over an existing image staging `manifest.json`
- image selection updates
- image extraction into `images/<key>.png`
- audio review over an existing audio staging `manifest.json`
- audio status updates
- audio extraction into `audio/<lang>/<key>.wav`
- `.wav` and `.ogg` approved audio recognition
- package audio output as WAV or Ogg Opus
- package creation through `generate_db.py`
- image generation jobs through `tools/itemimages/generate_images.py`
- audio candidate/synthesis jobs through `tools/itemaudio/generate_candidates.py`
- job list, job logs, SSE log streaming, progress summaries, cancellation, and image failed-item retry
- persistent local UI settings in `tools/webui/state.json`
- corpus filters, item detail view, and per-key generate shortcuts
- copyable equivalent CLI commands
- approved-image comparison during image review
- exportable maintenance report

Not implemented yet:

- corpus editing

## Recommendations and Decisions

- Use Flask and plain static HTML/JavaScript for now. This matches the existing review tools and avoids adding a build system before the UI needs one.
- Keep existing CLI tools as the source of behavior. The UI wraps `generate_db.py` and directly uses the same manifest formats as the image/audio review scripts.
- Bind to `127.0.0.1` by default. This is intended as a trusted local maintenance tool, not a network service.
- Keep approved media overwrites explicit. Extraction fails when `images/<key>.png` or `audio/<lang>/<key>.wav` already exists unless the UI sends the overwrite flag.
- Allow word-list paths outside the Wordbank repo. A downstream app may own its `word-list.yml`, so paths outside the repo are accepted.
- Allow package output paths outside the repo. The documented workflow uses `/tmp/...`, and package outputs are generated artifacts rather than corpus source files.
- Persist only local workflow settings in `tools/webui/state.json`; do not store credentials or generated reports there.
- Stream package job logs with Server-Sent Events. SSE is simpler than WebSockets and is enough for append-only command output.
- Keep review manifests in place after extraction. They remain useful review history and approval evidence.
- Run generation scripts with unbuffered Python (`-u`) so progress appears while a job is running.
- Keep generation command construction explicit in the backend. The browser sends form options; the server maps them to known script flags.
- Default image generation to `tools/itemimages/items.yml`, but require the file to exist. This keeps setup ownership with `tools/itemimages/setup.py --switch SET` and avoids silently inventing prompts.
- Treat ComfyUI reachability as a health check, not a hard preflight. Image generation can auto-start ComfyUI when a directory is provided, so the host may be unreachable before the job starts.
- Limit retry automation to image jobs with `failed.txt`. Audio candidate generation either succeeds in writing a manifest or fails as a command; finer-grained audio retry can be added once synthesis logs expose stable per-clip failure data.
- Keep corpus editing out of Phase 3. The UI is now useful for inspection and workflow control, while schema-preserving YAML edits remain better handled deliberately in a separate pass.
- Export reports as generated Markdown text copied to the clipboard. This avoids inventing a report storage lifecycle while still making review/package state easy to share.

## Default Paths

- image staging: `tools/itemimages/output`
- image items YAML: `tools/itemimages/items.yml`
- image workflow: `tools/itemimages/workflow_api.json`
- ComfyUI directory: `tools/itemimages/ComfyUI`
- audio staging: `tools/itemaudio/output_audio`
- approved images: `images/`
- approved audio: `audio/<lang>/` with `.wav` or `.ogg`
- local UI state: `tools/webui/state.json`

## Equivalent Package Command

The package form runs the equivalent of:

```bash
python generate_db.py \
  --wordlist PATH_TO_WORDLIST \
  --langs en,de,fr,es \
  --out /tmp/wordbank-assets \
  --format assets \
  --audio-format ogg
```

Optional toggles add:

- `--clean`
- `--fail-on-drop`
- `--quiet-drops`

Audio package formats:

- `wav` copies approved WAV files when available
- `ogg` writes Ogg Opus files, converting WAV sources with `ffmpeg` when needed

## Equivalent Generation Commands

Image generation runs the equivalent of:

```bash
python -u tools/itemimages/generate_images.py \
  --items tools/itemimages/items.yml \
  --workflow tools/itemimages/workflow_api.json \
  --output tools/itemimages/output \
  --host 127.0.0.1:8188
```

Audio generation runs the equivalent of:

```bash
python -u tools/itemaudio/generate_candidates.py \
  --wordlist PATH_TO_WORDLIST \
  --langs en,de,fr,es \
  --takes 1 \
  --output tools/itemaudio/output_audio
```

The audio form adds `--synthesize` when requested.
