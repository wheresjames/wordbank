# Coqui YAML TTS test package

## Files

- `coqui_local_tts.py` - main script
- `config.yml` - global model/runtime config
- `phrases.yml` - phrase list with per-item language tags

## Usage

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip espeak-ng ffmpeg

chmod +x coqui_local_tts.py

./coqui_local_tts.py --setup
COQUI_PYTHON=python3.11 ./coqui_local_tts.py --download
./coqui_local_tts.py --process phrases.yml --output ./audio
./coqui_local_tts.py --translations translations.yml --output ./audio
```

Coqui TTS currently needs Python `>=3.9, <3.12`. If `python3` is Python 3.12
or newer on your system, set `COQUI_PYTHON` to a compatible interpreter when
running `--download`.

The current `TTS` release also needs `transformers` on the 4.x line for XTTS
model loading. The script pins that automatically during `--download`.

On Kali Rolling and other systems where `apt` does not provide `python3.11`,
use `uv` instead:

```bash
uv python install 3.11
rm -rf ./coqui_local/.venv
COQUI_PYTHON="$(uv python find 3.11)" ./coqui_local_tts.py --download
```

## XTTS speaker sample

For XTTS-v2, provide a short clean WAV speaker sample in `config.yml`:

```yaml
default_speaker_wav: ./speaker.wav
```

Without either `speaker_wav` or `speaker`, multi-speaker models such as
`xtts_v2` will not synthesize audio.

## Built-in voice

If you do not want voice cloning, use single-speaker models instead of XTTS.
For mixed-language phrase files, route each language to a model:

```yaml
model_name: tts_models/en/ljspeech/vits
language_models:
  en: tts_models/en/ljspeech/vits
  de: tts_models/de/thorsten/vits
  fr: tts_models/fr/css10/vits
  es: tts_models/es/css10/vits
  it: tts_models/it/mai_female/vits
  ja: tts_models/ja/kokoro/tacotron2-DDC
  zh-cn: tts_models/zh-CN/baker/tacotron2-DDC-GST
skip_unsupported_languages: true
```

Single-speaker models have built-in voices and do not need `speaker_wav`.
Coqui does not currently list a Korean single-language model in this install,
so `ko` items are skipped when `skip_unsupported_languages` is true. Use a
different engine/model or XTTS with a speaker reference for Korean.

## Translations

`translations.yml` can contain English keys with translated values. The
translation command writes each supported language to its own folder and keeps
English filenames:

```bash
COQUI_PYTHON="$(uv python find 3.11)" ./coqui_local_tts.py --translations translations.yml --output ./audio
```

Example output paths:

```text
audio/en/cat.wav
audio/de/cat.wav
audio/fr/cat.wav
```

Unsupported languages are skipped. Chinese keys (`cn`, `zh`, `zh-cn`) are
ignored because the available Coqui Chinese model is low quality in this setup.

You can also override it per phrase item:

```yaml
items:
  - text: Guten Tag
    language: de
    speaker_wav: ./speaker_de.wav
```


## Wordbank candidate and review flow

From the Wordbank root, generate a candidate manifest and Coqui phrase file:

```bash
python tools/itemaudio/generate_candidates.py \
  --langs en,de,fr,es \
  --takes 2 \
  --output tools/itemaudio/output_audio
```

Add `--wordlist path/to/word-list.yml` to limit generation to a subset of keys.

Use `--format ogg` to stage Ogg candidates instead of WAV. Package generation can
also convert approved WAV corpus audio to Ogg Opus with `generate_db.py
--audio-format ogg`.

The generated spoken text comes from the selected corpus: English uses the bare
key with underscores converted to spaces, while `de`, `fr`, and `es` compose
`art + tr`. Elided articles such as `l'` are joined without an extra space.

Run synthesis with `--synthesize`, then review and approve clips in the Wordbank
web UI (Audio tab), which reads/writes `output/audio`:

```bash
python tools/itemaudio/generate_candidates.py --langs en,de,fr,es --takes 2 --output output/audio --synthesize
python tools/webui/app.py   # review + approve under the Audio tab
```
