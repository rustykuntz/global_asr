# global_asr

`global_asr` is a small runtime for speaking directly into app inputs. You press a key, talk, and the transcription is inserted right where your cursor already is: Terminal, Cursor, WhatsApp, Google Docs, chat boxes, editors, wherever. The point is not to make another note-taking app or another voice recorder. The point is to make voice-to-text feel native inside the software you already use.

It started as a tool for talking to coding agents and other LLM workflows, but it generalizes well to anything with a text field. In `MANUAL` mode it behaves like push-to-talk dictation. In `AUTO` mode it listens with VAD and only inserts when the focused UI context looks safe. On macOS it can also duck system playback while recording, so you do not have to manually pause music or YouTube every time you want to speak. There is also a lightweight replacements system for recurring terms, names, and code words that speech models like to mangle.

## Why this is useful
- Dictate straight into the current app instead of recording somewhere else first.
- Keep one hotkey flow across terminals, editors, docs, messengers, and browser text boxes.
- Choose between explicit start/stop control (`MANUAL`) and hands-free speech segmentation (`AUTO`).
- Reduce bad inserts in `AUTO` mode with focus validation, confidence checks, and context gating.
- Smoothly duck playback audio during recording on macOS.
- Teach the system your recurring corrections with custom replacements.

## What It Does
- Records from your microphone.
- Transcribes locally or through OpenAI.
- Types the result into the focused app.
- Can send the message directly in some apps when appropriate.
- Can keep listening continuously in `AUTO` mode.

## Modes
- `MANUAL` mode (default):
  - Press `F4` to start recording.
  - Press `F4` again to stop, transcribe, and insert.
  - No app/field restrictions by design. You are driving.

- `AUTO` mode:
  - Uses VAD to segment speech automatically.
  - Validates focused UI context before insertion.
  - Applies blocking rules to reduce false positives:
    - blocks disallowed apps
    - allows only supported text-input roles (or trusted app exceptions)
    - aborts when focus changes during/after speech capture
    - drops low-energy / low-confidence / garbage transcriptions

## Hotkeys
- `F6`: switch mode (`AUTO` / `MANUAL`)
- `F4` in `MANUAL`: start/stop recording
- `F4` in `AUTO`: toggle auto listening ON/OFF
- `ESC` in `MANUAL`: cancel current recording

Hotkeys are configurable in `.env`:
- `ASR_ACTION_KEY` (default: `f4`)
- `ASR_MODE_KEY` (default: `f6`)
- `ASR_CANCEL_KEY` (default: `esc`)

Accepted key values:
- any `pynput.keyboard.Key` name such as `f1`-`f20`, `esc`, `tab`, `enter`, `space`
- a single character such as `a`, `/`, `;`

## STT Backends
- Local backend:
  - macOS: Whisper Turbo (MLX)
  - Windows: faster-whisper
  - Linux: whisper.cpp
- OpenAI Audio Transcriptions API

## Repository Layout
- `global_asr.py`: main runtime
- `setup_asr.py`: interactive setup flow
- `requirements.txt`: Python dependencies
- `overlay.py`: macOS overlay UI
- `get_focus`: macOS focus detector used by `AUTO`
- `whisper-turbo-mlx/`: local backend runtime files
- `whisper.cpp/`: Linux local backend checkout created by setup when needed

## Requirements
- Python 3.9+
- Microphone access
- Global keyboard event access (OS permission)

Platform notes:
- macOS:
  - `AUTO` mode uses `get_focus`
  - local backend requires MLX stack and typically `ffmpeg`
- Windows:
  - `AUTO` mode requires `uiautomation`
  - local backend uses `faster-whisper`
  - `MANUAL` mode works without UI focus integration
- Linux:
  - local backend uses `whisper.cpp`
  - setup requires `libportaudio2`, `git`, `cmake`, and a C/C++ build toolchain
  - Ubuntu/Debian packages: `sudo apt install -y libportaudio2 git cmake build-essential`
  - if `nvcc` is available, setup builds `whisper.cpp` with CUDA (`GGML_CUDA=ON`)
  - if `sounddevice` must rebuild locally, also install `portaudio19-dev`
  - `MANUAL` mode works without UI focus integration

## Quick Start
```bash
python setup_asr.py
python global_asr.py
```

## Setup Flow
`setup_asr.py` will:
1. Install dependencies from `requirements.txt`.
2. Ask you to choose STT backend (`local` or `openai`).
3. If `openai` is selected, prompt for `OPENAI_API_KEY`.
4. If `local` is selected, optionally prepare the local model/runtime (macOS MLX, Windows faster-whisper, or Linux whisper.cpp).
5. Save configuration to `.env`.

## Run Options
```bash
python global_asr.py [options]
```

Options:
- `--stt-backend {local,openai}`
- `--lang LANG` (default: `auto`)
- `--openai-model OPENAI_MODEL` (default: `whisper-1`)
- `--openai-prompt OPENAI_PROMPT`
- `--silence-wait {normal,long}` (default: `normal`)
- `--duck-output-audio` / `--no-duck-output-audio` (macOS only; default: on)
- `--duck-output-volume VOLUME` (macOS only; default: `0`)
- `--duck-fade-ms MS` (macOS only; default: `180`)
- `--context` (enable context engine in `AUTO` mode)

Language examples:
```bash
# Auto-detect language (default)
python global_asr.py --lang auto

# Start with English
python global_asr.py --lang en

# Start with English and longer AUTO silence wait
python global_asr.py --lang en --silence-wait long

# Force Spanish with OpenAI backend
python global_asr.py --stt-backend openai --lang es
```

## Configuration
Configuration is read from `.env` in this folder.
Use `.env.example` as a starting point.

Common keys:
- `ASR_ACTION_KEY=f4`
- `ASR_MODE_KEY=f6`
- `ASR_CANCEL_KEY=esc`
- `STT_BACKEND=local|openai`
- `OPENAI_API_KEY=...`
- `OPENAI_WHISPER_MODEL=whisper-1`
- `OPENAI_WHISPER_PROMPT=...`
- `ASR_SILENCE_WAIT=normal|long`
- `FASTER_WHISPER_MODEL=large-v3-turbo` (optional, Windows local backend)
- `FASTER_WHISPER_DEVICE=auto` (optional, Windows local backend)
- `FASTER_WHISPER_COMPUTE_TYPE=int8` (optional, Windows local backend)
- `WHISPER_CPP_DIR=whisper.cpp` (optional, Linux local backend)
- `WHISPER_CPP_MODEL=large-v3-turbo` (optional, Linux local backend)
- `WHISPER_CPP_MODEL_PATH=whisper.cpp/models/ggml-large-v3-turbo.bin` (optional, Linux local backend)
- `WHISPER_CPP_BINARY=whisper.cpp/build/bin/whisper-cli` (optional, Linux local backend)
- `WHISPER_CPP_DEVICE=0` (optional, Linux CUDA device)
- `WHISPER_CPP_THREADS=8` (optional, Linux local backend)
- `WHISPER_CPP_BEAM_SIZE=1` (optional, Linux local backend)
- `WHISPER_CPP_BEST_OF=1` (optional, Linux local backend)
- `ASR_REPLACEMENTS_FILE=transcription_replacements.txt` (optional)
- `ASR_DUCK_OUTPUT_AUDIO=1` (macOS only; lower system output volume during manual recording)
- `ASR_DUCK_OUTPUT_VOLUME=0` (macOS only; volume percent while recording)
- `ASR_DUCK_FADE_MS=180` (macOS only; fade duration for lowering/restoring output audio)
- `VAD_*` and `ASR_*` thresholds

## Custom Word Replacements
- Purpose: fix recurring Whisper mis-transcriptions after transcription and before text insertion.
- Execution point: replacements are applied after transcript cleanup and right before `keyboard_controller.type(...)`.
- Default file: `transcription_replacements.txt` in the project root.
- The repo ships a starter `transcription_replacements.txt` with a few examples.
- Optional override: set `ASR_REPLACEMENTS_FILE` in `.env`.

Format:
- one rule per line
- `wrong term => correct term` (default `mode=exact`)
- optional mode: `wrong term => correct term | mode=all`
- lines starting with `#` are comments
- `mode=exact`: case-sensitive source matching
- `mode=all` / `mode=match_all`: case-insensitive source matching
- replacement text is always inserted exactly as written in the rule target
- optional wrapping quotes are allowed (example: `'Rossie' => 'Rocie'`)
- spaces inside a source phrase are whitespace-tolerant (` `, tabs, newlines)

Example:
```txt
Global ASA => global_asr | mode=all
skill md => SKILL.md | mode=all
toolmd => TOOL.md | mode=all
```

## Input Device Behavior
- Uses the OS default input device when available.
- If no default input device is set, picks the first valid microphone.
- Re-checks for device changes while running and reopens the stream automatically.
- Also reopens on audio stream errors (for example, unplugged device).
- Clears stale buffered audio when switching streams.

## Troubleshooting
- `OPENAI_API_KEY is required`:
  - set key in `.env` or rerun `setup_asr.py`
- `AUTO mode unavailable` on Windows:
  - install dependency: `pip install uiautomation`
- local backend import/load issues:
  - run `python setup_asr.py` again and select local backend
  - on Windows, verify `faster-whisper` installed in the project venv
  - on Linux, verify `whisper.cpp/build/bin/whisper-cli` and `whisper.cpp/models/ggml-large-v3-turbo.bin` exist
- no microphone input:
  - verify OS microphone permission and input device selection
