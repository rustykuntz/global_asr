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
  - Press `F4` again, or `Enter`, to stop, transcribe, and insert.
  - No app/field restrictions by design. You are driving.

- `AUTO` mode:
  - Uses VAD to segment speech automatically.
  - Validates focused UI context before insertion.
  - Applies blocking rules to reduce false positives:
    - blocks disallowed apps
    - allows only supported text-input roles (or trusted app exceptions)
    - aborts when focus changes during/after speech capture
    - drops low-energy / low-confidence / garbage transcriptions

- `OFF` mode:
  - Closes the microphone and ignores the action key.
  - Press `F6` again to return to `MANUAL`.

## Hotkeys
- `F6`: cycle mode (`MANUAL` -> `AUTO` -> `OFF`)
- `F4` in `MANUAL`: start/stop recording
- `Enter` in `MANUAL`: stop current recording
- `F4` in `AUTO`: toggle auto listening ON/OFF
- `ESC` in `MANUAL`: cancel current recording

Hotkeys are configurable in `.env`:
- `ASR_ACTION_KEY` (default: `f4`)
- `ASR_MODE_KEY` (default: `f6`)
- `ASR_STOP_KEY` (default: `enter`)
- `ASR_CANCEL_KEY` (default: `esc`)

Accepted key values:
- key names such as `f1`-`f20`, `esc`, `tab`, `enter`, `space`
- a single character such as `a`, `/`, `;`

## STT Backends
- Local backend:
  - macOS: Whisper Turbo (MLX)
  - Windows: faster-whisper
  - Linux: faster-whisper with prebuilt CTranslate2 CPU/CUDA wheels
- OpenAI Audio Transcriptions API

## Repository Layout
- `global_asr.py`: main runtime
- `setup_asr.py`: interactive setup flow
- `requirements.txt`: Python dependencies
- `requirements-linux-cuda.txt`: prebuilt cuBLAS/cuDNN runtime packages for NVIDIA Linux systems
- `overlay.py`: macOS overlay UI
- `get_focus`: macOS focus detector used by `AUTO`
- `whisper-turbo-mlx/`: local backend runtime files

## Requirements
- Python 3.9+
- Microphone access
- Global keyboard event access (OS permission)

Platform notes:
- macOS:
  - `AUTO` mode uses `get_focus`
  - local backend requires MLX stack and typically `ffmpeg`
- Linux:
  - install `python3-venv` or the versioned package such as `python3.14-venv`
  - setup installs `libportaudio2`, matching Python development headers, build tools, and `wl-clipboard` when needed
  - Wayland global keys use `evdev` and `/dev/uinput`; setup installs a udev rule and adds your user to the `input` group after an explicit prompt
  - sign out and back in once after Wayland input permissions are configured
  - this input-group access can read keyboard events and inject key events, which is why setup does not enable it silently
  - `AUTO` focus validation is not supported; `F6` cycles `MANUAL` -> `OFF` -> `MANUAL`
  - local transcription uses `large-v3-turbo` through faster-whisper and prebuilt CTranslate2 wheels; no CMake or local compilation is required
  - if `nvidia-smi -L` lists a GPU, setup installs prebuilt CUDA 12 cuBLAS/cuDNN packages and verifies that CTranslate2 can see the GPU
  - CUDA uses FP16 by default and refuses to silently fall back to CPU
  - overlays automatically follow desktop/monitor DPI; set `ASR_OVERLAY_SCALE=1.5` in `.env` to override detection
  - install `portaudio19-dev` only if `sounddevice` itself must build locally
  - install `rustc cargo` only if pip has to build a Rust-based package from source
- Windows:
  - `AUTO` mode requires `uiautomation`
  - local backend uses `faster-whisper`
  - `MANUAL` mode works without UI focus integration

## Quick Start
```bash
python setup_asr.py
python global_asr.py
```

## Setup Flow
`setup_asr.py` will:
1. On Debian/Ubuntu, detect missing system packages and offer to install them.
2. On Wayland, offer to configure the input-device permissions required by global hotkeys.
3. Install dependencies from `requirements.txt`.
4. Ask you to choose STT backend (`local` or `openai`).
5. If `openai` is selected, prompt for `OPENAI_API_KEY`.
6. If `local` is selected, optionally prepare the local model/runtime (macOS MLX or Windows/Linux faster-whisper).
7. Save configuration to `.env`.

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
- `ASR_STOP_KEY=enter`
- `ASR_CANCEL_KEY=esc`
- `STT_BACKEND=local|openai`
- `OPENAI_API_KEY=...`
- `OPENAI_WHISPER_MODEL=whisper-1`
- `OPENAI_WHISPER_PROMPT=...`
- `ASR_SILENCE_WAIT=normal|long`
- `FASTER_WHISPER_MODEL=large-v3-turbo` (Windows/Linux local backend)
- `FASTER_WHISPER_DEVICE=cuda` (Linux NVIDIA default; use `cpu` when needed)
- `FASTER_WHISPER_COMPUTE_TYPE=float16` (Linux NVIDIA default; CPU/Windows default is `int8`)
- `FASTER_WHISPER_BEAM_SIZE=1`
- `FASTER_WHISPER_BEST_OF=1`
- `FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT=0`
- `FASTER_WHISPER_WITHOUT_TIMESTAMPS=1`
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
- `ensurepip is not available` while creating `.venv` on Ubuntu/Debian:
  - install the venv package for your Python, for example `sudo apt install -y python3.14-venv`
  - if the versioned package is unavailable, try `sudo apt install -y python3-venv`
  - rerun `python3 setup_asr.py`
- `can't find Rust compiler` while installing `tiktoken`:
  - rerun setup after pulling the latest requirements; newer `tiktoken` has Python 3.14 wheels
  - upgrade pip inside the venv: `.venv/bin/python -m pip install --upgrade pip setuptools wheel`
  - if pip still builds from source, install Rust: `sudo apt install -y rustc cargo`
- `Python.h: No such file or directory` while installing `evdev`:
  - rerun `python3 setup_asr.py`; setup will offer to install the matching Python development package
  - for Python 3.12, the manual command is `sudo apt install -y python3.12-dev`
- `PortAudio library not found` while starting Global ASR:
  - rerun `python3 setup_asr.py`; setup will offer to install `libportaudio2`
  - the manual command is `sudo apt install -y libportaudio2`
- F4/F6 do not respond on Ubuntu Wayland:
  - rerun `python3 setup_asr.py` and accept Wayland keyboard permission setup
  - sign out of Ubuntu and sign back in so the new `input` group membership becomes active
  - verify access with `test -r /dev/input/event0 && test -w /dev/uinput && echo ready`
- Linux local transcription cannot initialize CUDA:
  - run `nvidia-smi -L`; any listed GPU makes setup select the CUDA lane
  - rerun `python3 setup_asr.py`; it reinstalls the prebuilt cuBLAS/cuDNN runtime and verifies CTranslate2 before completing
  - `nvcc`, CMake, and a local whisper.cpp build are not used
- `OPENAI_API_KEY is required`:
  - set key in `.env` or rerun `setup_asr.py`
- `AUTO mode unavailable` on Windows:
  - install dependency: `pip install uiautomation`
- local backend import/load issues:
  - run `python setup_asr.py` again and select local backend
  - on Windows/Linux, verify `faster-whisper` and `ctranslate2` are installed in the project venv
- no microphone input:
  - verify OS microphone permission and input device selection
