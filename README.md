# global_asr

`global_asr` is a cross-platform voice input runtime focused on dictation for CLI agents and other LLM workflows. It supports fast manual push-to-talk dictation and safer automatic dictation with context gating.

## Why this exists
- Dictate directly into coding assistants, terminals, editors, and chat inputs.
- Switch between explicit control (`MANUAL`) and hands-free capture (`AUTO`).
- Reduce accidental inserts with validation in `AUTO` mode.

## Modes
- `MANUAL` mode (default):
  - Press `F4` to start recording.
  - Press `F4` again to stop, transcribe, and insert.
  - No app/field restrictions by design; user is in control.

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

## STT Backends
- Local Whisper Turbo (best on macOS with MLX)
- OpenAI Audio Transcriptions API

## Repository Layout
- `global_asr.py`: main runtime
- `setup_asr.py`: interactive setup flow
- `requirements.txt`: Python dependencies
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
- Windows:
  - `AUTO` mode requires `uiautomation`
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
4. If `local` is selected on macOS, optionally warm/download local model.
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
- `--context` (enable context engine in `AUTO` mode)

## Configuration
Configuration is read from `.env` in this folder.

Common keys:
- `STT_BACKEND=local|openai`
- `OPENAI_API_KEY=...`
- `OPENAI_WHISPER_MODEL=whisper-1`
- `OPENAI_WHISPER_PROMPT=...`
- `VAD_*` and `ASR_*` thresholds

## Troubleshooting
- `OPENAI_API_KEY is required`:
  - set key in `.env` or rerun `setup_asr.py`
- `AUTO mode unavailable` on Windows:
  - install dependency: `pip install uiautomation`
- local backend import/load issues:
  - run `python setup_asr.py` again and select local backend
- no microphone input:
  - verify OS microphone permission and input device selection

