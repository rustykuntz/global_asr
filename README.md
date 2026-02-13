# Voice Input Runtime

Cross-platform desktop voice input with two interaction modes and selectable transcription backend.

## Features
- `MANUAL` mode (default): push-to-record and transcribe on demand.
- `AUTO` mode: VAD-driven automatic segmentation with context validation before insertion.
- STT backend selection:
  - Local Whisper Turbo (MLX path, best on macOS)
  - OpenAI Audio Transcriptions API
- Global hotkeys:
  - `F6`: switch mode (`AUTO` / `MANUAL`)
  - `F4` in `MANUAL`: start/stop recording
  - `F4` in `AUTO`: toggle auto listening

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

## Testing Checklist
1. Launch app and confirm startup mode is `MANUAL`.
2. Press `F4` to record/stop and verify text insertion.
3. Press `F6` to switch to `AUTO` and verify mode overlay/state.
4. In `AUTO`, confirm insertion works in editable fields and is blocked in invalid contexts.
5. If using OpenAI backend, verify transcription succeeds with your API key.

## Troubleshooting
- `OPENAI_API_KEY is required`:
  - set key in `.env` or rerun `setup_asr.py`
- `AUTO mode unavailable` on Windows:
  - install dependency: `pip install uiautomation`
- local backend import/load issues:
  - run `python setup_asr.py` again and select local backend
- no microphone input:
  - verify OS microphone permission and input device selection

## Security
- `.env` contains secrets and should not be committed.
- Rotate API keys if exposed.
