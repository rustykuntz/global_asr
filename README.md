# Global ASR Runtime

This folder contains a production-focused runtime with only required files.

## Included
- `global_asr.py` (single ASR app)
- `setup_asr.py` (interactive setup)
- `requirements.txt`
- `overlay.py` (macOS visual overlays)
- `get_focus` (macOS focus detector for Auto mode)
- `whisper-turbo-mlx/` runtime files for local backend

## Setup
```bash
cd asr_runtime
python setup_asr.py
```

## Run
```bash
cd asr_runtime
python global_asr.py
```

## Notes
- Default mode is `MANUAL`.
- `AUTO` mode requires focus detection:
  - macOS: included via `get_focus`
  - Windows: install `uiautomation` (already in requirements with Windows marker)
- Local backend on macOS uses MLX and may require `ffmpeg` installed on system.
# global_asr
