import argparse
import collections
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from pynput.keyboard import Controller, Key, Listener


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))
GET_FOCUS_BIN = os.path.join(_SCRIPT_DIR, "get_focus")
OVERLAY_PY = os.path.join(_SCRIPT_DIR, "overlay.py")
WHISPER_TURBO_DIR = os.path.join(_SCRIPT_DIR, "whisper-turbo-mlx")
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform.startswith("win")


# Audio and ASR config
SAMPLE_RATE = 16000
VAD_POSITIVE_THRESHOLD = float(os.getenv("VAD_POSITIVE_THRESHOLD", 0.3))
VAD_NEGATIVE_THRESHOLD = float(os.getenv("VAD_NEGATIVE_THRESHOLD", 0.25))
MIN_SPEECH_DURATION_MS = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", 400))
MIN_SILENCE_DURATION_MS = int(os.getenv("VAD_REDEMPTION_DURATION_MS", 1000))
SPEECH_PAD_MS = int(os.getenv("VAD_PRE_SPEECH_PADDING_MS", 800))

ASR_DICTATION_THRESHOLD = float(os.getenv("ASR_DICTATION_THRESHOLD", -0.12))
ASR_COMMAND_THRESHOLD = float(os.getenv("ASR_COMMAND_THRESHOLD", -0.2))
ASR_ENERGY_THRESHOLD = float(os.getenv("ASR_ENERGY_THRESHOLD", -45.6))
TOOL_INSTRUCTION_DURATION = float(os.getenv("TOOL_INSTRUCTION_DURATION", 2.5))
ASR_SEND_WINDOW_DURATION_S = float(os.getenv("ASR_SEND_WINDOW_DURATION_S", 5.0))

# Context filters (auto mode only)
DISAPPROVED_APPS = ["Calculator", "System Settings", "Finder"]
ALLOWED_ROLES = ["AXTextArea", "AXTextField", "AXComboBox"]
WINDOWS_ALLOWED_ROLES = ["EditControl", "DocumentControl", "ComboBoxControl"]
TRUSTED_APPS_WITHOUT_ROLE = os.getenv(
    "TRUSTED_APPS_WITHOUT_ROLE", "Code,Terminal,iTerm2,Cursor,Hyper"
).split(",")
CMD_ENTER_APPS = os.getenv("CMD_ENTER_APPS", "Code,Cursor,Google Chrome,Arc").split(",")

# Keybindings
ACTION_KEY = Key.f4
MODE_KEY = Key.f6

# Runtime state
keyboard_controller = Controller()
processing_lock = threading.Lock()

MODE = "manual"  # manual | auto
AUTO_ASR_ENABLED = True

SESSION_COUNTER = 0
last_paste_time = 0.0

SELECTED_LANGUAGE = "auto"
STT_BACKEND = "local"  # local | openai
OPENAI_MODEL = "whisper-1"
OPENAI_PROMPT = ""

# Optional engine in auto mode
context_engine = None

# Audio runtime objects
vad_audio = None
manual_recording_overlay_proc = None

# Lazy-loaded backend/runtime dependencies
_torch = None
_vad_model = None
_local_transcribe = None
_local_load_model = None
_openai_client = None
_win_uia = None
_win_uia_import_error_logged = False


def show_overlay_text(text, color="green", duration=None):
    if not IS_MAC or not os.path.exists(OVERLAY_PY):
        return None
    try:
        cmd = [
            sys.executable,
            OVERLAY_PY,
            "--text",
            text,
            "--color",
            color,
        ]
        if duration is not None:
            cmd.extend(["--duration", str(duration)])
        return subprocess.Popen(cmd)
    except Exception:
        return None


def show_overlay_success():
    if not IS_MAC or not os.path.exists(OVERLAY_PY):
        return
    try:
        subprocess.Popen([sys.executable, OVERLAY_PY, "--success"])
    except Exception:
        pass


def start_manual_recording_overlay():
    global manual_recording_overlay_proc
    stop_manual_recording_overlay()
    manual_recording_overlay_proc = show_overlay_text("REC ●", "red", duration=86400)


def stop_manual_recording_overlay():
    global manual_recording_overlay_proc
    if manual_recording_overlay_proc is None:
        return
    if manual_recording_overlay_proc.poll() is None:
        manual_recording_overlay_proc.terminate()
        try:
            manual_recording_overlay_proc.wait(timeout=0.2)
        except Exception:
            manual_recording_overlay_proc.kill()
    manual_recording_overlay_proc = None


def ensure_vad_model():
    global _torch, _vad_model
    if _vad_model is not None:
        return True
    try:
        import torch as torch_module

        _torch = torch_module
    except ImportError:
        print("Error: torch is required for AUTO mode VAD. Install: pip install torch torchaudio")
        return False

    print("Loading Silero VAD...")
    _vad_model, _ = _torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    print("Silero VAD loaded.")
    return True


def ensure_local_backend():
    global _local_transcribe, _local_load_model
    if _local_transcribe is not None and _local_load_model is not None:
        return True

    if not os.path.isdir(WHISPER_TURBO_DIR):
        print(f"Error: local backend not found: {WHISPER_TURBO_DIR}")
        return False

    if WHISPER_TURBO_DIR not in sys.path:
        sys.path.append(WHISPER_TURBO_DIR)

    try:
        from whisper_turbo import load_model as local_load_model, transcribe as local_transcribe
    except ImportError as e:
        print(f"Error importing local backend whisper_turbo: {e}")
        return False

    _local_transcribe = local_transcribe
    _local_load_model = local_load_model
    return True


def ensure_openai_backend():
    global _openai_client
    if _openai_client is not None:
        return True

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAI_API_KEY is required for --stt-backend openai")
        return False

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package is required for --stt-backend openai. Install: pip install openai")
        return False

    _openai_client = OpenAI(api_key=api_key)
    return True


def transcribe_audio_local(audio_data):
    if not ensure_local_backend():
        return None

    _local_load_model()
    result = _local_transcribe(path_audio=audio_data, lang=SELECTED_LANGUAGE)

    text = str(result.get("text", "")).strip()
    avg_logprob = result.get("avg_logprob")
    detected_lang = result.get("language", "unknown")
    return {
        "text": text,
        "avg_logprob": avg_logprob,
        "language": detected_lang,
    }


def write_temp_wav(audio_data):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    pcm = np.clip(audio_data, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    return path


def transcribe_audio_openai(audio_data):
    if not ensure_openai_backend():
        return None

    tmp_path = write_temp_wav(audio_data)
    try:
        with open(tmp_path, "rb") as audio_file:
            kwargs = {
                "model": OPENAI_MODEL,
                "file": audio_file,
                "response_format": "verbose_json",
            }
            if SELECTED_LANGUAGE and SELECTED_LANGUAGE != "auto":
                kwargs["language"] = SELECTED_LANGUAGE
            if OPENAI_PROMPT:
                kwargs["prompt"] = OPENAI_PROMPT

            response = _openai_client.audio.transcriptions.create(**kwargs)

        text = ""
        language = "unknown"

        if isinstance(response, str):
            text = response.strip()
        else:
            text = getattr(response, "text", "")
            language = getattr(response, "language", "unknown")
            if not text and isinstance(response, dict):
                text = str(response.get("text", "")).strip()
                language = response.get("language", "unknown")

        return {
            "text": str(text).strip(),
            "avg_logprob": None,
            "language": language,
        }
    except Exception as e:
        print(f"OpenAI transcription failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def transcribe_audio(audio_data):
    if STT_BACKEND == "openai":
        return transcribe_audio_openai(audio_data)
    return transcribe_audio_local(audio_data)


def _get_windows_process_name(pid):
    if not pid:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return f"pid-{pid}"
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            if ok:
                exe = os.path.basename(buf.value)
                if exe.lower().endswith(".exe"):
                    exe = exe[:-4]
                return exe
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        pass
    return f"pid-{pid}"


def ensure_windows_focus_backend():
    global _win_uia, _win_uia_import_error_logged
    if not IS_WINDOWS:
        return False
    if _win_uia is not None:
        return True

    try:
        import uiautomation as uia
    except ImportError:
        if not _win_uia_import_error_logged:
            print("AUTO mode on Windows requires UI Automation package: pip install uiautomation")
            _win_uia_import_error_logged = True
        return False

    _win_uia = uia
    return True


def get_focused_element_windows():
    if not ensure_windows_focus_backend():
        return None, None, None, None, None, None

    try:
        control = _win_uia.GetFocusedControl()
        if control is None:
            return None, None, None, None, None, None

        pid = getattr(control, "ProcessId", 0)
        app_name = _get_windows_process_name(pid)
        role = getattr(control, "ControlTypeName", None) or "Unknown"
        subrole = getattr(control, "ClassName", None) or "None"
        ax_desc = getattr(control, "Name", None) or "None"

        title = "Unknown"
        try:
            top = control.GetTopLevelControl()
            top_name = getattr(top, "Name", None) if top is not None else None
            if top_name:
                title = top_name
            elif ax_desc and ax_desc != "None":
                title = ax_desc
        except Exception:
            if ax_desc and ax_desc != "None":
                title = ax_desc

        return app_name, role, title, "None", subrole, ax_desc
    except Exception as e:
        print(f"Windows focus check failed: {e}")
        return None, None, None, None, None, None


def get_focused_element():
    """
    Returns (app, role, title, selection, subrole, ax_desc) or (None, ...).
    """
    if IS_WINDOWS:
        return get_focused_element_windows()

    if not IS_MAC or not os.path.exists(GET_FOCUS_BIN):
        return None, None, None, None, None, None

    try:
        result = subprocess.run(
            [GET_FOCUS_BIN],
            capture_output=True,
            text=True,
            timeout=2,
        )
        output = result.stdout.strip()

        if "|" in output and "App:" in output:
            parts = output.split("|")

            def get_val(p):
                return p.split(":", 1)[1].strip() if ":" in p else "Unknown"

            app_part = get_val(parts[0])
            role_part = get_val(parts[1])
            title_part = get_val(parts[2]) if len(parts) > 2 else "Unknown"
            selection_part = get_val(parts[3]) if len(parts) > 3 else "None"
            subrole_part = get_val(parts[4]) if len(parts) > 4 else "None"
            ax_desc_part = get_val(parts[5]) if len(parts) > 5 else "None"
            return app_part, role_part, title_part, selection_part, subrole_part, ax_desc_part
    except Exception as e:
        print(f"Focus check failed: {e}")

    return None, None, None, None, None, None


def validate_context(app_name, role):
    if not app_name:
        return False
    if app_name in DISAPPROVED_APPS:
        return False
    if app_name in TRUSTED_APPS_WITHOUT_ROLE:
        return True
    if IS_WINDOWS:
        return role in WINDOWS_ALLOWED_ROLES
    return role in ALLOWED_ROLES


def get_best_input_device():
    try:
        input_device = sd.default.device[0]
        if input_device is None or input_device < 0:
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    input_device = i
                    break
        if input_device is None or input_device < 0:
            return None, None

        input_device = int(input_device)
        dev_info = sd.query_devices(input_device, "input")
        return input_device, dev_info
    except Exception as e:
        print(f"Error finding input device: {e}")
        return None, None


def clean_text(text):
    text = re.sub(r"(?i)\s*Продолжение следует\.{3}.*$", "", text)
    text = re.sub(r"(?i)\s*Thank you[\.!]*\s*$", "", text)
    text = text.strip()

    lower_text = text.lower()
    garbage_phrases = ["clears throat", "cough", "ahem"]
    garbage_len = sum(lower_text.count(phrase) * len(phrase) for phrase in garbage_phrases)
    hmm_len = lower_text.count("hmm") * len("hmm")

    if len(text) > 0 and (hmm_len / len(text)) > 0.6:
        return ""
    if len(text) > 0 and (garbage_len / len(text)) > 0.5:
        return ""
    return text


def send_message(current_app):
    if current_app in CMD_ENTER_APPS:
        mod_key = Key.cmd if IS_MAC else Key.ctrl
        with keyboard_controller.pressed(mod_key):
            keyboard_controller.press(Key.enter)
            keyboard_controller.release(Key.enter)
    else:
        keyboard_controller.press(Key.enter)
        keyboard_controller.release(Key.enter)
    show_overlay_success()


def paste_content(content):
    keyboard_controller.type(content + " ")


def has_low_confidence(avg_logprob, threshold):
    return avg_logprob is not None and avg_logprob < threshold


def process_audio(audio_data, start_context=None, end_context=None, source_mode="auto"):
    global last_paste_time, SESSION_COUNTER

    with processing_lock:
        SESSION_COUNTER += 1

        start_app = "MANUAL"
        start_role = "manual"
        start_title = ""
        end_app = None
        end_role = None
        end_title = None

        if source_mode == "auto":
            if not start_context or not end_context:
                return

            start_app, start_role, start_title, _, _, _ = start_context
            end_app, end_role, end_title, _, _, _ = end_context

            if not validate_context(start_app, start_role):
                return

            if (start_app, start_role, start_title) != (end_app, end_role, end_title):
                print(f"Focus changed during speech ({start_app}:{start_title} -> {end_app}:{end_title}).")
                return

        duration = len(audio_data) / SAMPLE_RATE
        rms = np.sqrt(np.mean(audio_data ** 2))
        db_fs = 20 * np.log10(rms + 1e-9)
        if db_fs < ASR_ENERGY_THRESHOLD:
            return

        try:
            start_time = time.perf_counter()
            result = transcribe_audio(audio_data)
            inference_time = time.perf_counter() - start_time
            if not result:
                return

            text = clean_text(str(result.get("text", "")).strip())
            avg_logprob = result.get("avg_logprob")

            if not text or text.lower() == "you":
                return

            action_type = "PASTE"
            action_payload = text

            if source_mode == "auto":
                current_app, current_role, current_title, _, _, _ = get_focused_element()
                if (current_app, current_role, current_title) != (end_app, end_role, end_title):
                    print(f"Focus moved ({end_app}:{end_title} -> {current_app}:{current_title}).")
                    return

                can_send = (time.time() - last_paste_time) < ASR_SEND_WINDOW_DURATION_S
                available_tools = ["insert_text"] + (["send_message"] if can_send else [])

                should_use_engine = (
                    context_engine is not None
                    and len(available_tools) > 1
                    and duration <= TOOL_INSTRUCTION_DURATION
                )

                if should_use_engine:
                    if has_low_confidence(avg_logprob, ASR_COMMAND_THRESHOLD):
                        action_type = "IGNORE"
                    else:
                        ce_action, ce_payload = context_engine.process(text, allowed_tools=available_tools)
                        if ce_action == "tool":
                            action_type = "TOOL"
                            action_payload = ce_payload
                        elif has_low_confidence(avg_logprob, ASR_DICTATION_THRESHOLD):
                            action_type = "IGNORE"
                        else:
                            action_payload = ce_payload
                elif has_low_confidence(avg_logprob, ASR_DICTATION_THRESHOLD):
                    action_type = "IGNORE"
            else:
                if has_low_confidence(avg_logprob, ASR_DICTATION_THRESHOLD):
                    action_type = "IGNORE"

            if action_type == "IGNORE":
                return

            if action_type == "TOOL":
                if action_payload == "send_message":
                    send_message(start_app)
                return

            paste_content(action_payload)
            last_paste_time = time.time()

            conf_part = f"conf={avg_logprob:.3f}" if avg_logprob is not None else "conf=n/a"
            print(
                f"OK #{SESSION_COUNTER} mode={source_mode} backend={STT_BACKEND} "
                f"dur={duration:.2f}s {conf_part} infer={inference_time:.2f}s"
            )

        except Exception as e:
            print(f"Transcription failed: {e}")


def auto_mode_supported():
    if IS_MAC:
        return os.path.exists(GET_FOCUS_BIN)
    if IS_WINDOWS:
        return ensure_windows_focus_backend()
    return False


class VADAudio:
    def __init__(self, callback):
        self.callback = callback
        self.buffer_queue = queue.Queue()
        self.manual_recording = False
        self.manual_buffer = []

    def start_manual_recording(self):
        self.manual_buffer = []
        self.manual_recording = True
        print("Manual recording started.")
        start_manual_recording_overlay()

    def stop_manual_recording(self):
        if not self.manual_recording:
            return

        self.manual_recording = False
        stop_manual_recording_overlay()
        show_overlay_text("REC STOP", "green")

        if not self.manual_buffer:
            print("Manual recording empty.")
            return

        full_audio = np.concatenate(self.manual_buffer)
        self.manual_buffer = []

        duration = len(full_audio) / SAMPLE_RATE
        if duration < (MIN_SPEECH_DURATION_MS / 1000.0):
            print("Manual recording too short.")
            return

        threading.Thread(
            target=self.callback,
            args=(full_audio, None, None, "manual"),
            daemon=True,
        ).start()

    def read_audio(self):
        def callback(indata, _frames, _time_info, status):
            if status:
                print(status, file=sys.stderr)
            self.buffer_queue.put(indata.copy())

        input_device, dev_info = get_best_input_device()
        if input_device is None:
            print("Error: no valid microphone found.")
            sys.exit(1)

        print(f"Input device: {dev_info['name']}")
        print("Listening...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=input_device,
            channels=1,
            callback=callback,
            blocksize=512,
        ):
            self.process_stream()

    def process_stream(self):
        triggered = False
        speech_buffer = []
        silence_counter = 0
        start_context = (None, None, None, None, None, None)
        ring_buffer = collections.deque(maxlen=max(1, int(SPEECH_PAD_MS / 32)))

        while True:
            chunk = self.buffer_queue.get()
            chunk_flat = chunk.flatten()

            if MODE == "manual":
                if triggered:
                    triggered = False
                    speech_buffer = []
                    silence_counter = 0
                    ring_buffer.clear()
                if self.manual_recording:
                    self.manual_buffer.append(chunk_flat)
                continue

            if not AUTO_ASR_ENABLED:
                if triggered:
                    triggered = False
                    speech_buffer = []
                    silence_counter = 0
                ring_buffer.clear()
                continue

            if _vad_model is None and not ensure_vad_model():
                continue

            speech_prob = _vad_model(_torch.from_numpy(chunk_flat), SAMPLE_RATE).item()

            if triggered:
                speech_buffer.append(chunk_flat)
                if speech_prob < VAD_NEGATIVE_THRESHOLD:
                    silence_counter += 32
                else:
                    silence_counter = 0

                if silence_counter > MIN_SILENCE_DURATION_MS:
                    triggered = False
                    end_context = get_focused_element()

                    chunks_per_ms = 1 / 32.0
                    chunks_to_keep = int(5 * chunks_per_ms)
                    silence_chunks = int(MIN_SILENCE_DURATION_MS * chunks_per_ms)
                    trim_amount = max(0, silence_chunks - chunks_to_keep)
                    if trim_amount > 0 and trim_amount < len(speech_buffer):
                        speech_buffer = speech_buffer[:-trim_amount]

                    if speech_buffer:
                        full_audio = np.concatenate(speech_buffer)
                        duration = len(full_audio) / SAMPLE_RATE
                        if duration >= (MIN_SPEECH_DURATION_MS / 1000.0):
                            threading.Thread(
                                target=self.callback,
                                args=(full_audio, start_context, end_context, "auto"),
                                daemon=True,
                            ).start()

                    speech_buffer = []
                    silence_counter = 0
            else:
                ring_buffer.append(chunk_flat)
                if speech_prob > VAD_POSITIVE_THRESHOLD:
                    triggered = True
                    start_context = get_focused_element()
                    speech_buffer = list(ring_buffer)
                    speech_buffer.append(chunk_flat)
                    silence_counter = 0


def toggle_mode():
    global MODE

    if MODE == "manual":
        if vad_audio and vad_audio.manual_recording:
            vad_audio.stop_manual_recording()

        if not auto_mode_supported():
            if IS_WINDOWS:
                print("AUTO mode unavailable: install Windows UIA dependency: pip install uiautomation")
            else:
                print("AUTO mode is only supported on macOS (with ./get_focus) or Windows (with UIA).")
            show_overlay_text("AUTO unsupported", "red")
            return

        if not ensure_vad_model():
            print("AUTO mode unavailable: VAD initialization failed.")
            show_overlay_text("AUTO unavailable", "red")
            return

        MODE = "auto"
        print("Mode: AUTO")
        show_overlay_text("MODE: AUTO", "green")
    else:
        MODE = "manual"
        print("Mode: MANUAL")
        show_overlay_text("MODE: MANUAL", "red")


def on_press(key):
    global AUTO_ASR_ENABLED

    try:
        if key == MODE_KEY:
            toggle_mode()
            return

        if key != ACTION_KEY:
            return

        if MODE == "auto":
            AUTO_ASR_ENABLED = not AUTO_ASR_ENABLED
            state_text = "AUTO: ON" if AUTO_ASR_ENABLED else "AUTO: OFF"
            color = "green" if AUTO_ASR_ENABLED else "red"
            print(state_text)
            show_overlay_text(state_text, color)
            return

        if vad_audio is None:
            return

        if not vad_audio.manual_recording:
            vad_audio.start_manual_recording()
        else:
            vad_audio.stop_manual_recording()

    except AttributeError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Global ASR (manual + auto)")
    parser.add_argument(
        "--stt-backend",
        choices=["local", "openai"],
        default=os.getenv("STT_BACKEND", "local"),
        help="Speech-to-text backend.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        help="Language code (e.g. en, fr). Default: auto.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=os.getenv("OPENAI_WHISPER_MODEL", "whisper-1"),
        help="OpenAI transcription model (used when --stt-backend openai).",
    )
    parser.add_argument(
        "--openai-prompt",
        type=str,
        default=os.getenv("OPENAI_WHISPER_PROMPT", ""),
        help="Optional prompt/hint for OpenAI transcription.",
    )
    parser.add_argument("--context", action="store_true", help="Enable Context Engine in AUTO mode.")
    return parser.parse_args()


def print_startup_intro():
    print("Global ASR")
    print("")
    print("Controls")
    print(f"- {MODE_KEY}: switch mode (AUTO <-> MANUAL)")
    print(f"- {ACTION_KEY} in AUTO: toggle auto listening ON/OFF")
    print(f"- {ACTION_KEY} in MANUAL: start/stop recording")
    print("")
    print("Modes")
    print("- AUTO: VAD detects speech and transcribes automatically.")
    print("        Also validates focused UI context before insertion.")
    print("- MANUAL: records only between start/stop keypresses, then transcribes once.")
    print("")
    print(f"Current mode: {MODE.upper()}")
    if MODE == "manual":
        print("Started in MANUAL mode.")
    if IS_WINDOWS:
        print("Windows AUTO mode requires: pip install uiautomation")
    print(f"STT backend: {STT_BACKEND}")
    if STT_BACKEND == "openai":
        print(f"OpenAI model: {OPENAI_MODEL}")


def init_optional_context_engine(enable_context):
    global context_engine
    if not enable_context:
        context_engine = None
        print("Context engine: OFF")
        return

    try:
        from context_engine import ContextEngine
    except Exception as e:
        print(f"Context engine unavailable: {e}")
        context_engine = None
        return

    try:
        context_engine = ContextEngine()
        context_engine.start_server()
        print("Context engine: ON")
    except Exception as e:
        print(f"Failed to start context engine: {e}")
        context_engine = None


def main():
    global SELECTED_LANGUAGE, STT_BACKEND, OPENAI_MODEL, OPENAI_PROMPT, vad_audio

    args = parse_args()
    SELECTED_LANGUAGE = args.lang
    STT_BACKEND = args.stt_backend
    OPENAI_MODEL = args.openai_model
    OPENAI_PROMPT = args.openai_prompt

    print_startup_intro()

    init_optional_context_engine(args.context)

    if STT_BACKEND == "local" and not ensure_local_backend():
        sys.exit(1)

    if STT_BACKEND == "openai" and not ensure_openai_backend():
        sys.exit(1)

    show_overlay_text("MODE: MANUAL", "red")

    listener = Listener(on_press=on_press)
    listener.start()

    vad_audio = VADAudio(callback=process_audio)

    try:
        vad_audio.read_audio()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_manual_recording_overlay()
        if context_engine:
            try:
                context_engine.stop_server()
            except Exception:
                pass
        listener.stop()


if __name__ == "__main__":
    main()
