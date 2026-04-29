import getpass
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
REQ_PATH = BASE_DIR / "requirements.txt"
WHISPER_DIR = BASE_DIR / "whisper-turbo-mlx"
WHISPER_CPP_DIR = BASE_DIR / "whisper.cpp"
WHISPER_CPP_MODEL = "large-v3-turbo"
WHISPER_CPP_MODEL_PATH = WHISPER_CPP_DIR / "models" / f"ggml-{WHISPER_CPP_MODEL}.bin"
FASTER_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_VENV_DIR = BASE_DIR / ".venv"


def ask_choice(title, options, default_index=1):
    print(title)
    for i, (_, label) in enumerate(options, start=1):
        marker = " (default)" if i == default_index else ""
        print(f"  {i}. {label}{marker}")

    while True:
        raw = input("Select option: ").strip()
        if not raw:
            return options[default_index - 1][0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print("Invalid selection. Please enter a valid number.")


def ask_yes_no(question, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = input(f"{question} {suffix} ").strip().lower()
        if not raw:
            return default_yes
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y or n.")


def run_cmd(cmd, cwd=None):
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def print_linux_system_dependency_notes():
    print("Linux system dependencies:")
    print("  sudo apt update")
    print("  sudo apt install -y libportaudio2 git cmake build-essential")
    print("")
    print("If sounddevice needs to be rebuilt locally, also install:")
    print("  sudo apt install -y portaudio19-dev")
    print("")


def venv_python_path(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def read_env(path):
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def write_env(path, updates):
    env_data = read_env(path)
    env_data.update(updates)

    keys = sorted(env_data.keys())
    lines = [f"{k}={env_data[k]}" for k in keys]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warm_local_model_mlx(python_bin):
    if not WHISPER_DIR.exists():
        print(f"Missing local backend folder: {WHISPER_DIR}")
        return False

    print("Downloading/loading local Whisper model (one-time warmup)...")
    try:
        code = (
            "import sys\n"
            f"sys.path.append({str(WHISPER_DIR)!r})\n"
            "from whisper_turbo import load_model\n"
            "load_model()\n"
            "print('Local model is ready.')\n"
        )
        subprocess.run([str(python_bin), "-c", code], check=True)
        return True
    except Exception as e:
        print(f"Failed to download/load local model: {e}")
        return False


def warm_local_model_faster_whisper(python_bin):
    print("Downloading/loading faster-whisper model (one-time warmup)...")
    try:
        code = (
            "from faster_whisper import WhisperModel\n"
            f"WhisperModel({FASTER_WHISPER_MODEL!r}, device='auto', compute_type='int8')\n"
            "print('faster-whisper model is ready.')\n"
        )
        subprocess.run([str(python_bin), "-c", code], check=True)
        return True
    except Exception as e:
        print(f"Failed to download/load faster-whisper model: {e}")
        return False


def ensure_whisper_cpp_linux():
    missing = [name for name in ("git", "cmake", "bash") if shutil.which(name) is None]
    if missing:
        print(f"Missing required command(s) for whisper.cpp setup: {', '.join(missing)}")
        print("Install them with your system package manager, then re-run setup.")
        return False

    if not WHISPER_CPP_DIR.exists():
        print("Cloning whisper.cpp...")
        if not run_cmd(["git", "clone", "https://github.com/ggml-org/whisper.cpp.git", str(WHISPER_CPP_DIR)]):
            print("Failed to clone whisper.cpp.")
            return False
    else:
        print(f"Using existing whisper.cpp checkout: {WHISPER_CPP_DIR}")

    cmake_cmd = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    if shutil.which("nvcc"):
        print("CUDA toolkit detected; building whisper.cpp with GGML_CUDA=ON.")
        cmake_cmd.append("-DGGML_CUDA=ON")
    else:
        print("CUDA toolkit not detected; building whisper.cpp CPU-only.")

    print("Building whisper.cpp...")
    if not run_cmd(cmake_cmd, cwd=WHISPER_CPP_DIR):
        print("Failed to configure whisper.cpp with cmake.")
        return False
    if not run_cmd(["cmake", "--build", "build", "--parallel", "--config", "Release"], cwd=WHISPER_CPP_DIR):
        print("Failed to build whisper.cpp.")
        return False

    if not WHISPER_CPP_MODEL_PATH.exists():
        print(f"Downloading whisper.cpp model: {WHISPER_CPP_MODEL}")
        if not run_cmd(["bash", "./models/download-ggml-model.sh", WHISPER_CPP_MODEL], cwd=WHISPER_CPP_DIR):
            print("Failed to download whisper.cpp model.")
            return False
    else:
        print(f"Using existing whisper.cpp model: {WHISPER_CPP_MODEL_PATH}")

    print("whisper.cpp is ready.")
    return True


def main():
    os_name = platform.system()
    venv_dir = DEFAULT_VENV_DIR
    venv_python = venv_python_path(venv_dir)
    using_venv = venv_python.exists()

    print("ASR Setup")
    print(f"Detected OS: {os_name}")
    print("")
    if os_name == "Linux":
        print_linux_system_dependency_notes()

    if not using_venv:
        print("Project venv not found.")
        print("Suggested commands:")
        print("  python -m venv .venv")
        if os_name == "Windows":
            print("  .venv\\Scripts\\activate")
        else:
            print("  source .venv/bin/activate")
        print("")
        if ask_yes_no("Create .venv now?", default_yes=True):
            ok = run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
            if not ok:
                print("Failed to create .venv.")
                sys.exit(1)
            using_venv = True
            venv_python = venv_python_path(venv_dir)
            print(f"Created venv at: {venv_dir}")
            print("")

    install_python = str(venv_python) if using_venv else sys.executable

    if ask_yes_no("Install Python dependencies now?", default_yes=True):
        ok = run_cmd([install_python, "-m", "pip", "install", "-r", str(REQ_PATH)])
        if not ok:
            print("Dependency install failed. Resolve errors, then re-run setup.")
            sys.exit(1)

    print("")
    if os_name == "Darwin":
        backend = ask_choice(
            "Choose speech backend:",
            [
                ("local", "Local Whisper Turbo (recommended on macOS)"),
                ("openai", "OpenAI Whisper API"),
            ],
            default_index=1,
        )
    elif os_name == "Linux":
        backend = ask_choice(
            "Choose speech backend:",
            [
                ("local", "Local Whisper Turbo via whisper.cpp (recommended on Linux)"),
                ("openai", "OpenAI Whisper API"),
            ],
            default_index=1,
        )
    elif os_name == "Windows":
        backend = ask_choice(
            "Choose speech backend:",
            [
                ("openai", "OpenAI Whisper API (recommended)"),
                ("local", "Local Whisper Turbo via faster-whisper"),
            ],
            default_index=1,
        )
    else:
        backend = ask_choice(
            "Choose speech backend:",
            [
                ("openai", "OpenAI Whisper API (recommended)"),
                ("local", "Local Whisper (unsupported on this OS)"),
            ],
            default_index=1,
        )

    if backend == "local" and os_name not in {"Darwin", "Windows", "Linux"}:
        print("Local backend is supported on macOS, Windows, and Linux only.")
        backend = "openai"

    updates = {
        "STT_BACKEND": backend,
        "ASR_VENV_PATH": ".venv",
    }
    if backend == "local" and os_name == "Linux":
        updates.update(
            {
                "WHISPER_CPP_DIR": "whisper.cpp",
                "WHISPER_CPP_MODEL": WHISPER_CPP_MODEL,
                "WHISPER_CPP_MODEL_PATH": f"whisper.cpp/models/ggml-{WHISPER_CPP_MODEL}.bin",
                "WHISPER_CPP_BINARY": "whisper.cpp/build/bin/whisper-cli",
                "WHISPER_CPP_DEVICE": "0",
                "WHISPER_CPP_BEAM_SIZE": "1",
                "WHISPER_CPP_BEST_OF": "1",
                "WHISPER_CPP_TEMPERATURE": "0",
                "WHISPER_CPP_TEMPERATURE_INC": "0",
                "WHISPER_CPP_MAX_CONTEXT": "0",
                "WHISPER_CPP_NO_FALLBACK": "1",
                "WHISPER_CPP_SUPPRESS_NST": "1",
            }
        )

    if backend == "openai":
        while True:
            api_key = getpass.getpass("Enter OPENAI_API_KEY: ").strip()
            if api_key:
                updates["OPENAI_API_KEY"] = api_key
                break
            print("API key cannot be empty.")

    write_env(ENV_PATH, updates)
    print(f"Saved configuration: {ENV_PATH}")

    if backend == "local" and os_name in {"Darwin", "Windows", "Linux"}:
        print("")
        if ask_yes_no("Prepare local backend now?", default_yes=True):
            if os_name == "Darwin":
                ok = warm_local_model_mlx(install_python)
            elif os_name == "Windows":
                ok = warm_local_model_faster_whisper(install_python)
            else:
                ok = ensure_whisper_cpp_linux()
            if not ok:
                print("Local backend setup failed. You can retry later by running setup again.")

    print("\nSetup complete.")
    print("Run:")
    if using_venv:
        if os_name == "Windows":
            print("  .venv\\Scripts\\activate")
        else:
            print("  source .venv/bin/activate")
    print("  python global_asr.py")
    print("Controls:")
    print("  F6 = switch mode (AUTO <-> MANUAL)")
    print("  F4 in MANUAL = start/stop recording")
    print("  F4 in AUTO   = toggle auto listening")


if __name__ == "__main__":
    main()
