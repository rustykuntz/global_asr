import getpass
import os
import platform
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
REQ_PATH = BASE_DIR / "requirements.txt"
WHISPER_DIR = BASE_DIR / "whisper-turbo-mlx"
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


def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


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


def warm_local_model(python_bin):
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


def main():
    os_name = platform.system()
    venv_dir = DEFAULT_VENV_DIR
    venv_python = venv_python_path(venv_dir)
    using_venv = venv_python.exists()

    print("ASR Setup")
    print(f"Detected OS: {os_name}")
    print("")

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
    else:
        backend = ask_choice(
            "Choose speech backend:",
            [
                ("openai", "OpenAI Whisper API (recommended)"),
                ("local", "Local Whisper Turbo"),
            ],
            default_index=1,
        )

    if backend == "local" and os_name != "Darwin":
        print("Local Whisper Turbo is optimized for macOS/MLX.")
        if not ask_yes_no("Keep local backend anyway?", default_yes=False):
            backend = "openai"

    updates = {
        "STT_BACKEND": backend,
        "ASR_VENV_PATH": ".venv",
    }

    if backend == "openai":
        while True:
            api_key = getpass.getpass("Enter OPENAI_API_KEY: ").strip()
            if api_key:
                updates["OPENAI_API_KEY"] = api_key
                break
            print("API key cannot be empty.")

    write_env(ENV_PATH, updates)
    print(f"Saved configuration: {ENV_PATH}")

    if backend == "local" and os_name == "Darwin":
        print("")
        if ask_yes_no("Download local model now?", default_yes=True):
            ok = warm_local_model(install_python)
            if not ok:
                print("Model warmup failed. You can retry later by running setup again.")

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
