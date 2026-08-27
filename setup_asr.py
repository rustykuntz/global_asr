import getpass
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
REQ_PATH = BASE_DIR / "requirements.txt"
CUDA_REQ_PATH = BASE_DIR / "requirements-linux-cuda.txt"
WHISPER_DIR = BASE_DIR / "whisper-turbo-mlx"
FASTER_WHISPER_MODEL = "large-v3-turbo"
DEFAULT_VENV_DIR = BASE_DIR / ".venv"
LEGACY_WHISPER_CPP_ENV_KEYS = {
    "WHISPER_CPP_DIR",
    "WHISPER_CPP_MODEL",
    "WHISPER_CPP_MODEL_PATH",
    "WHISPER_CPP_BINARY",
    "WHISPER_CPP_DEVICE",
    "WHISPER_CPP_THREADS",
    "WHISPER_CPP_BEAM_SIZE",
    "WHISPER_CPP_BEST_OF",
    "WHISPER_CPP_TEMPERATURE",
    "WHISPER_CPP_TEMPERATURE_INC",
    "WHISPER_CPP_MAX_CONTEXT",
    "WHISPER_CPP_NO_FALLBACK",
    "WHISPER_CPP_SUPPRESS_NST",
}


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


def install_python_dependencies(python_bin, os_name):
    print("Upgrading pip/setuptools/wheel...")
    if not run_cmd([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]):
        print("Failed to upgrade pip/setuptools/wheel.")
        return False

    ok = run_cmd([
        python_bin,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "eager",
        "--prefer-binary",
        "-r",
        str(REQ_PATH),
    ])
    if ok:
        return True

    print_dependency_failure_help(os_name, python_bin)
    return False


def python_venv_package_name():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"python{version}-venv"


def python_runtime_info(python_bin):
    code = (
        "import pathlib, sys, sysconfig\n"
        "include = sysconfig.get_path('include') or ''\n"
        "header = pathlib.Path(include) / 'Python.h'\n"
        "print(f'{sys.version_info.major}.{sys.version_info.minor}')\n"
        "print(header)\n"
        "raise SystemExit(0 if header.is_file() else 1)\n"
    )
    result = subprocess.run(
        [str(python_bin), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    version = lines[0] if lines else None
    header_path = lines[1] if len(lines) > 1 else None
    return version, header_path, result.returncode == 0


def python_library_available(python_bin, library_name):
    code = (
        "import ctypes.util, sys\n"
        "raise SystemExit(0 if ctypes.util.find_library(sys.argv[1]) else 1)\n"
    )
    result = subprocess.run(
        [str(python_bin), "-c", code, library_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def is_wayland_session():
    return (
        os.getenv("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.getenv("WAYLAND_DISPLAY"))
    )


def root_command_prefix():
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    return [sudo] if sudo else None


def ensure_linux_system_dependencies(python_bin):
    version, header_path, headers_available = python_runtime_info(python_bin)
    portaudio_available = python_library_available(python_bin, "portaudio")
    packages = []
    versioned_package = None

    if not headers_available:
        if not version:
            print("Could not determine the Python version used by the environment.")
            return False
        versioned_package = f"python{version}-dev"
        packages.append(versioned_package)

    if not portaudio_available:
        packages.append("libportaudio2")

    command_packages = {
        "cc": "build-essential",
    }
    for command, package in command_packages.items():
        if shutil.which(command) is None and package not in packages:
            packages.append(package)

    if is_wayland_session() and shutil.which("wl-copy") is None:
        packages.append("wl-clipboard")

    if not packages:
        return True

    print("")
    print("Missing Linux system dependencies:")
    if not headers_available:
        print(f"  Python development headers ({header_path or 'Python.h not found'})")
    if not portaudio_available:
        print("  PortAudio runtime library")
    if "wl-clipboard" in packages:
        print("  wl-clipboard (Unicode text insertion on Wayland)")
    missing_build_packages = [
        package for package in ("build-essential",) if package in packages
    ]
    if missing_build_packages:
        print(f"  Build tools: {', '.join(missing_build_packages)}")

    apt_get = shutil.which("apt-get")
    if not apt_get:
        print("Install these packages with your distribution package manager:")
        print(f"  {' '.join(packages)}")
        return False

    command_prefix = root_command_prefix()
    if command_prefix is None:
        print("sudo was not found. Install the packages as root:")
        print(f"  apt-get install -y {' '.join(packages)}")
        return False

    package_list = " ".join(packages)
    if not ask_yes_no(f"Install missing packages ({package_list}) now?", default_yes=True):
        print("Global ASR cannot run until the missing system dependencies are installed.")
        return False

    print("Updating apt package metadata...")
    if not run_cmd([*command_prefix, apt_get, "update"]):
        print("apt-get update failed; trying the existing package metadata.")

    candidates = [packages]
    if versioned_package and versioned_package != "python3-dev":
        candidates.append([
            "python3-dev" if package == versioned_package else package
            for package in packages
        ])

    for candidate_packages in candidates:
        print(f"Installing {' '.join(candidate_packages)}...")
        if not run_cmd([*command_prefix, apt_get, "install", "-y", *candidate_packages]):
            continue
        _, _, current_headers_available = python_runtime_info(python_bin)
        current_portaudio_available = python_library_available(python_bin, "portaudio")
        commands_available = all(shutil.which(command) for command in command_packages)
        wayland_tools_available = not is_wayland_session() or shutil.which("wl-copy")
        if current_headers_available and current_portaudio_available and commands_available and wayland_tools_available:
            print("Linux system dependencies are ready.")
            return True

    print("System dependency installation did not satisfy all requirements.")
    if not python_runtime_info(python_bin)[2]:
        print(f"  Python header still missing: {header_path or 'unknown'}")
    if not python_library_available(python_bin, "portaudio"):
        print("  PortAudio library is still unavailable.")
    if is_wayland_session() and not shutil.which("wl-copy"):
        print("  wl-copy is still unavailable.")
    return False


def configure_wayland_input_permissions():
    if not is_wayland_session():
        return True

    import grp

    target_user = os.getenv("SUDO_USER") or getpass.getuser()
    try:
        input_group = grp.getgrnam("input")
        configured_member = target_user in input_group.gr_mem
        active_member = input_group.gr_gid in os.getgroups()
    except KeyError:
        configured_member = False
        active_member = False

    readable_keyboard = any(
        os.access(path, os.R_OK) for path in Path("/dev/input").glob("event*")
    )
    writable_uinput = os.path.exists("/dev/uinput") and os.access("/dev/uinput", os.W_OK)
    if readable_keyboard and writable_uinput:
        print("Wayland keyboard permissions are ready.")
        return True

    print("")
    print("Wayland global hotkeys require access to physical keyboard events and /dev/uinput.")
    print("This grants the local input group permission to read keyboard devices and inject keys.")
    if not ask_yes_no("Configure Wayland keyboard permissions now?", default_yes=True):
        print("F4/F6 cannot work globally on Wayland without these permissions.")
        return False

    command_prefix = root_command_prefix()
    if command_prefix is None:
        print("sudo was not found. Configure the input group and udev permissions as root.")
        return False

    if not run_cmd([*command_prefix, "groupadd", "-f", "input"]):
        print("Failed to create or confirm the input group.")
        return False
    if not configured_member and not run_cmd(
        [*command_prefix, "usermod", "-aG", "input", target_user]
    ):
        print(f"Failed to add {target_user} to the input group.")
        return False
    if not run_cmd([*command_prefix, "modprobe", "uinput"]):
        print("Failed to load the uinput kernel module.")
        return False

    rule = (
        'SUBSYSTEM=="input", KERNEL=="event*", GROUP="input", MODE="0660"\n'
        'SUBSYSTEM=="misc", KERNEL=="uinput", GROUP="input", MODE="0660", '
        'OPTIONS+="static_node=uinput"\n'
    )
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False) as f:
            f.write(rule)
            tmp_path = f.name
        if not run_cmd([
            *command_prefix,
            "install",
            "-m",
            "0644",
            tmp_path,
            "/etc/udev/rules.d/70-global-asr-input.rules",
        ]):
            print("Failed to install the Global ASR udev rule.")
            return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not run_cmd([*command_prefix, "udevadm", "control", "--reload-rules"]):
        print("Failed to reload udev rules.")
        return False
    if not run_cmd([*command_prefix, "udevadm", "trigger", "--subsystem-match=input"]):
        print("Failed to apply permissions to input devices.")
        return False
    if not run_cmd([*command_prefix, "udevadm", "trigger", "--subsystem-match=misc"]):
        print("Failed to apply permissions to /dev/uinput.")
        return False
    print("Wayland keyboard permissions configured.")
    if not active_member:
        print("IMPORTANT: sign out of Ubuntu and sign back in before running Global ASR.")
    return True


def print_linux_system_dependency_notes():
    venv_package = python_venv_package_name()
    dev_package = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
    print("Linux system dependencies:")
    print("  sudo apt update")
    print(
        f"  sudo apt install -y {venv_package} {dev_package} "
        "libportaudio2 build-essential wl-clipboard"
    )
    if venv_package != "python3-venv":
        print("")
        print("If your distro does not provide the versioned venv package, try:")
        print("  sudo apt install -y python3-venv")
    print("")
    print("If sounddevice needs to be rebuilt locally, also install:")
    print("  sudo apt install -y portaudio19-dev")
    print("")
    print("If a Python package needs a Rust source build, also install:")
    print("  sudo apt install -y rustc cargo")
    print("")


def print_venv_failure_help(os_name):
    print("Failed to create .venv.")
    if os_name != "Linux":
        return

    venv_package = python_venv_package_name()
    print("")
    print("On Debian/Ubuntu this usually means ensurepip is missing.")
    print("Install the venv package for the Python version you are running:")
    print(f"  sudo apt install -y {venv_package}")
    if venv_package != "python3-venv":
        print("")
        print("If that package is not available, try the generic package:")
        print("  sudo apt install -y python3-venv")
    print("")
    print("Then rerun:")
    print("  python3 setup_asr.py")


def print_dependency_failure_help(os_name, python_bin):
    print("Dependency install failed.")
    if os_name != "Linux":
        print("Resolve the pip error above, then re-run setup.")
        return

    version, _, _ = python_runtime_info(python_bin)
    dev_package = f"python{version}-dev" if version else "python3-dev"
    print("")
    print("On fresh Ubuntu/Debian machines, common fixes are:")
    print("  .venv/bin/python -m pip install --upgrade pip setuptools wheel")
    print(f"  sudo apt install -y {dev_package}")
    print("")
    print("Only Rust-based source builds require:")
    print("  sudo apt install -y rustc cargo")
    print("")
    print("Then rerun:")
    print("  python3 setup_asr.py")


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


def write_env(path, updates, remove_keys=None):
    env_data = read_env(path)
    for key in remove_keys or ():
        env_data.pop(key, None)
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


def nvidia_gpu_names():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]


def install_linux_cuda_runtime(python_bin):
    print("Installing the prebuilt CUDA runtime for faster-whisper...")
    if not run_cmd([
        str(python_bin),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "eager",
        "--prefer-binary",
        "-r",
        str(CUDA_REQ_PATH),
    ]):
        print("Failed to install the faster-whisper CUDA runtime packages.")
        return False
    cuda_dirs = python_cuda_library_dirs(python_bin)
    if len(cuda_dirs) < 2:
        print("The CUDA packages installed, but their cuBLAS/cuDNN library directories were not found.")
        return False
    print("Prebuilt faster-whisper CUDA runtime is ready.")
    return True


def python_cuda_library_dirs(python_bin):
    code = (
        "import os, pathlib, site\n"
        "paths = []\n"
        "for root in site.getsitepackages():\n"
        "    for package in ('cublas', 'cudnn', 'cuda_nvrtc'):\n"
        "        path = pathlib.Path(root) / 'nvidia' / package / 'lib'\n"
        "        if path.is_dir(): paths.append(str(path))\n"
        "print(os.pathsep.join(paths))\n"
    )
    proc = subprocess.run(
        [str(python_bin), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [path for path in proc.stdout.strip().split(os.pathsep) if path]


def faster_whisper_subprocess_env(python_bin):
    env = os.environ.copy()
    cuda_dirs = python_cuda_library_dirs(python_bin)
    if cuda_dirs:
        current_dirs = [path for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys([*cuda_dirs, *current_dirs])
        )
    return env


def warm_local_model_faster_whisper(python_bin, device, compute_type):
    print("Downloading/loading faster-whisper large-v3-turbo (one-time warmup)...")
    code = (
        "import ctranslate2, faster_whisper\n"
        "from faster_whisper import WhisperModel\n"
        "print(f'faster-whisper {faster_whisper.__version__}')\n"
        "print(f'CTranslate2 {ctranslate2.__version__}')\n"
    )
    if device == "cuda":
        code += (
            "count = ctranslate2.get_cuda_device_count()\n"
            "print(f'CTranslate2 CUDA devices: {count}')\n"
            "if count < 1: raise SystemExit('No CUDA device available; CPU fallback is disabled.')\n"
        )
    code += (
        f"WhisperModel({FASTER_WHISPER_MODEL!r}, device={device!r}, "
        f"compute_type={compute_type!r})\n"
        "print('faster-whisper large-v3-turbo is ready.')\n"
    )
    try:
        subprocess.run(
            [str(python_bin), "-c", code],
            check=True,
            env=faster_whisper_subprocess_env(python_bin),
        )
        return True
    except Exception as e:
        print(f"Failed to download/load faster-whisper model: {e}")
        return False


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
                print_venv_failure_help(os_name)
                sys.exit(1)
            using_venv = True
            venv_python = venv_python_path(venv_dir)
            print(f"Created venv at: {venv_dir}")
            print("")

    install_python = str(venv_python) if using_venv else sys.executable

    if os_name == "Linux" and not ensure_linux_system_dependencies(install_python):
        sys.exit(1)

    if os_name == "Linux" and not configure_wayland_input_permissions():
        sys.exit(1)

    if ask_yes_no("Install Python dependencies now?", default_yes=True):
        ok = install_python_dependencies(install_python, os_name)
        if not ok:
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
                ("local", "Local Whisper large-v3-turbo via faster-whisper (recommended on Linux)"),
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
    local_device = "auto"
    local_compute_type = "int8"
    if backend == "local" and os_name == "Linux":
        gpus = nvidia_gpu_names()
        if gpus:
            print("")
            print("NVIDIA GPU detected:")
            for gpu in gpus:
                print(f"  {gpu}")
            if not install_linux_cuda_runtime(install_python):
                sys.exit(1)
            local_device = "cuda"
            local_compute_type = "float16"
        else:
            print("nvidia-smi -L reported no NVIDIA GPU; using faster-whisper on CPU.")
            local_device = "cpu"
            local_compute_type = "int8"
        updates.update(
            {
                "FASTER_WHISPER_MODEL": FASTER_WHISPER_MODEL,
                "FASTER_WHISPER_DEVICE": local_device,
                "FASTER_WHISPER_COMPUTE_TYPE": local_compute_type,
                "FASTER_WHISPER_BEAM_SIZE": "1",
                "FASTER_WHISPER_BEST_OF": "1",
                "FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT": "0",
                "FASTER_WHISPER_WITHOUT_TIMESTAMPS": "1",
            }
        )

    if backend == "openai":
        while True:
            api_key = getpass.getpass("Enter OPENAI_API_KEY: ").strip()
            if api_key:
                updates["OPENAI_API_KEY"] = api_key
                break
            print("API key cannot be empty.")

    write_env(ENV_PATH, updates, remove_keys=LEGACY_WHISPER_CPP_ENV_KEYS)
    print(f"Saved configuration: {ENV_PATH}")

    if backend == "local" and os_name in {"Darwin", "Windows", "Linux"}:
        print("")
        if ask_yes_no("Prepare local backend now?", default_yes=True):
            if os_name == "Darwin":
                ok = warm_local_model_mlx(install_python)
            else:
                ok = warm_local_model_faster_whisper(
                    install_python,
                    local_device,
                    local_compute_type,
                )
            if not ok:
                print("Local backend setup failed. Re-run setup after resolving the error above.")
                sys.exit(1)

    print("\nSetup complete.")
    print("Run:")
    if using_venv:
        if os_name == "Windows":
            print("  .venv\\Scripts\\activate")
        else:
            print("  source .venv/bin/activate")
    print("  python global_asr.py")
    print("Controls:")
    print("  F6 = cycle mode (MANUAL -> AUTO -> OFF)")
    print("  F4 in MANUAL = start/stop recording")
    print("  F4 in AUTO   = toggle auto listening")


if __name__ == "__main__":
    main()
