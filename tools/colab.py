"""Install locked dependencies while retaining the notebook’s numerical stack."""
from pathlib import Path
import importlib.util
from importlib.metadata import requires, version
import os
import shutil
import subprocess
import sys
import tempfile

UV_VERSION = "0.13.1"


def ensure_uv():
    """Use the pinned uv; pip is used only to bootstrap uv itself if necessary."""
    binary = shutil.which("uv")
    if binary:
        result = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True)
        if result.stdout.split()[:2] == ["uv", UV_VERSION]:
            return binary
    target = Path(tempfile.mkdtemp(prefix="decision-rerank-uv-"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", str(target), f"uv=={UV_VERSION}"],
        check=True,
    )
    binary = target / "bin" / "uv"
    if not binary.is_file():
        raise RuntimeError("Expected the uv executable for the Linux/macOS notebook runtime")
    return str(binary)


def setup_notebook():
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError('Use Python 3.12–3.14.')
    loaded = {name: getattr(sys.modules.get(name), '__version__', None)
              for name in ('numpy', 'scipy')}
    root = Path(__file__).resolve().parents[1]
    uv = ensure_uv()
    command = [uv, "export", "--locked", "--no-default-groups", "--no-emit-project", "--group", "models"]
    if importlib.util.find_spec('torch') is None:
        command += ["--group", "cpu"]
    else:
        command += ["--prune", "torch"]  # Keep the runtime's CUDA build and its dependencies.
    if all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy')):
        command += ["--prune", "numpy", "--prune", "scipy"]
    exported = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--require-hashes", "-r", "-"],
        input=exported.stdout, check=True, text=True,
    )
    subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--editable", str(root)],
        check=True,
    )
    changed = [name for name, previous in loaded.items() if previous and previous != version(name)]
    if changed:
        raise RuntimeError(
            f"Setup replaced already loaded {', '.join(changed)}. "
            "In Colab, choose Runtime > Restart session, then rerun Setup and the remaining cells. "
            "Do not disconnect and delete the runtime."
        )
    from packaging.requirements import Requirement
    for text in [*requires('decision-rerank'), *requires('scipy')]:
        requirement = Requirement(text)
        if requirement.name in {'numpy', 'scipy'} and version(requirement.name) not in requirement.specifier:
            raise RuntimeError(f"Installed {requirement.name}=={version(requirement.name)} does not satisfy {requirement}")
    # Use this checkout even if the kernel started with another editable install.
    sys.path.insert(0, str(root / 'src'))
    importlib.invalidate_caches()
    os.environ['DECISION_RERANK_ROOT'] = str(root)
    print(f"Installed decision-rerank with uv {UV_VERSION}; NumPy {version('numpy')}, SciPy {version('scipy')}")
