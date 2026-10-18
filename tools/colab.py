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


def setup_notebook(*, linear_kernels=False):
    if linear_kernels:
        import torch
        if sys.platform != 'linux' or not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
            raise RuntimeError('FLA trial needs Linux and an NVIDIA GPU with compute capability 8.0+ (e.g. L4/A100); T4 is unsupported.')
        from packaging.version import Version
        if Version(torch.__version__.split('+')[0]) < Version('2.7') or Version(version('triton')) < Version('3.3'):
            raise RuntimeError('FLA trial needs the existing torch >=2.7 and Triton >=3.3; select a compatible runtime.')
        if 'transformers' in sys.modules or 'fla' in sys.modules:
            raise RuntimeError('Restart the session, then select the FLA setup before importing transformers or fla.')
    # Optional JAX in bm25s must not reserve VRAM intended for PyTorch models.
    os.environ["JAX_PLATFORMS"] = "cpu"
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError('Use Python 3.12–3.14.')
    loaded = {name: getattr(sys.modules.get(name), '__version__', None)
              for name in ('numpy', 'scipy')}
    root = Path(__file__).resolve().parents[1]
    uv = ensure_uv()
    # These runs use unquantized weights; Colab's old torchao breaks PEFT's optional probe.
    if importlib.util.find_spec('torchao') is not None:
        subprocess.run([uv, "pip", "uninstall", "--python", sys.executable, "torchao"], check=True)
    command = [uv, "export", "--locked", "--no-default-groups", "--no-emit-project", "--group", "models"]
    if linear_kernels:
        command += ['--group', 'linear-kernels']
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
    if linear_kernels:
        # Verify imports in a subprocess so the notebook's Transformers cache stays fresh.
        subprocess.run([sys.executable, '-c',
            'from transformers.models.qwen3_5.modeling_qwen3_5 import chunk_gated_delta_rule; '
            'assert chunk_gated_delta_rule is not None, "FLA DeltaNet kernel import unavailable"'], check=True)
    # Use this checkout even if the kernel started with another editable install.
    sys.path.insert(0, str(root / 'src'))
    importlib.invalidate_caches()
    os.environ['DECISION_RERANK_ROOT'] = str(root)
    print(f"Installed decision-rerank with uv {UV_VERSION}; NumPy {version('numpy')}, SciPy {version('scipy')}")
