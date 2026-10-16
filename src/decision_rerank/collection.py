"""Run selected benchmarks × methods and export portable results."""
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import sys
import zipfile

from . import evaluate, load_benchmark, run
from .results import write_json

SUPPORTED_METHODS = {'bm25', 'tfidf-cpu', 'medcpt-cpu', 'qwen3-4b', 'qwen3-8b', 'medcpt',
                     'jina-v35', 'biodecision', 'qwen35-base', 'clef-flash', 'jev', 'openai-decisions'}


def benchmark(name, *, panel=None, limit=None, root=None, offline=False):
    """Load a benchmark selection, downloading missing data automatically."""
    root = root or os.environ.get('DECISION_RERANK_ROOT', Path(__file__).resolve().parents[2])
    return load_benchmark(root, name, panel=panel, limit=limit, offline=offline)


def _ensure_api_key(name):
    key = {'jev': 'OPENROUTER_API_KEY', 'openai-decisions': 'OPENAI_API_KEY'}.get(name)
    if key is None or os.environ.get(key):
        return
    try:
        from google.colab import userdata
        os.environ[key] = userdata.get(key)
    except Exception:
        pass  # Missing/inaccessible Colab Secrets fall back to a hidden prompt.
    if not os.environ.get(key):
        from getpass import getpass
        value = getpass(key + ' (hidden): ').strip()
        if not value:
            raise ValueError('No value supplied for ' + key)
        os.environ[key] = value


@dataclass(frozen=True)
class MethodChoice:
    """Defer model loading and credential lookup until this method actually runs."""
    name: str
    options: dict = field(default_factory=dict)

    def build(self, folder):
        if self.name == 'bm25':
            from .methods import OriginalOrder
            return OriginalOrder(**self.options)
        if self.name == 'tfidf-cpu':
            from .methods import CandidateTfidf
            return CandidateTfidf(**self.options)
        if self.name in {'medcpt', 'medcpt-cpu'}:
            from .medcpt import MedCPT
            device = 'cpu' if self.name == 'medcpt-cpu' else 'cuda'
            return MedCPT(**{'device': device, 'dtype': 'float32', 'batch_size': 16, **self.options})
        if self.name in {'qwen3-4b', 'qwen3-8b'}:
            from .qwen import QwenReranker
            variant = self.name.removeprefix('qwen3-')
            return QwenReranker(**{'variant': variant, 'batch_size': 2 if variant == '4b' else 1, **self.options})
        if self.name in {'biodecision', 'qwen35-base'}:
            from .local_decisions import LocalDecisions
            return LocalDecisions(self.name, **self.options)
        if self.name == 'jina-v35':
            from .native_models import JinaReranker
            return JinaReranker(**self.options)
        if self.name == 'clef-flash':
            from .native_models import ClefFlash
            return ClefFlash(**self.options)
        if self.name not in {'jev', 'openai-decisions'}:
            raise ValueError(f'Unknown method: {self.name}')
        from .hosted import HostedDecisions
        _ensure_api_key(self.name)
        provider = 'jev' if self.name == 'jev' else 'openai'
        return HostedDecisions(provider, calls=Path(folder) / 'calls.jsonl', **self.options)


def method(name, **options):
    """Choose a runnable method without loading weights or making API calls."""
    if name not in SUPPORTED_METHODS:
        raise ValueError(f'Unknown method: {name}; choose from {sorted(SUPPORTED_METHODS)}')
    return MethodChoice(name, options)


def collect_results(benchmarks, methods, *, output='results', max_seconds=None):
    """Run benchmarks × methods, checkpoint each pair and export one portable ZIP.

    Methods can be lazy choices from method() or existing Method implementations.
    Methods run in list order, each across all benchmarks. Completed work is reused.
    A failed pair is recorded while remaining pairs run.
    """
    benchmarks, methods = tuple(benchmarks), tuple(methods)
    if not benchmarks or not methods:
        raise ValueError('Select at least one benchmark and one method.')
    if len({(b.name, b.fingerprint) for b in benchmarks}) != len(benchmarks):
        raise ValueError('Duplicate benchmark selections.')
    if len({m.name for m in methods}) != len(methods):
        raise ValueError('Duplicate methods; use a separate output folder for different settings.')
    for choice in methods:
        if isinstance(choice, MethodChoice):
            _ensure_api_key(choice.name)
    output = Path(output).resolve()
    print(f'{len(benchmarks)} benchmarks × {len(methods)} methods = {len(benchmarks) * len(methods)} runs')
    selections = []
    for selected in benchmarks:
        folder = output / selected.name / selected.fingerprint
        session_path = folder / 'session.json'
        statuses = json.loads(session_path.read_text())['status'] if session_path.exists() else {}
        statuses.update(dict.fromkeys((m.name for m in methods), 'pending'))
        session = {'benchmark': selected.name, 'benchmark_fingerprint': selected.fingerprint,
                   'status': statuses,
                   'queries': [{'collection': item.collection, 'query_id': item.query_id, 'query': item.query}
                               for item in selected.inputs()]}
        write_json(session_path, session)
        selections.append((selected, folder, session))
    for choice in methods:
        for selected, folder, session in selections:
            statuses = session['status']
            algorithm = None
            print(f'{selected.name} ({len(selected.inputs())} queries) → {choice.name}', flush=True)
            try:
                algorithm = choice.build(folder / choice.name) if isinstance(choice, MethodChoice) else choice
                result = collect(selected, algorithm, folder / choice.name, max_seconds=max_seconds)
                statuses[choice.name] = 'complete' if len(result.results) == len(selected.inputs()) else 'partial'
                print(' ', statuses[choice.name], f'({len(result.results)}/{len(selected.inputs())} queries)', flush=True)
            except Exception as error:
                statuses[choice.name] = 'failed'
                message = str(error)
                for key in ['OPENAI_API_KEY', 'OPENROUTER_API_KEY']:
                    if os.environ.get(key):
                        message = message.replace(os.environ[key], '[REDACTED]')
                print(f'  failed: {type(error).__name__}: {message[:1000]}', flush=True)
            finally:
                del algorithm
                gc.collect()
                if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
                    sys.modules['torch'].cuda.empty_cache()
                write_json(folder / 'session.json', session)
    archive = export_results(output, output.with_name(output.name + '.zip'))
    print(f'Results: {output}\nArchive: {archive}')
    return archive


def download_results(archive):
    """Download in Colab; show the archive location in a local notebook."""
    try:
        from google.colab import files
    except ImportError:
        print(Path(archive).resolve())
    else:
        files.download(str(archive))


def collect(benchmark, method, folder, *, max_seconds=None):
    """Resume rankings and evaluate only when every selected query is complete."""
    folder = Path(folder)
    result = run(benchmark, method, checkpoint=folder / 'run.json', max_seconds=max_seconds)
    if len(result.results) == len(benchmark.inputs()):
        evaluate(benchmark, result).save(folder / 'run.evaluation.json')
    return result


def export_results(folder, destination):
    """Zip portable results without flattening subfolders or including credentials."""
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    names = {'run.json', 'run.evaluation.json', 'calls.jsonl', 'session.json'}
    paths = [p for p in sorted(folder.rglob('*')) if p.is_file() and not p.is_symlink() and p.name in names]
    if not paths:
        raise ValueError('No collected result files to export')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(folder))
    return destination
