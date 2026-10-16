"""Pinned native Jina reranking and Kev decision-head adapters."""
import importlib.util
from pathlib import Path
import re
import sys
from urllib.request import urlopen

from .hosted import expected_grade, request_body
from .inference import runtime
from .methods import Pointwise


class JinaReranker(Pointwise):
    def __init__(self, *, device='cuda', dtype='float16'):
        self._torch, self._dtype, settings = runtime(device, dtype)
        self._device, self._model = device, None
        super().__init__('jina_v35', self._score, {
            'model': 'jinaai/jina-reranker-v3.5', 'revision': 'e8a93f33f0b22108f8c2364f8484ce3422552fbc',
            **settings, 'score': 'native_listwise_cosine', 'code_revision': 'e8a93f33f0b22108f8c2364f8484ce3422552fbc',
            'evidence_policy': 'reject native truncation or sanitization; preserve original strings',
        })

    def _score(self, query, candidates):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            model, revision = self.parameters['model'], self.parameters['revision']
            self._model = AutoModel.from_pretrained(model, revision=revision, code_revision=revision,
                trust_remote_code=True, dtype=self._dtype, token=False, attn_implementation='sdpa').to(self._device).eval()
            # The publisher's lazy tokenizer otherwise follows the moving main branch.
            self._model._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, token=False, padding_side='left')
        tokenizer = self._model._tokenizer
        texts = [doc.text for doc in candidates]
        if len(tokenizer.encode(query)) >= 1024 or any(len(tokenizer.encode(text)) >= 8192 for text in texts):
            raise ValueError('Shared evidence would be truncated by Jina native rerank')
        special = self._model.special_tokens.values()
        if any(token in text for token in special for text in [query, *texts]):
            raise ValueError('Evidence contains special token strings removed by the native Jina formatter')
        with self._torch.inference_mode():
            rows = self._model.rerank(query, texts)
        if len(rows) != len(texts) or {int(r['index']) for r in rows} != set(range(len(texts))):
            raise ValueError('Jina did not return exactly one score per candidate')
        indexed = {int(r['index']): float(r['relevance_score']) for r in rows}
        return [indexed[i] for i in range(len(texts))]


KEV_CODE_REVISION = '1f04cfa28e6a708485b28a9585010b9e436b776e'


def kev_runtime():
    """Load only the publisher's checkpoint/model modules from a pinned commit."""
    from huggingface_hub import cached_assets_path
    directory = Path(cached_assets_path('decision-rerank', namespace='kev', subfolder=KEV_CODE_REVISION))
    for name in ('model.py', 'checkpoint.py'):
        target = directory / name
        if not target.exists():
            url = f'https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_CODE_REVISION}/kev/{name}'
            with urlopen(url, timeout=60) as response:
                source = response.read()
            target.write_bytes(source)
    name = '_decision_rerank_kev_' + KEV_CODE_REVISION[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory/'checkpoint.py',
                                                     submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


class Kev4B(Pointwise):
    def __init__(self, *, batch_size=1, device='cuda', dtype='float16'):
        if batch_size < 1:
            raise ValueError('Use a positive batch size')
        self._torch, self._dtype, settings = runtime(device, dtype)
        self._device, self._batch_size = device, batch_size
        self._model = self._tokenizer = None
        super().__init__('kev_4b_pointwise', self._score, {
            'model': 'jaredpalmer/kev-4b', 'revision': '6cfce5c2fa4b4bd64026336ab649c5ca78857d52',
            'base_model': 'Qwen/Qwen3.5-4B-Base', 'base_revision': '1001bb4d826a52d1f399e183466143f4da7b741b',
            **settings, 'batch_size': batch_size, 'code_revision': KEV_CODE_REVISION,
            'score': 'expected_grade_native_pointer_head', 'temperature': 2.406050072164233,
            'max_input_tokens': 8192, 'schema': request_body('jev', 'jaredpalmer/kev-4b', '', '')['questions'],
            'load_device': 'cpu', 'merge_lora': True, 'fused_kernels': False, 'cuda_graphs': False,
        })

    def _score(self, query, candidates):
        if self._model is None:
            from huggingface_hub import snapshot_download
            native = kev_runtime()
            path = snapshot_download(self.parameters['model'], revision=self.parameters['revision'],
                                     allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'head.pt'])
            checkpoint = native.Checkpoint(path)
            if (checkpoint.meta.base, checkpoint.meta.base_revision, checkpoint.meta.temperature) != (
                self.parameters['base_model'], self.parameters['base_revision'], self.parameters['temperature']
            ):
                raise ValueError('Kev checkpoint base or calibration differs from recorded settings')
            # Merge on CPU so the temporary LoRA merge buffers never consume VRAM.
            self._tokenizer, model = checkpoint.load('cpu', native.LoadOptions(
                dtype=self._dtype, merge=True, backend='torch', fused=False, cuda_graphs=False,
                attn='sdpa' if self._device == 'cuda' else 'eager',
            ))
            self._model = model.to(self._device).eval()
            self._model.device = self._device
        scores = []
        for start in range(0, len(candidates), self._batch_size):
            encoded = []
            for document in candidates[start:start+self._batch_size]:
                body = request_body('jev', self.parameters['model'], query, document.text)
                # The native encoder rewrites special-token strings; reject rather than alter shared evidence.
                if re.search(r'<\|[A-Za-z0-9_]+\|>', body['state']):
                    raise ValueError('Evidence contains special token strings rewritten by the native Kev formatter')
                question = body['questions']['relevance']
                record = {'state': body['state'], 'questions': [{
                    'instr': question['instructions'], 'options': question['criteria'], 'label': 0,
                }]}
                encoded.append(self._model.encode(self._tokenizer, record, max_state=8192, max_branch=8192, strict=True))
            with self._torch.inference_mode():
                outputs = self._model.forward_batch(encoded)
            for row in outputs:
                probabilities = row[0].float().softmax(-1).cpu().tolist()
                scores.append(expected_grade('jev', {'answers': {'relevance': {
                    'type': 'score', 'probabilities': {str(i): p for i, p in enumerate(probabilities)},
                }}}))
        return scores
