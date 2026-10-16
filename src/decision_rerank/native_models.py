"""Pinned native listwise Jina and native-head Clef adapters."""
import importlib.util
from pathlib import Path
import sys

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


class ClefFlash(Pointwise):
    def __init__(self, *, batch_size=1, device='cuda', dtype='float16'):
        if batch_size < 1:
            raise ValueError('Use a positive batch size')
        self._torch, self._dtype, settings = runtime(device, dtype)
        self._device, self._batch_size = device, batch_size
        self._model = self._processor = self._native = None
        super().__init__('clef_flash_pointwise', self._score, {
            'model': 'Cloudflare/clef-flash', 'revision': '8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3',
            **settings, 'batch_size': batch_size, 'score': 'expected_grade_native_schema_head',
            'max_input_tokens': 16384, 'schema': request_body('jev', 'Cloudflare/clef-flash', '', '')['questions'],
        })

    def _score(self, query, candidates):
        if self._model is None:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(self.parameters['model'], revision=self.parameters['revision'], token=False))
            spec = importlib.util.spec_from_file_location('decision_rerank_clef_native', path/'joint_schema_model.py')
            self._native = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = self._native
            spec.loader.exec_module(self._native)
            self._model, self._processor = self._native.load_release_model(path, device=self._device, dtype=self._dtype)
        scores = []
        for start in range(0, len(candidates), self._batch_size):
            records = [request_body('jev', self.parameters['model'], query, doc.text)
                       for doc in candidates[start:start+self._batch_size]]
            # Encode without state cropping, then enforce the declared context limit.
            encoded = [self._native.encode_record(self._processor.tokenizer, r, processor=self._processor, max_length=1_000_000)
                       for r in records]
            if any(len(row.input_ids) > 16384 for row in encoded):
                raise ValueError('Shared evidence exceeds Clef context; refusing state truncation')
            batch = self._native.collate_records(encoded, self._processor.tokenizer.pad_token_id, self._torch.device(self._device))
            with self._torch.inference_mode():
                outputs = self._model(batch)
            for row, record in zip(outputs, records):
                probabilities = row[0].float().softmax(-1).cpu().tolist()
                scores.append(expected_grade('jev', {'answers': {'relevance': {
                    'type': 'score', 'probabilities': {str(i): p for i, p in enumerate(probabilities)},
                }}}))
        return scores
