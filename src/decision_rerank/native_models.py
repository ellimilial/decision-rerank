"""Pinned native Jina reranking and Kev decision-head adapters."""
import importlib.util
from pathlib import Path
import re
import sys
from urllib.request import urlopen

from .hosted import expected_grade, request_body
from .inference import runtime
from .methods import Pointwise


JINA_ATTENTION_CHUNK = 256


def jina_attention(module, query, key, value, attention_mask, *, scaling=None,
                   sliding_window=None, dropout=0., **kwargs):
    """Exact causal attention in query blocks; documents remain one joint sequence."""
    import torch
    if attention_mask is not None or query.shape[0] != 1 or query.shape[2] != key.shape[2] or dropout:
        raise ValueError('Jina attention requires one unpadded sequence, without cache or dropout')
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    output = torch.empty_like(query)
    for start in range(0, query.shape[2], JINA_ATTENTION_CHUNK):
        end = min(start + JINA_ATTENTION_CHUNK, query.shape[2])
        left = max(0, start - sliding_window + 1) if sliding_window else 0
        q_positions = torch.arange(start, end, device=query.device)[:, None]
        k_positions = torch.arange(left, end, device=query.device)[None, :]
        mask = k_positions <= q_positions
        if sliding_window:
            mask &= k_positions > q_positions - sliding_window
        output[:, :, start:end] = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:end], key[:, :, left:end], value[:, :, left:end],
            attn_mask=mask, scale=scaling, dropout_p=0.,
        )
    return output.transpose(1, 2).contiguous(), None


def jina_attention_mask(*, batch_size, q_length, kv_length, q_offset, kv_offset,
                        attention_mask=None, **kwargs):
    """Validate the native single-sequence input without allocating square masks."""
    if (batch_size != 1 or q_length != kv_length or q_offset != 0 or kv_offset != 0
        or (attention_mask is not None and not attention_mask.all())):
        raise ValueError('Jina attention requires one unpadded sequence, without cache')
    return None


class JinaReranker(Pointwise):
    def __init__(self, *, device='cuda', dtype='float16'):
        self._torch, self._dtype, settings = runtime(device, dtype)
        self._device, self._model = device, None
        super().__init__('jina_v35', self._score, {
            'model': 'jinaai/jina-reranker-v3.5', 'revision': 'e8a93f33f0b22108f8c2364f8484ce3422552fbc',
            **settings, 'attention': 'query_chunked_sdpa_v1', 'attention_query_chunk': JINA_ATTENTION_CHUNK,
            'score': 'native_listwise_cosine', 'code_revision': 'e8a93f33f0b22108f8c2364f8484ce3422552fbc',
            'evidence_policy': 'reject native truncation or sanitization; preserve original strings',
        })

    def _score(self, query, candidates):
        if self._model is None:
            from transformers import AttentionInterface, AttentionMaskInterface, AutoModel, AutoTokenizer
            AttentionInterface.register('decision_rerank_jina', jina_attention)
            AttentionMaskInterface.register('decision_rerank_jina', jina_attention_mask)
            model, revision = self.parameters['model'], self.parameters['revision']
            self._model = AutoModel.from_pretrained(model, revision=revision, code_revision=revision,
                trust_remote_code=True, dtype=self._dtype, token=False,
                attn_implementation='decision_rerank_jina').to(self._device).eval()
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


def load_kev_model(native, checkpoint, device, dtype):
    """Load weights onto the target device; stage only one LoRA layer on CPU."""
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer

    meta = checkpoint.meta
    if checkpoint.full or meta.weights_dtype != 'fp32' or checkpoint.adapter_config().get('trainable_token_indices'):
        raise ValueError('Expected the pinned Kev FP32 LoRA checkpoint without trained token embeddings')
    tokenizer = native.load_tokenizer(meta.base, revision=meta.base_revision)
    model = native.DecisionModel(meta.base, tokenizer, device, revision=meta.base_revision,
        head_dim=meta.head_dim, option_isolation=meta.option_isolation, dtype=dtype,
        attn='sdpa' if device == 'cuda' else 'eager', direct_load=True)
    # Keep PEFT's normal FP32 adapter loading; meta loading can round adapter values to FP16.
    model.lm = PeftModel.from_pretrained(model.lm, checkpoint.path, torch_device=device)
    # Keep the original CPU merge arithmetic without a full host copy of the backbone.
    with torch.no_grad():
        for layer in model.lm.modules():
            if isinstance(layer, LoraLayer):
                layer.to('cpu')
                layer.merge()
                layer.to(device)
    model.lm = model.lm.unload().to(dtype)
    model.head.load_state_dict(meta.head)
    model.head.temperature = meta.temperature
    return tokenizer, model.eval()


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
            'load_device': device, 'merge_lora': True, 'merge_device': 'cpu', 'merge_scope': 'one_lora_layer',
            'fused_kernels': False, 'cuda_graphs': False,
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
            self._tokenizer, self._model = load_kev_model(native, checkpoint, self._device, self._dtype)
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
