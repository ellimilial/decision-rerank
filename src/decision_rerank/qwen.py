"""Pinned Qwen3 reranker using the publisher's yes/no scoring format."""
from .inference import LocalPointwise, runtime

MODEL = 'Qwen/Qwen3-Reranker-4B'
REVISION = '22e683669bc0f0bd69640a1354a6d0aebcfeede5'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class QwenReranker(LocalPointwise):
    def __init__(self, *, batch_size=2, dtype='float16'):
        if batch_size < 1 or dtype not in {'float16', 'bf16'}:
            raise ValueError('Use a positive batch size and float16/bf16 precision')
        self._torch, self._dtype, settings = runtime('cuda', dtype)
        self._batch_size = batch_size
        self._model = self._tokenizer = None
        super().__init__('qwen3_reranker_4b', self._score, {
            'model': MODEL, 'revision': REVISION, 'batch_size': batch_size, 'dtype': dtype,
            'instruction': INSTRUCTION, 'prefix': PREFIX, 'suffix': SUFFIX,
            'score': 'yes_logit_minus_no_logit', 'max_input_tokens': 8192,
            **settings,
        })

    def _score(self, query, candidates):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left', token=False)
            self._model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=self._dtype, token=False,
                attn_implementation='sdpa', device_map='cuda',
            ).eval()
        tokenizer = self._tokenizer
        yes, no = [tokenizer.encode(word, add_special_tokens=False) for word in ['yes', 'no']]
        if len(yes) != 1 or len(no) != 1:
            raise ValueError('Expected single-token yes/no labels')
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        scores = []
        for start in range(0, len(candidates), self._batch_size):
            pairs = [f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {d.text}'
                     for d in candidates[start:start + self._batch_size]]
            ids = tokenizer(pairs, add_special_tokens=False, truncation=False)['input_ids']
            ids = [prefix + row + suffix for row in ids]
            if max(map(len, ids)) > 8192:
                raise ValueError('Evidence exceeds Qwen input limit; refusing silent truncation')
            inputs = tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            with self._torch.inference_mode():
                logits = self._model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                scores.extend((logits[:, yes[0]] - logits[:, no[0]]).cpu().tolist())
        return scores
