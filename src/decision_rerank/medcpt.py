"""Pinned biomedical cross-encoder on CPU or GPU, imported only when selected."""
from .methods import Pointwise
from .inference import runtime

MODEL = "ncbi/MedCPT-Cross-Encoder"
REVISION = "71caf65d4927987813984f54c284405a13fcca49"


class MedCPT(Pointwise):
    def __init__(self, *, batch_size: int = 32, dtype: str = "bf16", device: str = "cuda"):
        if batch_size < 1:
            raise ValueError("Use a positive batch size")
        self._torch, self._dtype, settings = runtime(device, dtype)
        self._device = device
        self._batch_size = batch_size
        self._model = self._tokenizer = None
        super().__init__("medcpt_cpu" if device == "cpu" else "medcpt", self._score, {
            "model": MODEL, "revision": REVISION, "batch_size": batch_size, "dtype": dtype,
            **settings,
        })

    def _score(self, query, candidates):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, revision=REVISION, token=False, dtype=self._dtype,
            ).to(self._device).eval()
        scores = []
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start:start + self._batch_size]
            encoded = self._tokenizer([[query, doc.text] for doc in batch], padding=True,
                                      truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > 512:
                raise ValueError("Shared evidence exceeds MedCPT context; do not silently truncate")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                scores.extend(self._model(**encoded).logits.squeeze(-1).float().cpu().tolist())
        return scores
