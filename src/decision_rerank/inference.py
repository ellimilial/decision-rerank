"""Shared loading, cleanup and precision policy for optional local models."""
from .methods import Pointwise


class LocalPointwise(Pointwise):
    def close(self):
        """Drop loaded assets without making a CPU copy; rank() can reload them."""
        self._model = self._tokenizer = None


def qwen35_kernels():
    """Record the actual Transformers paths, including partial kernel availability."""
    from importlib.metadata import PackageNotFoundError, version
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    paths = {
        'delta_rule': 'fla' if qwen.chunk_gated_delta_rule is not None else 'torch',
        'causal_conv': 'causal_conv1d' if qwen.causal_conv1d_fn is not None else 'torch',
        'gated_norm': 'fla' if qwen.FusedRMSNormGated is not None else 'torch',
    }
    packages = {}
    for name in ('flash-linear-attention', 'fla-core', 'causal-conv1d', 'triton'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    print('Qwen3.5 kernels: ' + ', '.join(f'{key}={value}' for key, value in paths.items()), flush=True)
    return {**paths, 'versions': packages}


def runtime(device='cuda', dtype='float16'):
    import torch
    import transformers
    if device not in {'cpu', 'cuda'} or dtype not in {'float32', 'float16', 'bf16'}:
        raise ValueError('Choose cpu/cuda and float32/float16/bf16')
    if device == 'cpu' and dtype != 'float32':
        raise ValueError('Use float32 for CPU inference')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Choose a GPU runtime, or an explicit CPU method')
    if device == 'cuda' and dtype == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bf16; use float16 or float32')
    precision = {'float32': torch.float32, 'float16': torch.float16, 'bf16': torch.bfloat16}[dtype]
    parameters = {'device': device, 'dtype': dtype, 'torch': torch.__version__,
                  'transformers': transformers.__version__,
                  'cuda': torch.version.cuda if device == 'cuda' else None,
                  'gpu': torch.cuda.get_device_name() if device == 'cuda' else None}
    return torch, precision, parameters
