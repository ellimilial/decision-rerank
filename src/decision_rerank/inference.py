"""Shared device/precision policy for optional local model adapters."""
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
