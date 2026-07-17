from layer_norm import LayerNorm
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    x = torch.randn(100, 10, device=DEVICE, dtype=torch.float32, requires_grad=True)
    dy = .1 * torch.randn_like(x)
    y = torch.softmax(x, dim=-1)
    y.backward(dy)

print(prof.key_averages().table(sort_by="cuda_time_total"))