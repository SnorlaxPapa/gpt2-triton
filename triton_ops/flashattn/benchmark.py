from flash import Attention
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver
from torch._functorch import config
from torch.nn.attention import sdpa_kernel, SDPBackend


DEVICE = triton.runtime.driver.active.get_active_torch_device()
config.donated_buffer = False

#hoisting to module level so we dont recompile each time
compiled_fn = torch.compile(nn.functional.scaled_dot_product_attention)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[512 * i for i in range(2, 16)],
        line_arg='provider',
        line_vals=(['triton', 'torch', 'compiled']),
        line_names=(['Triton', 'Torch', 'Torch Compiled']),
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='TFLOP/s',
        plot_name='attn-forward',
        args={'B': 2, 'H': 4, 'N': 64, 'dtype': torch.float16, 'mode': 'forward'},
    ))


def bench_layer_norm(B, H, M, N, dtype, provider, mode='backward', device=DEVICE):
    # create data
    q = torch.randn((B, H, M, N), device=device, dtype=dtype) #B, H, SEQ, DIM
    k = torch.randn((B, H, M, N), device=device, dtype=dtype)  #B, H, SEQ, DIM
    v = torch.randn((B, H, M, N), device=device, dtype=dtype)  #B, H, SEQ, DIM
    scale = q.shape[-1] ** -0.5 
    quantiles = [0.5, 0.2, 0.8]
    
    def y_fwd():

        if provider == "triton":
            return Attention.apply(q, k, v, scale, False)

        if provider == "torch":
            return nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=False,
                        scale=scale,
                    )  
        
        if provider == "compiled":
            return compiled_fn(
                        q,
                        k,
                        v,
                        is_causal=False,
                        scale=scale,
                    )  

    # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
        flops = 4 * B * H * M**2 * N
        perf = lambda runtime_ms: flops * 1e-12 / (runtime_ms * 1e-3)
    # # backward pass
    # if mode == 'backward':
    #     y = y_fwd()
    #     gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
    #                                                  grad_to_none=[x], rep=500)
    return perf(ms), perf(max_ms), perf(min_ms)


bench_layer_norm.run(save_path='.', print_data=True)


