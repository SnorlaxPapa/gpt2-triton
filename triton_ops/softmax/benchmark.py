import torch
import triton
import triton.language as tl
from triton.runtime import driver
from softmax_triton import Softmax

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(1, 65)], 
        line_arg="provider",
        line_vals=["triton", "torch", "compiled"],
        line_names=["Triton", "Torch", "Compiled Torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  
        ylabel="GB/s",  
        plot_name="softmax-performance",  
        args={'M': 1024}, #the argument that stays constant and the value we provide for it
    )
)
def forward_benchmark(M, N, provider):
    return benchmark(M, N, provider, dir="forward")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(1, 65)],
        line_arg="provider",
        line_vals=["triton", "torch", "compiled"],
        line_names=["Triton", "Torch", "Compiled Torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  
        ylabel="GB/s",  
        plot_name="softmax-performance",  
        args={'M': 512}, #the argument that stays constant and the value we provide for it
    )
)
def backward_benchmark(M, N, provider):
    return benchmark(M, N, provider, dir="backward")


def benchmark(M, N, provider, dir="forward"):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    dy = .1 * torch.randn_like(x)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        fn = lambda: torch.softmax(x, dim=-1)
    if provider == 'triton':
        fn = lambda: Softmax.apply(x)
    if provider == 'compiled':
        compiled = torch.compile(torch.softmax)
        fn = lambda: compiled(x, dim=-1)

    #we run it 200 times to minimize variance 
    if dir == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(fn, rep=50, quantiles=quantiles)
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    if dir == "backward":
        y_out = fn()
        backward = lambda: y_out.backward(dy, retain_graph=True)
        ms, min_ms, max_ms = triton.testing.do_bench(backward, quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        
    return gbps(ms), gbps(max_ms), gbps(min_ms)
        




backward_benchmark.run(show_plots=False, print_data=True, save_path=".")
# power_of_2_benchmark.run(show_plots=False, print_data=True, save_path="./two")