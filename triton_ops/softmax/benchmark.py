import torch
import triton
import triton.language as tl
from triton.runtime import driver
from softmax_triton import Softmax

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]

    return ret



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(1, 70)], #roughly 1300 to 7800
        line_arg="provider",
        line_vals=["triton", "naive_softmax", "torch"],
        line_names=["Triton", "Naive Softmax", "Torch Softmax"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  
        ylabel="GB/s",  
        plot_name="softmax-performance",  
        args={'M': 1024}, #the argument that stays constant and the value we provide for it
    )
)
def forward_benchmark(M, N, provider):
    return benchmark(M, N, provider)



def benchmark(M, N, provider, n_runs=5):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        fn = lambda: torch.softmax(x, dim=-1)
    if provider == 'triton':
        fn = lambda: Softmax.apply(x)
    if provider == 'naive_softmax':
        fn = lambda: naive_softmax(x)

    #we run it a total of five times to minimize variance 
    total = [triton.testing.do_bench(fn) for _ in range (n_runs)]
    ms = sum(total)/len(total)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


forward_benchmark.run(show_plots=False, print_data=True, save_path=".")
# power_of_2_benchmark.run(show_plots=False, print_data=True, save_path="./two")