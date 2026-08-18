# Triton GPU Kernels

Custom forward and backward Triton kernels for Softmax, LayerNorm, and FlashAttention-2, benchmarked against PyTorch on an NVIDIA RTX 4080.

# Project Overview

This repository contains custom Triton implementations of Softmax, Layer Normalization, and FlashAttention-2, developed from first principles to explore GPU kernel optimization. Each kernel includes forward and backward passes, was numerically validated against its PyTorch equivalent, and was benchmarked against eager and compiled PyTorch implementations. The project’s primary goal was to develop a deeper understanding of Triton and GPU kernel optimization.

# Implemented Kernels and Optimization Highlights

In this project, I implemented the following kernels:
- **Softmax**:
    - Fuses row-wise maximum, exponentiation, normalization, and output storage into a single kernel, avoiding intermediate global-memory writes. A detailed walkthrough of the forward and backward passes, as well as the benchmarked results can be found [here](triton_ops/softmax/README.md).
- **LayerNorm**:
    - Fuses the input-gradient computation and uses grouped partial-gradient buffers with atomic synchronization, followed by a second-stage reduction for the weight and bias gradients. Similarly, a detailed walkthrough of the algorithm and benchmark results can be found [here](triton_ops/layer_norm/README.md).
- **FlashAttention-2**:
    - Uses tiled online softmax to avoid materializing the quadratic attention matrix. The backward pass reconstructs attention tiles on-chip and computes `dQ`, `dK`, and `dV` in one main traversal. For a detailed explanation of the code and results, please click [here](triton_ops/flashattn/README.md).
- **Autotuning:** Uses Triton autotuning to select block sizes, warp counts, and pipeline stages for different input configurations.

# Performance Summary

The results below were measured using FP16 inputs on an NVIDIA RTX 4080. Reported bandwidth and FLOP rates are algorithmic throughput estimates rather than hardware-counter measurements. For full results, please refer to the respective READMEs for each kernel.

<table>
  <thead>
    <tr>
      <th>Kernel</th>
      <th>Benchmark Configuration</th>
      <th>Metric</th>
      <th>Custom Triton Performance</th>
      <th>Comparison with PyTorch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Softmax Forward</b></td>
      <td>
        Sequence Length = 512<br>
        Softmax Dimension = 128–8192
      </td>
      <td>Algorithmic Effective Bandwidth</td>
      <td>Approximately 350 GB/s throughout much of the tested range</td>
      <td>Faster than eager PyTorch and generally competitive with <code>torch.compile</code></td>
    </tr>
    <tr>
      <td><b>Softmax Backward</b></td>
      <td>
        Sequence Length = 512<br>
        Softmax Dimension = 128–8192
      </td>
      <td>Algorithmic Effective Bandwidth</td>
      <td>Approximately 300–370 GB/s throughout much of the tested range</td>
      <td>Generally faster than eager PyTorch and competitive with <code>torch.compile</code></td>
    </tr>
    <tr>
      <td><b>LayerNorm Forward</b></td>
      <td>
        Sequence Length = 4096<br>
        Hidden Dimension = 512–15,872
      </td>
      <td>Algorithmic Effective Bandwidth</td>
      <td>Approximately 365–380 GB/s throughout much of the tested range</td>
      <td>Near parity with <code>torch.compile</code> and faster than eager PyTorch at larger dimensions</td>
    </tr>
    <tr>
      <td><b>LayerNorm Backward</b></td>
      <td>
        Sequence Length = 4096<br>
        Hidden Dimension = 512–15,872
      </td>
      <td>Algorithmic Effective Bandwidth</td>
      <td>Approximately 350–375 GB/s throughout much of the tested range</td>
      <td>Faster than both PyTorch implementations, reaching roughly 1.5× compiled and 2× eager PyTorch at larger dimensions</td>
    </tr>
    <tr>
      <td><b>FlashAttention-2 Forward</b></td>
      <td>
        Sequence Length = 1024–7680<br>
        Batch Size = 2<br>
        Heads = 4<br>
        Head Dimension = 64
      </td>
      <td>Algorithmic TFLOP/s</td>
      <td>Typically 34–47 TFLOP/s, with a peak of approximately 47 TFLOP/s</td>
      <td>Near parity with PyTorch FlashAttention SDPA, with occasional small wins within benchmark variability</td>
    </tr>
    <tr>
      <td><b>FlashAttention-2 Backward</b></td>
      <td>
        Sequence Length = 1024–7680<br>
        Batch Size = 2<br>
        Heads = 4<br>
        Head Dimension = 64
      </td>
      <td>Algorithmic TFLOP/s</td>
      <td>Typically 24–34 TFLOP/s, with a peak of approximately 38 TFLOP/s</td>
      <td>Generally reaches 80–90% of PyTorch FlashAttention SDPA throughput, with a larger gap at the longest sequence lengths</td>
    </tr>
  </tbody>
</table>


# Scope and Limitations

- The kernels were primarily optimized and benchmarked for FP16 inputs on an NVIDIA RTX 4080.
- Performance results are specific to the tested shapes and hardware.
- The implementations assume CUDA tensors and may require contiguous inputs.
- This is an educational implementation intended to explore kernel optimization rather than a drop-in replacement for production libraries.
- Supported dimensions, causal masking, and other kernel-specific restrictions are documented in their respective READMEs.

# References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton Documentation and Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)