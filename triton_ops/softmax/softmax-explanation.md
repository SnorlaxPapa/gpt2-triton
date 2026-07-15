# Introduction

A naive softmax implementation without memory optimization and operation fusion performs multiple passes over global memory, materializing intermediate tensors after each operation. This results in approximately 8MN+4M element transfers between HBM and on-chip memory, making the kernel memory-bound. The Triton implementation fuses these operations into a single kernel, reducing the required global memory traffic to one read and one write per element. For a detailed explanation of the naive softmax explanation, see the [Softmax Explanation](#softmax-explanation).











# Softmax Explanation

The softtmax function is given by $\text{softmax}(x_i)=\frac{e^{x_i - x_\max}}{\sum_j e^{x_j - x_\max}}$.

- Softmax, when implemented naively without , operates as follows:
    1. For a (M, N) matrix, each row is loaded from HBM into registers/shared memory, the maximum value of that row is computed, and the resulting maximum is written to HBM. MN read and M writes.
    2. To calculate x - x_max, we read the (M, N) x matrix and (M, ) x_max matrix. We then write back a (M, N) matrix. We substract x_max to prevent floating-point overflow. So we read MN + M elements and write back MN elements.
    3. Then, we need to exponentiate x to find the numerator in the softmax function. This is another MN read AND a MN write. 
    4. Then we need to load the new numerator (M, N) matrix again, and sum each row to find the normalization factor/denominator, producing a (M, ) vector. This is a MN read and a M write as we write back the normalization factor back to the HBM.
    5. Finally, we need to load both our numerators (M, N) and denominators (M, ) from the HBM to SRAM. We calculate our softmax and write back a (M, N) array. This is a MN + M read from the HBM and a MN write back to the SRAM.
    
    In total, the naive implementation performs 5MN+2M reads and 3MN+2M writes, for a total of 8MN+4M element transfers between HBM and the GPU's on-chip memory. Since memory bandwidth is significantly lower than the computational throughput of the Streaming Multiprocessors (SMs), the kernel becomes memory-bound: the SMs spend a significant fraction of execution time waiting for data to arrive from global memory rather than performing arithmetic.

