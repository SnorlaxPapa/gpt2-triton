import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

"""
a quick side note. i was writing the code halfway under the presumption that our dimension could somehow exceed 16k, which is why i 
followed the methodology of breaking up the columns into BLOCK_SIZE (i dont really why the tutorial blocks off N > block_size as a safety guardrail, so i just removed it)
however upon further consideration i've realized that my intended purpose for my code will never exceed 16k (our element limit is below 16k for 64kb programs)
regardless, i have proceeded to code out the entire thing following this train of thought. who knows... maybe one day we'll have 16k dim embeddings
"""
@triton.autotune(
    [
        triton.Config({"num_stages": s}, num_warps=w)
        for s in [1, 2, 3, 4, 5, 6] for w in [2, 4, 8, 16]
    ],
    key=["N", "BLOCK_SIZE"]
)
@triton.jit
def layer_norm_fwd(
    x_pointer,
    y_pointer,
    stride,
    W,
    B,
    mean_pointer,
    rstd_pointer, #1/std we save it
    N, #N is the number of columns jic our stride is different from N (maybe for a transpose)
    eps,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
  # we calculate our row position
  row_index = tl.program_id(0)
  row_position = x_pointer + row_index * stride
  
  #initialize our row sum accumulator
  _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for col_block in tl.range(0, N, BLOCK_SIZE, num_stages=num_stages): #num stages to preload
    #identify the column offsets and capture our row
    off = col_block + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    cols = row_position + off
  
    #load our row 
    row = tl.load(cols, mask=mask, other=0.).to(tl.float32) #we load other as 0 as we need to find the mean
    #we accumulate our rows before we can find the mean
    _mean += row

  #once sum is accumulated for our entire row, we find the mean
  mean = tl.sum(_mean, axis=0)/N

  #using this mean, we calculate our variance which is sum_i(x_i - mean)**2/N
  #again we must load our blocks
  _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for col_block in tl.range(0, N, BLOCK_SIZE, num_stages=num_stages): 
    off = col_block + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    cols = row_position + off
  
    #load our row 
    row = tl.load(cols, mask=mask, other=0.).to(tl.float32) 
    #accumulate (xi - mean)**2
    x_minus_mean = tl.where(mask, row - mean, 0.)
    _var += x_minus_mean * x_minus_mean
  
  #once we have calculated sum_i(x_i - mean)**2, we divide by N
  var = tl.sum(_var, axis=0) / N

  #compute rstd
  rstd = 1/tl.sqrt(var + eps)

  #sync our rstd and mean pointers
  mean_pointer = mean_pointer + row_index
  rstd_pointer = rstd_pointer + row_index

  #store our rstd and mean
  tl.store(mean_pointer, mean)
  tl.store(rstd_pointer, rstd)

  #normalize and compute linear transformation
  for col_block in tl.range(0, N, BLOCK_SIZE, num_stages=num_stages):
    off = col_block + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    cols = row_position + off
  
    #load our row 
    x = tl.load(cols, mask=mask).to(tl.float32) 
    weight = tl.load(W+off, mask=mask) #dont need a others as the padded rows and columns won't be accessed and saved for .store
    bias = tl.load(B+off, mask=mask)

    x_hat = (x - mean) * rstd
    y = x_hat * weight + bias

    #write to that specific (row, BLOCK_SIZE):
    y_row = y_pointer + row_index * stride
    y_index = y_row + off
    tl.store(y_index, y, mask=mask)




class LayerNorm(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, weight, bias, eps): 
    """
    forward call for layer norm. we don't need to ini num_stages here as it will be auto tuned 
    """
    #we flatten our tensor into 2d, maintaining its num columns
    x_reshape = x.reshape(-1, x.shape[-1])

    #extract our shape and initialize our normalized vector
    n_rows, N = x_reshape.shape
    x_norm = torch.empty_like(x, dtype=torch.float32)

    stride = x_reshape.stride(0)
    
    #we calculate our BLOCK_SIZE. Each program should <64KB of space at a time. float32 4bytes, float16 2bytes etc
    MAX_SIZE = 65536 // x.element_size() #64Kb = 64 * 1024 = 65536 bytes
    BLOCK_SIZE = min(MAX_SIZE, triton.next_power_of_2(N))

    #initialize our mean and rstd
    mean_pointer = torch.empty((n_rows), dtype=torch.float32, device=x.device) #torch.empty doesnt copy device unlike .zeros_like
    rstd_pointer = torch.empty((n_rows), dtype=torch.float32, device=x.device) 

    #launch our fused kernel with a n_rows, 1, 1 grid size, i.e. one row per program
    layer_norm_fwd[(n_rows, 1, 1)](
      x,
      x_norm,
      stride,
      weight,
      bias,
      mean_pointer,
      rstd_pointer, 
      N, 
      eps,
      BLOCK_SIZE,
    )

    ctx.save_for_backward(x, weight, bias, mean_pointer, rstd_pointer)
    ctx.eps = eps

    return x_norm


def check_layernorm():
    batch, row, col = (2, 10, 128)
    x = torch.randn(batch, row, col, device='cuda', requires_grad=True, dtype=torch.float32)
    w = torch.randn(col, device='cuda', requires_grad=True, dtype=torch.float32)
    b = torch.randn(col, device='cuda', requires_grad=True, dtype=torch.float32)
    eps = 1e-5

    # forward
    y_triton = LayerNorm.apply(x, w, b, eps)
    y_torch = torch.nn.functional.layer_norm(x, (col, ), w, b, eps).to(torch.float32)
    print("Forward match:", torch.allclose(y_triton, y_torch, atol=1e-5))
    print("Forward diff:", (y_triton - y_torch).abs().max().item())

check_layernorm()
