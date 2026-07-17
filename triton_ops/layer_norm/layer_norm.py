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
        triton.Config({}, num_warps=w)
        for w in [2, 4, 8, 16]
    ],
    key=["N"]
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
):
  # we calculate our row position
  row_index = tl.program_id(0)
  row_position = x_pointer + row_index * stride

  #identify the column offsets and capture our row
  off = tl.arange(0, BLOCK_SIZE)
  mask = off < N
  cols = row_position + off

  #load our row
  row = tl.load(cols, mask=mask, other=0.).to(tl.float32) #we load other as 0 as we need to find the mean
  #once sum is accumulated for our entire row, we find the mean
  mean = tl.sum(row, axis=0)/N

  #calculate our vaariance
  x_minus_mean = tl.where(mask, row - mean, 0.)
  x_minus_squared = x_minus_mean * x_minus_mean
  var = tl.sum(x_minus_squared, axis=0)/N

  #compute rstd
  rstd = 1/tl.sqrt(var + eps)

  #sync our rstd and mean pointers
  mean_pointer = mean_pointer + row_index
  rstd_pointer = rstd_pointer + row_index

  #store our rstd and mean
  tl.store(mean_pointer, mean)
  tl.store(rstd_pointer, rstd)

  #we compute our linear transformation
  weight = tl.load(W+off, mask=mask) #dont need a others as the padded rows and columns won't be accessed and saved for .store
  bias = tl.load(B+off, mask=mask)

  x_hat = (row - mean) * rstd
  y = x_hat * weight + bias

  #write to that specific (row, BLOCK_SIZE):
  y_row = y_pointer + row_index * stride
  y_index = y_row + off
  tl.store(y_index, y, mask=mask)


@triton.autotune(
    [
        triton.Config({}, num_warps=w)
        for w in [2, 4, 8, 16]
    ],
    key=["N"]
)
@triton.jit
def backward_dw_db_dx_fused(
    dy,
    dx,
    dw,
    db,
    x,
    weight,
    stride,
    N,
    mean_pointer,
    rstd_pointer,
    lock, #pointer to our lock
    BLOCK_SIZE: tl.constexpr, 
    GROUP_SIZE_M: tl.constexpr #autotuned
):
  """
  stage 1 of the backward pass splits our dy into different groups. 
  each group accumulates their dw and db in their respective buffers before they are added together in stage 2
  we split into group to reduce the queue for atomic operations
  we implement a lock and atomic cas to ensure indivisible read and write operations and prevent race conditions
  we also compute dx 
  """

  row_index = tl.program_id(0)
  lock_id = row_index % GROUP_SIZE_M 

  #initialize our x, y, dy pointers
  x_ptr = x + stride * row_index
  dy_ptr = dy + stride * row_index

  #set our cols and mask
  cols = tl.arange(0, BLOCK_SIZE)
  mask = cols < N

  #load our tensors, mean, weight, and rstd
  x = tl.load(x_ptr + cols, mask=mask, other=0.) #we use others=0 to not affect our dot product
  dy = tl.load(dy_ptr + cols, mask=mask, other=0.)
  mean = tl.load(mean_pointer + row_index)
  rstd = tl.load(rstd_pointer + row_index)
  w = tl.load(weight + cols, mask=mask).to(tl.float32)

  #initialize our lock, count, dw and db pointers. each group's buffer is size [N] and serves as an accumulator
  lock = lock + lock_id
  count_point = lock + GROUP_SIZE_M
  dw = dw + lock_id * N  + cols
  db = db + lock_id * N + cols

  #we compute dx
  #compute our xhat and wdy first
  xhat = tl.where(mask, (x - mean) * rstd, 0.)
  wdy = tl.where(mask, w * dy, 0.)

  #compute c1
  c1 = 1/N * tl.sum(xhat * wdy, axis=0)
  #compute c2
  c2 = 1/N * tl.sum(wdy, axis=0)

  #compute our dx
  dx_val = rstd * (wdy - c2 - c1 * xhat)
  #Write our dx back
  dx = dx + stride * row_index
  tl.store(dx + cols, dx_val, mask=mask)

  #calc partial dw and db
  dw_partial = (dy * xhat).to(w.dtype)
  db_partial = dy.to(w.dtype)

  #loop until lock opens
  while tl.atomic_cas(lock, 0, 1) == 1:
    pass

  #once lock opens, if buffer doesnt exist yet we initialize it
  count = tl.load(count_point)
  if count == 0:
    tl.atomic_xchg(count_point, 1)
  else:
    db_partial += tl.load(db, mask=mask)
    dw_partial += tl.load(dw, mask=mask)

  tl.store(db, db_partial, mask=mask)
  tl.store(dw, dw_partial, mask=mask)
  
  tl.debug_barrier() #ensure all threads have finished. each thread is executing different lines and need to ensure all storing has finished before unlocking

  tl.atomic_xchg(lock, 0) #unlock our lock


#no autotune cause this kernel is so tiny
@triton.jit
def _layer_norm_bwd_dwdb(DW,  
                         DB,  
                         FINAL_DW,  
                         FINAL_DB,  
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, 
                         BLOCK_SIZE_N: tl.constexpr):
  
    # we map our program id to the correct column in our buffers
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #initialize our accumulators
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the rows of DW and DB to get partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)

    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)



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
    x_norm = torch.empty_like(x)

    stride = x_reshape.stride(0)

    BLOCK_SIZE = triton.next_power_of_2(N)

    #initialize our mean and rstd
    mean_pointer = torch.empty((n_rows), dtype=torch.float32, device=x.device) #torch.empty doesnt copy device unlike .zeros_like
    rstd_pointer = torch.empty((n_rows), dtype=torch.float32, device=x.device)

    #launch our fused kernel with a n_rows, 1, 1 grid size, i.e. one row per program
    layer_norm_fwd[(n_rows, 1, 1)](
      x_reshape,
      x_norm,
      stride,
      weight,
      bias,
      mean_pointer,
      rstd_pointer,
      N,
      eps,
      BLOCK_SIZE=BLOCK_SIZE,
    )

    ctx.save_for_backward(x, weight, bias, mean_pointer, rstd_pointer)

    return x_norm

  @staticmethod
  def backward(ctx, dy):
    #load our context and get necessary dimensions
    x, weight, bias, mean_pointer, rstd_pointer = ctx.saved_tensors
    dy_reshaped = dy.reshape(-1, dy.shape[-1])
    stride = dy_reshaped.stride(0)

    M, N = dy_reshaped.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    stride = dy_reshaped.shape[-1]

    GROUP_SIZE_M = 64
    if N <= 8192: GROUP_SIZE_M = 96
    if N <= 4096: GROUP_SIZE_M = 128
    if N <= 1024: GROUP_SIZE_M = 256

    #initialize any additional pointers
    dx = torch.empty_like(x)
    dw_buffer = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=x.device)
    db_buffer = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=x.device)

    lock = torch.zeros(2*GROUP_SIZE_M, dtype=torch.int32, device=x.device)

    #stage 1 to calculate dx, reduce dw and db into different partial sums
    backward_dw_db_dx_fused[(M,)](
      dy_reshaped,
      dx,
      dw_buffer,
      db_buffer,
      x,
      weight,
      stride,
      N,
      mean_pointer,
      rstd_pointer,
      lock, 
      BLOCK_SIZE=BLOCK_SIZE,
      GROUP_SIZE_M=GROUP_SIZE_M,
    )

    #stage 2, reduce partial dw and db into final
    dw_final = torch.zeros([N], dtype=x.dtype, device=x.device)
    db_final = torch.zeros([N], dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    _layer_norm_bwd_dwdb[grid](dw_buffer,  
                         db_buffer,  
                         dw_final,  
                         db_final,  
                         GROUP_SIZE_M,  
                         N,  
                         BLOCK_SIZE_M=32, 
                         BLOCK_SIZE_N=128)
    
    return dx, dw_final, db_final, None


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

    #backward
    dy = torch.randn_like(x)
    y_triton.backward(dy)
    grad_triton = x.grad.clone()

    x.grad = None
    y_torch = torch.nn.functional.layer_norm(x, (col, ), w, b, eps).to(torch.float32)
    y_torch.backward(dy)
    grad_torch = x.grad.clone()
    print("Backward match:", torch.allclose(grad_triton, grad_torch, atol=1e-5))
    print("Max diff:", (grad_triton - grad_torch).abs().max().item())

