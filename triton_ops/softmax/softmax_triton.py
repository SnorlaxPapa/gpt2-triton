import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver


"""
softmax kernel processes BLOCK_SIZE_M * DIM. code is written for BLOCK_SIZE_M to be power of 2. 
For non power of 2 sequences and dimensions, padding can be used with triton's next power of 2 fn
"""

@triton.autotune(
  configs=[triton.Config(
            {"BLOCK_SIZE_M": BM}, 
            num_warps=w)
            for BM in [1, 2, 4]
            for w in [1, 2, 4, 8]
          ],
  key=["rows" , "BLOCK_SIZE_N"],
  cache_results=True,
)
@triton.jit
def _softmax_bwd(
   dOut,
   rows,
   dim,
   out,
   dx,
   BLOCK_SIZE_M: tl.constexpr,
   BLOCK_SIZE_N: tl.constexpr,
   num_warps: tl.constexpr,
):
  group_pos = tl.program_id(0)
  row_off = tl.arange(0, BLOCK_SIZE_M)
  col_off = tl.arange(0, BLOCK_SIZE_N)

  dOut_idx = (
    dOut + 
    group_pos * BLOCK_SIZE_M * dim + 
    row_off[:, None] * dim + 
    col_off[None, :]
  )

  out_idx = (
    out + 
    group_pos * BLOCK_SIZE_M * dim + 
    row_off[:, None] * dim + 
    col_off[None, :]
  )

  mask = ((group_pos * BLOCK_SIZE_M + row_off)[:, None] < rows) & (col_off[None, :] < dim)

  dOut_tile = tl.load(dOut_idx, mask=mask, other=0.0).to(tl.float32)
  out_tile = tl.load(out_idx, mask=mask, other=0.0).to(tl.float32)

  dot = tl.sum(dOut_tile * out_tile, axis=1)
  dx_tile = out_tile * (dOut_tile - dot[:, None])

  dx_idx = (
    dx + 
    group_pos * BLOCK_SIZE_M * dim + 
    row_off[:, None] * dim + 
    col_off[None, :]
  )

  tl.store(dx_idx, dx_tile, mask=mask)




@triton.autotune(
  configs=[triton.Config(
            {"BLOCK_SIZE_M": BM}, 
            num_warps=w)
            for BM in [1, 2, 4]
            for w in [1, 2, 4, 8]
          ],
  key=["length" , "BLOCK_SIZE_N"],
  cache_results=True,
)
@triton.jit
def _softmax_fwd(
   x,
   out,
   length,
   BLOCK_SIZE_M: tl.constexpr,
   BLOCK_SIZE_N: tl.constexpr,
   DIM, 
   num_warps: tl.constexpr,
):
  group_pos = tl.program_id(0)
  row_offs = tl.arange(0, BLOCK_SIZE_M)
  col_offs = tl.arange(0, BLOCK_SIZE_N)

  x_idx = (
    x + 
    group_pos * BLOCK_SIZE_M * DIM +
    row_offs[:, None] * DIM + 
    col_offs[None, :]
  )

  out_idx = (
    out + 
    group_pos * BLOCK_SIZE_M * DIM +
    row_offs[:, None] * DIM + 
    col_offs[None, :]
  )

  rows = group_pos * BLOCK_SIZE_M + row_offs
  mask = (rows[:, None] < length) & (col_offs[None, :] < DIM)
  x_tile = tl.load(x_idx, mask=mask, other=float("-inf")).to(tl.float32)

  numer = tl.exp(x_tile - tl.max(x_tile, axis=1)[:, None])
  denom = tl.sum(numer, axis=1)

  softmax = numer/denom[:, None]

  tl.store(out_idx, softmax, mask=mask)

   
class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x): #function expects a tensor of shape (ROWS, DIM)
      assert x.ndim == 2  #enforce (ROW, DIM)
      assert x.is_contiguous()

      rows, dim = x.shape
      out = torch.empty_like(x)

      grid = lambda meta: (((rows + meta["BLOCK_SIZE_M"] - 1)//meta["BLOCK_SIZE_M"]), 1, 1)
      BLOCK_SIZE_N = triton.next_power_of_2(dim)

      _softmax_fwd[grid](
        x=x,
        out=out,
        length=rows,
        DIM=dim, 
        BLOCK_SIZE_N=BLOCK_SIZE_N,
      )
      ctx.save_for_backward(out)

      return out

    @staticmethod
    def backward(ctx, dOut):
      rows, dim = dOut.shape
      dx = torch.empty_like(dOut)
      BLOCK_SIZE_N = triton.next_power_of_2(dim)
      
      (out, ) = ctx.saved_tensors
      grid = lambda meta: (((rows + meta["BLOCK_SIZE_M"] - 1)//meta["BLOCK_SIZE_M"]), 1, 1)

      _softmax_bwd[grid](
        dOut=dOut,
        rows=rows,
        dim=dim,
        out=out,
        dx=dx,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
      )

      return dx
    



def test_softmax():
  #initialize softmax layers. we don't initialize a softmax instance for triton. rather, we just call the class. this is because its a static method and .apply integrates with torch.autograd
  x = torch.randn(1005, 32, device='cuda', requires_grad=True)

  # forward
  y_triton = Softmax.apply(x)
  y_torch = torch.softmax(x, dim=-1)
  print("Forward match:", torch.allclose(y_triton, y_torch, atol=1e-5))

  # backward
  dy = torch.randn_like(x)
  y_triton.backward(dy)
  grad_triton = x.grad.clone()

  x.grad = None
  y_torch = torch.softmax(x, dim=-1)
  y_torch.backward(dy)
  grad_torch = x.grad.clone()
  print("Backward match:", torch.allclose(grad_triton, grad_torch, atol=1e-5))
  print("Max diff:", (grad_triton - grad_torch).abs().max().item())


