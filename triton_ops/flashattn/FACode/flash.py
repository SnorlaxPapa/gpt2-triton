import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver
  
from forward import _forward_q
from backward import _preprocess, _backward_dq_outer, _backward_dkdv_outer

class Attention(torch.autograd.Function):

  @staticmethod
  def forward(ctx, q, k, v, sqrt_dk, causal):
    """
    input shape (B, num_head, seq_len, head_dim)
    launch cdiv(seq_len, BLOCK_SIZE_M) * num_head programs to compute Qi * KjVj
    output new P@V which has the same shape as (B, num_head, seq_len, head_dim)
    """
    #get our necessary dimensions and check
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_V == HEAD_DIM_K

    B, NUM_HEAD, SEQ_LEN, HEAD_DIM = q.shape

    #define our grid (query_group, num_heads)
    grid = lambda meta: (
                          triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_M"]),
                          B * NUM_HEAD,
                          1
                        )

    out = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    #Determine the stage needed. 1 for non-causal, 3 for causal
    stage = 3 if causal else 1

    #launch our kernel
    _forward_q[grid](
        q,
        k,
        v,
        out,
        M,
        sqrt_dk,
        SEQ_LEN,
        HEAD_DIM_Q,
        NUM_HEAD,
        stage=stage,
    )

  
    ctx.save_for_backward(q, k, v, out, M)
    ctx.sqrt_dk = sqrt_dk
    ctx.HEAD_DIM = HEAD_DIM_Q
    ctx.causal = causal

    return out

  @staticmethod
  def backward(ctx, dO):
    """
    input dy (B, H, N, C)
    flow: initialize my dd vector size (B, H, N, ) one value per row 
    launch a preprocessing grid with BLOCK_SIZE_M to compute dd
    """
    B, H, N, C = dO.shape
    assert C == ctx.HEAD_DIM

    q, k, v, out, M = ctx.saved_tensors
    sqrt_dk = ctx.sqrt_dk
    causal = ctx.causal
  
    D = torch.empty((B, H, N), device=dO.device, dtype=torch.float32)
    grid = lambda meta: ((N+meta["BLOCK_SIZE_M"] - 1)//meta["BLOCK_SIZE_M"], B*H, 1)

    _preprocess[grid](
      out,
      dO,
      D,   
      N,
      C,
    )

    dq = torch.empty_like(q)
    dv = torch.empty_like(v)
    dk = torch.empty_like(k)

    grid_dkdv = lambda meta: ((N+meta["BLOCK_SIZE_M"] - 1)//meta["BLOCK_SIZE_M"], B*H, 1)
    _backward_dkdv_outer[grid_dkdv](
      q=q,
      k=k,
      v=v,
      N=N,
      dk=dk,
      dv=dv,
      dq=dq,
      C=C,
      sqrt_dk=sqrt_dk,
      out=out,
      dO=dO,
      M=M,
      D=D,
      causal=causal,
    )

    grid_q = lambda meta: ((N+meta["BLOCK_SIZE_M2"]-1)//meta["BLOCK_SIZE_M2"], B*H, 1)
    _backward_dq_outer[grid_q](
      q=q,
      k=k,
      v=v,
      N=N,
      dq=dq,
      C=C,
      sqrt_dk=sqrt_dk,
      dO=dO,
      M=M,
      D=D,
      causal=causal,
    )

    return dq, dk, dv, None, None
    





def check_attention():
  B, H, C, N = 4, 8, 31, 16
  q = torch.randn((B, H, C, N), device="cuda", dtype=torch.float16, requires_grad=True) #B, H, SEQ, DIM
  k = torch.randn((B, H, C, N), device="cuda", dtype=torch.float16, requires_grad=True) #B, H, SEQ, DIM
  v = torch.randn((B, H, C, N), device="cuda", dtype=torch.float16, requires_grad=True) #B, H, SEQ, DIM
  scale = q.shape[-1] ** -0.5 

  y_torch = nn.functional.scaled_dot_product_attention(
      q,
      k,
      v,
      is_causal=False,
      scale=scale,
  )
  y_triton = Attention.apply(q, k, v, scale, False)

  #forward
  torch.allclose(y_torch, y_triton)
  print("max difference:", (y_torch - y_triton).abs().max().item())

  #backward
  fake_gradient = torch.randn_like(y_torch) 
  y_torch.backward(fake_gradient)
  torch_dq = q.grad.clone()
  torch_dk = k.grad.clone()
  torch_dv = v.grad.clone()

  q.grad = None
  k.grad = None
  v.grad = None

  y_triton.backward(fake_gradient)
  triton_dq = q.grad.clone()
  triton_dk = k.grad.clone()
  triton_dv = v.grad.clone()

  print("Backward match:", torch.allclose(torch_dq, triton_dq, atol=1e-2, rtol=1e-2))
  print("dq max difference:", (torch_dq - triton_dq).abs().max().item())
  print("Relative diff: ", ((torch_dq - triton_dq).abs().max() / torch_dq.abs().max()).item())
  print("Backward match:", torch.allclose(torch_dk, triton_dk, atol=1e-2, rtol=1e-2))
  print("dk max difference:", (torch_dk - triton_dk).abs().max().item())
  print("Relative diff: ", ((torch_dk - triton_dk).abs().max() / torch_dk.abs().max()).item())
  print("Backward match:", torch.allclose(torch_dv, triton_dv, atol=1e-2, rtol=1e-2))
  print("dv max difference:", (torch_dv - triton_dv).abs().max().item())
  print("Relative diff: ", ((torch_dv - triton_dv).abs().max() / torch_dv.abs().max()).item())

